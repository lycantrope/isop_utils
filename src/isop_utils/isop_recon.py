"""
isop_recon.py  --  ISOP 顕微鏡 (三上研) 生データの読み込みと 3D 再構築の Python 実装

LabVIEW プログラム
  3Danalysis240909 cropcalibration with real-time waveforms.vi
  reconstruct 3D oblique image without config-2 (SubVI).vi  (stack method = "0pad to z")
の処理をブロック図の解析と、実際の .3DU16 出力との照合 (t=0, 2ch, 線形相関 0.97) から再現したもの。

処理の流れ (1 ボリューム・1 チャンネル):
  1. img-chN.bin から Nv=8 フレーム (300 x 2304, uint16 BE) を読む。
  2. 各フレームを Nzf=16 本の帯 (XY フレーム, 幅 Nx=141 px) に切り取る。
     帯 k (k = frame*16 + strip) の切り取り開始 x は cropcalib.csv 第2列 (= cropparameters.csv 第2列 + 50)。
     帯内の歪み (傾き r = y/x ≈ 7) は、行 y ごとに開始位置を x0 + y / r に整数丸めして
     幅 Nx を切り出すことで補正する (create3Dimg3.vi の "x correct fast" 経路。センサ外は 100 で埋める)。
     cropcalib.csv 第3列 (y) は create3Dimg3 では使われない (コメント「y方向の補正は校正データを使わずにマニュアル補正」)。
     輝度補正:  I' = (I - 100) * (1 + c) + 100   (c = 第4列)。
     ch2 はカメラ画像を左右反転 (reverse h 2) してから同じ座標で切り取る。
  3. [rot 方式 = LabVIEW の "0pad to x and rot"] 各 y について、帯 (反転) を k*(pz/pxy)*sinθ ずつずらして
     並べた 2D 画像 (スライス x 幅) をスライス軸方向に伸ばし、θ=45° 回転して帯状領域を切り出す
     (reconstruct_volume_rot を参照)。出力は 99 x 297 x 510、0.4095 µm 等方。
     [roll 方式 = 旧近似] 幅方向を sinθ 倍に縮小して z とし、X へ 3.203 voxel 間隔で並べ、行を循環シフト (互換用)。
  4. y 方向は y shift (imgcalib.txt 6行目, -3) の分だけ行を削って 297 行にする。ch2 はその分だけ上にずらす (ch1/ch2 の位置合わせ)。
注意: 2026-11 に .3DU16 の読み方を修正した。.3DU16 は 52 B のグローバルヘッダの後、時点ごとに 16 B のヘッダ (C,Z,Y,X) +
データが並ぶ (Volumes3DU16 を参照)。以前はこの 16 B を飛ばしていなかったため t=0 で 88 ボクセルぶん行方向にずれた像と
比較しており、「LabVIEW 出力には -88 px の回り込みがある」と誤解していた (imaq_shift=-87 はその補正だった)。正しく読むと
LabVIEW 出力に回り込みは無く、素直な配置 (imaq_shift=0) で一致する: ch1 t=1 で NCC 0.992、t=0 で 0.946、ch2 t=0 で 0.947。
出力配列の軸順は [t][ch][z=99][y=297][x=510]、ボクセルは 0.4095 µm 等方。

速度: rot 方式は 1 ボリューム 1ch で約 2 秒 (297 断面の回転)。既定の fast 方式は同じ写像を
疎行列 (build_rot_operator, 初回のみ構築してキャッシュ) にして行列積 1 回で計算し、約 0.1 秒。
rot との一致 0.998、LabVIEW 出力との一致は rot と同じ (ch1 t=0 で 0.985)。

使い方:
  python isop_recon.py <データフォルダ> [--ch 1] [--t 0] [--compare] [--method fast|rot|roll]
  python から:  V = isop_recon.reconstruct(folder, ch=1, t=0)   # (99, 297, 510) float32
"""

import argparse
import csv
import os
import struct
import sys
from pathlib import Path
from typing import Final

import numpy as np
from scipy.ndimage import map_coordinates

# ---------- ファイル読み込み ----------
_ZIP_CACHE = {}
ZIP_MAX_BYTES = int(
    4e9
)  # .bin.zip を丸ごとメモリに展開してよい上限 (これ以上は unzip か isop_pack.py を使う)


def _open_bin(path):
    """img-chN.bin を bytes 風の uint8 配列で返す。
    .bin が無く .bin.zip (または同名 .zip) があれば、zip の中身を一度だけ全部展開してメモリに置く
    (132 MB で約 0.5 秒。zip は途中から読めないので、丸ごと展開して以後はキャッシュを使う)。"""
    if os.path.exists(path):
        return np.memmap(path, dtype=np.uint8, mode="r")

    for zp in (path + ".zip", os.path.splitext(path)[0] + ".zip"):
        if not os.path.exists(zp):
            continue
        key = (zp, os.path.getmtime(zp))
        if key in _ZIP_CACHE:
            return _ZIP_CACHE[key]

        import zipfile

        with zipfile.ZipFile(zp) as zf:
            names = [
                n
                for n in zf.namelist()
                if n.endswith(".bin") and not n.startswith("__MACOSX")
            ]
            if not names:
                raise FileNotFoundError(f"{zp} の中に .bin がありません")
            usize = zf.getinfo(names[0]).file_size
            if usize > ZIP_MAX_BYTES:
                raise MemoryError(
                    f"{zp} の展開後サイズ {usize / 1e9:.1f} GB は上限 {ZIP_MAX_BYTES / 1e9:.0f} GB を超えています。"
                    " zip は途中から読めないため、大きなファイルは unzip して .bin のまま置くか、"
                    " isop_pack.py でボリューム単位の圧縮形式に変換してください (isop_recon.ZIP_MAX_BYTES で上限変更可)"
                )
            _ZIP_CACHE.clear()  # 同時に抱えるのは 1 ファイルだけ
            _ZIP_CACHE[key] = np.frombuffer(zf.read(names[0]), dtype=np.uint8)
        return _ZIP_CACHE[key]
    raise FileNotFoundError(f"{path} (または .zip) が見つかりません")


def read_bin_frames(path, indices=None):
    """img-chN.bin (または img-chN.bin.zip) から指定フレーム (0 始まり) を float32 配列 (n, ny, nx) で返す。"""
    m = _open_bin(path)
    size = m.size
    ny, nx = struct.unpack(">ii", m[:8].tobytes())
    fb = 8 + ny * nx * 2
    nframes = size // fb
    if indices is None:
        indices = range(nframes)
    out = np.empty((len(indices), ny, nx), np.float32)
    for i, k in enumerate(indices):
        hdr = struct.unpack(">ii", m[k * fb : k * fb + 8].tobytes())
        if hdr != (ny, nx):
            raise ValueError(f"frame {k}: header {hdr} != ({ny},{nx})  (ファイル破損?)")
        out[i] = np.frombuffer(
            m[k * fb + 8 : (k + 1) * fb].tobytes(),
            dtype=">u2",
        ).reshape(ny, nx)
    return out, nframes


def read_timestamp_header(path):
    with open(path) as fh:
        v = fh.read().split("\t")
    keys = ["Nv", "Nzf", "Nx", "r", "g2_period_us", "dead_time_us"]
    return dict(zip(keys, map(float, v[:6])))


def read_imgcalib(path):
    with open(path) as fd:
        data = fd.read()
    v = [float(x) for x in data.split()]
    return dict(pxy=v[0], pz=v[1], a=v[2], b=v[3], ofst=v[4], yshift=v[5])


_CSV_CACHE = {}


def _read_csv_cached(path):
    """cropparameters.csv は時点数 x 256 行 (5000 時点で 50 MB) あるので、一度読んだらメモリに置く。"""
    key = (os.path.abspath(path), os.path.getmtime(path))
    if key not in _CSV_CACHE:
        _CSV_CACHE.clear()
        _CSV_CACHE[key] = np.loadtxt(path, delimiter=",", dtype=np.float64, ndmin=2)
    return _CSV_CACHE[key]


def find_calib_folder(folder):
    """同じ親フォルダにある、同じ日付 (先頭 8 桁) で名前に 'calib' を含むフォルダのうち、
    cropcalib.csv を持つものを返す (複数あればデータの直前に撮ったもの)。見つからなければ None。"""
    folder = os.path.normpath(folder)
    name = os.path.basename(folder)
    parent = os.path.dirname(folder) or "."
    date = name[:8]
    cands = []
    for d in sorted(os.listdir(parent)):
        full = os.path.join(parent, d)
        if (
            d != name
            and os.path.isdir(full)
            and "calib" in d.lower()
            and d[:8] == date
            and os.path.exists(os.path.join(full, "cropcalib.csv"))
        ):
            cands.append(d)
    if not cands:
        return None
    before = [d for d in cands if d[:15] <= name[:15]]  # YYYYMMDD-HHMMSS で比較
    return os.path.join(parent, (before or cands)[-1])


def resolve_calib(folder, calib=None):
    """切り取りパラメータの所在を決める。戻り値 (kind, path):
    kind = 'cropparameters' (データフォルダ内, LabVIEW 書き出し), 'cropcalib' (cropcalib.csv の場所)。"""
    folder = Path(folder)
    if isinstance(calib, (str, os.PathLike)):  # 明示された校正フォルダが最優先
        p = Path(calib) / "cropcalib.csv"
        if not p.is_file():
            raise FileNotFoundError(
                f"校正フォルダ {calib} に cropcalib.csv がありません"
            )
        return "cropcalib", os.fspath(p)

    for p_type in ("cropparameters", "cropcalib"):
        p = folder / f"{p_type}.csv"
        if p.is_file():
            return p_type, os.fspath(p)

    auto = find_calib_folder(folder)
    if not auto:
        raise FileNotFoundError(
            f"{folder}: cropparameters.csv も cropcalib.csv も無く、同じ日の校正フォルダ (…calib) も見つかりません。"
            " calib= で校正フォルダを指定してください"
        )
    return "cropcalib", os.path.join(auto, "cropcalib.csv")


def read_crop_params(folder, nslices, t, ch, calib=None):
    """切り取りパラメータ (nslices, 4) [r, x0_cropcalib, dy, c] を返す。
    優先順: 明示された calib フォルダの cropcalib.csv (256 行, 全時点共通)
          > データフォルダの cropparameters.csv (時点ごと 256 行, x0 が -50 されているので戻す)
          > データフォルダの cropcalib.csv
          > 同じ日の校正フォルダを自動で探す (find_calib_folder)。
    cropparameters.csv と cropcalib.csv は y 列 (未使用) 以外同じ値なので、どちらを使っても結果は同じ。"""
    kind, p = resolve_calib(folder, calib)
    rows = _read_csv_cached(p)
    if kind == "cropparameters":
        blk = rows[(t * 2 + (ch - 1)) * nslices : (t * 2 + ch) * nslices].copy()
        if blk.shape[0] != nslices:
            raise IndexError(
                f"cropparameters.csv に時点 {t} の行がありません (行数 {rows.shape[0]})"
            )
        blk[:, 1] += 50.0
        return blk
    return rows[(ch - 1) * nslices : ch * nslices].copy()


def find_file(folder, name, calib=None):
    """データフォルダに name が無ければ校正フォルダ (指定 or 自動探索) のものを使う (imgcalib.txt など)。"""
    p = os.path.join(folder, name)
    if os.path.exists(p):
        return p
    for d in ([calib] if calib else []) + [find_calib_folder(folder) or ""]:
        if d and os.path.exists(os.path.join(d, name)):
            return os.path.join(d, name)
    raise FileNotFoundError(f"{name} が {folder} にも校正フォルダにもありません")


class Volumes3DU16:
    """.3DU16 の読み手。ファイル構造 (big-endian):
        グローバルヘッダ 52 B: int32 ndim(=3), double x3 (ボクセルサイズ um), int32 5, int32 T, C, Z, Y, X
        その後、ボリューム (時点) ごとに  16 B ヘッダ (uint32 x4 = C, Z, Y, X) + データ (C*Z*Y*X の uint16, 順序 [c][z][y][x])
    時点ごとに 16 B のヘッダが挟まるので、全体を 1 つの配列として memmap はできない。v[t] で (C, Z, Y, X) の memmap を返す。
    (以前の実装はこの 16 B を無視して先頭から連続配列として読んでいたため、t=0 で 88 ボクセル、t=1 で 80 ボクセル…と
     行方向にずれた像になっていた。「IMAQ Rotate 固有の -88 px の回り込み」と解釈していたのはこの読み違いだった。)"""

    GLOBAL_HEADER: Final[int] = 52
    VOL_HEADER: Final[int] = 16

    def __init__(self, path):
        self.path = path
        with open(path, "rb") as fd:
            h = fd.read(Volumes3DU16.GLOBAL_HEADER)
        self.ndim = struct.unpack(">i", h[:4])[0]
        self.voxel = struct.unpack(">3d", h[4:28])
        _, T, C, Z, Y, X = struct.unpack(">6i", h[28:52])
        self.T, self.C, self.Z, self.Y, self.X = T, C, Z, Y, X
        self.shape = (T, C, Z, Y, X)
        self.vol_bytes = C * Z * Y * X * 2
        expected = Volumes3DU16.GLOBAL_HEADER + T * (
            Volumes3DU16.VOL_HEADER + self.vol_bytes
        )
        size = os.path.getsize(path)
        if size != expected:
            raise ValueError(
                f"{path}: サイズ {size} がヘッダから期待される {expected} と一致しません"
            )
        self._m = np.memmap(path, dtype=np.uint8, mode="r")

    def header(self, t):
        o = self.GLOBAL_HEADER + t * (self.VOL_HEADER + self.vol_bytes)
        return struct.unpack(">4I", self._m[o : o + self.VOL_HEADER].tobytes())

    def __len__(self):
        return self.T

    def __getitem__(self, t):
        """v[t] -> (C, Z, Y, X) big-endian uint16 の memmap。v[t, c] も可。"""
        if isinstance(t, tuple):
            t, rest = t[0], t[1:]
        else:
            rest = ()
        if not (0 <= t < self.T):
            raise IndexError(t)
        o = (
            self.GLOBAL_HEADER
            + t * (self.VOL_HEADER + self.vol_bytes)
            + self.VOL_HEADER
        )
        v = np.memmap(
            self.path,
            dtype=">u2",
            mode="r",
            offset=o,
            shape=(self.C, self.Z, self.Y, self.X),
        )
        return v[rest] if rest else v


def read_3du16(path):
    """.3DU16 を開く。戻り値 (Volumes3DU16, voxel)。v[t, c] で (Z, Y, X) の配列が取れる (以前と同じ添字で使える)。"""
    v = Volumes3DU16(path)
    return v, v.voxel


# ---------- 再構築 ----------
def extract_strips(
    frames, cp, nx=141, dx=0.5, intensity=True, mirror=False, integer_shift=True
):
    """フレーム群から補正済みの帯を取り出す。戻り値 (nslices, ny, nx)。
    mirror=True はカメラ画像を左右反転してから切り取る (LabVIEW の "reverse h 2"。ch2 で使用)。"""
    nv, ny, _ = frames.shape
    nslices = cp.shape[0]
    nzf = nslices // nv
    yy = np.arange(ny, dtype=np.float32)[:, None]
    ss = np.arange(nx, dtype=np.float32)[None, :]
    S = np.zeros((nslices, ny, nx), np.float32)
    for k in range(nslices):
        f = k // nzf
        r, x0, dy, c = cp[k]
        if integer_shift:
            # LabVIEW (create3Dimg3 / extract single slice img) は行ごとに開始位置を
            # 整数画素に丸めて Array Subset で切り出す ("x correct fast" 経路)
            xs = np.round(x0 + dx + yy / r) + ss
        else:
            xs = x0 + dx + ss + yy / r  # サブピクセル補間版
        if mirror:
            xs = (frames.shape[2] - 1) - xs
        if integer_shift:
            # 整数座標なので直接参照 (map_coordinates order=1, mode='constant' と同値)
            W = frames.shape[2]
            xi = xs.astype(int)
            img = frames[f][np.arange(ny)[:, None], np.clip(xi, 0, W - 1)]
            img = np.where((xi >= 0) & (xi < W), img, 0.0)
        else:
            ys = np.broadcast_to(yy, xs.shape)
            img = map_coordinates(
                frames[f], [ys, xs], order=1, mode="constant", cval=0.0
            )
        if intensity:
            img = np.where(img > 0, (img - 100.0) * (1.0 + c) + 100.0, 0.0)
        S[k] = img
    return S


def reconstruct_volume(
    S, pxy=0.4095, pz=1.311666, theta_deg=45.0, yshift=-3, roll0=12, delta=0.5, pad=None
):
    """帯 (nslices, ny, nx) から [z][y][x] ボリュームを作る (LabVIEW "0pad to z" 方式)。"""
    nslices, ny, nx = S.shape
    sin_t = np.sin(np.radians(theta_deg))
    nz = int(np.floor(nx * sin_t))  # 141 -> 99
    if pad is None:
        pad = int(np.round(nx * sin_t))  # 100
    step = pz / pxy  # 3.203 voxel / slice
    ndata = int(np.round(nslices * step))  # 410
    nxout = ndata + pad  # 510
    ny_out = ny - abs(int(yshift))  # 297
    y0 = 0 if yshift <= 0 else int(yshift)
    # 幅方向を sin(theta) 倍に縮小 -> z
    zs = np.arange(nz) / sin_t
    Sz = np.zeros((nslices, nz, ny_out), np.float32)
    for k in range(nslices):
        Sz[k] = map_coordinates(
            S[k].T,
            [
                np.broadcast_to(zs[:, None], (nz, ny)),
                np.broadcast_to(np.arange(ny)[None, :], (nz, ny)),
            ],
            order=1,
            mode="nearest",
        )[:, y0 : y0 + ny_out]
    # X 方向へ線形補間しながら積層
    X = np.arange(ndata, dtype=np.float32)
    kf = (X - delta) / step
    k0 = np.floor(kf).astype(int)
    valid = (k0 >= 0) & (k0 <= nslices - 1)
    k0c = np.clip(k0, 0, nslices - 2)
    w = (kf - k0c).astype(np.float32)
    V = np.zeros((nz, ny_out, nxout), np.float32)
    for z in range(nz):
        a = Sz[k0c, z, :]
        b = Sz[k0c + 1, z, :]
        row = np.zeros((ny_out, nxout), np.float32)
        row[:, :ndata] = np.where(
            valid[None, :], ((1 - w)[:, None] * a + w[:, None] * b).T, 0
        )
        V[z] = np.roll(row, roll0 - z, axis=1)  # 階段ずらし (循環シフト)
    return V


def reconstruct_volume_rot(
    S, pxy=0.4095, pz=1.311666, theta_deg=45.0, yshift=-3, imaq_shift=0, phase=0.0
):
    """LabVIEW の実際の経路 "0pad to x and rot" を再現する (推奨)。
    1. 帯 k を幅方向 (s) に反転し、k * (pz/pxy) * sin(theta) 画素だけ右へずらして並べる (整数分はゼロ挿入、端数は線形補間)。
    2. スライス軸を (pz/pxy)*sin(theta) 倍に伸ばす (IMAQ Resample 相当)。128 枚 -> 290 行。
    3. できた 2D 画像 (290 x 431) を theta 回転 (IMAQ Rotate, 出力は外接 510 x 510)。
    4. 行 mi = ny*cos(theta) から ml = nx*sin(theta) - ny*cos(theta) 行を切り出す (fBox の式そのまま)。
    5. IMAQ Rotate 固有の横ずれ (このデータでは -87 画素, 循環) を加える。
    出力 [z][y][x]。LabVIEW 出力との線形相関 0.988 (ch1, t=0)。"""
    from scipy.ndimage import rotate, zoom

    nslices, ny, nx = S.shape
    th = np.radians(theta_deg)
    fe = pz / pxy * np.sin(th)
    shifts = fe * (np.arange(nslices) + phase)
    nint = np.floor(shifts).astype(int)
    frac = shifts - nint  # phase: 帯の並べ位置の位相 (スライス単位, 0〜0.5)
    L = nx + int(np.ceil(fe * nslices))
    H = int(round(fe * nslices))
    mi = H * np.cos(th)
    ml = L * np.sin(th) - mi
    start = int(round(mi)) + 1
    nz = int(np.floor(ml))
    ny_out = ny - abs(int(yshift))
    y0 = 0 if yshift <= 0 else int(yshift)
    nxo = int(np.ceil((L + H) * np.sin(th)))  # 外接正方形 (510)
    V = np.zeros((nz, ny_out, nxo), np.float32)
    xs = np.arange(nx)
    for y in range(y0, y0 + ny_out):
        plane = S[:, y, ::-1]
        A = np.zeros((nslices, L), np.float32)
        for i in range(nslices):
            A[i, nint[i] : nint[i] + nx] = np.interp(
                xs - frac[i], xs, plane[i], left=0, right=0
            )
        B = zoom(A, (H / nslices, 1.0), order=1)[:H, :L]
        R = rotate(B, theta_deg, reshape=True, order=1, cval=0.0)
        V[:, y - y0, :] = np.roll(R[start : start + nz, :nxo], imaq_shift, axis=1)
    return V


# ---------- 高速版 (疎行列) ----------
_ROT_CACHE = {}


def build_rot_operator(
    nslices, nx, pxy=0.4095, pz=1.311666, theta_deg=45.0, imaq_shift=0
):
    """reconstruct_volume_rot の 1 断面分の処理 (帯を並べる -> Z 伸長 -> 回転 -> 切り出し) は
    線形写像で、どの y 断面でも同じ。そこで座標画像 (全 1、スライス番号、帯内位置) をその処理に通して
    「出力の各ボクセルが帯のどの点をどの重みで参照するか」を数値的に求め、疎行列 W にまとめる。
        出力 (nz*nxo, ny) = W ((nz*nxo) x (nslices*nx)) @ 帯 (s 反転, (nslices*nx, ny))
    W の各行は双一次補間 4 点の重み x 被覆率 (端のゼロ埋めの効果を再現)。同じ設定では結果をキャッシュする。
    戻り値 (W, nz, nxo)。"""
    key = (
        int(nslices),
        int(nx),
        round(float(pxy), 6),
        round(float(pz), 6),
        float(theta_deg),
        int(imaq_shift),
    )
    if key in _ROT_CACHE:
        return _ROT_CACHE[key]
    import scipy.sparse as sp

    def run(plane):
        # reconstruct_volume_rot は S[:, y, ::-1] を使うので、plane を「反転済みの帯」として渡す
        S1 = np.ascontiguousarray(plane[:, None, ::-1]).astype(np.float32)
        return reconstruct_volume_rot(
            S1, pxy, pz, theta_deg, yshift=0, imaq_shift=imaq_shift
        )[:, 0, :].astype(np.float64)

    ii, ss = np.meshgrid(np.arange(nslices), np.arange(nx), indexing="ij")
    cov = run(np.ones((nslices, nx)))  # 被覆率 (内側は 1、端は <1、外は 0)
    ci = run(ii.astype(np.float64))  # 参照するスライス番号 x 被覆率
    cs = run(ss.astype(np.float64))  # 参照する帯内位置 x 被覆率
    nz, nxo = cov.shape
    oz, ox = np.nonzero(cov > 1e-3)
    c = cov[oz, ox]
    si = np.clip(ci[oz, ox] / c, 0, nslices - 1)
    sv = np.clip(cs[oz, ox] / c, 0, nx - 1)
    i0 = np.minimum(np.floor(si).astype(int), nslices - 2)
    fi = si - i0
    s0 = np.minimum(np.floor(sv).astype(int), nx - 2)
    fs = sv - s0
    rows, cols, vals = [], [], []
    for di, ds, w in (
        (0, 0, (1 - fi) * (1 - fs)),
        (1, 0, fi * (1 - fs)),
        (0, 1, (1 - fi) * fs),
        (1, 1, fi * fs),
    ):
        rows.append(oz * nxo + ox)
        cols.append((i0 + di) * nx + (s0 + ds))
        vals.append(w * c)
    W = sp.csr_matrix(
        (
            np.concatenate(vals).astype(np.float32),
            (np.concatenate(rows), np.concatenate(cols)),
        ),
        shape=(nz * nxo, nslices * nx),
    )
    _ROT_CACHE[key] = (W, nz, nxo)
    return W, nz, nxo


def reconstruct_volume_fast(
    S, pxy=0.4095, pz=1.311666, theta_deg=45.0, yshift=-3, imaq_shift=0
):
    """reconstruct_volume_rot と同じ幾何を疎行列 1 回の積で計算する (約 100 倍速, 一致 0.998)。"""
    nslices, ny, nx = S.shape
    W, nz, nxo = build_rot_operator(nslices, nx, pxy, pz, theta_deg, imaq_shift)
    ny_out = ny - abs(int(yshift))
    y0 = 0 if yshift <= 0 else int(yshift)
    S2 = np.ascontiguousarray(
        S[:, y0 : y0 + ny_out, ::-1].transpose(0, 2, 1), dtype=np.float32
    ).reshape(nslices * nx, ny_out)
    V = (W @ S2).reshape(nz, nxo, ny_out).transpose(0, 2, 1)
    return np.ascontiguousarray(V, dtype=np.float32)


def reconstruct(folder, ch=1, t=0, method="fast", imaq_shift=0, calib=None):
    """folder: 見たい動画のフォルダ (img-chN.bin)。calib: 校正撮影のフォルダ (cropcalib.csv, imgcalib.txt)。
    calib を省略すると、データフォルダ内の cropparameters.csv、無ければ同じ日の校正フォルダを自動で使う。
    imaq_shift: 出力の x 方向の循環シフト (通常 0)。以前は -87 を既定にしていたが、それは .3DU16 の読み違い
    (ボリュームごとの 16 B ヘッダを飛ばしていなかった) を打ち消すための値で、LabVIEW 出力に回り込みは無い。
    ch2 は imgcalib.txt の y shift (6 行目, -3) ぶんだけ上にずらしてから再構築する (LabVIEW と同じ。ch1 は末尾の行を落とすだけ)。"""
    # 先頭6値の設定情報は timestamp-ch1.csv にしか入っていない (ch2 側は別の値)
    ts = read_timestamp_header(find_file(folder, "timestamp-ch1.csv", calib))
    ic = read_imgcalib(find_file(folder, "imgcalib.txt", calib))
    nv, nzf = int(ts["Nv"]), int(ts["Nzf"])
    frames, nframes = read_bin_frames(
        os.path.join(folder, f"img-ch{ch}.bin"), range(t * nv, (t + 1) * nv)
    )
    cp = read_crop_params(folder, nv * nzf, t, ch, calib)
    S = extract_strips(frames, cp, nx=int(np.floor(ts["Nx"])), mirror=(ch == 2))
    vs = int(ic["yshift"])
    if (
        ch == 2 and vs != 0
    ):  # ch2 を v shift ぶん平行移動 (行 j <- j - vs; vs=-3 なら 3 行上へ)
        S2 = np.zeros_like(S)
        if vs < 0:
            S2[:, :vs, :] = S[:, -vs:, :]
        else:
            S2[:, vs:, :] = S[:, :-vs, :]
        S = S2
    if method == "fast":
        return reconstruct_volume_fast(
            S, pxy=ic["pxy"], pz=ic["pz"], yshift=ic["yshift"], imaq_shift=imaq_shift
        )
    if method == "rot":
        return reconstruct_volume_rot(
            S, pxy=ic["pxy"], pz=ic["pz"], yshift=ic["yshift"], imaq_shift=imaq_shift
        )
    return reconstruct_volume(S, pxy=ic["pxy"], pz=ic["pz"], yshift=ic["yshift"])


def reconstruct_u16(folder, ch, t, method="fast", imaq_shift=-87, calib=None):
    """reconstruct の uint16 版 (別プロセスでの先読み用。戻り値をそのままキャッシュに入れる)。"""
    v = reconstruct(folder, ch, t, method, imaq_shift, calib)
    return np.clip(v, 0, 65535).astype(np.uint16)


def ncc(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-9))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--ch", type=int, default=1)
    ap.add_argument("--t", type=int, default=0)
    ap.add_argument("--compare", action="store_true", help="同名の .3DU16 と比較する")
    ap.add_argument("--out", help="結果を .npy で保存する")
    ap.add_argument(
        "--calib",
        help="校正撮影のフォルダ (cropcalib.csv がある所)。省略時はデータフォルダ内の cropparameters.csv か、同じ日の …calib フォルダを自動で使う",
    )
    ap.add_argument(
        "--method",
        default="fast",
        choices=["fast", "rot", "roll"],
        help="fast: 回転方式の疎行列版 (既定, 0.1 秒/ボリューム), rot: 回転方式の逐次版 (参照実装, 2 秒), roll: 行シフト近似",
    )
    a = ap.parse_args()
    kind, p = resolve_calib(a.folder, a.calib)
    print("切り取りパラメータ:", p)
    V = reconstruct(a.folder, a.ch, a.t, a.method, calib=a.calib)
    print("volume", V.shape, "mean", V.mean())
    if a.out:
        np.save(a.out, V.astype(np.uint16))
    if a.compare:
        name = os.path.basename(os.path.normpath(a.folder))
        ref, voxel = read_3du16(os.path.join(a.folder, name + ".3DU16"))
        R = np.asarray(ref[a.t, a.ch - 1]).astype(np.float32)
        print(
            "ref",
            R.shape,
            "voxel",
            voxel,
            "NCC(lin) = %.4f  NCC(log) = %.4f  zero-mask agreement = %.4f"
            % (ncc(V, R), ncc(np.log1p(V), np.log1p(R)), ((V == 0) == (R == 0)).mean()),
        )
