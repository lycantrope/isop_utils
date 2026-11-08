"""
isop_viewer.py  --  img-ch1.bin / img-ch2.bin をその場で再構築して見るビューア

.3DU16 を作らずに、生データ (img-chN.bin) と校正 (cropcalib.csv / cropparameters.csv, imgcalib.txt,
timestamp-ch1.csv) から isop_recon.reconstruct() で 1 時点 1ch あたり約 0.1 秒で再構築し、napari で表示する。

フォルダは 2 種類ある:
  データフォルダ  = 見たい動画 (例 20261020-194904beads)。img-ch1.bin / img-ch2.bin がある。
  校正フォルダ    = その日の校正撮影 (例 20261020-172105beads-cropbulkcalib)。cropcalib.csv, imgcalib.txt がある。
開くのはデータフォルダ。校正は、データフォルダ内に LabVIEW が書いた cropparameters.csv があればそれ、
無ければ --calib (画面の「校正フォルダを選ぶ…」) で指定したもの、それも無ければ同じ日の「…calib」フォルダを自動で探す。
時点 (t) と z のスライダーで動かせ、ch1 をマゼンタ、ch2 を緑で重ねる。3D 表示にも切り替えられる。

使い方:
  python isop_viewer.py                                # 起動後にダイアログでフォルダを選ぶ (画面内のボタンで切り替え可)
  python isop_viewer.py <データフォルダ>                 # 全時点 (遅延読み込み: 開いた時点だけ再構築)
  python isop_viewer.py <データフォルダ> --preload       # 先に全時点を再構築してから開く (12 時点なら数秒)
  python isop_viewer.py <データフォルダ> --t 0 3         # 時点 0〜3 だけ
  python isop_viewer.py <データフォルダ> --backend mpl   # napari が無いときの簡易版 (matplotlib)
  python isop_viewer.py <データフォルダ> --save out.tif  # 表示せず全時点を tiff (TZCYX) に書き出す

操作 (napari):
  下のスライダー: t と z。矢印キーでも動く。
  左のレイヤー一覧でチャンネルの表示/非表示、コントラスト、不透明度を変えられる。左下の「2D/3D」ボタンで 3D 表示。
  右の「ISOP」パネル: データフォルダと校正フォルダの選択 (校正は自動検出した結果が表示され、手動で差し替え/自動に戻せる)、
  表示する時点範囲、先読み (--preload)、tiff 書き出し (--save)。
  右の「ヒストグラム」パネル: 表示中の時点の ch1/ch2 の輝度分布と飽和 (サチり) の有無。スライダーが止まってから更新。
  「ch2 の位置合わせ」: ch1/ch2 のずれをカメラ画像の h/v (px) で合わせる (再構築後は h が 45° 方向、v が y に動く)。
  「校正フォルダに保存」でその日の全データに共通の chshift.csv として保存 (推奨)、「データフォルダに保存」でそのデータだけ。
  開くときはデータフォルダ > 校正フォルダの順に chshift.csv を探して自動適用。tiff 書き出しにも反映。
  時系列スライダーを動かすと、前後の時点を裏で先に再構築する (prefetch) のでキャッシュが追いつく範囲ではなめらかに動く。
"""
import os, sys, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import isop_recon as recon


class IsopDataset:
    """フォルダを開き、時点ごと・チャンネルごとに再構築して返す (結果はメモリにキャッシュ)。"""

    def __init__(self, folder, method='fast', clean=True, max_cache_gb=2.0, calib=None):
        self.folder = folder
        self.method = method
        self.calib = calib                          # 校正撮影のフォルダ (None なら自動)
        self.calib_kind, self.calib_path = recon.resolve_calib(folder, calib)   # 無ければここで例外
        self.imaq_shift = 0                        # (旧 'clean' オプションは廃止: LabVIEW 出力に回り込みは無かった)
        self.max_cache_bytes = int(max_cache_gb * 1e9)   # 再構築結果のキャッシュ上限 (超えたら古いものから捨てる)
        ts = recon.read_timestamp_header(recon.find_file(folder, 'timestamp-ch1.csv', calib))
        self.ic = recon.read_imgcalib(recon.find_file(folder, 'imgcalib.txt', calib))
        self.nv = int(ts['Nv'])
        self.channels = [ch for ch in (1, 2) if os.path.exists(os.path.join(folder, f'img-ch{ch}.bin'))]
        _, nframes = recon.read_bin_frames(os.path.join(folder, 'img-ch1.bin'), [])
        self.nt = nframes // self.nv                      # 完全なボリュームの数
        from collections import OrderedDict
        import threading
        self._cache = OrderedDict()
        self._lock = threading.RLock()
        self._pending = set()
        self._pool = None
        v = self.get(self.channels[0], 0)
        self.shape = v.shape                               # (Z, Y, X)
        self.voxel = self.ic['pxy']

    def get(self, ch, t):
        key = (ch, t)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        v = recon.reconstruct(self.folder, ch, t, self.method, imaq_shift=self.imaq_shift, calib=self.calib)
        v = np.clip(v, 0, 65535).astype(np.uint16)
        with self._lock:
            self._cache[key] = v
            while len(self._cache) > 1 and sum(a.nbytes for a in self._cache.values()) > self.max_cache_bytes:
                self._cache.popitem(last=False)              # 最も古いものから捨てる (LRU)
        return v

    def put(self, ch, t, v):
        with self._lock:
            self._cache[(ch, t)] = v
            while len(self._cache) > 1 and sum(a.nbytes for a in self._cache.values()) > self.max_cache_bytes:
                self._cache.popitem(last=False)

    def prefetch(self, ts, chs=None, workers=6):
        """指定した時点を裏で先に再構築してキャッシュに入れる (スライダー操作をなめらかにする)。
        再構築は別プロセス (numpy/scipy が GIL を握るのでスレッドでは 2 倍程度しか速くならない)。"""
        from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
        chs = self.channels if chs is None else chs
        if self._pool is None:
            try:
                import multiprocessing as mp
                self._pool = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'))
                self._pool_kind = 'process'
            except Exception:
                self._pool = ThreadPoolExecutor(max_workers=workers)
                self._pool_kind = 'thread'
        for t in ts:
            if not (0 <= t < self.nt):
                continue
            for ch in chs:
                key = (ch, t)
                with self._lock:
                    if key in self._cache or key in self._pending:
                        continue
                    self._pending.add(key)
                try:
                    fut = self._pool.submit(recon.reconstruct_u16, self.folder, ch, t, self.method, self.imaq_shift, self.calib)
                except Exception:                      # プロセスプールが壊れたらスレッドに切り替える
                    self._pool = ThreadPoolExecutor(max_workers=workers); self._pool_kind = 'thread'
                    fut = self._pool.submit(recon.reconstruct_u16, self.folder, ch, t, self.method, self.imaq_shift, self.calib)
                fut.add_done_callback(lambda f, key=key: self._prefetch_done(key, f))

    def _prefetch_done(self, key, fut):
        try:
            v = fut.result()
            self.put(key[0], key[1], v)
        except Exception as e:
            print(f'prefetch {key} failed: {e}')
        finally:
            with self._lock:
                self._pending.discard(key)

    def close(self):
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None

    def preload(self, ts=None):
        import time
        ts = range(self.nt) if ts is None else ts
        need = len(ts) * len(self.channels) * int(np.prod(self.shape)) * 2
        if need > self.max_cache_bytes:
            print(f'注意: {len(ts)} 時点 x {len(self.channels)} ch = {need / 1e9:.1f} GB はキャッシュ上限 '
                  f'{self.max_cache_bytes / 1e9:.1f} GB を超えるので、古い時点から捨てられます (--max-cache-gb で変更)')
        t0 = time.time()
        for t in ts:
            for ch in self.channels:
                self.get(ch, t)
            print(f'\r再構築中 t={t}/{self.nt - 1}  ({time.time() - t0:.1f} 秒)', end='', flush=True)
        print()

    def stack(self, ch, ts):
        return np.stack([self.get(ch, t) for t in ts])


CH_COLOR = {1: 'magenta', 2: 'green'}


def contrast_limits(v, mode='auto'):
    """表示コントラストの初期値。'auto': 背景の下端〜99.99% (明るい点の 0.01% だけ白飛び)、'full': 0〜最大値 (白飛びなし)。"""
    if mode == 'full':
        return [0.0, float(max(v.max(), 1))]
    lo = float(np.percentile(v, 0.5)); hi = float(np.percentile(v, 99.99))
    return [lo, max(hi, lo + 1)]


class _LazyT:
    """時点軸だけ遅延する配列もどき。arr[t] で ds.get(ch, ts[t]) を返す (dask.array.from_array 用)。"""
    def __init__(self, ds, ch, ts):
        self.ds, self.ch, self.ts = ds, ch, ts
        self.shape = (len(ts),) + tuple(ds.shape)
        self.dtype = np.dtype(np.uint16)
        self.ndim = 4

    def __getitem__(self, idx):
        if not isinstance(idx, tuple):
            idx = (idx,)
        t, rest = idx[0], idx[1:]
        if isinstance(t, slice):
            out = np.stack([self.ds.get(self.ch, self.ts[i]) for i in range(*t.indices(self.shape[0]))])
            return out[(slice(None),) + rest]
        return self.ds.get(self.ch, self.ts[int(t)])[rest]


def lazy_array(ds, ch, ts):
    """napari 用: 開いた時点だけ再構築する dask 配列 (dask が無ければ全部先に作る)。
    from_delayed + stack だと時点数ぶんのタスクグラフを毎回最適化して 1 回 200 ms かかるので、
    from_array に時点 1 つ = 1 チャンクの配列もどきを渡す (1 回 7 ms)。"""
    try:
        import dask.array as da
    except ImportError:
        return ds.stack(ch, ts)
    return da.from_array(_LazyT(ds, ch, ts), chunks=(1,) + tuple(ds.shape), asarray=False, fancy=False, name=f'isop-ch{ch}-{id(ds)}')


REQUIRED_FILES = ('timestamp-ch1.csv', 'imgcalib.txt')


def check_folder(folder, calib=None):
    """ISOP のデータフォルダとして読めるか確認し、問題があれば理由を返す (問題なければ None)。
    校正ファイル (cropcalib.csv, imgcalib.txt) はデータフォルダに無くても、calib か同じ日の …calib フォルダにあればよい。"""
    if not folder or not os.path.isdir(folder):
        return 'フォルダが選ばれていません'
    if not any(os.path.exists(os.path.join(folder, f)) for f in ('img-ch1.bin', 'img-ch1.bin.zip', 'img-ch1.zip')):
        return 'img-ch1.bin がありません (動画のフォルダを選んでください)'
    missing = []
    for f in REQUIRED_FILES:
        try:
            recon.find_file(folder, f, calib)
        except FileNotFoundError:
            missing.append(f)
    try:
        recon.resolve_calib(folder, calib)
    except FileNotFoundError:
        missing.append('cropcalib.csv (校正フォルダ)')
    if missing:
        return ('次のファイルがデータフォルダにも校正フォルダにもありません: ' + ', '.join(missing)
                + '。「校正フォルダを選ぶ…」で校正撮影のフォルダを指定してください')
    return None


def choose_folder_qt(start=None):
    from qtpy.QtWidgets import QFileDialog
    d = QFileDialog.getExistingDirectory(None, 'ISOP データフォルダ (img-ch1.bin があるフォルダ) を選択', start or os.getcwd())
    return d or None


def choose_folder_tk(start=None):
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk(); root.withdraw()
    d = filedialog.askdirectory(title='ISOP データフォルダ (img-ch1.bin があるフォルダ) を選択', initialdir=start or os.getcwd())
    root.destroy()
    return d or None


def load_into_viewer(viewer, folder, opts, ts=None):
    """viewer のレイヤーを folder のデータに入れ替える。戻り値 (IsopDataset, ts)。"""
    import napari
    problem = check_folder(folder, opts.get('calib'))
    if problem:
        napari.utils.notifications.show_warning(f'{folder}: {problem}')
        print(f'{folder}: {problem}')
        return None, None
    ds = IsopDataset(folder, opts['method'], clean=opts['clean'], max_cache_gb=opts['max_cache_gb'], calib=opts.get('calib'))
    print(f'切り取りパラメータ: {ds.calib_path}')
    ts = list(range(ds.nt)) if ts is None else [t for t in ts if t < ds.nt]
    if opts['preload']:
        ds.preload(ts)
    viewer.layers.clear()
    for ch in ds.channels:
        arr = ds.stack(ch, ts) if opts['preload'] else lazy_array(ds, ch, ts)
        viewer.add_image(arr, name=f'ch{ch}', colormap=CH_COLOR[ch], blending='additive',
                         contrast_limits=contrast_limits(ds.get(ch, ts[0])),
                         scale=(1, ds.voxel, ds.voxel, ds.voxel), units=('pixel', 'um', 'um', 'um'), multiscale=False)
    viewer.dims.axis_labels = ('t', 'z', 'y', 'x')
    viewer.canvas.overlays.scale_bar.visible = True     # 単位はレイヤーの scale (um) から自動
    viewer.reset_view()
    viewer.title = f'ISOP {os.path.basename(os.path.normpath(folder))}'
    print(f'{folder}: {ds.nt} 時点, ch {ds.channels}, ボリューム {ds.shape} (z, y, x), ボクセル {ds.voxel} um, 表示 t={ts[0]}..{ts[-1]}')
    return ds, ts


CHSHIFT_FILE = 'chshift.csv'
COS45 = float(np.cos(np.pi / 4))


def camera_to_volume_shift(h, v, mirrored=True):
    """カメラ画像の平行移動 (h: 列方向, 右が正 / v: 行方向, 下が正; 単位 px) が再構築後にどこへ動くか。
    実測 (toy1): ch1 (反転なし) h=+6 -> (dz,dx)=(+4,-4), v=+6 -> dy=+6。ch2 は左右反転するので h の符号が逆。
    戻り値 (dz, dy, dx) ボクセル。"""
    sgn = -1.0 if mirrored else 1.0
    return sgn * h * COS45, float(v), -sgn * h * COS45


def volume_to_camera_shift(dz, dy, dx, mirrored=True):
    sgn = -1.0 if mirrored else 1.0
    return sgn * (dz - dx) / (2 * COS45), float(dy)


def calib_dir_of(folder, calib=None):
    """このデータに使う校正フォルダ (cropcalib.csv のある所)。見つからなければ None。"""
    try:
        kind, path = recon.resolve_calib(folder, calib)
    except FileNotFoundError:
        return None
    if kind == 'cropcalib':
        return os.path.dirname(os.path.abspath(path))
    return calib or recon.find_calib_folder(folder)


def read_chshift(folder, calib=None):
    """chshift.csv を読む。データフォルダにあればそれ、無ければ校正フォルダのもの (その日の全データに共通)。
    戻り値 ({ch: (dz, dy, dx)} ボクセル, 読んだファイルのパス or None)。"""
    import csv
    cands = [os.path.join(folder, CHSHIFT_FILE)]
    cd = calib_dir_of(folder, calib)
    if cd:
        cands.append(os.path.join(cd, CHSHIFT_FILE))
    for p in cands:
        if os.path.exists(p):
            out = {}
            for row in csv.DictReader(open(p)):
                out[int(row['ch'])] = (float(row['dz_voxel']), float(row['dy_voxel']), float(row['dx_voxel']))
            return out, p
    return {}, None


def write_chshift(folder, shifts):
    """{ch: (dz, dy, dx)} を chshift.csv に書く (ボクセル単位と、換算したカメラ画像の h/v も併記)。"""
    import csv
    p = os.path.join(folder, CHSHIFT_FILE)
    with open(p, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['ch', 'dz_voxel', 'dy_voxel', 'dx_voxel', 'h_camera_px', 'v_camera_px', 'note'])
        for ch, (dz, dy, dx) in sorted(shifts.items()):
            h, v = volume_to_camera_shift(dz, dy, dx, mirrored=(ch == 2))
            w.writerow([ch, f'{dz:.3f}', f'{dy:.3f}', f'{dx:.3f}', f'{h:.3f}', f'{v:.3f}',
                        'translation applied to this channel in the viewer (voxel = 0.4095 um; z,y,x of the reconstructed volume)'])
    return p


class IsopPanel:
    """napari 右側のパネル。データフォルダと校正フォルダの 2 つを選び、表示オプションを操作する。
    opts: method / clean / max_cache_gb / preload / calib。state: ds / folder / ts。"""

    def __init__(self, viewer, opts, state):
        from qtpy.QtWidgets import (QPushButton, QLabel, QWidget, QVBoxLayout, QHBoxLayout, QCheckBox,
                                    QSpinBox, QGroupBox, QScrollArea)
        self.viewer, self.opts, self.state = viewer, opts, state
        opts.setdefault('calib', None)
        w = QWidget(); lay = QVBoxLayout(w)

        # --- データフォルダ ---
        g1 = QGroupBox('データ (見たい動画のフォルダ)'); l1 = QVBoxLayout(g1)
        self.btn_data = QPushButton('データフォルダを開く…'); self.btn_data.clicked.connect(lambda: self.open_data())
        self.lbl_data = QLabel('(未選択)'); self.lbl_data.setWordWrap(True)
        l1.addWidget(self.btn_data); l1.addWidget(self.lbl_data)

        # --- 校正フォルダ ---
        g2 = QGroupBox('校正 (その日の …cropbulkcalib フォルダ)'); l2 = QVBoxLayout(g2)
        row = QHBoxLayout()
        self.btn_calib = QPushButton('校正フォルダを選ぶ…'); self.btn_calib.clicked.connect(lambda: self.open_calib())
        self.btn_auto = QPushButton('自動に戻す'); self.btn_auto.clicked.connect(self.auto_calib)
        row.addWidget(self.btn_calib); row.addWidget(self.btn_auto)
        self.lbl_calib = QLabel('(自動)'); self.lbl_calib.setWordWrap(True)
        l2.addLayout(row); l2.addWidget(self.lbl_calib)

        # --- 表示オプション ---
        g3 = QGroupBox('表示オプション'); l3 = QVBoxLayout(g3)
        row2 = QHBoxLayout(); row2.addWidget(QLabel('時点')); self.sp_t0 = QSpinBox(); self.sp_t1 = QSpinBox()
        for sp in (self.sp_t0, self.sp_t1):
            sp.setRange(0, 0); sp.setEnabled(False)
        row2.addWidget(self.sp_t0); row2.addWidget(QLabel('〜')); row2.addWidget(self.sp_t1)
        self.btn_t = QPushButton('適用'); self.btn_t.clicked.connect(self.apply_t); row2.addWidget(self.btn_t)
        self.btn_preload = QPushButton('表示中の時点を先に全部再構築 (--preload)'); self.btn_preload.clicked.connect(self.preload)
        self.btn_save = QPushButton('表示中の時点を tiff に書き出す… (--save)'); self.btn_save.clicked.connect(self.save)
        l3.addLayout(row2); l3.addWidget(self.btn_preload); l3.addWidget(self.btn_save)

        # --- ch2 の位置合わせ ---
        from qtpy.QtWidgets import QDoubleSpinBox, QGridLayout
        g4 = QGroupBox('ch2 の位置合わせ (ch1 基準)'); l4 = QGridLayout(g4)
        self.sp_h = QDoubleSpinBox(); self.sp_h.setRange(-100, 100); self.sp_h.setSingleStep(0.5); self.sp_h.setDecimals(2)
        self.sp_v = QDoubleSpinBox(); self.sp_v.setRange(-100, 100); self.sp_v.setSingleStep(0.5); self.sp_v.setDecimals(2)
        self.sp_h.setToolTip('カメラ画像の横方向のずれ (px, 右が正)。再構築後は 45° 方向 (x と z) に動く')
        self.sp_v.setToolTip('カメラ画像の縦方向のずれ (px, 下が正)。再構築後は y に動く')
        for sp in (self.sp_h, self.sp_v):
            sp.valueChanged.connect(lambda _v: self.on_hv_changed())
        l4.addWidget(QLabel('カメラ h'), 0, 0); l4.addWidget(self.sp_h, 0, 1)
        l4.addWidget(QLabel('カメラ v'), 0, 2); l4.addWidget(self.sp_v, 0, 3)
        self.chk_cam = QCheckBox('カメラ面に拘束 (h, v の 2 自由度)'); self.chk_cam.setChecked(True)
        self.chk_cam.setToolTip('ch1/ch2 のずれは光路の差なのでカメラ面内の平行移動で表せる。外すと dx/dy/dz を自由に動かせる')
        self.chk_cam.toggled.connect(self.on_constrain_toggled)
        l4.addWidget(self.chk_cam, 0, 4, 1, 2)
        self.sp_shift = {}
        for col, name in enumerate(('dx', 'dy', 'dz')):
            sp = QDoubleSpinBox(); sp.setRange(-100, 100); sp.setSingleStep(0.5); sp.setDecimals(2); sp.setValue(0.0)
            sp.valueChanged.connect(lambda _v, n=name: self.on_xyz_changed())
            sp.setEnabled(False)
            l4.addWidget(QLabel(name), 1, col * 2); l4.addWidget(sp, 1, col * 2 + 1)
            self.sp_shift[name] = sp
        row4 = QHBoxLayout()
        self.btn_shift_save_calib = QPushButton('校正フォルダに保存'); self.btn_shift_save_calib.setToolTip('その日の全データに共通のずれとして校正フォルダの chshift.csv に保存する (推奨)')
        self.btn_shift_save_calib.clicked.connect(lambda: self.save_shift('calib'))
        self.btn_shift_save = QPushButton('データフォルダに保存'); self.btn_shift_save.setToolTip('このデータだけの値として保存する (校正フォルダの値より優先される)')
        self.btn_shift_save.clicked.connect(lambda: self.save_shift('data'))
        self.btn_shift_reset = QPushButton('0 に戻す'); self.btn_shift_reset.clicked.connect(self.reset_shift)
        row4.addWidget(self.btn_shift_save_calib); row4.addWidget(self.btn_shift_save); row4.addWidget(self.btn_shift_reset)
        l4.addLayout(row4, 2, 0, 1, 6)
        self.lbl_shift = QLabel(''); self.lbl_shift.setWordWrap(True); l4.addWidget(self.lbl_shift, 3, 0, 1, 6)
        self._shift_src = None

        self.lbl_status = QLabel(''); self.lbl_status.setWordWrap(True)
        for x in (g1, g2, g4, g3, self.lbl_status):
            lay.addWidget(x)
        lay.addStretch()
        sc = QScrollArea(); sc.setWidgetResizable(True); sc.setWidget(w)
        viewer.window.add_dock_widget(sc, name='ISOP', area='right')

    # ---- 表示の更新 ----
    def refresh_labels(self):
        ds, folder = self.state.get('ds'), self.state.get('folder')
        if ds is not None:
            ts = self.state['ts']
            # 長いパスで表示が切れるので、名前だけ出して全体はツールチップに
            self.lbl_data.setText(f'{os.path.basename(os.path.normpath(folder))}\n{ds.nt} 時点, ch {ds.channels}, {ds.shape} (z, y, x)\n表示: t={ts[0]}〜{ts[-1]}')
            self.lbl_data.setToolTip(folder)
            src = '手動' if self.opts.get('calib') else ('データ内の cropparameters.csv' if ds.calib_kind == 'cropparameters' else '自動検出')
            self.lbl_calib.setText(f'[{src}] {os.path.basename(os.path.dirname(ds.calib_path))}/{os.path.basename(ds.calib_path)}')
            self.lbl_calib.setToolTip(ds.calib_path)
            for sp, v in ((self.sp_t0, ts[0]), (self.sp_t1, ts[-1])):
                sp.setRange(0, ds.nt - 1); sp.setValue(v); sp.setEnabled(True)
        elif folder:
            self.lbl_data.setText(folder)
        if self.opts.get('calib') and ds is None:
            self.lbl_calib.setText(f'[手動] {self.opts["calib"]}')

    def status(self, msg):
        self.lbl_status.setText(msg); print(msg)

    # ---- ch2 の位置合わせ ----
    def current_shift(self):
        return tuple(self.sp_shift[n].value() for n in ('dz', 'dy', 'dx'))

    def _set_boxes(self, dz, dy, dx, h=None, v=None):
        if h is None:
            h, v = volume_to_camera_shift(dz, dy, dx, mirrored=True)
        for sp, val in ((self.sp_shift['dz'], dz), (self.sp_shift['dy'], dy), (self.sp_shift['dx'], dx), (self.sp_h, h), (self.sp_v, v)):
            sp.blockSignals(True); sp.setValue(float(val)); sp.blockSignals(False)

    def set_shift(self, dz, dy, dx, apply=True):
        self._set_boxes(dz, dy, dx)
        if apply:
            self.apply_shift()

    def on_hv_changed(self):
        dz, dy, dx = camera_to_volume_shift(self.sp_h.value(), self.sp_v.value(), mirrored=True)
        self._set_boxes(dz, dy, dx, self.sp_h.value(), self.sp_v.value())
        self.apply_shift()

    def on_xyz_changed(self):
        dz, dy, dx = self.current_shift()
        self._set_boxes(dz, dy, dx)
        self.apply_shift()

    def on_constrain_toggled(self, checked):
        for sp in self.sp_shift.values():
            sp.setEnabled(not checked)
        if checked:                                   # 拘束に戻すときは今の h/v に丸める
            self.on_hv_changed()

    def apply_shift(self):
        ds = self.state.get('ds')
        if ds is None or 'ch2' not in self.viewer.layers:
            return
        dz, dy, dx = self.current_shift()
        vox = ds.voxel
        self.viewer.layers['ch2'].translate = (0, dz * vox, dy * vox, dx * vox)
        src = f'  (読み込み元: {self._shift_src})' if self._shift_src else ''
        self.lbl_shift.setText(f'ch2 を (dz, dy, dx) = ({dz:+.2f}, {dy:+.2f}, {dx:+.2f}) voxel 動かしています'
                               f' (カメラ h={self.sp_h.value():+.2f}, v={self.sp_v.value():+.2f} px){src}')

    def reset_shift(self):
        self._shift_src = None
        self.set_shift(0, 0, 0)

    def save_shift(self, where='calib'):
        if not self.state.get('folder'):
            return
        if where == 'calib':
            target = calib_dir_of(self.state['folder'], self.opts.get('calib'))
            if not target:
                self.status('校正フォルダが見つからないので、データフォルダに保存してください'); return
        else:
            target = self.state['folder']
        p = write_chshift(target, {2: self.current_shift()})
        self._shift_src = p
        self.status(f'保存しました: {p}')
        self.apply_shift()

    def load_shift(self):
        sh, src = read_chshift(self.state['folder'], self.opts.get('calib')) if self.state.get('folder') else ({}, None)
        self._shift_src = src
        if 2 in sh:
            self.set_shift(*sh[2])
            self.status(f'{src} の ch2 シフトを適用しました: {tuple(round(x, 2) for x in sh[2])}')
        else:
            self.set_shift(0, 0, 0)

    # ---- 操作 ----
    def load(self, folder, ts=None):
        ds, ts2 = load_into_viewer(self.viewer, folder, self.opts, ts if ts is not None else self.state.get('ts'))
        if ds is not None:
            if self.state.get('ds') is not None and self.state['ds'] is not ds:
                self.state['ds'].close()                   # 前のデータの先読みプロセスを止める
            self.state['ds'], self.state['folder'], self.state['ts'] = ds, folder, ts2
            self.status('')
            self.load_shift()
            if getattr(self, 'on_loaded', None):
                self.on_loaded()
        else:
            self.status(check_folder(folder, self.opts.get('calib')) or '')
        self.refresh_labels()
        return ds

    def open_data(self, folder=None):
        d = folder or choose_folder_qt(os.path.dirname(self.state['folder']) if self.state.get('folder') else None)
        if d:
            self.state['ts'] = None                        # 新しいデータは全時点
            return self.load(d)

    def open_calib(self, folder=None):
        d = folder or choose_folder_qt(os.path.dirname(self.state['folder']) if self.state.get('folder') else None)
        if d:
            if not os.path.exists(os.path.join(d, 'cropcalib.csv')):
                self.status(f'{d} に cropcalib.csv がありません (校正撮影のフォルダを選んでください)'); return
            self.opts['calib'] = d
            self.refresh_labels()
            if self.state.get('folder'):
                self.load(self.state['folder'])            # 新しい校正で読み直す

    def auto_calib(self):
        self.opts['calib'] = None
        if self.state.get('folder'):
            self.load(self.state['folder'])
        else:
            self.lbl_calib.setText('(自動)')

    def on_clean(self, checked):
        self.opts['clean'] = not bool(checked)
        if self.state.get('folder'):
            self.load(self.state['folder'])

    def apply_t(self):
        if self.state.get('ds') is None:
            return
        t0, t1 = sorted((self.sp_t0.value(), self.sp_t1.value()))
        self.load(self.state['folder'], list(range(t0, t1 + 1)))

    def preload(self):
        ds = self.state.get('ds')
        if ds is None:
            return
        ds.preload(self.state['ts'])
        self.status(f'{len(self.state["ts"])} 時点 x {len(ds.channels)} ch を再構築してメモリに置きました')

    def save(self):
        from qtpy.QtWidgets import QFileDialog
        ds = self.state.get('ds')
        if ds is None:
            return
        name = os.path.basename(os.path.normpath(self.state['folder'])) + '.tif'
        path, _ = QFileDialog.getSaveFileName(None, 'tiff に書き出す', os.path.join(os.path.dirname(self.state['folder']), name), 'TIFF (*.tif *.tiff)')
        if path:
            save_tiff(ds, self.state['ts'], path, shifts={2: self.current_shift()})
            self.status(f'書き出しました: {path} (ch2 のシフトを反映)')


class IsopHistogram:
    """右側の「ヒストグラム」パネル。表示中の時点について、ch1 (マゼンタ) / ch2 (緑) の輝度ヒストグラムと
    サチり (飽和) の有無を表示する。スライダーが止まってから HIST_DELAY_MS 後に更新するので操作は重くならない。
    - 上の数字: 生のカメラ画像 (再構築前) の最大値と、閾値以上の画素数。サチりの判定はこちらで行う
      (再構築は補間でピークを平均化するので、再構築後の値だけ見ると飽和を見落とす)。
    - グラフ: 再構築後のボリューム (1/8 に間引き) のヒストグラム (縦軸は対数)。破線は各レイヤーの表示コントラスト範囲で、
      右の破線より右にある画素は画面上では白飛びして見える (センサーの飽和とは別)。"""
    HIST_DELAY_MS = 150

    def __init__(self, viewer, state):
        from qtpy.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox, QCheckBox
        from qtpy.QtCore import QTimer
        import matplotlib
        matplotlib.use('qtagg', force=False)
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure
        self.viewer, self.state = viewer, state
        w = QWidget(); lay = QVBoxLayout(w)
        row = QHBoxLayout()
        self.chk_live = QCheckBox('自動更新'); self.chk_live.setChecked(True); row.addWidget(self.chk_live)
        row.addWidget(QLabel('飽和の閾値'))
        self.sp_sat = QSpinBox(); self.sp_sat.setRange(1, 65535); self.sp_sat.setValue(65535)
        self.sp_sat.valueChanged.connect(lambda _: self.schedule()); row.addWidget(self.sp_sat)
        row.addStretch(); lay.addLayout(row)
        from qtpy.QtWidgets import QPushButton
        row2 = QHBoxLayout()
        self.btn_fix = QPushButton('ビンと軸を全時点で固定'); self.btn_fix.setToolTip('時点をまたいで比較できるよう、全時点から最大値を調べて (最大 40 時点をサンプリング) ビン区切りと軸を固定する')
        self.btn_fix.clicked.connect(self.fix_bins)
        self.btn_unfix = QPushButton('時点ごと'); self.btn_unfix.clicked.connect(self.unfix_bins)
        row2.addWidget(self.btn_fix); row2.addWidget(self.btn_unfix); row2.addStretch(); lay.addLayout(row2)
        row3 = QHBoxLayout(); row3.addWidget(QLabel('表示範囲'))
        self.btn_full = QPushButton('0〜最大 (白飛びなし)'); self.btn_full.setToolTip('各チャンネルの表示コントラストを 0〜全時点の最大値にする (サンプリングした時点の最大)')
        self.btn_full.clicked.connect(lambda: self.set_contrast('full'))
        self.btn_auto = QPushButton('自動 (0.5〜99.99%)'); self.btn_auto.clicked.connect(lambda: self.set_contrast('auto'))
        row3.addWidget(self.btn_full); row3.addWidget(self.btn_auto); row3.addStretch(); lay.addLayout(row3)
        self.fixed = None          # {'xmax': {ch: 最大値}, 'ymax': 最大カウント} 固定モードのとき
        self.lbl = QLabel('(データ未選択)'); self.lbl.setWordWrap(True); lay.addWidget(self.lbl)
        self.fig = Figure(figsize=(3.6, 2.4), dpi=90, facecolor='#262930')
        self.canvas = FigureCanvasQTAgg(self.fig); self.canvas.setMinimumHeight(220); lay.addWidget(self.canvas)
        self.ax = self.fig.add_subplot(111)
        viewer.window.add_dock_widget(w, name='ヒストグラム', area='right')
        self.timer = QTimer(); self.timer.setSingleShot(True); self.timer.timeout.connect(self.update)
        viewer.dims.events.current_step.connect(lambda e: self.schedule())

    def sample_max(self, n=40):
        """全時点から最大 n 時点を等間隔に選び、生画像の最大値をチャンネルごとに返す (1 時点 2ch で約 40 ms)。"""
        ds = self.state.get('ds'); ts = self.state.get('ts')
        if ds is None or not ts:
            return None
        pick = ts if len(ts) <= n else [ts[int(i)] for i in np.linspace(0, len(ts) - 1, n)]
        mx = {}
        for ch in ds.channels:
            m = 0
            for t in pick:
                raw, _ = recon.read_bin_frames(os.path.join(ds.folder, f'img-ch{ch}.bin'), range(t * ds.nv, (t + 1) * ds.nv))
                m = max(m, int(raw.max()))
            mx[ch] = m
        return mx, len(pick)

    def fix_bins(self):
        r = self.sample_max()
        if r is None:
            return
        mx, n = r
        self.fixed = {'xmax': mx, 'n': n, 'ymax': None}
        self.lbl.setText(f'ビン固定: {n} 時点をサンプリング、最大値 ' + ', '.join(f'ch{c}={v}' for c, v in mx.items()))
        self.update()

    def unfix_bins(self):
        self.fixed = None
        self.update()

    def set_contrast(self, mode):
        """各レイヤーの表示コントラストを一括設定。'full' は全時点サンプルの最大値まで。"""
        ds = self.state.get('ds'); t = self.current_t()
        if ds is None or t is None:
            return
        if mode == 'full':
            r = self.sample_max(); mx = r[0] if r else {}
        for ch in ds.channels:
            layer = self.viewer.layers[f'ch{ch}']
            if mode == 'full':
                hi = float(mx.get(ch, ds.get(ch, t).max()))
                layer.contrast_limits_range = [0.0, hi]
                layer.contrast_limits = [0.0, hi]
            else:
                layer.contrast_limits = contrast_limits(ds.get(ch, t), 'auto')
        self.schedule()

    def schedule(self):
        if self.chk_live.isChecked():
            self.timer.start(self.HIST_DELAY_MS)       # 連続で動いている間は延期し、止まってから更新する

    def connect_layers(self):
        for layer in self.viewer.layers:
            layer.events.contrast_limits.connect(lambda e: self.schedule())

    def current_t(self):
        ts = self.state.get('ts')
        if not ts:
            return None
        i = int(self.viewer.dims.current_step[0])
        return ts[i] if 0 <= i < len(ts) else None

    def update(self):
        ds = self.state.get('ds'); t = self.current_t()
        if ds is None or t is None:
            return
        keys = [(ch, t) for ch in ds.channels]
        with ds._lock:
            ready = all(k in ds._cache for k in keys)
        if not ready:                                  # まだ再構築中なら少し待ってもう一度
            ds.prefetch([t]); self.timer.start(200); return
        sat = self.sp_sat.value()
        self.ax.clear()
        lines = []
        colors = {1: '#ff40ff', 2: '#40ff40'}
        xmax = 1
        for ch in ds.channels:
            V = ds.get(ch, t)
            sub = V[::2, ::2, ::2]
            vmax = int(V.max())
            xmax = max(xmax, vmax)
            # 生のカメラ画像 (この時点の 8 フレーム) で飽和を判定
            try:
                raw, _ = recon.read_bin_frames(os.path.join(ds.folder, f'img-ch{ch}.bin'), range(t * ds.nv, (t + 1) * ds.nv))
                rmax = int(raw.max()); nsat = int((raw >= sat).sum()); frac = nsat / raw.size
                rawtxt = f'生画像 最大 {rmax}' + (f'  <b>飽和 {nsat} 画素 ({frac * 100:.4f}%)</b>' if nsat else '  飽和なし')
            except Exception as e:
                rawtxt = f'生画像 読めず ({e})'
            # ビンはデータの範囲で決める (飽和閾値 65535 を基準にすると 1 ビン 256 階調になり分布が潰れる)
            # 固定モードでは全時点サンプルの最大値で区切るので、時点をまたいで同じビンになる
            top = int(self.fixed['xmax'].get(ch, vmax)) if self.fixed else vmax
            top = max(top, vmax)
            nb = int(min(256, top + 1))
            h, edges = np.histogram(sub, bins=nb, range=(0, top + 1))
            if self.fixed:
                xmax = max(xmax, top)
                self.fixed['ymax'] = max(self.fixed.get('ymax') or 0, int(h.max()))
            self.ax.step(edges[:-1], np.maximum(h, 0.5), where='post', color=colors.get(ch, 'w'), lw=1.2, label=f'ch{ch}')
            try:
                lo, hi = self.viewer.layers[f'ch{ch}'].contrast_limits
                self.ax.axvline(lo, color=colors.get(ch, 'w'), ls='--', lw=0.8, alpha=0.6)
                self.ax.axvline(hi, color=colors.get(ch, 'w'), ls='--', lw=0.8, alpha=0.6)
                clip = float((V[::4, ::4, ::4] >= hi).mean())
                disptxt = f'表示範囲 {int(lo)}〜{int(hi)} (上限以上 {clip:.2%} が白飛び表示)'
            except Exception:
                disptxt = ''
            lines.append(f'<span style="color:{colors.get(ch, "w")}">ch{ch}</span>: {rawtxt} / 再構築後 最大 {vmax} / {disptxt}')
        self.ax.set_yscale('log'); self.ax.set_xlim(0, xmax * 1.05)
        if self.fixed and self.fixed.get('ymax'):
            self.ax.set_ylim(0.5, self.fixed['ymax'] * 2)
        if sat <= xmax * 1.05:
            self.ax.axvline(sat, color='red', ls=':', lw=1.0)
        bw = max(1, (xmax + 1) // 256)   # グラフ内の文字は ASCII (日本語フォントが無い環境で豆腐になる)
        title_extra = f'  (bin width {bw}, sat. threshold {sat} off scale)' if sat > xmax * 1.05 else f'  (bin width {bw})'
        self.ax.set_facecolor('#1e2026'); self.ax.tick_params(colors='#dddddd', labelsize=7)
        for sp in self.ax.spines.values():
            sp.set_color('#888888')
        self.ax.set_xlabel('intensity', color='#dddddd', fontsize=8); self.ax.set_ylabel('voxels', color='#dddddd', fontsize=8)   # 日本語はフォントが無い環境で豆腐になる
        self.ax.legend(fontsize=7, loc='upper right', facecolor='#262930', labelcolor='#dddddd')
        self.ax.set_title(f't = {t}' + title_extra + ('  [fixed bins]' if self.fixed else ''), color='#dddddd', fontsize=8)
        self.fig.subplots_adjust(left=0.16, right=0.98, top=0.9, bottom=0.2); self.canvas.draw_idle()
        if self.fixed:
            lines.insert(0, f"ビン固定中: {self.fixed['n']} 時点をサンプリングした最大値 " + ', '.join(f'ch{c}={v}' for c, v in self.fixed['xmax'].items()))
        self.lbl.setText('<br>'.join(lines))


PREFETCH_AHEAD = 12    # スライダーの現在位置から先に何時点ぶん裏で再構築しておくか
PREFETCH_BEHIND = 2


def install_prefetch(viewer, state):
    """t スライダーが動いたら、前後の時点を裏で再構築しておく。"""
    def on_step(event=None):
        ds = state.get('ds'); ts = state.get('ts')
        if ds is None or not ts:
            return
        i = int(viewer.dims.current_step[0])
        if not (0 <= i < len(ts)):
            return
        order = [i + k for k in range(1, PREFETCH_AHEAD + 1)] + [i - k for k in range(1, PREFETCH_BEHIND + 1)]
        ds.prefetch([ts[j] for j in order if 0 <= j < len(ts)])
    viewer.dims.events.current_step.connect(on_step)
    return on_step


def run_napari(folder, ts, opts):
    """folder が None なら起動後にダイアログで選ぶ。右側のパネルでデータ/校正フォルダの切り替えとオプション操作ができる。"""
    import napari
    try:   # dask 配列の読み込みを UI と別スレッドにして、操作中に固まらないようにする
        from napari.settings import get_settings
        get_settings().experimental.async_ = True
    except Exception:
        pass
    viewer = napari.Viewer(title='ISOP viewer')
    state = {'ds': None, 'folder': folder, 'ts': ts}
    panel = IsopPanel(viewer, opts, state)
    panel.refresh_labels()
    install_prefetch(viewer, state)
    hist = IsopHistogram(viewer, state)
    panel.on_loaded = lambda: (hist.connect_layers(), hist.schedule())
    if folder:
        panel.load(folder, ts)
    else:
        from qtpy.QtCore import QTimer
        QTimer.singleShot(200, panel.open_data)      # 起動直後 (イベントループが回ってから) にダイアログを出す
    napari.run()


def run_mpl(ds, ts, preload):
    """napari が無いときの簡易ビューア。XY 断面 (または MIP) を ch1 マゼンタ / ch2 緑で重ねる。"""
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, Button
    if preload:
        ds.preload(ts)
    nz, ny, nx = ds.shape
    state = dict(t=ts[0], z=nz // 2, mip=False)
    lim = {ch: contrast_limits(ds.get(ch, ts[0])) for ch in ds.channels}

    def composite():
        rgb = np.zeros((ny, nx, 3), np.float32)
        for ch in ds.channels:
            v = ds.get(ch, state['t'])
            img = v.max(0) if state['mip'] else v[state['z']]
            lo, hi = lim[ch]
            g = np.clip((img.astype(np.float32) - lo) / (hi - lo), 0, 1)
            if ch == 1:
                rgb[..., 0] += g; rgb[..., 2] += g      # マゼンタ
            else:
                rgb[..., 1] += g                        # 緑
        return np.clip(rgb, 0, 1)

    fig, ax = plt.subplots(figsize=(10, 7))
    plt.subplots_adjust(bottom=0.22)
    im = ax.imshow(composite(), interpolation='nearest')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    title = ax.set_title('')

    def refresh(_=None):
        im.set_data(composite())
        title.set_text(f"t={state['t']}   " + ('XY MIP' if state['mip'] else f"z={state['z']}/{nz - 1}")
                       + '   (ch1 magenta, ch2 green)')
        fig.canvas.draw_idle()

    ax_t = plt.axes([0.15, 0.10, 0.65, 0.03]); ax_z = plt.axes([0.15, 0.05, 0.65, 0.03])
    s_t = Slider(ax_t, 't', ts[0], ts[-1], valinit=ts[0], valstep=1)
    s_z = Slider(ax_z, 'z', 0, nz - 1, valinit=state['z'], valstep=1)
    b_mip = Button(plt.axes([0.85, 0.05, 0.1, 0.08]), 'MIP')

    def on_t(v): state['t'] = int(v); refresh()
    def on_z(v): state['z'] = int(v); refresh()
    def on_mip(_): state['mip'] = not state['mip']; refresh()
    s_t.on_changed(on_t); s_z.on_changed(on_z); b_mip.on_clicked(on_mip)

    def on_key(ev):
        if ev.key in ('right', 'left'):
            s_t.set_val(int(np.clip(state['t'] + (1 if ev.key == 'right' else -1), ts[0], ts[-1])))
        elif ev.key in ('up', 'down'):
            s_z.set_val(int(np.clip(state['z'] + (1 if ev.key == 'up' else -1), 0, nz - 1)))
        elif ev.key == 'm':
            on_mip(None)
    fig.canvas.mpl_connect('key_press_event', on_key)
    refresh()
    plt.show()


def save_tiff(ds, ts, path, shifts=None):
    """shifts: {ch: (dz, dy, dx)} ボクセル。指定があればそのチャンネルを平行移動してから書く (線形補間、端は 0)。"""
    import tifffile
    from scipy.ndimage import shift as nd_shift
    ds.preload(ts)
    def vol(ch, t):
        v = ds.get(ch, t)
        if shifts and ch in shifts and any(abs(x) > 1e-6 for x in shifts[ch]):
            v = nd_shift(v.astype(np.float32), shifts[ch], order=1, mode='constant', cval=0.0)
            v = np.clip(v, 0, 65535).astype(np.uint16)
        return v
    arr = np.stack([np.stack([vol(ch, t) for ch in ds.channels]) for t in ts])   # (T, C, Z, Y, X)
    arr = arr.transpose(0, 2, 1, 3, 4)                                               # ImageJ 順 TZCYX
    tifffile.imwrite(path, arr, imagej=True,
                     resolution=(1 / ds.voxel, 1 / ds.voxel),
                     metadata={'axes': 'TZCYX', 'spacing': ds.voxel, 'unit': 'um'})
    print(f'書き出し: {path}  {arr.shape} (T, Z, C, Y, X)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('folder', nargs='?', help='データフォルダ (省略するとダイアログで選ぶ)')
    ap.add_argument('--t', nargs=2, type=int, metavar=('T0', 'T1'), help='表示する時点の範囲 (両端含む)')
    ap.add_argument('--calib', help='校正撮影のフォルダ (cropcalib.csv がある所)。省略時はデータ内の cropparameters.csv か同じ日の …calib フォルダを自動で使う')
    ap.add_argument('--backend', default='auto', choices=['auto', 'napari', 'mpl'])
    ap.add_argument('--method', default='fast', choices=['fast', 'rot', 'roll'])
    ap.add_argument('--preload', action='store_true', help='先に全時点を再構築する')
    ap.add_argument('--max-cache-gb', type=float, default=2.0, help='再構築結果をメモリに置く上限 (GB, 既定 2)。1 時点 1ch = 30 MB')
    ap.add_argument('--labview', action='store_true', help=argparse.SUPPRESS)   # 旧オプション (無効)
    ap.add_argument('--clean', action='store_true', help=argparse.SUPPRESS)   # 旧オプション (今は既定)
    ap.add_argument('--save', metavar='OUT.tif', help='表示せず ImageJ 形式の tiff に書き出す')
    a = ap.parse_args()
    opts = dict(method=a.method, clean=not a.labview, max_cache_gb=a.max_cache_gb, preload=a.preload, calib=a.calib)
    ts = list(range(a.t[0], a.t[1] + 1)) if a.t else None

    backend = a.backend
    if backend == 'auto':
        try:
            import napari  # noqa
            backend = 'napari'
        except ImportError:
            print('napari が見つからないので matplotlib 版で開きます (conda activate isop で napari 版)')
            backend = 'mpl'

    if backend == 'napari' and not a.save:
        if a.folder and check_folder(a.folder, a.calib):
            sys.exit(f'{a.folder}: {check_folder(a.folder, a.calib)}')
        run_napari(a.folder, ts, opts)
        sys.exit(0)

    folder = a.folder or choose_folder_tk()
    problem = check_folder(folder, a.calib)
    if problem:
        sys.exit(f'{folder}: {problem}')
    ds = IsopDataset(folder, a.method, clean=not a.labview, max_cache_gb=a.max_cache_gb, calib=a.calib)
    print(f'切り取りパラメータ: {ds.calib_path}')
    ts = list(range(ds.nt)) if ts is None else [t for t in ts if t < ds.nt]
    print(f'{folder}: {ds.nt} 時点, ch {ds.channels}, ボリューム {ds.shape}, 表示 t={ts[0]}..{ts[-1]}')
    if a.save:
        save_tiff(ds, ts, a.save, shifts=read_chshift(folder, a.calib)[0])
    else:
        run_mpl(ds, ts, a.preload)
