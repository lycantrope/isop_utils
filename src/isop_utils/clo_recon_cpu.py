"""Closed-loop-optogenetics (utils_v2 / reconstruction_v2) のセクション3までの処理を
numpy/scipy だけで忠実に再現する (cupy 部分は CPU 経路に置換)。"""
import sys, os, math, struct, csv
import numpy as np
from scipy.ndimage import affine_transform
sys.path.insert(0, "/Users/yutaro/Data/Toyoshima_lab/Data/ISOP")
from isop_recon import read_bin_frames, read_3du16, ncc

DATA = "/Users/yutaro/Data/Toyoshima_lab/Data/ISOP/20261020-194904beads"
CALIB = "/Users/yutaro/Data/Toyoshima_lab/Data/ISOP/20261020-172105beads-cropbulkcalib"
OUT = "/private/tmp/claude-501/-Users-yutaro-Data-Toyoshima-lab-Data-ISOP/abcd4edb-9477-42fa-9735-d5cbc5e2b752/scratchpad"

def read_imgcalib_txt_offline(folder):
    lines = open(f"{folder}/imgcalib.txt").readlines()
    return dict(voxel_size_xy=float(lines[0]), voxel_size_z=float(lines[1]), y_shift_comp=float(lines[2]),
                y_shift_comp_ch2=float(lines[3]), h_shift=float(lines[4]), v_shift=float(lines[5]))

def crop_param_with_yshift(path):
    rows = [list(map(float, r)) for r in csv.reader(open(path)) if r]
    # caluculate_y_shit_crop_parameter 相当 (5列目を追加)
    out = []
    for m in range(len(rows) // 16):
        x1 = rows[m*16][1]; xn = rows[m*16+15][1]
        for i in range(16):
            r = rows[m*16+i]; xi = r[1]; dy = r[2]
            if dy > 0: ys = round((xi - x1) * abs(dy) / 2304)
            elif dy < 0: ys = round((xn - xi) * abs(dy) / 2304)
            else: ys = 0
            out.append(r[:4] + [ys])
    return np.ascontiguousarray(np.array(out, dtype=np.float64))

def cutting_frame_cython8(img, crop_param, new_stack, m, ch):
    """crop_cython.pyx の cutting_frame_cython8 を素直に移植"""
    img_h, img_w = img.shape
    ofs = crop_param.shape[0] // 2 if ch == 2 else 0
    base = ofs + m * 16
    for k in range(16):
        r, xb, _, _, yo = crop_param[base + k]
        for j in range(300):
            xn = int(round(xb + j / r))
            y = int(j + yo)
            if 0 <= y < img_h and 0 <= xn and xn + 140 <= img_w and 0 <= y < new_stack.shape[1]:
                new_stack[16*m+k, y] = img[y, xn:xn+140]
    return new_stack

class DoingFullyReconstCPU:
    def __init__(self, shear_factor, sheet_angle):
        self.shear_factor = shear_factor; self.sheet_angle = sheet_angle
    def _dims(self, aff, shape):
        d0, d1, d2 = [s - 1 for s in shape]
        c = np.array([[0,0,0,1],[d0,0,0,1],[0,d1,0,1],[0,0,d2,1],[d0,d1,0,1],[d0,0,d2,1],[0,d1,d2,1],[d0,d1,d2,1]], np.float32)
        tr = (aff @ c.T).T
        dims = tr.max(0) - tr.min(0) + 1
        dims = (2 * np.ceil(dims / 2.0)).astype(np.int32)
        return tuple(dims[:3].tolist())
    def _shift(self, shape, direction=-1.0):
        s = np.eye(4); s[0:3, 3] = direction * np.array(shape) / 2; return s
    def scale_pixel_z(self, v):
        sc = np.eye(4, dtype=np.float32); sc[0, 0] = np.float32(self.shear_factor)
        v = np.asarray(v, np.float32)
        return affine_transform(v, np.linalg.inv(sc), output_shape=self._dims(sc, v.shape), order=1, mode='constant')
    def rotate(self, v):
        v = np.asarray(v, np.float32)
        a = np.deg2rad(-self.sheet_angle)
        R = np.eye(4, dtype=np.float32); R[0,0]=np.cos(a); R[0,2]=np.sin(a); R[2,0]=-np.sin(a); R[2,2]=np.cos(a)
        sh = self._shift(v.shape)
        osh = self._dims(R @ sh, v.shape)
        un = self._shift(osh, 1.0)
        M = un @ R @ sh
        return affine_transform(v, np.linalg.inv(M), output_shape=osh, order=1, mode='constant')
    def trim_data(self, r):
        mx = r.max(axis=(1, 2)); nz = mx > 0
        s = nz.tolist().index(True); e = len(mx) - nz[::-1].tolist().index(True)
        return r[s:e], s, e
    def zscale_rotate_trim(self, v):
        return self.trim_data(self.rotate(self.scale_pixel_z(v)))

def make_reconst_volume(ch, t, imgcalib, crop_param, frames_per_vol=8, slices_per_frame=16,
                        intensity_correct=False, offset=100, affine=True):
    z = frames_per_vol * slices_per_frame
    frames, _ = read_bin_frames(f"{DATA}/img-ch{ch}.bin", range(t*frames_per_vol, (t+1)*frames_per_vol))
    frames = frames.astype(np.uint16)
    h_shift = int(imgcalib["h_shift"]); v_shift = int(imgcalib["v_shift"])
    raw_h, raw_w, depth = 300, 140, z
    shear = np.cos(np.deg2rad(45)) * (imgcalib["voxel_size_z"] / imgcalib["voxel_size_xy"])
    start_w = [math.ceil(shear * i) for i in range(depth)]
    stop_w = [math.ceil(shear * i + raw_w) for i in range(depth)]
    new_width = math.ceil(raw_w + shear * (depth - 1))
    reconst = np.zeros([depth, raw_h, new_width], np.uint16)
    new_stack = np.zeros((z, 300, 140), np.uint16)
    delta = crop_param[:, 3]
    for j in range(frames_per_vol):
        img = frames[j]
        if ch == 2:
            img = np.fliplr(img)
            hs = np.zeros_like(img); vs = np.zeros_like(img)
            if h_shift > 0: hs[:, h_shift:] = img[:, :-h_shift]
            elif h_shift < 0: hs[:, :h_shift] = img[:, -h_shift:]
            else: hs = img
            if v_shift > 0: vs[v_shift:, :] = hs[:-v_shift, :]
            elif v_shift < 0: vs[:v_shift, :] = hs[-v_shift:, :]
            else: vs = hs
            img = np.ascontiguousarray(vs)
        new_stack = cutting_frame_cython8(img, crop_param, new_stack, j, ch)
        for k in range(slices_per_frame):
            idx = j*slices_per_frame + k
            sl = new_stack[idx]
            if intensity_correct:
                sl = np.clip((sl.astype(np.float32) - 100) * (1 + delta[idx]) + 100, 0, 65535)
            reconst[idx, 0:raw_h, start_w[idx]:stop_w[idx]] = np.fliplr(sl)
    print(f"ch{ch} t{t}: deskewed stack {reconst.shape}, shear={shear:.4f}, new_width={new_width}")
    if not affine:
        return reconst
    fr = DoingFullyReconstCPU(shear, 45)
    full, s, e = fr.zscale_rotate_trim(reconst)
    print(f"   z-scaled/rotated/trimmed: {full.shape}  (trim z {s}:{e})")
    return full - offset

if __name__ == "__main__":
    ic = read_imgcalib_txt_offline(DATA)
    print("imgcalib:", ic)
    cp_ = crop_param_with_yshift(f"{CALIB}/cropcalib.csv")
    print("crop_param", cp_.shape, "y-shift col unique:", np.unique(cp_[:, 4]))
    for ch in (1, 2):
        V = make_reconst_volume(ch, 0, ic, cp_)
        np.save(f"{OUT}/clo_ch{ch}_t0.npy", V.astype(np.float32))
        D = make_reconst_volume(ch, 0, ic, cp_, affine=False)
        np.save(f"{OUT}/clo_deskew_ch{ch}_t0.npy", D)
