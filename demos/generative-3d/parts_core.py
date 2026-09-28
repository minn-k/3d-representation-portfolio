"""생성 과정 신호 → 3D 부위 라벨 (NumPy · SciPy 만, GPU 불필요). lift_parts.py 가 부르는 핵심 계산.

입력 (gen3d_sem.py · seg2d.py 출력):
  sem.npz      tok_coords (T,3) 32³ 토큰 격자 · attn_<b> (T, 5+37²) cross-attention · feat_<b> (T,1024) DiT 특징
               · slat_coords (V,3) 64³ 복셀 격자
  parts2d.npz  label (H,W) 0 = 배경/미분류, 1..K = 부위 · names · alpha
  cond.png · gaussian.ply  (색 검사용 — 선택)

mode "proj" (기본):
  1 attention 투표      S_attn = 앞쪽 블록의 상대 cross-attention · 패치별 부위 비율
  2 입력 카메라 추정     토큰 중심 → attention 무게중심 affine 카메라(IRLS) + 방위 · 고도 · 원근 전역 탐색 후보들을 물체
                        실루엣(발자국 Chamfer)으로 다듬고, 실루엣 IoU + 색 상관이 가장 높은 것 (색이 앞 · 뒤를 가른다)
  3 투영 투표            복셀을 z-buffer 로 그려 '입력 사진에 보이는' 복셀만 그 픽셀의 2D 부위(경계 침식)를 받는다
  4 전파                DiT 특징 유사도를 가중치로 한 26-이웃 토큰 그래프. 보이는 · 순도 높은 토큰은 고정(clamp)
  5 정리                같은 부위의 작은 조각(연결 성분)은 이웃 부위로
  6 복셀 다듬기          보이는 복셀은 자기 투영 라벨을 섞어 경계를 복셀(64³) 해상도로
  7 가려진 쪽            껍질을 속이 찬 부피로 채우고, 사진에 라벨이 없는 복셀은 '어느 부위의 부피에 속하나' 로:
                        D(표면까지 거리)를 따라 올라가 닿는 속심이 같은 부위 씨앗과 가까운지 + 부피 안 측지 거리(두꺼운
                        속은 싸고 얇은 목은 비쌈). 끝으로 복셀 단위 작은 조각 정리
mode "attn": 예전 방식 그대로 (attention 투표 → 반경 4 토큰 · 특징 k-NN 그래프 전파, 고정 · 정리 없음).

왜 바꿨나 (곰 인형 한쪽 팔이 섞이던 원인):
  - attention 은 '대응' 이 아니라 특징 검색이라, 털처럼 균일한 질감에서는 머리 · 팔 · 다리 패치를 두루 본다 (잡음).
  - 2D → 3D 투표에 가림(visibility)이 없다: 발 위에 얹힌 앞발, 팔 안쪽처럼 같은 광선 위의 토큰이 같은 패치로 투표한다.
  - 전파 그래프 반경 4 토큰(= 복셀 8칸, 물체 크기의 1/8)이 팔 · 몸통 · 다리 사이의 틈을 건너뛰고, 같은 털이라 특징
    가중치도 경계를 막지 못한다 → 잡음이 팔 안쪽 띠로 번진다.
"""
import numpy as np
from scipy import ndimage
from scipy.optimize import minimize
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

N_PATCH = 37            # DINOv2 ViT-L/14, 518² → 37×37 패치 토큰
N_EXTRA = 5             # TRELLIS 조건 열: cls 1 + register 4 가 패치 앞에 온다
TOK_RES, VOX_RES = 32, 64
# 부위 색: 0 = 모름(회색), 1.. = 부위 순서 (parts2d 의 names 순서)
PART_COLORS = np.array([[0.75, 0.75, 0.75], [0.90, 0.30, 0.25], [0.25, 0.55, 0.90], [0.30, 0.75, 0.35],
                        [0.95, 0.75, 0.20], [0.65, 0.40, 0.85], [0.20, 0.80, 0.80], [0.95, 0.50, 0.70],
                        [0.55, 0.35, 0.20], [0.45, 0.45, 0.80]])


def part_colors(ids):
    """부위 번호 (0..K−1, −1 = 모름) → RGB."""
    ids = np.asarray(ids, np.int64)
    return PART_COLORS[np.where(ids < 0, 0, 1 + ids % (len(PART_COLORS) - 1))]


# ------------------------------------------------------------------------------------------ 공통 입력
def patch_fractions(label, K):
    """37×37 패치마다 부위 비율 (배경 · 미분류 제외) → (1369, K)."""
    H, W = label.shape
    ph, pw = H / N_PATCH, W / N_PATCH
    M = np.zeros((N_PATCH * N_PATCH, K + 1), np.float32)
    for r in range(N_PATCH):
        for c in range(N_PATCH):
            blk = label[int(r * ph):int((r + 1) * ph), int(c * pw):int((c + 1) * pw)]
            M[r * N_PATCH + c] = np.bincount(blk.ravel(), minlength=K + 1)[:K + 1] / max(blk.size, 1)
    return M[:, 1:]


def vox_to_tok(sem):
    """64³ 복셀 → 그 복셀을 품은 32³ 토큰 번호."""
    tc = sem["tok_coords"].astype(np.int64)
    vc = sem["slat_coords"].astype(np.int64) // 2
    lut = np.full(TOK_RES ** 3, -1, np.int64)
    lut[(tc[:, 0] * TOK_RES + tc[:, 1]) * TOK_RES + tc[:, 2]] = np.arange(len(tc))
    v2t = lut[(vc[:, 0] * TOK_RES + vc[:, 1]) * TOK_RES + vc[:, 2]]
    if (v2t < 0).any():
        raise ValueError(f"토큰이 없는 복셀 {int((v2t < 0).sum())} 개 — sem.npz 의 tok_coords/slat_coords 가 맞지 않는다")
    return v2t


def token_centers(sem):
    return (sem["tok_coords"].astype(np.float64) + 0.5) / TOK_RES - 0.5


def voxel_centers(sem):
    return (sem["slat_coords"].astype(np.float64) + 0.5) / VOX_RES - 0.5


def dit_features(sem, blocks):
    F = np.concatenate([(lambda f: (f - f.mean(0)) / (f.std(0) + 1e-6))(sem[f"feat_{b}"].astype(np.float32))
                        for b in blocks], 1)
    return F / np.linalg.norm(F, axis=1, keepdims=True)


def voxel_colors(col, op, n_vox, per=32):
    """가우시안 색(복셀 순서, 복셀당 per 개) → 복셀 색 (불투명도 가중 평균)."""
    c = col[:n_vox * per].reshape(n_vox, per, 3)
    w = op[:n_vox * per].reshape(n_vox, per, 1)
    return (c * w).sum(1) / np.maximum(w.sum(1), 1e-9)


def kmeans(X, k, iters=30, seed=0):
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), k, replace=False)]
    for _ in range(iters):
        lab = np.argmin(((X[:, None] - C[None]) ** 2).sum(-1), 1)
        C = np.stack([X[lab == j].mean(0) if np.any(lab == j) else C[j] for j in range(k)])
    return lab


# ------------------------------------------------------------------------------------------ 1 attention
def relative_attention(sem, blocks):
    """토큰 × 패치 attention (cls · register 제외), 토큰 평균보다 '특별히 더 본' 몫만 남겨 정규화."""
    A = sum(sem[f"attn_{b}"].astype(np.float32)[:, N_EXTRA:] for b in blocks) / len(blocks)
    A = A / np.maximum(A.sum(1, keepdims=True), 1e-9)
    Ar = A / np.maximum(A.mean(0, keepdims=True), 1e-9)
    Ar = np.maximum(Ar - 1.0, 0)
    return Ar / np.maximum(Ar.sum(1, keepdims=True), 1e-9)


def attention_votes(Ar, M):
    S = Ar @ M
    return S / np.maximum(S.sum(1, keepdims=True), 1e-9)


def patch_centers(size):
    g = (np.arange(N_PATCH) + 0.5) * size / N_PATCH
    py, px = np.meshgrid(g, g, indexing="ij")                  # DINOv2 패치 순서 = 행 우선
    return np.stack([px.ravel(), py.ravel()], 1)


def attention_centroids(Ar, size, topk=16):
    """토큰마다 가장 많이 더 본 topk 패치의 무게중심 (픽셀 x, y) · 퍼짐 · 쓸 수 있는지."""
    pc = patch_centers(size)
    k = min(topk, Ar.shape[1] - 1)
    idx = np.argpartition(-Ar, k, axis=1)[:, :k]
    w = np.take_along_axis(Ar, idx, 1).astype(np.float64)
    pts = pc[idx]
    ws = w.sum(1)
    u = (w[..., None] * pts).sum(1) / np.maximum(ws, 1e-12)[:, None]
    spread = np.sqrt((w * ((pts - u[:, None]) ** 2).sum(-1)).sum(1) / np.maximum(ws, 1e-12))
    return u, spread, ws > 1e-6


# ------------------------------------------------------------------------------------------ 2 입력 카메라
def rodrigues(w):
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3)
    k = np.asarray(w, float) / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * Kx @ Kx


class Camera:
    """u = c + s·(x, y) / (1 + κ·z),  (x, y, z) = R (X − x0).  이미지 x 오른쪽 · y 아래, z = 깊이(멀수록 +).
    κ = 1 / (카메라 ~ x0 거리), 0 이면 약원근. s = x0 깊이에서 단위 길이당 픽셀."""

    def __init__(self, R, s, c, kappa, x0):
        self.R = np.asarray(R, float)
        self.s, self.kappa = float(s), float(kappa)
        self.c, self.x0 = np.asarray(c, float), np.asarray(x0, float)

    def project(self, X):
        Xc = (np.asarray(X, float) - self.x0) @ self.R.T
        den = np.maximum(1.0 + self.kappa * Xc[:, 2], 1e-3)
        return self.c + self.s * Xc[:, :2] / den[:, None], Xc[:, 2], den

    @staticmethod
    def from_affine(A, b, x0):
        U, sig, Vt = np.linalg.svd(A, full_matrices=False)     # 가장 가까운 '회전 × 등방 배율'
        R12 = U @ Vt
        R = np.vstack([R12, np.cross(R12[0], R12[1])])          # det +1, z = x × y (카메라 앞)
        return Camera(R, sig.mean(), A @ np.asarray(x0, float) + b, 0.0, x0)

    def perturbed(self, th):
        """th = (회전 벡터 3, log 배율, 중심 이동/100 px 2, κ)."""
        return Camera(rodrigues(th[:3]) @ self.R, self.s * np.exp(th[3]), self.c + 100.0 * np.asarray(th[4:6]),
                      th[6], self.x0)

    def as_dict(self):
        return {"R": np.round(self.R, 5).tolist(), "s_px_per_unit": round(self.s, 2),
                "c_px": np.round(self.c, 2).tolist(), "kappa": round(self.kappa, 4), "x0": np.round(self.x0, 4).tolist(),
                "focal_px": None if self.kappa < 1e-6 else round(self.s / self.kappa, 1)}


def fit_affine(X, u, w0, iters=12):
    """u ≈ A X + b (가중 최소제곱 + Huber 재가중). X (n,3) · u (n,2) → A (2,3), b (2,), 잔차 (n,)."""
    Xh = np.concatenate([X, np.ones((len(X), 1))], 1)
    w = w0.astype(np.float64).copy()
    for _ in range(iters):
        sw = np.sqrt(w)[:, None]
        sol = np.linalg.lstsq(Xh * sw, u * sw, rcond=None)[0]      # (4, 2)
        r = np.linalg.norm(Xh @ sol - u, axis=1)
        delta = 1.345 * max(float(np.median(r)) / 0.6745, 1e-6)
        w = w0 * np.where(r <= delta, 1.0, delta / np.maximum(r, 1e-12))
    return sol[:3].T, sol[3], r


def splat_zbuffer(u, depth, rad, shape):
    """점마다 반지름 rad[px] 원판을 그려 가장 가까운 깊이만 남긴다 → (H,W), 빈 곳 inf."""
    H, W = shape
    zb = np.full(H * W, np.inf)
    if len(u) == 0:
        return zb.reshape(H, W)
    ui = np.round(u).astype(np.int64)
    r = int(np.ceil(float(rad.max())))
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            m = dx * dx + dy * dy <= rad ** 2
            x, y = ui[:, 0] + dx, ui[:, 1] + dy
            m &= (x >= 0) & (x < W) & (y >= 0) & (y < H)
            np.minimum.at(zb, y[m] * W + x[m], depth[m])
    return zb.reshape(H, W)


def splat_colors(u, depth, rad, colors, shape):
    """가장 앞의 점 색으로 원판을 칠한다 → (H,W,3) 이미지, 칠해진 마스크."""
    H, W = shape
    zb = splat_zbuffer(u, depth, rad, shape)
    img = np.zeros((H, W, 3))
    if len(u) == 0:
        return img, np.isfinite(zb)
    ui = np.round(u).astype(np.int64)
    r = int(np.ceil(float(rad.max())))
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            m = np.nonzero(dx * dx + dy * dy <= rad ** 2)[0]
            x, y = ui[m, 0] + dx, ui[m, 1] + dy
            ok = (x >= 0) & (x < W) & (y >= 0) & (y < H)
            m, x, y = m[ok], x[ok], y[ok]
            front = depth[m] <= zb[y, x] + 1e-12
            img[y[front], x[front]] = colors[m[front]]
    return img, np.isfinite(zb)


def visible_mask(u, depth, zb, tol):
    H, W = zb.shape
    ui = np.round(u).astype(np.int64)
    inside = (ui[:, 0] >= 0) & (ui[:, 0] < W) & (ui[:, 1] >= 0) & (ui[:, 1] < H)
    vis = np.zeros(len(u), bool)
    vis[inside] = depth[inside] <= zb[ui[inside, 1], ui[inside, 0]] + tol
    return vis


def render_voxels(cam, Xv, shape, grow=1.25):
    """복셀 중심을 카메라로 투영해 z-buffer → (덮인 픽셀, u, 깊이, 보임)."""
    u, z, den = cam.project(Xv)
    rad = 0.5 * grow * cam.s * (1.0 / VOX_RES) / den
    zb = splat_zbuffer(u, z, rad, shape)
    return np.isfinite(zb), u, z, visible_mask(u, z, zb, 2.0 / VOX_RES)


def iou(a, b):
    return float((a & b).sum() / max((a | b).sum(), 1))


class SilhouetteCost:
    """카메라가 물체 실루엣과 얼마나 안 맞나 (픽셀). 복셀 '중심' 이 아니라 반지름 r = margin·(복셀 한 칸의 투영 크기) 의
    '발자국' 이 실루엣과 맞도록:
      안쪽 항  복셀 중심이 실루엣을 r 만큼 침식한 영역 안에 있어야 한다 (밖 거리 + 모자란 안쪽 여유)
      바깥 항  실루엣 경계 픽셀마다 가장 가까운 복셀 중심까지 거리 − r (넘는 만큼만)
    중심끼리만 맞추면 경계 픽셀이 늘 반 칸쯤 떨어져 있어 카메라가 부풀려진다 (합성 시험에서 IoU 0.95 → 0.88)."""

    def __init__(self, alpha, margin=0.6, n_bnd=4000, seed=0):
        rng = np.random.default_rng(seed)
        H, W = alpha.shape
        self.pad = pad = max(H, W) // 2
        big = np.zeros((H + 2 * pad, W + 2 * pad), bool)
        big[pad:pad + H, pad:pad + W] = alpha
        self.dt_out = ndimage.distance_transform_edt(~big)
        self.dt_in = ndimage.distance_transform_edt(big)
        by, bx = np.nonzero(alpha & ~ndimage.binary_erosion(alpha))
        bnd = np.stack([bx, by], 1).astype(np.float64)
        self.bnd = bnd[rng.choice(len(bnd), n_bnd, replace=False)] if len(bnd) > n_bnd else bnd
        ys, xs = np.nonzero(alpha)
        self.center = np.array([xs.mean(), ys.mean()])
        self.radius = float(np.sqrt(((xs - xs.mean()) ** 2 + (ys - ys.mean()) ** 2).mean()))
        self.margin = margin

    def __call__(self, cam, X):
        u, _, den = cam.project(X)
        if not np.all(np.isfinite(u)):
            return 1e9
        r = self.margin * cam.s / VOX_RES / den
        at = [u[:, 1] + self.pad, u[:, 0] + self.pad]
        f = (ndimage.map_coordinates(self.dt_out, at, order=1, mode="nearest")
             + np.maximum(r - ndimage.map_coordinates(self.dt_in, at, order=1, mode="nearest"), 0)).mean()
        d, j = cKDTree(u).query(self.bnd)
        return float(f + np.maximum(d - r[j], 0).mean())


def refine_camera(cam0, Xv, alpha, X_att=None, u_att=None, w_att=None, lam=0.01, margin=0.6, seed=0, sil=None,
                  kappa_starts=(0.0, 0.3), kappa_max=0.7, maxfev=6000):
    """실루엣에 맞춰 카메라 다듬기 (Powell). lam·(attention 무게중심 잔차, Huber) 는 대칭인 해 사이의 선택만 돕도록 작게
    둔다 (무게중심은 가운데로 쏠려 있다). κ ≤ kappa_max: TRELLIS 학습 시점의 화각 10~70° 에 해당하는 범위."""
    sil = sil or SilhouetteCost(alpha, margin, seed=seed)
    H, W = alpha.shape
    use_att = X_att is not None and lam > 0 and len(X_att) > 0
    if use_att:
        wa = w_att / w_att.sum()
        d_h = 0.04 * max(H, W)                                   # Huber 폭 [px]

    def cost(th):
        cam = cam0.perturbed(th)
        f = sil(cam, Xv)
        if use_att and f < 1e8:
            ua, _, _ = cam.project(X_att)
            r = np.linalg.norm(ua - u_att, axis=1)
            f += lam * float(wa @ np.where(r < d_h, 0.5 * r * r / d_h, r - 0.5 * d_h))
        return float(f)

    bounds = [(-0.6, 0.6)] * 3 + [(-0.8, 0.8), (-2.0, 2.0), (-2.0, 2.0), (0.0, kappa_max)]
    best = None
    for k0 in kappa_starts:
        th0 = np.zeros(7)
        th0[6] = min(max(k0, 0.0), kappa_max)
        res = minimize(cost, th0, method="Powell", bounds=bounds,
                       options={"xtol": 1e-4, "ftol": 1e-7, "maxfev": maxfev})
        if best is None or res.fun < best[1]:
            best = (cam0.perturbed(res.x), float(res.fun))
    return best


def look_rotation(az, el):
    """물체를 보는 카메라 회전 (내부 좌표 z 위, 굴림 0). az: z 축 둘레 [rad], el: 내려다보는 각 [rad]."""
    f = np.array([np.sin(az) * np.cos(el), np.cos(az) * np.cos(el), -np.sin(el)])
    r1 = np.cross(f, [0.0, 0.0, 1.0])
    r1 /= np.linalg.norm(r1)
    return np.stack([r1, np.cross(f, r1), f])


def search_cameras(Xv, sil, n_az=24, els_deg=(-10, 5, 20, 35, 50), kappas=(0.25, 0.5), n_sub=4000, top=6, seed=0):
    """attention 시작점이 틀렸을 때를 위한 전역 탐색: 방위 × 고도 × 원근 격자마다 실루엣 중심 · 크기를 맞춘 카메라를
    만들어 실루엣 비용이 가장 낮은 top 개 → [(비용, 카메라, (방위°, 고도°, κ))]."""
    rng = np.random.default_rng(seed)
    Xs = Xv[rng.choice(len(Xv), min(n_sub, len(Xv)), replace=False)]
    x0 = Xv.mean(0)
    out = []
    for az in np.arange(n_az) * 2 * np.pi / n_az:
        for el in np.radians(els_deg):
            R = look_rotation(az, el)
            for kp in kappas:
                u, _, _ = Camera(R, 1.0, (0.0, 0.0), kp, x0).project(Xs)
                m = u.mean(0)
                sc = 1.1 * sil.radius / max(float(np.sqrt(((u - m) ** 2).sum(1).mean())), 1e-9)   # 껍질 투영은 가장자리가 짙다
                cam = Camera(R, sc, sil.center - sc * m, kp, x0)
                out.append((sil(cam, Xs), cam, (round(float(np.degrees(az))), round(float(np.degrees(el))), kp)))
    out.sort(key=lambda t: t[0])
    return out[:top]


def estimate_camera(Xv, alpha, cam_att, att=None, vox_col=None, cond_rgb=None, color_w=0.5, n_sub=4000, seed=0,
                    log=print):
    """입력 사진의 카메라: attention affine 시작점 + 전역 탐색 후보들을 (복셀 일부로) 다듬고, 실루엣 IoU + color_w·색 상관이
    가장 높은 것을 모든 복셀로 한 번 더 다듬는다. 색 상관은 실루엣이 같은 앞 · 뒤(좌우 반전)를 가른다.
    → (카메라, 통계)."""
    rng = np.random.default_rng(seed)
    Xs = Xv[rng.choice(len(Xv), min(n_sub, len(Xv)), replace=False)]
    sil = SilhouetteCost(alpha, seed=seed)
    att = att or (None, None, None)
    starts = [("attention", cam_att, (0.0, 0.3))]
    starts += [(f"search az{a} el{e} k{k}", c, (k,)) for _, c, (a, e, k) in search_cameras(Xv, sil, seed=seed)]
    tried = []
    for name, c0, ks in starts:
        cam, _ = refine_camera(c0, Xs, alpha, *att, sil=sil, kappa_starts=ks, maxfev=1500)
        fp, u, _, vis = render_voxels(cam, Xv, alpha.shape)
        ci = color_agreement(vox_col, cond_rgb, u, vis) if vox_col is not None and cond_rgb is not None else 0.0
        tried.append({"start": name, "iou": round(iou(fp, alpha), 4), "color_corr": round(ci, 4),
                      "score": iou(fp, alpha) + color_w * max(ci, 0.0), "cam": cam})
    tried.sort(key=lambda t: -t["score"])
    best = tried[0]
    cam, cost = refine_camera(best["cam"], Xv, alpha, *att, sil=sil, kappa_starts=(best["cam"].kappa,))
    log("[parts] camera starts: " + " | ".join(f"{t['start']}: IoU {t['iou']:.3f} colour {t['color_corr']:.2f}"
                                               for t in tried[:4]))
    return cam, {"chosen_start": best["start"], "fit_cost": round(cost, 3),
                 "starts": [{k: v for k, v in t.items() if k not in ("cam", "score")} for t in tried]}


# ------------------------------------------------------------------------------------------ 3 투영 투표
def erode_labels(label, K, px):
    """부위 마스크마다 px 만큼 침식 (부위 경계 · 실루엣 경계의 몇 픽셀은 '모름' 으로)."""
    if px <= 0:
        return label.copy()
    out = np.zeros_like(label)
    for k in range(1, K + 1):
        m = label == k
        if m.any():
            out[ndimage.binary_erosion(m, iterations=px)] = k
    return out


def projection_labels(u, vis, label):
    """보이는 복셀이 투영된 픽셀의 2D 라벨 (0 = 모름)."""
    H, W = label.shape
    ui = np.round(u).astype(np.int64)
    ok = vis & (ui[:, 0] >= 0) & (ui[:, 0] < W) & (ui[:, 1] >= 0) & (ui[:, 1] < H)
    lab = np.zeros(len(u), np.int64)
    lab[ok] = label[ui[ok, 1], ui[ok, 0]]
    return lab


def color_agreement(vox_col, img, u, vis):
    """보이는 복셀 색 vs 입력 사진 색: 밝기 상관계수 (카메라가 맞으면 높다 — 참고 지표)."""
    H, W = img.shape[:2]
    ui = np.round(u).astype(np.int64)
    ok = vis & (ui[:, 0] >= 0) & (ui[:, 0] < W) & (ui[:, 1] >= 0) & (ui[:, 1] < H)
    if ok.sum() < 10:
        return 0.0
    lum = np.array([0.299, 0.587, 0.114])
    a = vox_col[ok] @ lum
    b = img[ui[ok, 1], ui[ok, 0]].astype(np.float64) @ lum
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


# ------------------------------------------------------------------------------------------ 4 전파 · 5 정리
def legacy_feature_graph(tc, F, knn, radius=4.0):
    """예전 그래프 그대로: 반경 안에서 특징이 가장 비슷한 knn 개, exp((cos−1)/0.1), 대칭 · 행 정규화."""
    tree = cKDTree(tc)
    nb = tree.query_ball_point(tc, r=radius)
    rows, cols, vals = [], [], []
    for i, js in enumerate(nb):
        js = np.array([j for j in js if j != i])
        if len(js) == 0:
            continue
        sim = F[js] @ F[i]
        top = js[np.argsort(-sim)[:knn]]
        w = np.exp((F[top] @ F[i] - 1) / 0.1)
        rows += [i] * len(top)
        cols += list(top)
        vals += list(w)
    Wm = csr_matrix((vals, (rows, cols)), shape=(len(tc), len(tc)))
    Wm = Wm + Wm.T
    d = np.asarray(Wm.sum(1)).ravel()
    return csr_matrix(Wm.multiply(1 / np.maximum(d, 1e-9)[:, None]))


def surface_graph(tc, F, radius=1.8, tau=0.1):
    """맞닿은 토큰(26-이웃)끼리만 잇고 DiT 특징 유사도로 가중 → (행 정규화 W, 간선 쌍)."""
    n = len(tc)
    pairs = cKDTree(tc).query_pairs(r=radius, output_type="ndarray")
    if len(pairs) == 0:
        return csr_matrix((n, n)), np.zeros((0, 2), np.int64)
    w = np.exp(((F[pairs[:, 0]] * F[pairs[:, 1]]).sum(1) - 1) / tau)
    Wm = coo_matrix((np.r_[w, w], (np.r_[pairs[:, 0], pairs[:, 1]], np.r_[pairs[:, 1], pairs[:, 0]])),
                    shape=(n, n)).tocsr()
    d = np.asarray(Wm.sum(1)).ravel()
    return csr_matrix(Wm.multiply(1 / np.maximum(d, 1e-9)[:, None])), pairs


def propagate(P0, Wn, alpha, iters, clamp=None):
    """Y ← α·W·Y + (1−α)·P0, clamp 된 행은 매번 P0 로 되돌린다 (Zhou et al. / Zhu & Ghahramani)."""
    Y = P0.copy()
    for _ in range(iters):
        Y = alpha * (Wn @ Y) + (1 - alpha) * P0
        if clamp is not None:
            Y[clamp] = P0[clamp]
    return Y / np.maximum(Y.sum(1, keepdims=True), 1e-9)


def cleanup_fragments(lab, pairs, K, min_size=16, rel=0.25, rounds=3):
    """같은 부위끼리 이어진 조각 중 (그 부위의 가장 큰 조각이 아니고) min_size 또는 rel×가장 큰 조각보다 작은 것은
    맞닿은 다른 부위 중 가장 많이 닿은 부위로 바꾼다. 양팔처럼 비슷한 크기의 조각 여러 개는 그대로 남는다."""
    lab = lab.copy()
    n = len(lab)
    changed = np.zeros(n, bool)
    if len(pairs) == 0:
        return lab, changed
    a, b = pairs[:, 0], pairs[:, 1]
    for _ in range(rounds):
        same = lab[a] == lab[b]
        G = coo_matrix((np.ones(int(same.sum())), (a[same], b[same])), shape=(n, n))
        nc, comp = connected_components(G, directed=False)
        size = np.bincount(comp, minlength=nc)
        clab = np.zeros(nc, np.int64)
        clab[comp] = lab
        largest = np.zeros(K, np.int64)
        np.maximum.at(largest, clab, size)
        small = (size != largest[clab]) & (size < np.maximum(min_size, rel * largest[clab]))
        if not small.any():
            break
        cross = comp[a] != comp[b]
        votes = np.zeros((nc, K))
        np.add.at(votes, (comp[a[cross]], lab[b[cross]]), 1.0)
        np.add.at(votes, (comp[b[cross]], lab[a[cross]]), 1.0)
        tgt = small & (votes.sum(1) > 0)
        if not tgt.any():
            break
        m = tgt[comp]
        lab[m] = votes.argmax(1)[comp[m]]
        changed |= m
    return lab, changed


# ------------------------------------------------------------------------------------------ 7 가려진 쪽: 부피
_OFF26 = np.array([o for o in ((a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)) if o != (0, 0, 0)])
_FWD = _OFF26[[tuple(o) > (0, 0, 0) for o in _OFF26]]           # 한쪽 13 방향 (간선 중복 없이)


def _grid_graph(mask, node_cost=None):
    """mask 복셀끼리 26-이웃 그래프 → (csr, 노드 번호 격자). 간선 길이 × 두 끝 비용 평균 (없으면 길이만)."""
    pts = np.argwhere(mask)
    idx = np.full(mask.shape, -1, np.int64)
    idx[pts[:, 0], pts[:, 1], pts[:, 2]] = np.arange(len(pts))
    rows, cols, w = [], [], []
    for o in _FWD:
        q = pts + o
        j = idx[q[:, 0], q[:, 1], q[:, 2]]
        ok = j >= 0
        a = np.nonzero(ok)[0]
        L = np.full(len(a), np.linalg.norm(o))
        rows.append(a)
        cols.append(j[ok])
        w.append(L if node_cost is None else L * 0.5 * (node_cost[a] + node_cost[j[ok]]))
    G = coo_matrix((np.concatenate(w), (np.concatenate(rows), np.concatenate(cols))), shape=(len(pts),) * 2).tocsr()
    return G, idx, pts


def _multi_source(G, sources_by_part, K, n):
    from scipy.sparse.csgraph import dijkstra
    d = np.full((n, K), np.inf)
    for k in range(K):
        s = np.unique(sources_by_part[k])
        if len(s):
            d[:, k] = dijkstra(G, directed=False, indices=s, min_only=True)
    return d


def volume_part_distances(vox_coords, seeds, K, power=2.0, core_frac=0.55, win=9, pad=2):
    """가려진 표면 복셀이 어느 부위의 '부피' 에 속하는지 — 보이는 씨앗(표면 복셀 번호, 부위별)에서 두 가지 거리.
    껍질을 속이 찬 부피 S 로 채우고 D = 표면까지 거리.
      d_geo   S 안 최단 경로, 비용 (1/D)^power: 두꺼운 속은 싸고 얇은 목(팔 · 몸통이 닿은 곳)은 비싸다
      d_core  각 복셀이 D 를 따라 올라가 닿는 '속심'(D ≥ core_frac × 주변 최대) 끼리의 거리: 배와 등은 같은 몸통 속심으로,
              팔 뒤는 팔 속심으로 간다
    → (d_geo (V,K), d_core (V,K), 통계) · 껍질이 닫혀 있지 않아 채워지지 않으면 None."""
    c = np.asarray(vox_coords, np.int64) + pad
    n = VOX_RES + 2 * pad
    O = np.zeros((n, n, n), bool)
    O[c[:, 0], c[:, 1], c[:, 2]] = True
    for ax in range(3):                                         # 격자 면에서 잘린 단면은 막는다 (물체가 격자 끝에 닿을 때)
        for i in (pad, pad + VOX_RES - 1):
            sl = [slice(None)] * 3
            sl[ax] = i
            O[tuple(sl)] |= ndimage.binary_fill_holes(O[tuple(sl)])
    for it in (1, 2):                                           # 1~2칸 틈은 메워 본다
        S = ndimage.binary_fill_holes(ndimage.binary_closing(O, iterations=it) | O)
        if S.sum() >= 1.15 * O.sum():
            break
    else:
        return None
    D = ndimage.distance_transform_edt(S)
    V = len(c)
    G, idx, _ = _grid_graph(S, 1.0 / np.maximum(D[S], 1.0) ** power)      # D[S] = np.argwhere(S) 순서
    node = idx[c[:, 0], c[:, 1], c[:, 2]]
    d_geo = _multi_source(G, [node[s] for s in seeds], K, G.shape[0])[node]
    core = S & (D >= core_frac * ndimage.maximum_filter(D, size=win)) & (D >= 1.5)
    P = c.copy()                                                # 가장 가파르게 D 를 올라가 속심에 닿을 때까지
    for _ in range(2 * VOX_RES):
        act = np.nonzero(~core[P[:, 0], P[:, 1], P[:, 2]])[0]
        if not len(act):
            break
        Q = P[act][:, None, :] + _OFF26[None]
        Dn = D[Q[..., 0], Q[..., 1], Q[..., 2]]
        j = Dn.argmax(1)
        mv = Dn[np.arange(len(act)), j] > D[P[act, 0], P[act, 1], P[act, 2]]
        if not mv.any():
            break
        P[act[mv]] = Q[np.nonzero(mv)[0], j[mv]]
    nodes = core.copy()
    nodes[P[:, 0], P[:, 1], P[:, 2]] = True
    Gc, idc, _ = _grid_graph(nodes)
    ent = idc[P[:, 0], P[:, 1], P[:, 2]]
    d_core = _multi_source(Gc, [ent[s] for s in seeds], K, Gc.shape[0])[ent]
    return d_geo, d_core, {"solid_voxels": int(S.sum()), "shell_voxels": int(V), "core_voxels": int(core.sum()),
                           "max_depth_vox": round(float(D.max()), 2)}


def soft_nearest(d, tau):
    """거리 (V,K) → 행마다 softmax(−(d − min)/tau). 모두 inf 인 행은 균등."""
    p = np.full(d.shape, 1.0 / d.shape[1])
    ok = np.isfinite(d).any(1)
    e = np.exp(-(d[ok] - d[ok].min(1, keepdims=True)) / tau)
    p[ok] = e / e.sum(1, keepdims=True)
    return p


# ------------------------------------------------------------------------------------------ 전체
def lift(sem, label2d, names, *, mode="proj", alpha_mask=None, cond_rgb=None, vox_col=None,
         attn_blocks=(4, 8, 12), feat_blocks=(6, 12), knn=12, alpha=0.9, iters=None, radius=None,
         erode=2, gate_iou=0.7, gate_color=0.2, clamp_purity=0.75, w_attn_vis=0.25, cleanup=True, vox_refine=0.5,
         geo=0.8, geo_power=2.0, geo_tau=0.3, core_tau=2.0, log=print):
    """→ dict: tok_prob (T,K) · tok_prob_attn · tok_part · vox_prob (V,K) · vox_part · vox_conf · v2t · (proj 이면)
    vox_visible · vox_proj_label · camera, 그리고 stats."""
    K = len(names)
    M = patch_fractions(label2d, K)
    Ar = relative_attention(sem, attn_blocks)
    S_attn = attention_votes(Ar, M)
    tc = sem["tok_coords"].astype(np.float32)
    F = dit_features(sem, feat_blocks)
    v2t = vox_to_tok(sem)
    T, V = len(tc), len(v2t)
    out = {"tok_prob_attn": S_attn.astype(np.float32), "v2t": v2t}
    st = {"mode": mode, "tokens": int(T), "voxels": int(V),
          "token_share_attn": {n: float(np.mean(S_attn.argmax(1) == k)) for k, n in enumerate(names)}}

    def legacy():
        Wn = legacy_feature_graph(tc, F, knn, radius or 4.0)
        return propagate(S_attn, Wn, alpha, iters or 40)

    if mode == "attn":
        Y = legacy()
        tok_part = Y.argmax(1)
        vox_prob = Y[v2t]
    else:
        size = label2d.shape[0]
        if alpha_mask is None:
            alpha_mask = label2d > 0
        Xt, Xv = token_centers(sem), voxel_centers(sem)
        u_att, spread, ok = attention_centroids(Ar, size)
        w0 = ok / (spread + size / N_PATCH)
        A, b, _ = fit_affine(Xt[ok], u_att[ok], w0[ok])
        cam0 = Camera.from_affine(A, b, Xv.mean(0))
        fp0, *_ = render_voxels(cam0, Xv, alpha_mask.shape)
        cam, cst = estimate_camera(Xv, alpha_mask, cam0, (Xt[ok], u_att[ok], w0[ok]), vox_col, cond_rgb, log=log)
        fp, u, z, vis = render_voxels(cam, Xv, alpha_mask.shape)
        st["camera"] = {**cam.as_dict(), "silhouette_iou_affine_init": round(iou(fp0, alpha_mask), 4),
                        "silhouette_iou": round(iou(fp, alpha_mask), 4), **cst}
        if vox_col is not None and cond_rgb is not None:
            st["camera"]["color_corr_visible"] = round(color_agreement(vox_col, cond_rgb, u, vis), 4)
        log(f"[parts] camera: silhouette IoU {st['camera']['silhouette_iou_affine_init']:.3f} (attention affine) -> "
            f"{st['camera']['silhouette_iou']:.3f} ({st['camera']['chosen_start']}, refined), kappa {cam.kappa:.3f}"
            + (f", colour corr {st['camera']['color_corr_visible']:.3f}" if "color_corr_visible" in st["camera"] else ""))
        out["camera"] = cam
        bad_color = st["camera"].get("color_corr_visible", 1.0) < gate_color
        if st["camera"]["silhouette_iou"] < gate_iou or bad_color:
            log(f"[parts] WARNING camera not trusted (silhouette IoU < {gate_iou} or colour corr < {gate_color}) "
                "-> attention-only (legacy) labels. Check camera_fit.png")
            st["fallback_to_attn"] = True
            Y = legacy()
            tok_part = Y.argmax(1)
            vox_prob = Y[v2t]
        else:
            st["fallback_to_attn"] = False
            lab_v = projection_labels(u, vis, erode_labels(label2d, K, erode))
            S_proj = np.zeros((T, K))
            m = lab_v > 0
            np.add.at(S_proj, (v2t[m], lab_v[m] - 1), 1.0)
            n_proj = S_proj.sum(1)
            Sp = S_proj / np.maximum(n_proj, 1e-9)[:, None]
            has = n_proj > 0
            P0 = S_attn.astype(np.float64).copy()
            P0[has] = Sp[has] + w_attn_vis * S_attn[has]
            P0 /= np.maximum(P0.sum(1, keepdims=True), 1e-9)
            clamp = (n_proj >= 2) & (Sp.max(1) >= clamp_purity)
            P0[clamp] = Sp[clamp]
            Wn, pairs = surface_graph(tc, F, radius or 1.8)
            Y = propagate(P0, Wn, alpha, iters or 120, clamp)
            tok_part = Y.argmax(1)
            n_changed = 0
            if cleanup:
                tok_part, changed = cleanup_fragments(tok_part, pairs, K)
                n_changed = int(changed.sum())
                Y[changed] = 0.4 * Y[changed] + 0.6 * np.eye(K)[tok_part[changed]]
            vox_prob = Y[v2t].copy()
            if vox_refine > 0:
                vm = lab_v > 0
                vox_prob[vm] = (1 - vox_refine) * vox_prob[vm] + vox_refine * np.eye(K)[lab_v[vm] - 1]
            # 6 사진에 라벨이 없는 복셀(가려진 뒤 · 옆, 2D 미분류): 부피 안 측지 거리로 가장 가까운 '보이는 부위' 씨앗
            #   표면 그래프 전파는 씨앗에서 표면을 따라 번져, 씨앗이 넓은 부위(팔)가 옆 · 뒤 몸통까지 먹는다.
            #   부피 안 경로는 두꺼운 속을 지나면 싸고 팔 · 몸통이 닿은 얇은 목은 비싸서, 몸통 뒤는 배 쪽 씨앗에 붙는다.
            if geo > 0:
                seed_ok = (lab_v > 0) & clamp[v2t] & (tok_part[v2t] == lab_v - 1)
                seeds = [np.nonzero(seed_ok & (lab_v == k + 1))[0] for k in range(K)]
                vd = volume_part_distances(sem["slat_coords"], seeds, K, power=geo_power)
                if vd is None:
                    log("[parts] shell is not closed -> volumetric step skipped")
                else:
                    d_geo, d_core, vst = vd
                    Pv = soft_nearest(d_core, core_tau) * soft_nearest(d_geo, geo_tau)
                    Pv /= np.maximum(Pv.sum(1, keepdims=True), 1e-12)
                    free = lab_v == 0
                    before = vox_prob[free].argmax(1)
                    vox_prob[free] = (1 - geo) * vox_prob[free] + geo * Pv[free]
                    st["volumetric"] = {**vst, "seeds": {n: int(len(seeds[k])) for k, n in enumerate(names)},
                                        "voxels_changed": int((vox_prob[free].argmax(1) != before).sum()),
                                        "weight": geo, "power": geo_power}
            if cleanup:                                             # 복셀 단위 작은 조각 (투영 · 부피 단계의 점 잡음)
                vpairs = cKDTree(sem["slat_coords"].astype(np.float32)).query_pairs(r=1.8, output_type="ndarray")
                vlab, vch = cleanup_fragments(vox_prob.argmax(1), vpairs, K, min_size=48)
                vox_prob[vch] = 0.4 * vox_prob[vch] + 0.6 * np.eye(K)[vlab[vch]]
                st["voxels_cleaned"] = int(vch.sum())
            out.update({"vox_visible": vis, "vox_proj_label": (lab_v - 1).astype(np.int8)})
            st.update({"visible_voxels": float(vis.mean()), "visible_labeled_voxels": float(m.mean()),
                       "tokens_with_projection": float(has.mean()), "tokens_clamped": float(clamp.mean()),
                       "tokens_cleaned": n_changed, "erode_px": erode, "graph_radius": radius or 1.8,
                       "iters": iters or 120})
    vox_part = vox_prob.argmax(1)
    out.update({"tok_prob": Y.astype(np.float32), "tok_part": tok_part, "vox_prob": vox_prob.astype(np.float32),
                "vox_part": vox_part, "vox_conf": vox_prob.max(1).astype(np.float32)})
    st.update({"voxel_share": {n: float(np.mean(vox_part == k)) for k, n in enumerate(names)},
               "token_share": {n: float(np.mean(tok_part == k)) for k, n in enumerate(names)},
               "changed_vs_attention": float(np.mean(S_attn.argmax(1) != tok_part)),
               "conf_median": float(np.median(Y.max(1)))})
    return out, st
