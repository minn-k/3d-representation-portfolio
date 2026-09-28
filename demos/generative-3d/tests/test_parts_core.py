"""parts_core 합성 시험 (GPU · 모델 불필요): 곰 모양 SDF 껍질 + 알려진 카메라로 2D 라벨 · 잡음 섞인 attention ·
DiT 비슷한 특징을 만들고, 예전 방식(mode attn)과 새 방식(mode proj)의 부위 정확도를 비교한다.

  python tests/test_parts_core.py          (또는 python -m pytest tests)

잡음 모형: 토큰 attention = 자기 투영 위치 가우시안 + 토큰마다 다른 '털 패치' 덩어리(외관 혼동) + 눈 · 코 · 나비넥타이
(모든 토큰이 보는 salient 패치). 가려진 토큰은 위치 성분이 약하다. 2D 분할은 몸통 옆구리를 비워 둔다 (SAM 이 배만 잡은 것처럼).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import parts_core as pc  # noqa: E402

NAMES = ["head", "arm", "body", "leg", "bow tie"]
H = W = 518


def _sd_sphere(X, c, r):
    return np.linalg.norm(X - c, axis=1) - r


def _sd_ellipsoid(X, c, r):
    q = (X - c) / r
    k = np.linalg.norm(q, axis=1)
    return (k - 1.0) * np.min(r)


def _sd_capsule(X, a, b, r):
    ab = b - a
    t = np.clip(((X - a) @ ab) / (ab @ ab), 0, 1)
    return np.linalg.norm(X - (a + t[:, None] * ab), axis=1) - r


def bear_sdf(X):
    """(부위별 SDF (N, 5) — 열 순서 NAMES)"""
    head = np.minimum.reduce([_sd_sphere(X, np.array([0, 0.02, 0.22]), 0.17),
                              _sd_sphere(X, np.array([0.12, 0.03, 0.36]), 0.06),
                              _sd_sphere(X, np.array([-0.12, 0.03, 0.36]), 0.06)])
    arm = np.minimum(_sd_capsule(X, np.array([0.13, -0.02, 0.06]), np.array([0.21, -0.13, -0.12]), 0.055),
                     _sd_capsule(X, np.array([-0.13, -0.02, 0.06]), np.array([-0.21, -0.13, -0.12]), 0.055))
    body = _sd_ellipsoid(X, np.array([0, 0, -0.06]), np.array([0.15, 0.12, 0.19]))
    leg = np.minimum(_sd_capsule(X, np.array([0.08, -0.02, -0.2]), np.array([0.13, -0.27, -0.25]), 0.075),
                     _sd_capsule(X, np.array([-0.08, -0.02, -0.2]), np.array([-0.13, -0.27, -0.25]), 0.075))
    bow = _sd_ellipsoid(X, np.array([0, -0.125, 0.05]), np.array([0.06, 0.02, 0.025]))
    return np.stack([head, arm, body, leg, bow], 1)


def look_camera(az_deg, el_deg, s, kappa, c):
    az, el = np.radians(az_deg), np.radians(el_deg)
    f = np.array([np.sin(az) * np.cos(el), np.cos(az) * np.cos(el), -np.sin(el)])     # 앞(-y)에서 +y 쪽을 본다
    r1 = np.cross(f, [0, 0, 1.0])
    r1 /= np.linalg.norm(r1)
    r2 = np.cross(f, r1)
    return pc.Camera(np.stack([r1, r2, f]), s, c, kappa, np.zeros(3))


def make_scene(seed=0):
    rng = np.random.default_rng(seed)
    g = (np.arange(64) + 0.5) / 64 - 0.5
    X = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    coords = np.stack(np.meshgrid(*(np.arange(64),) * 3, indexing="ij"), -1).reshape(-1, 3)
    D = bear_sdf(X)
    sd = D.min(1)
    shell = np.abs(sd) <= 0.9 / 64
    Xv, cv, gt_v = X[shell], coords[shell], D[shell].argmin(1)
    tok = np.unique(cv // 2, axis=0)
    sem = {"tok_coords": tok.astype(np.int16), "slat_coords": cv.astype(np.int16)}
    v2t = pc.vox_to_tok(sem)
    T = len(tok)
    gt_t = np.array([np.bincount(gt_v[v2t == t], minlength=5).argmax() for t in range(T)])

    # 색: 털 갈색, 나비넥타이 · 눈 · 코 어둡게
    col = np.tile([0.62, 0.46, 0.30], (len(Xv), 1)) + rng.normal(0, 0.03, (len(Xv), 3))
    col[gt_v == 4] = [0.08, 0.08, 0.12]
    for c in ([0.05, -0.14, 0.26], [-0.05, -0.14, 0.26], [0, -0.17, 0.2]):
        col[np.linalg.norm(Xv - c, axis=1) < 0.028] = [0.05, 0.04, 0.04]
    col = np.clip(col, 0, 1)

    cam = look_camera(-30, 12, 560.0, 0.35, np.array([259.0, 262.0]))
    u, z, den = cam.project(Xv)
    rad = 0.5 * 1.3 * cam.s / 64 / den
    zb = pc.splat_zbuffer(u, z, rad, (H, W))
    alpha = np.isfinite(zb)
    vis = pc.visible_mask(u, z, zb, 2.0 / 64)
    # 2D 라벨 · 사진: 보이는 복셀을 앞에서부터 칠한다
    label = np.zeros((H, W), np.int32)
    img = np.zeros((H, W, 3))
    order = np.argsort(-z)
    ui = np.round(u).astype(int)
    r = int(np.ceil(rad.max()))
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            m = order[np.hypot(dx, dy) <= rad[order]]
            x, y = ui[m, 0] + dx, ui[m, 1] + dy
            ok = (x >= 0) & (x < W) & (y >= 0) & (y < H)
            m, x, y = m[ok], x[ok], y[ok]
            front = z[m] <= zb[y, x] + 1e-9
            label[y[front], x[front]] = gt_v[m[front]] + 1
            img[y[front], x[front]] = col[m[front]]
    # SAM 흉내: 몸통은 배만 (옆구리 = 미분류)
    flank = np.zeros((H, W), bool)
    bx = np.nonzero((gt_v == 2) & vis & (np.abs(Xv[:, 0]) > 0.085))[0]
    flank[np.clip(ui[bx, 1], 0, H - 1), np.clip(ui[bx, 0], 0, W - 1)] = True
    from scipy import ndimage
    label[ndimage.binary_dilation(flank, iterations=4) & (label == 3)] = 0

    # attention (토큰 × (5 + 1369))
    Xt = pc.token_centers(sem)
    ut, zt, _ = cam.project(Xt)
    tvis = np.zeros(T, bool)
    np.logical_or.at(tvis, v2t, vis)
    P = pc.patch_centers(H)
    fur_patch = np.nonzero(np.isin(label[np.clip(P[:, 1].astype(int), 0, H - 1),
                                         np.clip(P[:, 0].astype(int), 0, W - 1)], [1, 2, 3, 4]))[0]
    sal = np.zeros(len(P))
    for c in ([0.05, -0.14, 0.26], [-0.05, -0.14, 0.26], [0, -0.17, 0.2], [0, -0.125, 0.05]):
        uc = cam.project(np.array([c]))[0][0]
        sal += np.exp(-np.sum((P - uc) ** 2, 1) / (2 * 14.0 ** 2))
    sal /= sal.sum()
    sig = 1.5 * H / 37
    attn = {}
    for b in (4, 8, 12):
        A = np.zeros((T, 5 + len(P)))
        blob_c = P[rng.choice(fur_patch, T)]
        for t in range(T):
            loc = np.exp(-np.sum((P - ut[t]) ** 2, 1) / (2 * sig ** 2))
            conf = np.exp(-np.sum((P - blob_c[t]) ** 2, 1) / (2 * (2.0 * sig) ** 2))
            wl = 0.5 if tvis[t] else 0.3
            a = wl * loc / loc.sum() + (0.8 - wl) * conf / conf.sum() + 0.2 * sal
            a *= rng.lognormal(0, 0.4, len(P))
            A[t, 5:] = 0.8 * a / a.sum()
            A[t, :5] = 0.04
        attn[b] = A
    for b in (4, 8, 12):
        sem[f"attn_{b}"] = attn[b].astype(np.float16)
    # DiT 특징: 털 공통 + 부위 + 위치 저주파 + 잡음 (부위 신호가 약해 같은 털끼리 비슷)
    E = rng.normal(0, 1, (6, 48))
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    Wf = rng.normal(0, 1, (3, 16)) * 3.0
    for b in (6, 12):
        pos = np.concatenate([np.sin(Xt @ Wf), np.cos(Xt @ Wf)], 1)
        fur = np.where(gt_t < 4, 1.0, 0.0)[:, None] * E[5]
        f = np.concatenate([0.35 * E[gt_t] + 0.6 * fur, 0.35 * pos[:, :16]], 1)
        sem[f"feat_{b}"] = (f + rng.normal(0, 0.25, f.shape)).astype(np.float16)
    return {"sem": sem, "label": label, "alpha": alpha, "img": img, "col": col, "gt_v": gt_v, "gt_t": gt_t,
            "cam": cam, "Xv": Xv, "vis": vis}


def _acc(pred, gt, k=None):
    m = np.ones(len(gt), bool) if k is None else gt == k
    return float(np.mean(pred[m] == gt[m]))


def run(verbose=True):
    sc = make_scene()
    log = print if verbose else (lambda *a, **k: None)
    res = {}
    for mode in ("attn", "proj"):
        out, st = pc.lift(sc["sem"], sc["label"], NAMES, mode=mode, alpha_mask=sc["alpha"], cond_rgb=sc["img"],
                          vox_col=sc["col"], log=log)
        vis = sc["vis"]
        res[mode] = {"vox_acc": _acc(out["vox_part"], sc["gt_v"]),
                     "visible_acc": float(np.mean(out["vox_part"][vis] == sc["gt_v"][vis])),
                     "hidden_acc": float(np.mean(out["vox_part"][~vis] == sc["gt_v"][~vis])),
                     "arm_acc": _acc(out["vox_part"], sc["gt_v"], 1),
                     "body_acc": _acc(out["vox_part"], sc["gt_v"], 2),
                     "leg_acc": _acc(out["vox_part"], sc["gt_v"], 3),
                     "tok_acc": _acc(out["tok_part"], sc["gt_t"])}
        if mode == "proj":
            cam = out["camera"]
            res[mode]["camera_px_err"] = float(np.median(np.linalg.norm(
                cam.project(sc["Xv"])[0] - sc["cam"].project(sc["Xv"])[0], axis=1)))
            res[mode]["silhouette_iou"] = st["camera"]["silhouette_iou"]
            res[mode]["visible_iou"] = float((out["vox_visible"] & sc["vis"]).sum() / max((out["vox_visible"] | sc["vis"]).sum(), 1))
        log(mode, {k: round(v, 4) for k, v in res[mode].items()})
    return res


def test_projection_lift_beats_attention_only():
    res = run(verbose=False)
    assert res["proj"]["silhouette_iou"] > 0.85
    assert res["proj"]["camera_px_err"] < 8.0
    assert res["proj"]["visible_acc"] > 0.95
    assert res["proj"]["vox_acc"] > res["attn"]["vox_acc"] + 0.05
    assert res["proj"]["arm_acc"] > res["attn"]["arm_acc"]
    assert res["proj"]["hidden_acc"] > res["attn"]["hidden_acc"] + 0.08          # 가려진 쪽 (부피 단계)
    assert res["proj"]["body_acc"] > res["attn"]["body_acc"] + 0.3


if __name__ == "__main__":
    run()
