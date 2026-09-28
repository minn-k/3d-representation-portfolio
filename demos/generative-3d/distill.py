"""생성 가우시안 줄이기 + 재최적화 (distillation): 원본(수십만 개) 렌더를 정답으로, 중요도 상위 N 개만 남겨 다시 맞춘다.

  C:\\anaconda\\anaconda3\\envs\\trellis\\python.exe distill.py --name bear --keep 100000 [--iters 3000]

- 입력: out/<name>/gaussian.ply (TRELLIS 원본, 위쪽 -y). 불투명도 0.02 미만은 먼저 뺀다 (to_model.py 와 같음).
- 초기값: 중요도(불투명도 × 두 큰 축 곱) 상위 N 개. 개수는 그대로 두고(분할·복제 없음) 위치·크기·회전·불투명도·색을 Adam 으로.
- 정답: 원본 전체를 무작위 시점(위 반구 + 약간 아래, 흰 배경, 512²)에서 렌더한 이미지. 손실 L1.
- 출력: out/<name>/distill_<N>.ply (gaussian.ply 와 같은 형식·좌표), distill_<N>.json (전/후 PSNR, 시간)
평가: 고정 8방향에서 전체 렌더 대비 PSNR — 잘라내기만 한 것(초기값)과 재최적화 후를 비교.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
from plyfile import PlyData, PlyElement

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from edit_demo import Cam, mat_to_quat, quat_to_mat  # noqa: E402
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer  # noqa: E402

SH_C0 = 0.28209479177387814
M = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)       # 회전: PLY 위쪽 -y → +z (det +1)
BG = torch.ones(3, device="cuda")


def raster(cam, xyz, scale, rot, opa, col):
    s = GaussianRasterizationSettings(image_height=cam.H, image_width=cam.W, tanfovx=cam.tanx, tanfovy=cam.tany, bg=BG,
                                      scale_modifier=1.0, viewmatrix=cam.view, projmatrix=cam.proj, sh_degree=0,
                                      campos=cam.center, prefiltered=False, debug=False, antialiasing=True)
    m2 = torch.zeros_like(xyz, requires_grad=True)
    return GaussianRasterizer(s)(means3D=xyz, means2D=m2, opacities=opa, colors_precomp=col, scales=scale,
                                 rotations=rot)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--keep", type=int, default=100000)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--res", type=int, default=512)
    args = ap.parse_args()
    out = os.path.join(HERE, "out", args.name)
    v = PlyData.read(os.path.join(out, "gaussian.ply"))["vertex"].data
    op = 1 / (1 + np.exp(-v["opacity"].astype(np.float64)))
    v = v[op >= 0.02]
    op = op[op >= 0.02]
    P = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64) @ M.T
    logS = np.stack([v[f"scale_{i}"] for i in range(3)], 1).astype(np.float64)
    q = np.stack([v[f"rot_{i}"] for i in range(4)], 1).astype(np.float64)
    q = mat_to_quat(M[None] @ quat_to_mat(q / np.linalg.norm(q, axis=1, keepdims=True)))
    fdc = np.stack([v[f"f_dc_{i}"] for i in range(3)], 1).astype(np.float64)
    ss = np.sort(np.exp(logS), 1)
    idx = np.argsort(-(op * ss[:, 2] * ss[:, 1]))[:args.keep]

    T = lambda a: torch.tensor(np.ascontiguousarray(a), dtype=torch.float32, device="cuda")  # noqa: E731
    full = dict(xyz=T(P), scale=T(np.exp(logS)), rot=T(q), opa=T(op[:, None]), col=T(np.clip(SH_C0 * fdc + 0.5, 0, 1)))
    c = P.mean(0)
    size = float(np.linalg.norm(P.max(0) - P.min(0)))
    R = args.res

    def rand_cam(rng):
        az = rng.uniform(0, 2 * math.pi)
        el = rng.uniform(-0.25, 1.1)
        r = size * rng.uniform(1.1, 1.6)
        eye = c + r * np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
        return Cam(eye, c + rng.normal(0, 0.05 * size, 3), R, R, rng.uniform(30, 45))

    eval_cams = [Cam(c + 1.3 * size * np.array([math.cos(a), math.sin(a), 0.35]), c, R, R, 35)
                 for a in np.linspace(0, 2 * math.pi, 8, endpoint=False)]
    with torch.no_grad():
        eval_ref = [raster(cm, **full) for cm in eval_cams]

    # 학습 변수 (활성화 전 값)
    xyz = torch.nn.Parameter(T(P[idx]))
    lsc = torch.nn.Parameter(T(logS[idx]))
    rot = torch.nn.Parameter(T(q[idx]))
    o = np.clip(op[idx], 1e-4, 1 - 1e-4)
    lop = torch.nn.Parameter(T(np.log(o / (1 - o))[:, None]))
    f = torch.nn.Parameter(T(fdc[idx]))
    opt = torch.optim.Adam([
        {"params": [xyz], "lr": 1e-4 * size}, {"params": [lsc], "lr": 5e-3}, {"params": [rot], "lr": 1e-3},
        {"params": [lop], "lr": 2.5e-2}, {"params": [f], "lr": 2.5e-3}])

    def cur():
        return dict(xyz=xyz, scale=torch.exp(lsc), rot=torch.nn.functional.normalize(rot, dim=1), opa=torch.sigmoid(lop),
                    col=(SH_C0 * f + 0.5).clamp(0, 1))

    masks = [((ref < 0.99).any(0)) for ref in eval_ref]          # 원본 렌더에서 물체가 있는 픽셀 (흰 배경 제외)

    def psnr_eval(save=None):
        """(전체 픽셀 PSNR, 물체 픽셀만 PSNR) — 흰 배경이 넓어 전체 PSNR 은 부풀려진다."""
        with torch.no_grad():
            ps, pm = [], []
            for k, (cm, ref, mk) in enumerate(zip(eval_cams, eval_ref, masks)):
                img = raster(cm, **cur())
                if save and k == 5:
                    from PIL import Image
                    Image.fromarray((torch.cat([ref, img], 2).clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255
                                     ).astype(np.uint8)).save(save)
                d2 = (img - ref) ** 2
                ps.append(10 * math.log10(1 / max(d2.mean().item(), 1e-12)))
                m = (mk | (img < 0.99).any(0)).float()
                pm.append(10 * math.log10(1 / max((d2 * m).sum().item() / (3 * m.sum().item()), 1e-12)))
            return float(np.mean(ps)), float(np.mean(pm))

    p0 = psnr_eval(os.path.join(out, f"distill_{args.keep // 1000}k_before.png"))
    rng = np.random.default_rng(0)
    torch.cuda.synchronize()
    t0 = time.time()
    for it in range(args.iters):
        cm = rand_cam(rng)
        with torch.no_grad():
            ref = raster(cm, **full)
        img = raster(cm, **cur())
        loss = (img - ref).abs().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if it in (args.iters // 2,):
            for g in opt.param_groups:
                g["lr"] *= 0.3
    torch.cuda.synchronize()
    dt = time.time() - t0
    p1 = psnr_eval(os.path.join(out, f"distill_{args.keep // 1000}k_after.png"))

    # 저장: 원래 좌표(위쪽 -y)로 되돌려 gaussian.ply 와 같은 필드
    with torch.no_grad():
        Pn = xyz.detach().cpu().numpy().astype(np.float64) @ M        # M 은 직교 → 역 = Mᵀ, 행벡터라 @ M
        qn = torch.nn.functional.normalize(rot, dim=1).detach().cpu().numpy().astype(np.float64)
        qn = mat_to_quat(M.T[None] @ quat_to_mat(qn))
        names = [p.name for p in PlyData.read(os.path.join(out, "gaussian.ply"))["vertex"].properties]
        el = np.zeros(len(Pn), dtype=[(k, "f4") for k in names])
        el["x"], el["y"], el["z"] = Pn[:, 0], Pn[:, 1], Pn[:, 2]
        fn = f.detach().cpu().numpy()
        for i in range(3):
            el[f"f_dc_{i}"] = fn[:, i]
            el[f"scale_{i}"] = lsc.detach().cpu().numpy()[:, i]
        for i in range(4):
            el[f"rot_{i}"] = qn[:, i]
        el["opacity"] = lop.detach().cpu().numpy()[:, 0]
    path = os.path.join(out, f"distill_{args.keep // 1000}k.ply")
    PlyData([PlyElement.describe(el, "vertex")]).write(path)
    st = {"name": args.name, "full": int(len(P)), "keep": args.keep, "iters": args.iters, "seconds": round(dt, 1),
          "psnr_prune_only": round(p0[0], 2), "psnr_distilled": round(p1[0], 2),
          "psnr_obj_prune_only": round(p0[1], 2), "psnr_obj_distilled": round(p1[1], 2), "res": R}
    json.dump(st, open(os.path.join(out, f"distill_{args.keep // 1000}k.json"), "w"), indent=2)
    print("[distill]", json.dumps(st), flush=True)


if __name__ == "__main__":
    main()
