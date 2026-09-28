"""생성 가우시안을 중요도 순으로 N 개만 남겼을 때 화질이 얼마나 떨어지나 (전체 렌더 대비 PSNR, 8방향).

  python prune_test.py bear plant robot
중요도 = 불투명도 × 화면 면적 근사(두 큰 축 곱). 아무것도 다시 학습하지 않은 단순 가지치기.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("APG_ROOT", os.path.dirname(HERE))       # 3DGS 작업 폴더 (output_1/, isaac_demo/, SIBR_viewers/)
sys.path.insert(0, HERE)
from edit_demo import Cam, render  # noqa: E402
from plyfile import PlyData  # noqa: E402

SH_C0 = 0.28209479177387814
res = {}
for name in sys.argv[1:]:
    v = PlyData.read(os.path.join(ROOT, "output_1", f"gen_{name}", "point_cloud", "iteration_1",
                                  "point_cloud.ply"))["vertex"].data
    P = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    S = np.exp(np.stack([v[f"scale_{i}"] for i in range(3)], 1))
    q = np.stack([v[f"rot_{i}"] for i in range(4)], 1).astype(np.float64)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    from edit_demo import quat_to_mat, mat_to_quat
    M = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)    # 회전: PLY 위쪽 -y → +z
    P = P @ M.T
    q = mat_to_quat(M[None] @ quat_to_mat(q))
    op = 1 / (1 + np.exp(-v["opacity"].astype(np.float64)))
    col = np.clip(SH_C0 * np.stack([v[f"f_dc_{i}"] for i in range(3)], 1) + 0.5, 0, 1)
    ss = np.sort(S, 1)
    imp = op * ss[:, 2] * ss[:, 1]
    order = np.argsort(-imp)
    c = P.mean(0)
    size = np.linalg.norm(P.max(0) - P.min(0))
    cams = [Cam(c + 1.3 * size * np.array([np.cos(a), np.sin(a), 0.35]), c, 512, 512, 35)
            for a in np.linspace(0, 2 * np.pi, 8, endpoint=False)]
    full = [render(cm, P, q, S, op, col).astype(np.float64) for cm in cams]
    rows = {}
    for N in (50_000, 100_000, 150_000, 200_000, 300_000):
        if N >= len(P):
            continue
        k = order[:N]
        ps = []
        for cm, ref in zip(cams, full):
            im = render(cm, P[k], q[k], S[k], op[k], col[k]).astype(np.float64)
            mse = np.mean((im - ref) ** 2) / 255 ** 2
            ps.append(10 * np.log10(1 / max(mse, 1e-12)))
        rows[N] = round(float(np.mean(ps)), 2)
        if N == 100_000:
            from PIL import Image
            Image.fromarray(np.concatenate([full[5].astype(np.uint8), render(cams[5], P[k], q[k], S[k], op[k], col[k])], 1)
                            ).save(os.path.join(HERE, "out", name, "prune_100k.png"))
    res[name] = {"total": int(len(P)), "psnr_vs_full": rows}
    print(name, res[name], flush=True)
json.dump(res, open(os.path.join(HERE, "out", "prune_test.json"), "w"), indent=2)
