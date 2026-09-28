"""TRELLIS 가 만든 gaussian.ply 를 3DGS 학습 결과 폴더(output_1/<run>) 형식으로 옮긴다.

  python to_model.py --name bear [--min-opacity 0.02] [--height 1.0]

- TRELLIS PLY 는 SH 0차(f_dc)만 있다 → f_rest 45개를 0 으로 채워 3차 형식으로 (SIBR 뷰어·prepare_splat 가 그대로 읽게).
- 좌표: TRELLIS 내부는 z-up 이고 save_ply 가 (x, -z, y) 로 바꿔 쓰므로 PLY 의 위쪽은 -y 다. 그대로 두고, 가짜 궤도 카메라
  (COLMAP 규약, 카메라 y 축 = 아래)를 -y 위쪽으로 둘러 놓아 prepare_splat.py 가 up 을 -y 로 찾게 한다.
- 거의 투명한 가우시안(--min-opacity 미만)은 뺀다 (그래프·XPBD 크기를 줄인다; 렌더에서는 안 보임).
출력: output_1/gen_<name>/{cfg_args, cameras.json, point_cloud/iteration_1/point_cloud.ply}, prune 통계 json
"""
import argparse
import json
import os

import numpy as np
from plyfile import PlyData, PlyElement

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("APG_ROOT", os.path.dirname(HERE))       # 3DGS 작업 폴더 (output_1/, apg_runtime/, SIBR_viewers/)


def orbit_cameras(n=36, radius=2.0, elev_deg=(15.0, 35.0), size=800, fov_deg=40.0):
    f = 0.5 * size / np.tan(np.radians(fov_deg) / 2)
    cams = []
    for i in range(n):
        yaw = 2 * np.pi * i / n
        el = np.radians(elev_deg[i % len(elev_deg)])
        c = radius * np.array([np.cos(el) * np.cos(yaw), -np.sin(el), np.cos(el) * np.sin(yaw)])
        z = -c / np.linalg.norm(c)                       # 카메라 앞 = 원점 쪽
        x = np.cross(z, [0.0, -1.0, 0.0])                # 위쪽 = -y
        x /= np.linalg.norm(x)
        y = np.cross(z, x)                               # COLMAP: y 아래
        R = np.stack([x, y, z], axis=1)                  # camera-to-world (열 = 카메라 축)
        cams.append({"id": i, "img_name": f"{i:04d}", "width": size, "height": size,
                     "position": c.tolist(), "rotation": R.tolist(), "fy": f, "fx": f})
    return cams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--min-opacity", type=float, default=0.02)
    ap.add_argument("--ply", default="", help="입력 PLY (기본 out/<name>/gaussian.ply — distill.py 결과 등)")
    ap.add_argument("--keep", type=int, default=0, help="중요도(불투명도 × 두 큰 축 곱) 큰 순으로 이 개수만 남긴다")
    args = ap.parse_args()

    src = args.ply or os.path.join(HERE, "out", args.name, "gaussian.ply")
    os.makedirs(os.path.join(HERE, "out", args.name), exist_ok=True)
    v = PlyData.read(src)["vertex"].data
    op = 1.0 / (1.0 + np.exp(-v["opacity"].astype(np.float64)))
    keep = op >= args.min_opacity
    if args.keep and keep.sum() > args.keep:
        ss = np.sort(np.exp(np.stack([v[f"scale_{i}"] for i in range(3)], 1)), 1)
        imp = np.where(keep, op * ss[:, 2] * ss[:, 1], -1.0)
        keep = np.zeros(len(v), bool)
        keep[np.argsort(-imp)[:args.keep]] = True
    v = v[keep]

    names = (["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"] + [f"f_rest_{i}" for i in range(45)]
             + ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"])
    el = np.zeros(len(v), dtype=[(k, "f4") for k in names])
    for k in names:
        if k in v.dtype.names:
            el[k] = v[k]
    q = np.stack([el[f"rot_{i}"] for i in range(4)], 1)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    for i in range(4):
        el[f"rot_{i}"] = q[:, i]

    run = os.path.join(ROOT, "output_1", f"gen_{args.name}")
    pc = os.path.join(run, "point_cloud", "iteration_1")
    os.makedirs(pc, exist_ok=True)
    PlyData([PlyElement.describe(el, "vertex")]).write(os.path.join(pc, "point_cloud.ply"))
    json.dump(orbit_cameras(), open(os.path.join(run, "cameras.json"), "w"))
    with open(os.path.join(run, "cfg_args"), "w") as f:
        f.write(f"Namespace(sh_degree=3, source_path='{os.path.join(HERE, 'out', args.name)}', "
                f"model_path='{run}', images='images', resolution=-1, white_background=True, "
                f"data_device='cuda', eval=False)")
    st = {"source": src, "total": int(len(keep)), "kept": int(keep.sum()), "min_opacity": args.min_opacity,
          "keep": args.keep}
    json.dump(st, open(os.path.join(HERE, "out", args.name, "to_model.json"), "w"), indent=2)
    P = np.stack([el["x"], el["y"], el["z"]], 1)
    print(f"[to_model] {args.name}: kept {keep.sum():,} / {len(keep):,} (opacity >= {args.min_opacity}), "
          f"bbox {np.round(P.min(0), 3).tolist()} .. {np.round(P.max(0), 3).tolist()} -> {run}")


if __name__ == "__main__":
    main()
