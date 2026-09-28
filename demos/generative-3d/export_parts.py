"""가우시안 part_id 를 파일로 내보낸다 — 다른 도구 · 뷰어에서 부위를 보고 쓰도록.

  python export_parts.py --asset gen_bear_sem_d100k --parts out/bear_sem/parts3d.npz [--sibr] [--hide arm,leg]

출력 <runtime>/<asset>/:
  <asset>_parts.ply     크롭 PLY 의 모든 속성 + part_id (uchar, 255 = 모름) + part_conf (float). 속성을 이름으로 읽는
                        도구(Python plyfile · Blender · 웹 3DGS 뷰어)용. SIBR 뷰어는 속성을 고정 순서로 읽어 이 파일은 못 연다
  <asset>_part_id.u8    크롭 PLY 순서의 part_id 원시 바이트 (C++ 뷰어에서 fread 한 번이면 된다)
  <asset>_parts.json    부위 이름 · 표시 색 · 개수 · 부위별 물성 표(기본값 — part_graph.py --materials 형식)
--sibr: <APG_ROOT>/output_1/<asset>_partcolor/          SIBR 뷰어 모델 폴더 (3DGS 표준 레이아웃, 색만 부위 색)
        <APG_ROOT>/output_1/<asset>_hide-<부위들>/      (--hide) 그 부위 불투명도 0 — 원래 색 그대로, 부위 끄기
  SIBR_gaussianViewer_app -m <APG_ROOT>/output_1/<asset>_partcolor
"""
import argparse
import json
import os
import sys

import numpy as np
from plyfile import PlyData, PlyElement

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gs_utils import SH_C0, runtime_root  # noqa: E402
from parts_core import part_colors  # noqa: E402

STD = (["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"] + [f"f_rest_{i}" for i in range(45)]
       + ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"])


def std_vertices(v):
    el = np.zeros(len(v), dtype=[(k, "f4") for k in STD])
    for k in STD:
        if k in v.dtype.names:
            el[k] = v[k]
    return el


def write_model(run, el, source):
    """SIBR 뷰어가 여는 모델 폴더 (to_model.py 와 같은 형식)."""
    from to_model import orbit_cameras
    pc = os.path.join(run, "point_cloud", "iteration_1")
    os.makedirs(pc, exist_ok=True)
    PlyData([PlyElement.describe(el, "vertex")]).write(os.path.join(pc, "point_cloud.ply"))
    json.dump(orbit_cameras(), open(os.path.join(run, "cameras.json"), "w"))
    with open(os.path.join(run, "cfg_args"), "w") as f:
        f.write(f"Namespace(sh_degree=3, source_path='{source}', model_path='{run}', images='images', resolution=-1, "
                f"white_background=True, data_device='cuda', eval=False)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", required=True, help="<APG runtime>/<asset> (예: gen_bear_sem_d100k)")
    ap.add_argument("--parts", required=True, help="lift_parts.py --asset 로 만든 parts3d.npz")
    ap.add_argument("--sibr", action="store_true", help="SIBR 뷰어 모델 폴더도 만든다 (부위 색 · --hide)")
    ap.add_argument("--hide", default="", help="끌 부위 이름들 (쉼표) — --sibr 와 함께")
    args = ap.parse_args()
    d = os.path.join(runtime_root(), args.asset)
    v = PlyData.read(os.path.join(d, f"{args.asset}_crop.ply"))["vertex"].data
    pz = np.load(args.parts)
    names = [str(n) for n in pz["names"]]
    part, conf = pz["asset_part"].astype(np.int64), pz["asset_conf"].astype(np.float32)
    if len(part) != len(v):
        raise SystemExit(f"parts3d 의 에셋({len(part):,})이 {args.asset}({len(v):,})와 다르다 — "
                         "lift_parts.py --asset 을 이 에셋으로 다시 돌릴 것")
    pid = np.where(part < 0, 255, part).astype(np.uint8)

    el = np.zeros(len(v), dtype=v.dtype.descr + [("part_id", "u1"), ("part_conf", "f4")])
    for k in v.dtype.names:
        el[k] = v[k]
    el["part_id"], el["part_conf"] = pid, conf
    PlyData([PlyElement.describe(el, "vertex")]).write(os.path.join(d, f"{args.asset}_parts.ply"))
    pid.tofile(os.path.join(d, f"{args.asset}_part_id.u8"))
    cols = part_colors(np.arange(len(names)))
    meta = {"asset": args.asset, "order": f"{args.asset}_crop.ply", "unknown_id": 255,
            "parts": [{"id": k, "name": n, "color": [round(float(c), 3) for c in cols[k]], "count": int((part == k).sum())}
                      for k, n in enumerate(names)],
            "materials": {n: {"stiffness": 1.0} for n in names}}
    json.dump(meta, open(os.path.join(d, f"{args.asset}_parts.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"[export] {args.asset}: {len(v):,} Gaussians, parts {dict((p['name'], p['count']) for p in meta['parts'])}",
          flush=True)

    if args.sibr:
        root = os.environ.get("APG_ROOT", os.path.dirname(HERE))
        std = std_vertices(v)
        pc_el = std.copy()
        c = part_colors(part)
        for i in range(3):
            pc_el[f"f_dc_{i}"] = ((c[:, i] - 0.5) / SH_C0).astype(np.float32)
        run = os.path.join(root, "output_1", f"{args.asset}_partcolor")
        write_model(run, pc_el, d)
        print(f"[export] SIBR model (part colours): {run}", flush=True)
        hide = [n.strip() for n in args.hide.split(",") if n.strip()]
        if hide:
            bad = [n for n in hide if n not in names]
            if bad:
                raise SystemExit(f"없는 부위 {bad} — 부위: {names}")
            h_el = std.copy()
            h_el["opacity"][np.isin(part, [names.index(n) for n in hide])] = -20.0     # sigmoid(−20) ≈ 0
            run = os.path.join(root, "output_1", f"{args.asset}_hide-{'-'.join(n.replace(' ', '_') for n in hide)}")
            write_model(run, h_el, d)
            print(f"[export] SIBR model (hidden {hide}): {run}", flush=True)


if __name__ == "__main__":
    main()
