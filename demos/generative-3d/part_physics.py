"""Generated Gaussian part labels을 XPBD constraint topology로 전달하는 작은 실행 도구.

예시:
  set APG_RUNTIME_ROOT=C:\gaussian-splatting\isaac_demo
  C:\anaconda\anaconda3\envs\trellis\python.exe part_physics.py --asset gen_bear_sem_d100k --parts out\bear_sem\parts3d.npz

`asset_part`는 distill된 PLY/graph와 같은 순서여야 한다. volume cluster는 같은 part 안에서만
만들고, shape matching은 (그래프 연결 성분, part)별로 따로 계산한다. 이 파일은 렌더 영상을
바꾸지 않고 positions/통계만 만든다.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gs_utils import runtime_root  # noqa: E402

RUNTIME = runtime_root()
sys.path.insert(0, RUNTIME)
from xpbd import XPBD, load_inputs  # noqa: E402


def parse_part_stiffness(text, names, default):
    """`head=0.9,arm=0.25` 형태를 part-id별 강도 배열로 바꾼다."""
    values = np.full(len(names), float(default), np.float32)
    if not text:
        return values
    for item in text.split(","):
        if not item.strip():
            continue
        name, value = item.split("=", 1)
        name = name.strip()
        if name not in names:
            raise ValueError(f"unknown part {name!r}; available: {', '.join(names)}")
        values[names.index(name)] = float(value)
    if np.any(~np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("part stiffness must be finite and in [0, 1]")
    return values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="gen_bear_sem_d100k", help="APG runtime asset directory name")
    ap.add_argument("--parts", default=os.path.join(HERE, "out", "bear_sem", "parts3d.npz"))
    ap.add_argument("--dll", default=os.path.join(RUNTIME, "xpbd_dll", "xpbd_isaac_part.dll"))
    ap.add_argument("--steps", type=int, default=2, help="validation steps; 0 only creates and configures")
    ap.add_argument("--ring-k", type=int, default=1)
    ap.add_argument("--max-members", type=int, default=64)
    ap.add_argument("--leader-min-hop", type=int, default=2)
    ap.add_argument("--shape-stiffness", type=float, default=0.25, help="default object shape strength")
    ap.add_argument("--part-stiffness", default="", help="comma-separated overrides, e.g. head=0.8,arm=0.2")
    ap.add_argument("--no-part-volume", action="store_true",
                    help="A/B only: do not restrict volume clusters by part")
    ap.add_argument("--no-part-shape", action="store_true",
                    help="A/B only: use ordinary object-level shape matching")
    ap.add_argument("--out", default="", help="optional .npz path for final positions")
    ap.add_argument("--out-ply", default="", help="optional deformed 3DGS PLY for SIBR viewer")
    ap.add_argument("--out-model", default="", help="optional SIBR model directory for the deformed result")
    ap.add_argument("--drive-part", default="", help="optional part name to translate after the first step")
    ap.add_argument("--drive-offset", default="0,0,0", help="source-coordinate translation for --drive-part")
    ap.add_argument("--drive-steps", type=int, default=0, help="steps while the selected part is held")
    ap.add_argument("--release-steps", type=int, default=0, help="steps after releasing the selected part")
    args = ap.parse_args()

    if args.steps < 0:
        ap.error("--steps must be non-negative")
    if args.steps == 0 and (args.out or args.out_ply or args.out_model):
        ap.error("--steps 0 cannot export a result; use --steps 1 or more")
    if not os.path.isfile(args.dll):
        raise FileNotFoundError(f"part-aware DLL not found: {args.dll}")

    part_data = np.load(args.parts)
    names = [str(x) for x in part_data["names"]]
    part_id = np.ascontiguousarray(part_data["asset_part"].astype(np.int32))
    inp = load_inputs(os.path.join(RUNTIME, args.asset), args.asset)
    if len(part_id) != len(inp["pos"]):
        raise ValueError(
            f"asset_part has {len(part_id):,} entries but {args.asset} graph has {len(inp['pos']):,}; "
            "use labels produced for the same distill asset"
        )

    sim = XPBD(args.dll)
    try:
        if not sim.has_part_constraints:
            raise RuntimeError("the selected DLL does not export part-aware XPBD functions")
        sim.create(inp)
        sim.set_solver(iters=12, dt=1.0 / 60.0, under_relax=0.6, vel_damping=0.002)
        sim.set_constraints(distance=True, shape=False, angle=False, volume=True)
        sim.set_volume(compliance=1e-6, ring_k=args.ring_k, max_members=args.max_members,
                       leader_min_hop=args.leader_min_hop)
        sim.set_object_shape(args.shape_stiffness)
        sim.set_part_ids(part_id, restrict_volume=not args.no_part_volume,
                         shape_groups=not args.no_part_shape)
        sim.set_part_shape(parse_part_stiffness(args.part_stiffness, names, args.shape_stiffness))
        drive = None
        if args.drive_part:
            if args.drive_part not in names:
                raise ValueError(f"unknown --drive-part {args.drive_part!r}; available: {', '.join(names)}")
            drive = np.flatnonzero(part_id == names.index(args.drive_part)).astype(np.int32)
            if not len(drive):
                raise ValueError(f"--drive-part {args.drive_part!r} has no Gaussians")
            offset = np.fromstring(args.drive_offset, sep=",", dtype=np.float32)
            if offset.shape != (3,) or not np.all(np.isfinite(offset)):
                raise ValueError("--drive-offset must be three finite comma-separated values")
            # C ABI attachment calls deliberately require a settled state; this step only initializes it.
            sim.step()
            sim.set_attached(drive, np.ascontiguousarray(inp["pos"][drive] + offset[None]), weight=1.0)
            for _ in range(args.drive_steps or args.steps):
                sim.step()
            sim.clear_attached()
            for _ in range(args.release_steps):
                sim.step()
        else:
            for _ in range(args.steps):
                sim.step()
        stats = sim.part_stats()
        report = {
            "asset": args.asset,
            "gaussians": int(sim.n),
            "parts": names,
            "steps": args.steps,
            "drive": ({"part": args.drive_part, "gaussians": int(len(drive)), "offset": offset.tolist(),
                       "hold_steps": args.drive_steps or args.steps, "release_steps": args.release_steps}
                      if drive is not None else None),
            "part_stats": {
                "cross_label_volume_clusters": stats[0],
                "fallback_groups": stats[1],
                "shape_groups": stats[2],
            },
            "part_constraints": {
                "volume_bounded": not args.no_part_volume,
                "shape_groups": not args.no_part_shape,
            },
        }
        print("[part_physics]", json.dumps(report, ensure_ascii=False), flush=True)
        if args.out:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            np.savez_compressed(args.out, pos=sim.positions(), part_id=part_id, names=np.asarray(names))
        if args.out_ply or args.out_model:
            from prepare_splat import read_ply, write_ply
            header, ply = read_ply(os.path.join(RUNTIME, args.asset, f"{args.asset}_crop.ply"))
            pos = sim.positions()
            scale, quat, _ = sim.shapes()
            if len(ply) != len(pos):
                raise ValueError("crop PLY and XPBD result have different sizes")
            ply = ply.copy()
            ply["x"], ply["y"], ply["z"] = pos[:, 0], pos[:, 1], pos[:, 2]
            for axis in range(3):
                ply[f"scale_{axis}"] = np.log(np.maximum(scale[:, axis], 1e-12))
            for axis in range(4):
                ply[f"rot_{axis}"] = quat[:, axis]
            if args.out_ply:
                os.makedirs(os.path.dirname(os.path.abspath(args.out_ply)), exist_ok=True)
                write_ply(args.out_ply, header, ply)
            if args.out_model:
                from export_parts import std_vertices, write_model
                write_model(args.out_model, std_vertices(ply), args.out_ply or os.path.join(RUNTIME, args.asset))
    finally:
        sim.destroy()


if __name__ == "__main__":
    main()
