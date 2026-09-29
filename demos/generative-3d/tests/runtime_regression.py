"""부위를 아는 DLL 이 부위 기능을 안 켜면 예전 DLL 과 똑같이 도는지 (위치 비트 단위 비교).

  python tests/runtime_regression.py --old xpbd_isaac_next.dll --new xpbd_isaac_part_v2.dll

런타임(APG_RUNTIME_ROOT)이 없으면 건너뛴다. 같은 입력 · 설정으로 발 고정 + 받침 흔들기를 --steps 스텝 돌리고
positions() 를 비교한다. 게시된 영상 · 수치(semantic_shake · semantic_drop)가 새 DLL 에서도 그대로라는 근거.
"""
import argparse
import json
import math
import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from gs_utils import runtime_root  # noqa: E402


def run(dll, inp, pin, ex, steps):
    from xpbd import XPBD
    sim = XPBD(dll)
    try:
        sim.create(inp)
        sim.set_solver(iters=16, dt=1 / 60, under_relax=0.6, vel_damping=0.002, dist_compliance=5e-5)
        sim.set_constraints(distance=True, shape=False, angle=False, volume=True)
        sim.set_volume(compliance=1e-6, ring_k=3, max_members=2048, leader_min_hop=2)
        sim.set_object_shape(0.1)
        sim.set_object_shape_gpu(True)
        sim.set_ground(False, (0, 0, 1), 0.0, gravity=0.0)
        sim.step()
        sim.reset()
        P0 = inp["pos"].astype(np.float64)
        for k in range(steps):
            base = 0.04 * math.sin(2 * math.pi * 2.0 * k / 60) * ex
            sim.set_attached(pin, np.ascontiguousarray((P0[pin] + base).astype(np.float32)), 1.0)
            sim.step()
        return sim.positions().astype(np.float64).copy()
    finally:
        sim.destroy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="gen_robot_sem_d100k")
    ap.add_argument("--old", default="xpbd_isaac_next.dll", help="xpbd_dll/ 안의 예전 DLL")
    ap.add_argument("--new", required=True, help="xpbd_dll/ 안의 부위를 아는 DLL")
    ap.add_argument("--steps", type=int, default=120)
    args = ap.parse_args()
    root = runtime_root()
    if not os.path.isfile(os.path.join(root, "xpbd.py")):
        print(f"[regression] skipped: no runtime at {root}")
        return 0
    sys.path.insert(0, root)
    from xpbd import load_inputs
    inp = load_inputs(os.path.join(root, args.asset), args.asset)
    xf = json.load(open(os.path.join(root, args.asset, f"{args.asset}_transform.json"), encoding="utf-8"))
    s, Rm, tw = float(xf["scale"]), np.array(xf["rotation_matrix"], float), np.array(xf["translate"], float)
    z = (s * (inp["pos"].astype(np.float64) @ Rm.T) + tw)[:, 2]  # 받침 고정은 월드 높이 기준 (semantic_shake.py 와 같게)
    pin = np.where(z < z.min() + 0.33 * np.ptp(z))[0].astype(np.int32)
    ex = Rm.T @ np.array([1.0, 0, 0]) / s
    tmp = tempfile.mkdtemp()
    res = {}
    for tag, name in (("old", args.old), ("new", args.new)):
        src = os.path.join(root, "xpbd_dll", name)
        dst = os.path.join(tmp, f"regress_{tag}.dll")               # DLL 전역 상태 → 따로 복사해 올린다
        shutil.copy2(src, dst)
        res[tag] = run(dst, inp, pin, ex, args.steps)
    diff = float(np.max(np.abs(res["old"] - res["new"])))
    print(f"[regression] {args.asset} {args.steps} steps | {args.old} vs {args.new} | max |Δpos| = {diff:.3e} "
          + ("(identical)" if diff == 0.0 else "(DIFFERENT — part DLL changes the default path)"))
    return 0 if diff == 0.0 else 1


if __name__ == "__main__":
    sys.exit(main())
