"""의미 부위로 하는 '부위 단위 편집' vs 기하 그래프만으로 하는 편집 — 같은 목표(팔을 어깨 기준으로 들어 올림), 나란히.

  C:\\anaconda\\anaconda3\\envs\\trellis\\python.exe semantic_pose.py --asset gen_robot_sem_d100k --parts out/robot_sem/parts3d.npz

의미 부위 (오른쪽): 가우시안의 part_id 로 '오른팔' 을 통째로 고르고, 팔과 몸통이 맞닿은 가우시안의 중심을 어깨 관절로 잡아
  팔 전체를 그 관절 둘레로 --angle 만큼 돌린 자리에 붙인다 (부착 = 운동학적 조작). 몸통 · 머리는 XPBD 가 따라간다.
기하 그래프만 (왼쪽): 부위를 모르므로 손끝 주변(--grab-r)만 잡아 '같은 손끝 경로' 로 끈다 (지금까지의 편집 방식).
둘 다 아래 --pin-h 를 고정, 같은 그래프 · 같은 솔버.
지표: 팔 모양 오차 (강체 맞춤 후 RMS, cm — 팔이 휘거나 늘어난 정도), 몸통 · 머리 평균 변위 (cm — 끌려온 정도),
      2배 넘게 늘어난 간선 비율.
출력 out/semantic_pose/<asset>.mp4, <asset>.json, <asset>_last.png
"""
import argparse
import json
import math
import os
import shutil
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from edit_demo import Cam, render, quat_to_mat, mat_to_quat  # noqa: E402
from edit_demo import RUNTIME, load_font  # noqa: E402
from prepare_splat import read_ply  # noqa: E402
from xpbd import XPBD, load_inputs  # noqa: E402
from semantic_drop import rigid_rms  # noqa: E402

SH_C0 = 0.28209479177387814


def rotmat(axis, ang):
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="gen_robot_sem_d100k")
    ap.add_argument("--parts", default=os.path.join(HERE, "out", "robot_sem", "parts3d.npz"))
    ap.add_argument("--part", default="arm")
    ap.add_argument("--body", default="torso", help="관절을 찾을 이웃 부위")
    ap.add_argument("--side", default="+x", help="같은 이름 부위가 둘일 때 (양팔) 고를 쪽: +x / -x (월드)")
    ap.add_argument("--axis", default="0,1,0", help="회전축 (월드). 오른팔(+x)을 y 축 둘레 −각도로 돌리면 위로")
    ap.add_argument("--angle", type=float, default=-60.0, help="[deg]")
    ap.add_argument("--pin-h", type=float, default=0.35)
    ap.add_argument("--grab-r", type=float, default=0.03)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--settle", type=int, default=60)
    ap.add_argument("--dist-compliance", type=float, default=1e-4)
    ap.add_argument("--eye", default="0,-1.05,0.36")
    ap.add_argument("--look", default="0,0,0.15")
    ap.add_argument("--fovy", type=float, default=26)
    ap.add_argument("--gap", type=float, default=0.42)
    ap.add_argument("--res", default="1600x800")
    ap.add_argument("--titles", default="기하 그래프만 · 손끝을 잡아 끎,의미 부위 · '오른팔' 을 어깨 관절로 들어 올림")
    args = ap.parse_args()
    vec = lambda t: np.array([float(v) for v in t.split(",")])  # noqa: E731
    W, H = (int(v) for v in args.res.split("x"))
    out = os.path.join(HERE, "out", "semantic_pose")
    os.makedirs(out, exist_ok=True)

    d = os.path.join(RUNTIME, args.asset)
    inp = load_inputs(d, args.asset)
    _, data = read_ply(os.path.join(d, f"{args.asset}_crop.ply"))
    col = np.clip(SH_C0 * np.stack([data["f_dc_0"], data["f_dc_1"], data["f_dc_2"]], 1) + 0.5, 0, 1)
    xf = json.load(open(os.path.join(d, f"{args.asset}_transform.json"), encoding="utf-8"))
    s, Rm, tw = float(xf["scale"]), np.array(xf["rotation_matrix"], float), np.array(xf["translate"], float)
    to_w = lambda P: s * (np.asarray(P, np.float64) @ Rm.T) + tw  # noqa: E731
    Wr = to_w(inp["pos"])
    pz = np.load(args.parts)
    names = [str(n) for n in pz["names"]]
    pa, pc = pz["asset_part"].astype(int), pz["asset_conf"]
    kp, kb = names.index(args.part), names.index(args.body)
    c = Wr.mean(0)
    sgn = 1.0 if args.side.startswith("+") else -1.0
    part = (pa == kp) & (pc > args.conf) & (sgn * (Wr[:, 0] - c[0]) > 0)          # 한쪽 팔
    # 그래프로 이어진 가장 큰 덩어리만 (팔로 잘못 붙은 안테나 조각 등 제외)
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    e_ = inp["edges"]
    m_ = part[e_[:, 0]] & part[e_[:, 1]]
    A_ = coo_matrix((np.ones(m_.sum()), (e_[m_, 0], e_[m_, 1])), shape=(len(Wr), len(Wr)))
    _, lab = connected_components(A_, directed=False)
    big = np.bincount(lab[part]).argmax()
    part &= lab == big
    body = (pa == kb) & (pc > args.conf)
    dn, _ = cKDTree(Wr[body]).query(Wr[part])
    touch = Wr[part][dn < 0.012]                                                     # 몸통과 맞닿은 팔 가우시안
    bc = Wr[body].mean(0)
    rr = np.linalg.norm((touch - bc)[:, [0, 2]], axis=1)                            # 몸 중심에서 가로·세로 거리
    joint_pts = touch[rr > np.percentile(rr, 80)]                                   # 가장 바깥쪽 = 어깨
    pivot = joint_pts.mean(0)
    axis = vec(args.axis)
    # 기하 쪽이 잡는 곳 = 팔 가우시안 중 관절에서 가장 먼 끝 (손끝)
    far = np.linalg.norm(Wr - pivot, axis=1)
    far_part = np.where(part)[0][np.argsort(-far[part])[:400]]
    tip = Wr[far_part].mean(0)
    z = Wr[:, 2]
    pin = np.where(z < z.min() + args.pin_h * np.ptp(z))[0]
    part_idx = np.where(part)[0]
    pin = np.setdiff1d(pin, part_idx)
    tip_idx = np.where((np.linalg.norm(Wr - tip, axis=1) < args.grab_r) & part)[0]
    tip_idx = np.setdiff1d(tip_idx, pin)
    print(f"[pose] part {args.part}{args.side}: {part.sum():,} Gaussians | joint pts {len(joint_pts)} at "
          f"{np.round(pivot, 3).tolist()} | tip grab {len(tip_idx)} | pinned {len(pin):,}", flush=True)

    src = max([os.path.join(RUNTIME, "xpbd_dll", n) for n in ("xpbd_isaac.dll", "xpbd_isaac_next.dll")
               if os.path.exists(os.path.join(RUNTIME, "xpbd_dll", n))], key=os.path.getmtime)
    sims = []
    for k in range(2):
        dll = os.path.join(out, f"xpbd_{k}.dll")
        if not os.path.exists(dll) or os.path.getmtime(dll) < os.path.getmtime(src):
            shutil.copy2(src, dll)
        sim = XPBD(dll)
        sim.create(inp)
        sim.set_solver(iters=20, dt=1 / 60, under_relax=0.6, vel_damping=0.1, dist_compliance=args.dist_compliance)
        sim.set_constraints(distance=True, shape=True, angle=False, volume=True)
        sim.set_volume(compliance=1e-6, ring_k=3, max_members=2048, leader_min_hop=2)
        sim.set_object_shape(0.0)
        sim.set_ground(False, (0, 0, 1), 0.0, gravity=0.0)
        sim.step()
        sim.reset()
        sims.append(sim)
    P0 = inp["pos"].astype(np.float64)
    to_src = lambda Pw: (((Pw - tw) / s) @ Rm)  # noqa: E731

    offs = [np.array([-args.gap / 2, 0, 0]), np.array([args.gap / 2, 0, 0])]
    cam = Cam(vec(args.eye), vec(args.look), W, H, args.fovy)
    font = load_font(26)
    titles = args.titles.split(",")
    frames = []
    total = args.steps + args.settle
    import imageio
    for k in range(total):
        f = min(1.0, (k + 1) / args.steps)
        f = f * f * (3 - 2 * f)
        Rk = rotmat(axis, math.radians(args.angle) * f)
        # 의미 부위: 팔 전체를 관절 둘레로 강체 회전한 자리에
        arm_target = (Wr[part_idx] - pivot) @ Rk.T + pivot
        # 기하: 손끝만, 같은 손끝 경로
        tip_target = (Wr[tip_idx] - pivot) @ Rk.T + pivot
        for sim, (idx, tgt) in zip(sims, ((tip_idx, tip_target), (part_idx, arm_target))):
            ids = np.concatenate([pin, idx]).astype(np.int32)
            pos = np.concatenate([P0[pin], to_src(tgt)]).astype(np.float32)
            sim.set_attached(ids, pos, 1.0)
            sim.step()
        if k % 3 == 0 or k == total - 1:
            Ps, Qs, Ss, Os, Cs = [], [], [], [], []
            for sim, off in zip(sims, offs):
                P = sim.positions()
                sc, q, _ = sim.shapes(1e-2)
                Ps.append(to_w(P) + off)
                Qs.append(mat_to_quat(Rm[None] @ quat_to_mat(q.astype(np.float64))))
                Ss.append(sc * s), Os.append(inp["opacity"]), Cs.append(col)
            img = render(cam, np.concatenate(Ps), np.concatenate(Qs), np.concatenate(Ss), np.concatenate(Os),
                         np.concatenate(Cs))
            im = Image.fromarray(img)
            dr = ImageDraw.Draw(im)
            for j, t in enumerate(titles):
                tw_ = dr.textlength(t, font=font)
                dr.text((W * (2 * j + 1) / 4 - tw_ / 2, 14), t, fill=(30, 30, 30), font=font)
            frames.append(np.asarray(im))
    res = {}
    e = inp["edges"]
    for sim, mode in zip(sims, ("geometry", "semantic")):
        P = sim.positions().astype(np.float64)
        disp = np.linalg.norm(P - P0, axis=1) * s * 100
        free = np.ones(len(P), bool)
        free[pin] = False
        free[part_idx if mode == "semantic" else tip_idx] = False
        r_ = np.linalg.norm(P[e[:, 0]] - P[e[:, 1]], axis=1) / np.maximum(inp["rest"], 1e-12)
        res[mode] = {"arm_shape_error_cm": round(rigid_rms(P[part_idx], P0[part_idx]) * s * 100, 3),
                     "arm_rotation_deg_achieved": None,
                     "disp_cm": {n: round(float(disp[(pa == k) & free & ~part].mean()), 3)
                                 for k, n in enumerate(names) if np.any((pa == k) & free & ~part)},
                     "edges_over_2x": float(np.mean(r_ > 2)), "edges_over_1p5x": float(np.mean(r_ > 1.5))}
        # 팔이 실제로 돈 각도 (관절 기준, 팔 중심 방향)
        a0 = Wr[part_idx].mean(0) - pivot
        a1 = to_w(P[part_idx]).mean(0) - pivot
        res[mode]["arm_rotation_deg_achieved"] = round(math.degrees(math.acos(
            np.clip(a0 @ a1 / np.linalg.norm(a0) / np.linalg.norm(a1), -1, 1))), 1)
    for sim in sims:
        sim.destroy()
    tag = args.asset.replace("gen_", "")
    imageio.mimsave(os.path.join(out, f"{tag}.mp4"), frames + [frames[-1]] * 30, fps=30, quality=8, macro_block_size=8)
    imageio.imwrite(os.path.join(out, f"{tag}_last.png"), frames[-1])
    st = {"asset": args.asset, "part": args.part, "side": args.side, "angle_deg": args.angle,
          "pivot_world": np.round(pivot, 4).tolist(), "joint_points": int(len(joint_pts)), "arm_gaussians": int(part.sum()),
          "tip_grab": int(len(tip_idx)), "pinned": int(len(pin)), "dist_compliance": args.dist_compliance, **res}
    json.dump(st, open(os.path.join(out, f"{tag}.json"), "w"), indent=2, ensure_ascii=False)
    print("[pose]", json.dumps(res, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
