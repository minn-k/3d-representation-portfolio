"""흔들기 시험 — 왼쪽: 기존 그래프 · 온몸 같은 물성 / 오른쪽: 의미 부위로 물성 배정 (양팔만 무름, 몸통 · 머리 · 다리 단단).

  C:\\anaconda\\anaconda3\\envs\\trellis\\python.exe semantic_shake.py          (기본값 = 포트폴리오 영상 robot_b)

발(아래 --pin-h)을 받침에 고정하고 받침을 좌우로 --shake-s 동안 흔든 뒤 멈춘다. 같은 그래프 · 같은 솔버.
물성 = 거리 간선의 강성 (α̃ = compliance / 강성 / dt² — 전역 compliance 가 0 이면 강성이 무효라 켠다)
     + 물체 단위 형상 유지.
의미 쪽은 part_id 가 팔이 아닌 가우시안(몸통 · 머리 · 다리)을 받침에 강체로 붙이고, 팔만 XPBD 연체로 푼다.
부위는 lift_parts.py 가 생성 과정의 신호로 가우시안마다 붙인 part_id.
지표: 부위별 흔들림 = 받침 이동을 뺀 변위의 시간 RMS (cm). 출력 out/semantic_shake/<tag>.mp4 · .json
"""
import argparse
import json
import math
import os
import shutil
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from edit_demo import Cam, render, quat_to_mat, mat_to_quat  # noqa: E402
from edit_demo import RUNTIME, load_font  # noqa: E402
from prepare_splat import read_ply  # noqa: E402
from xpbd import XPBD, load_inputs  # noqa: E402

SH_C0 = 0.28209479177387814


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="gen_robot_sem_d100k")
    ap.add_argument("--parts", default=os.path.join(HERE, "out", "robot_sem", "parts3d.npz"))
    ap.add_argument("--soft", default="arm", help="무르게 할 부위")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--compliance", type=float, default=5e-5, help="거리 제약 전역 compliance")
    ap.add_argument("--soft-stiff", type=float, default=0.2, help="의미 쪽 무른 부위(팔) 간선 강성 배율")
    ap.add_argument("--uniform-stiff", type=float, default=0.3, help="기존 그래프 쪽 온몸 간선 강성 배율 (찢어지지 않을 만큼 무름)")
    ap.add_argument("--uniform-shape", type=float, default=0.1, help="기존 그래프 쪽 물체 형상 유지 (온몸이 한 덩어리 젤리로)")
    ap.add_argument("--semantic-shape", type=float, default=0.08, help="의미 쪽 물체 형상 유지 (강체 몸체가 기준 → 팔이 스프링처럼 돌아온다)")
    ap.add_argument("--rigid-body", type=int, default=1, help="의미 쪽: 단단한 부위를 강체로 (받침과 함께 움직임), 무른 부위만 XPBD")
    ap.add_argument("--pin-h", type=float, default=0.33, help="받침에 붙이는 높이 (다리)")
    ap.add_argument("--amp", type=float, default=0.04, help="흔들기 폭 [m]")
    ap.add_argument("--freq", type=float, default=2.0, help="[Hz]")
    ap.add_argument("--shake-s", type=float, default=1.6)
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--damping", type=float, default=0.002)
    ap.add_argument("--gravity", type=float, default=0.0)
    ap.add_argument("--gap", type=float, default=0.42)
    ap.add_argument("--res", default="1600x800")
    ap.add_argument("--eye", default="0,-1.1,0.34")
    ap.add_argument("--look", default="0,0,0.15")
    ap.add_argument("--fovy", type=float, default=27)
    ap.add_argument("--tag", default="robot_b")
    args = ap.parse_args()
    vec = lambda t: np.array([float(v) for v in t.split(",")])  # noqa: E731
    W, H = (int(v) for v in args.res.split("x"))
    out = os.path.join(HERE, "out", "semantic_shake")
    os.makedirs(out, exist_ok=True)

    d = os.path.join(RUNTIME, args.asset)
    inp0 = load_inputs(d, args.asset)
    _, data = read_ply(os.path.join(d, f"{args.asset}_crop.ply"))
    col = np.clip(SH_C0 * np.stack([data["f_dc_0"], data["f_dc_1"], data["f_dc_2"]], 1) + 0.5, 0, 1)
    xf = json.load(open(os.path.join(d, f"{args.asset}_transform.json"), encoding="utf-8"))
    s, Rm, tw = float(xf["scale"]), np.array(xf["rotation_matrix"], float), np.array(xf["translate"], float)
    to_w = lambda P: s * (np.asarray(P, np.float64) @ Rm.T) + tw  # noqa: E731
    Wr = to_w(inp0["pos"])
    pz = np.load(args.parts)
    names = [str(n) for n in pz["names"]]
    pa, pc = pz["asset_part"].astype(int), pz["asset_conf"]
    soft_ids = [names.index(n) for n in args.soft.split(",") if n in names]
    soft = np.isin(pa, soft_ids) & (pc > args.conf)
    # 팔로 잘못 붙은 작은 조각(안테나 끝 등)은 빼고, 몸에 붙은 큰 덩어리만 팔로 본다
    e = inp0["edges"]
    m_ = soft[e[:, 0]] & soft[e[:, 1]]
    _, lab = connected_components(coo_matrix((np.ones(m_.sum()), (e[m_, 0], e[m_, 1])), shape=(len(Wr),) * 2),
                                  directed=False)
    sizes = np.bincount(lab[soft])
    keep_lab = np.where(sizes > 0.1 * sizes.max())[0]
    soft &= np.isin(lab, keep_lab)
    z = Wr[:, 2]
    pin = np.where(z < z.min() + args.pin_h * np.ptp(z))[0].astype(np.int32)
    body_rigid = np.where(~soft)[0].astype(np.int32)                  # 의미 쪽 강체 = 팔이 아닌 모든 가우시안
    print(f"[shake] soft ({args.soft}) {soft.sum():,} / {len(Wr):,} Gaussians in {len(keep_lab)} pieces | pinned {len(pin):,}",
          flush=True)

    src = max([os.path.join(RUNTIME, "xpbd_dll", n) for n in ("xpbd_isaac.dll", "xpbd_isaac_next.dll")
               if os.path.exists(os.path.join(RUNTIME, "xpbd_dll", n))], key=os.path.getmtime)
    sims = []
    for k, mode in enumerate(("uniform", "semantic")):
        inp = {kk: (v.copy() if isinstance(v, np.ndarray) else v) for kk, v in inp0.items()}
        if mode == "uniform":
            inp["stiff"] = np.ascontiguousarray((inp["stiff"] * args.uniform_stiff).astype(np.float32))
        else:
            either_soft = soft[e[:, 0]] | soft[e[:, 1]]
            inp["stiff"] = np.ascontiguousarray((inp["stiff"] * np.where(either_soft, args.soft_stiff, 1.0))
                                                .astype(np.float32))
        dll = os.path.join(out, f"xpbd_{k}.dll")
        if not os.path.exists(dll) or os.path.getmtime(dll) < os.path.getmtime(src):
            shutil.copy2(src, dll)
        sim = XPBD(dll)
        sim.create(inp)
        sim.set_solver(iters=16, dt=1 / 60, under_relax=0.6, vel_damping=args.damping, dist_compliance=args.compliance)
        sim.set_constraints(distance=True, shape=False, angle=False, volume=True)
        sim.set_volume(compliance=1e-6, ring_k=3, max_members=2048, leader_min_hop=2)
        sim.set_object_shape(args.uniform_shape if mode == "uniform" else args.semantic_shape)
        sim.set_object_shape_gpu(True)
        sim.set_ground(False, (0, 0, 1), 0.0, gravity=0.0)
        sim.step()
        sim.reset()
        if args.gravity > 0:
            sim.set_ground(False, tuple(Rm.T @ np.array([0, 0, 1.0])), 0.0, gravity=args.gravity / s)
        sims.append(sim)

    P0 = inp0["pos"].astype(np.float64)
    ex_src = Rm.T @ np.array([1.0, 0, 0]) / s                   # 월드 x 1 m → 원본 좌표
    offs = [np.array([-args.gap / 2, 0, 0]), np.array([args.gap / 2, 0, 0])]
    cam = Cam(vec(args.eye), vec(args.look), W, H, args.fovy)
    font = load_font(26)
    titles = ["기존 그래프 · 온몸 같은 연체", "의미 부위 · 몸체 강체 + 양팔만 연체"]
    frames, wob = [], {m: {n: [] for n in names} for m in ("uniform", "semantic")}
    n_steps = int(args.seconds * 60)
    import imageio
    for k in range(n_steps):
        t = k / 60
        amp = args.amp * math.sin(2 * math.pi * args.freq * t) if t < args.shake_s else 0.0
        amp *= min(1.0, t / 0.2)                                  # 부드럽게 시작
        base = amp * ex_src
        for sim, mode in zip(sims, ("uniform", "semantic")):
            ids = body_rigid if (mode == "semantic" and args.rigid_body) else pin
            sim.set_attached(ids, np.ascontiguousarray((P0[ids] + base).astype(np.float32)), 1.0)
            sim.step()
        if k % 2 == 0:
            Ps, Qs, Ss, Os, Cs = [], [], [], [], []
            for sim, off, mode in zip(sims, offs, ("uniform", "semantic")):
                P = sim.positions().astype(np.float64)
                dev = np.linalg.norm((P - (P0 + base)) , axis=1) * s * 100    # 받침을 따라간 강체 이동을 뺀 흔들림
                for kk, n in enumerate(names):
                    mm = (pa == kk) if n not in args.soft.split(",") else soft
                    if mm.sum() > 50:
                        wob[mode][n].append(float(np.mean(dev[mm])))
                sc, q, _ = sim.shapes(1e-2)
                Ps.append(to_w(P) + off)
                Qs.append(mat_to_quat(Rm[None] @ quat_to_mat(q.astype(np.float64))))
                Ss.append(sc * s), Os.append(inp0["opacity"]), Cs.append(col)
            img = render(cam, np.concatenate(Ps), np.concatenate(Qs), np.concatenate(Ss), np.concatenate(Os),
                         np.concatenate(Cs))
            im = Image.fromarray(img)
            dr = ImageDraw.Draw(im)
            for j, tt in enumerate(titles):
                w_ = dr.textlength(tt, font=font)
                dr.text((W * (2 * j + 1) / 4 - w_ / 2, 14), tt, fill=(30, 30, 30), font=font)
            if t < args.shake_s:
                dr.text((16, H - 44), "받침 흔드는 중", fill=(180, 60, 40), font=font)
            frames.append(np.asarray(im))
    for sim in sims:
        sim.destroy()
    imageio.mimsave(os.path.join(out, f"{args.tag}.mp4"), frames, fps=30, quality=8, macro_block_size=8)
    imageio.imwrite(os.path.join(out, f"{args.tag}_mid.png"), frames[int(len(frames) * 0.35)])
    rms = {m: {n: round(float(np.sqrt(np.mean(np.square(v)))), 3) for n, v in c.items() if v} for m, c in wob.items()}
    peak = {m: {n: round(float(np.max(v)), 3) for n, v in c.items() if v} for m, c in wob.items()}
    st = {k_: getattr(args, k_) for k_ in ("asset", "soft", "compliance", "soft_stiff", "uniform_stiff", "uniform_shape", "semantic_shape", "rigid_body", "pin_h", "amp",
                                            "freq", "shake_s", "seconds", "damping", "gravity")}
    st.update({"soft_gaussians": int(soft.sum()), "wobble_rms_cm": rms, "wobble_peak_cm": peak})
    json.dump(st, open(os.path.join(out, f"{args.tag}.json"), "w"), indent=2, ensure_ascii=False)
    print("[shake]", json.dumps({"rms": rms}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
