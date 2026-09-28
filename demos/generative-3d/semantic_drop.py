"""같은 로봇 두 개를 같은 자세로 떨어뜨린다 — 왼쪽: 기하 그래프 · 재질 하나 / 오른쪽: 의미 부위로 재질 배정.

  C:\\anaconda\\anaconda3\\envs\\trellis\\python.exe semantic_drop.py [--rigid head,torso,leg --soft arm,antenna]

부위는 lift_parts.py 가 생성 과정의 신호(cross-attention + DiT 특징)로 가우시안마다 붙인 것 (parts3d.npz 의 asset_part).
재질 = 부위별 물체 단위 형상 유지 세기 (DLL set_particle_weights 의 shape_stiffness). 기하 그래프(간선·부피 클러스터)는 두 쪽이 같다.
지표: 부위마다 매 프레임 '강체로 가장 잘 맞춘 뒤 남는 오차' (RMS, cm) = 그 부위 모양이 얼마나 망가졌나.
출력 out/semantic_drop/semantic_drop.mp4, semantic_drop.json
"""
import argparse
import json
import math
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from edit_demo import Cam, render, quat_to_mat, mat_to_quat, label  # noqa: E402
try:
    from edit_demo import RUNTIME  # noqa: E402   (저장소 판: APG_RUNTIME_ROOT)
except ImportError:
    from edit_demo import DEMO as RUNTIME  # noqa: E402   (로컬 작업 폴더 판)
from prepare_splat import read_ply  # noqa: E402
from xpbd import XPBD, load_inputs, world_ground  # noqa: E402

SH_C0 = 0.28209479177387814


def rigid_rms(P, P0):
    """P0 → P 강체 맞춤 후 남는 RMS (같은 단위)."""
    a, b = P0 - P0.mean(0), P - P.mean(0)
    U, _, Vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(U @ Vt))
    R = (U @ np.diag([1, 1, d]) @ Vt).T
    return float(np.sqrt(np.mean(np.sum((b - a @ R.T) ** 2, 1))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="gen_robot_sem_d100k")
    ap.add_argument("--parts", default=os.path.join(HERE, "out", "robot_sem", "parts3d.npz"))
    ap.add_argument("--rigid", default="head,torso,leg")
    ap.add_argument("--soft", default="arm,antenna")
    ap.add_argument("--base-shape", type=float, default=0.3, help="기하 그래프 · 전체 무름 쪽 형상 유지 (의미 쪽의 부위 없는 점도)")
    ap.add_argument("--stiff-shape", type=float, default=1.0, help="기하 그래프 · 전체 단단 쪽")
    ap.add_argument("--soft-fit", type=float, default=0.05, help="무른 부위의 물체 강체 맞춤 가중 (작을수록 몸 기준 자세를 안 흔든다)")
    ap.add_argument("--rigid-shape", type=float, default=1.0)
    ap.add_argument("--soft-shape", type=float, default=0.02)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--drop", type=float, default=0.3, help="[m]")
    ap.add_argument("--tilt-deg", type=float, default=15.0, help="앞으로 기울인 채 떨어뜨림 (옆으로 부딪히게)")
    ap.add_argument("--gravity", type=float, default=9.81)
    ap.add_argument("--restitution", type=float, default=0.4)
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--res", default="1800x720")
    ap.add_argument("--gap", type=float, default=0.36, help="로봇 사이 [m]")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    W, H = (int(v) for v in args.res.split("x"))
    out = os.path.join(HERE, "out", "semantic_drop")
    os.makedirs(out, exist_ok=True)
    d = os.path.join(RUNTIME, args.asset)
    inp = load_inputs(d, args.asset)
    _, data = read_ply(os.path.join(d, f"{args.asset}_crop.ply"))
    col = np.clip(SH_C0 * np.stack([data["f_dc_0"], data["f_dc_1"], data["f_dc_2"]], 1) + 0.5, 0, 1)
    xf = json.load(open(os.path.join(d, f"{args.asset}_transform.json"), encoding="utf-8"))
    s, Rm, tw = float(xf["scale"]), np.array(xf["rotation_matrix"], float), np.array(xf["translate"], float)
    up, h, g, _ = world_ground(d, args.asset, ground_z=-args.drop, gravity=args.gravity)
    Hs = float(np.ptp(inp["pos"].astype(np.float64) @ up))
    pz = np.load(args.parts)
    names = [str(n) for n in pz["names"]]
    pa, pc = pz["asset_part"].astype(int), pz["asset_conf"]
    assert len(pa) == len(inp["pos"]), "parts3d 의 asset 이 이 에셋과 다르다"
    rigid = [names.index(n) for n in args.rigid.split(",") if n in names]
    soft = [names.index(n) for n in args.soft.split(",") if n in names]

    src = max([os.path.join(RUNTIME, "xpbd_dll", n) for n in ("xpbd_isaac.dll", "xpbd_isaac_next.dll")
               if os.path.exists(os.path.join(RUNTIME, "xpbd_dll", n))], key=os.path.getmtime)
    sims = []
    MODES = ("geometry_soft", "geometry_stiff", "semantic")
    for k, mode in enumerate(MODES):
        dll = os.path.join(out, f"xpbd_{k}.dll")
        if not os.path.exists(dll) or os.path.getmtime(dll) < os.path.getmtime(src):
            shutil.copy2(src, dll)
        sim = XPBD(dll)
        sim.create(inp)
        sim.set_solver(iters=16, dt=1 / 60, under_relax=0.6, vel_damping=0.01)
        sim.set_constraints(distance=True, shape=False, angle=False, volume=True)
        sim.set_volume(compliance=1e-6, ring_k=3, max_members=2048, leader_min_hop=2)
        sim.set_object_shape(args.stiff_shape if mode == "geometry_stiff" else args.base_shape)
        sim.set_object_shape_gpu(True)
        sim.set_ground(False, tuple(up), 0.0, gravity=0.0)
        sim.step()
        sim.reset()
        if mode == "semantic":
            first = True
            for grp, val, fw in ((rigid, args.rigid_shape, 1.0), (soft, args.soft_shape, args.soft_fit)):
                idx = np.where(np.isin(pa, grp) & (pc > args.conf))[0].astype(np.int32)
                sim.set_particle_weights(idx, inv_mass_scale=1.0, fit_weight=fw, shape_stiffness=val, append=not first)
                first = False
        sim.set_ground(True, tuple(up), h, friction=0.6, restitution=args.restitution,
                       contact_radius=0.003 * Hs, contact_slop=0.002 * Hs, gravity=g)
        tilt = -math.radians(args.tilt_deg) * (Rm.T @ np.array([1.0, 0.0, 0.0]))   # 월드 x 축 둘레 → 앞으로 숙임
        sim.launch((0, 0, 0), (0, 0, 0), tuple(tilt))
        sims.append(sim)

    P0 = inp["pos"].astype(np.float64)
    offs = [np.array([(k - 1) * args.gap, 0, args.drop]) for k in range(len(MODES))]

    def world(P, q, sc, off):
        Pw = s * (P.astype(np.float64) @ Rm.T) + tw + off
        qw = mat_to_quat(Rm[None] @ quat_to_mat(q.astype(np.float64)))
        return Pw, qw, sc * s

    cam = Cam(np.array([0.0, -1.0, 0.42]), np.array([0.0, 0.0, 0.13]), W, H, 30)
    frames, curves = [], {m: {n: [] for n in names} for m in MODES}
    titles = ["기하 그래프 · 전체 무름 (%.2g)" % args.base_shape, "기하 그래프 · 전체 단단 (%.2g)" % args.stiff_shape,
              "의미 부위 · 몸 단단 + 팔 무름"]
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype("C:/Windows/Fonts/malgunbd.ttf", 26)
    import imageio
    n_frames = int(args.seconds * args.fps)
    spf = max(1, round(60 / args.fps))
    for fi in range(n_frames):
        Ps, Qs, Ss, Os, Cs = [], [], [], [], []
        for sim, off, mode in zip(sims, offs, MODES):
            if fi > 0:
                for _ in range(spf):
                    sim.step()
            P = sim.positions()
            sc, q, _ = sim.shapes(1e-2)
            for k, n in enumerate(names):
                m = pa == k
                if m.sum() > 50:
                    curves[mode][n].append(rigid_rms(P[m].astype(np.float64), P0[m]) * s * 100)
            Pw, qw, sw = world(P, q, sc, off)
            Ps.append(Pw), Qs.append(qw), Ss.append(sw), Os.append(inp["opacity"]), Cs.append(col)
        img = render(cam, np.concatenate(Ps), np.concatenate(Qs), np.concatenate(Ss), np.concatenate(Os),
                     np.concatenate(Cs))
        im = Image.fromarray(img)
        dr = ImageDraw.Draw(im)
        for k, t in enumerate(titles):
            tw_ = dr.textlength(t, font=font)
            dr.text((W * (2 * k + 1) / 6 - tw_ / 2, 14), t, fill=(30, 30, 30), font=font)
        frames.append(np.asarray(im))
    imageio.mimsave(os.path.join(out, f"semantic_drop{args.tag}.mp4"), frames + [frames[-1]] * 20, fps=args.fps,
                    quality=8, macro_block_size=8)
    imageio.imwrite(os.path.join(out, f"semantic_drop{args.tag}_last.png"), frames[-1])
    st = {"asset": args.asset, "parts": names, "modes": list(MODES), "rigid": args.rigid, "soft": args.soft,
          "base_shape": args.base_shape, "stiff_shape": args.stiff_shape, "soft_fit": args.soft_fit,
          "rigid_shape": args.rigid_shape, "soft_shape": args.soft_shape, "drop_m": args.drop, "tilt_deg": args.tilt_deg,
          "gravity": args.gravity, "restitution": args.restitution, "seconds": args.seconds,
          "part_counts": {n: int((pa == k).sum()) for k, n in enumerate(names)},
          "max_shape_error_cm": {m: {n: round(max(v), 3) for n, v in c.items() if v} for m, c in curves.items()},
          "final_shape_error_cm": {m: {n: round(v[-1], 3) for n, v in c.items() if v} for m, c in curves.items()}}
    json.dump(st, open(os.path.join(out, f"semantic_drop{args.tag}.json"), "w"), indent=2, ensure_ascii=False)
    np.savez_compressed(os.path.join(out, f"semantic_drop{args.tag}_curves.npz"),
                        **{f"{m}_{n}": np.array(v) for m, c in curves.items() for n, v in c.items()})
    for sim in sims:
        sim.destroy()
    print("[semdrop]", json.dumps({k: st[k] for k in ("max_shape_error_cm", "final_shape_error_cm")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
