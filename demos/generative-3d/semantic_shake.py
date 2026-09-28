"""흔들기 시험 — 왼쪽: 기존 그래프 · 온몸 같은 물성 / 오른쪽: 의미 부위로 물성 배정 (팔은 더 말랑, 몸체는 덜 말랑).

  C:\\anaconda\\anaconda3\\envs\\trellis\\python.exe semantic_shake.py --tag robot_dangle

발(아래 --pin-h)을 받침에 고정하고 받침을 좌우로 --shake-s 동안 흔든 뒤 멈춘다. 같은 그래프 · 같은 솔버 · 같은 모양.
물성 = 거리 간선의 강성 (α̃ = compliance / 강성 / dt² — 전역 compliance 가 0 이면 강성이 무효라 켠다)
     + 형상 유지 (물체 단위 강체 맞춤으로 되돌리는 세기, set_particle_weights 로 가우시안마다).
의미 쪽도 발만 고정한 한 덩어리 연체이고, 물성만 부위마다 다르다: part_id 가 팔인 가우시안은 --soft-stiff ·
--soft-shape (더 말랑), 나머지는 --body-stiff · --body-shape (기존 그래프 쪽보다 덜 말랑). 부위 경계에서는 물성이
그래프 이웃을 따라 --blend-hops 번 섞여 몇 가우시안에 걸쳐 이어진다. 경계를 강체에 붙이거나 한 줄에서 물성을 끊으면
그 줄만 늘어나 잘린 것처럼 보인다.
--elbow auto 를 주면 팔 안에서도 팔꿈치 너머만 말랑하게 한다 (find_arms). --rigid-body 1 은 예전 방식
(팔이 아닌 부분을 받침에 강체로 붙임).
중력은 쓰지 않는다: 이 솔버에서 중력은 무른 팔을 기둥처럼 늘인다.
부위는 lift_parts.py 가 생성 과정의 신호로 가우시안마다 붙인 part_id.
지표: 부위별 흔들림 = 받침 이동을 뺀, 쉬는 자세에서 벗어난 거리의 시간 RMS (cm). 출력 out/semantic_shake/<tag>.mp4 · .json

팔을 더 출렁이게 하려면 --soft-shape 를 낮추고 (0.03), 몸체를 더 단단하게 하려면 --body-stiff · --body-shape 를 올린다.
경계가 보이면 --blend-hops 를 올린다 (10). 흔드는 힘은 --amp (폭, 힘에 비례).
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
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from edit_demo import Cam, render, quat_to_mat, mat_to_quat  # noqa: E402
from edit_demo import RUNTIME, load_font  # noqa: E402
from prepare_splat import read_ply  # noqa: E402
from xpbd import XPBD, load_inputs  # noqa: E402

SH_C0 = 0.28209479177387814


def _seg_dist(X, A, B):
    AB = B - A
    t = np.clip(((X - A) @ AB) / max(AB @ AB, 1e-12), 0, 1)
    return np.linalg.norm(X - (A + t[:, None] * AB), axis=1)


def find_arms(Wr, arm, body, edges, elbow="auto", blend=0.25, touch_r=0.012, bins=24):
    """좌 · 우 팔마다 어깨 S · 팔꿈치 E · 손끝 T 를 찾고, 팔 가우시안마다 무른 정도 w (0 = 강체, 1 = 무름) 를 준다.
    S = 몸통에 닿은 팔 가우시안 중 가장 바깥쪽 (semantic_pose.py 와 같은 방법). 어깨에서 팔 그래프를 따라 잰 거리로
    팔을 줄 세우고, T = 가장 먼 5% 의 중심. E = S–T 직선에서 가장 먼 3% 의 중심 (elbow="auto", 어깨→손끝의
    15~75% 사이, |ST| 의 15% 넘게 꺾였을 때. 곧은 팔이면 45% 지점). 숫자를 주면 그 비율 지점.
    팔 위치 u = 꺾은선 S–E–T 위 가장 가까운 점까지의 길이 비율 (0 = 어깨, 1 = 손끝; 팔꿈치 안쪽에서 두 토막이 맞닿아도
    섞이지 않는다). w 는 u 가 팔꿈치 ± blend/2 를 지나며 0 → 1 로 매끄럽게 오른다."""
    bc = Wr[body].mean(0)
    tree = cKDTree(Wr[body])
    e = np.unique(np.sort(edges.astype(np.int64), 1), axis=0)
    arms = []
    for side in (-1, 1):
        idx = np.nonzero(arm & (side * (Wr[:, 0] - bc[0]) > 0))[0]
        if len(idx) < 50:
            continue
        X = Wr[idx]
        dn, _ = tree.query(X)
        near = dn < touch_r
        if near.sum() < 10:                                      # 몸통 라벨과 떨어져 있으면 가장 가까운 2%
            near = dn <= np.percentile(dn, 2)
        t = np.nonzero(near)[0]
        rr = np.linalg.norm((X[t] - bc)[:, [0, 2]], axis=1)
        sh = t[rr >= np.percentile(rr, 80)]
        loc = -np.ones(len(Wr), np.int64)
        loc[idx] = np.arange(len(idx))
        m = (loc[e[:, 0]] >= 0) & (loc[e[:, 1]] >= 0)
        i, j = loc[e[m, 0]], loc[e[m, 1]]
        G = coo_matrix((np.linalg.norm(X[i] - X[j], axis=1) + 1e-9, (i, j)), shape=(len(idx),) * 2).tocsr()
        d = dijkstra(G, directed=False, indices=sh, min_only=True)
        ok = np.isfinite(d)
        if not ok.all():                                         # 어깨와 안 이어진 조각은 가장 가까운 가우시안 값
            _, nn = cKDTree(X[ok]).query(X[~ok])
            d[~ok] = d[ok][nn]
        dmax = np.percentile(d, 99)
        S, T = X[sh].mean(0), X[d >= np.percentile(d, 95)].mean(0)
        b = np.clip((d / dmax * bins).astype(int), 0, bins - 1)
        cnt = np.bincount(b, minlength=bins)
        C = np.array([X[b == k].mean(0) if cnt[k] >= 10 else np.full(3, np.nan) for k in range(bins)])
        fr = (np.arange(bins) + 0.5) / bins
        valid = ~np.isnan(C[:, 0])
        E = None
        if elbow == "auto":
            ST = T - S
            h = np.linalg.norm(np.cross(X - S, ST), axis=1) / np.linalg.norm(ST)
            mid = (d >= 0.15 * dmax) & (d <= 0.75 * dmax)
            if mid.sum() >= 30:
                top = mid & (h >= np.percentile(h[mid], 97))    # 꺾인 곳의 바깥 면 (구간 중심은 안쪽으로 끌린다)
                if np.median(h[top]) > 0.15 * np.linalg.norm(ST):
                    E = X[top].mean(0)
            target = 0.45
        else:
            target = float(elbow)
        if E is None and target <= 0:                            # 팔 전체를 무르게
            E, w = S, np.ones(len(idx))
        else:
            if E is None:
                E = C[np.nonzero(valid)[0][np.argmin(np.abs(fr[valid] - target))]]
            L1, L2 = np.linalg.norm(E - S), np.linalg.norm(T - E)
            t1 = np.clip(((X - S) @ (E - S)) / max(L1 * L1, 1e-12), 0, 1)
            t2 = np.clip(((X - E) @ (T - E)) / max(L2 * L2, 1e-12), 0, 1)
            u = np.where(_seg_dist(X, S, E) <= _seg_dist(X, E, T), t1 * L1, L1 + t2 * L2) / max(L1 + L2, 1e-12)
            f = np.clip((u - (L1 / max(L1 + L2, 1e-12) - blend / 2)) / max(blend, 1e-6), 0, 1)
            w = f * f * (3 - 2 * f)
        u, v = S - E, T - E
        bend = 180.0 - math.degrees(math.acos(np.clip(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12), -1, 1)))
        _, ne = cKDTree(X).query(E, min(20, len(X)))
        arms.append({"side": "+x" if side > 0 else "-x", "idx": idx, "w": w,
                     "elbow_frac": round(float(np.median(d[ne]) / dmax), 3), "bend_deg": round(bend, 1),
                     "shoulder": S, "elbow": E, "tip": T})
    return arms


def graph_blend(w, edges, hops):
    """간선 이웃 평균을 hops 번 — 부위 경계에서 물성이 몇 가우시안에 걸쳐 이어지게 (경계에서 먼 곳은 그대로)."""
    N = len(w)
    i, j = edges[:, 0].astype(np.int64), edges[:, 1].astype(np.int64)
    A = coo_matrix((np.ones(2 * len(i)), (np.r_[i, j], np.r_[j, i])), shape=(N, N)).tocsr()
    deg = np.maximum(np.asarray(A.sum(1)).ravel(), 1)
    w = w.astype(np.float64)
    for _ in range(hops):
        w = 0.5 * w + 0.5 * (A @ w) / deg
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="gen_robot_sem_d100k")
    ap.add_argument("--parts", default=os.path.join(HERE, "out", "robot_sem", "parts3d.npz"))
    ap.add_argument("--soft", default="arm", help="무르게 할 부위")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--compliance", type=float, default=5e-5, help="거리 제약 전역 compliance")
    ap.add_argument("--soft-stiff", type=float, default=0.2, help="의미 쪽 팔 간선 강성 배율 (기존 그래프 쪽보다 무름)")
    ap.add_argument("--soft-shape", type=float, default=0.05, help="의미 쪽 팔 형상 유지 (낮을수록 더 출렁)")
    ap.add_argument("--soft-fit", type=float, default=0.3, help="팔의 강체 맞춤 가중 (작을수록 팔이 몸 기준 자세를 덜 흔든다)")
    ap.add_argument("--body-stiff", type=float, default=0.6, help="의미 쪽 몸체 간선 강성 배율 (기존 그래프 쪽보다 단단)")
    ap.add_argument("--body-shape", type=float, default=0.15, help="의미 쪽 몸체 형상 유지")
    ap.add_argument("--blend-hops", type=int, default=6, help="부위 경계에서 물성을 그래프 이웃으로 섞는 횟수")
    ap.add_argument("--uniform-stiff", type=float, default=0.3, help="기존 그래프 쪽 온몸 간선 강성 배율 (찢어지지 않을 만큼 무름)")
    ap.add_argument("--uniform-shape", type=float, default=0.1, help="기존 그래프 쪽 물체 형상 유지 (온몸이 한 덩어리 젤리로)")
    ap.add_argument("--semantic-shape", type=float, default=0.08, help="--rigid-body 1 일 때 의미 쪽 물체 형상 유지")
    ap.add_argument("--rigid-body", type=int, default=0, help="1 = 예전 방식: 팔이 아닌 부분을 받침에 강체로 붙이고 팔만 XPBD")
    ap.add_argument("--elbow", default="none",
                    help="none = 팔 클래스 전체가 말랑 / auto (팔이 가장 꺾인 곳부터) / 어깨→손끝 (팔을 따라 잰) 거리의 비율")
    ap.add_argument("--elbow-blend", type=float, default=0.25,
                    help="--elbow 를 줄 때 팔꿈치 앞뒤로 물성이 이어지는 폭 (어깨→손끝 길이의 비율)")
    ap.add_argument("--body", default="torso", help="어깨를 찾을 이웃 부위")
    ap.add_argument("--pin-h", type=float, default=0.33, help="받침에 붙이는 높이 (다리)")
    ap.add_argument("--amp", type=float, default=0.04, help="흔들기 폭 [m]")
    ap.add_argument("--freq", type=float, default=2.0, help="[Hz]")
    ap.add_argument("--shake-s", type=float, default=2.4)
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--damping", type=float, default=0.002)
    ap.add_argument("--titles", default="기존 그래프 · 온몸 같은 연체,의미 부위 · 팔은 더 말랑 + 몸체는 덜 말랑")
    ap.add_argument("--gap", type=float, default=0.42)
    ap.add_argument("--res", default="1600x800")
    ap.add_argument("--eye", default="0,-1.1,0.34")
    ap.add_argument("--look", default="0,0,0.15")
    ap.add_argument("--fovy", type=float, default=27)
    ap.add_argument("--tag", default="robot")
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
    arm = np.isin(pa, soft_ids) & (pc > args.conf)
    # 팔로 잘못 붙은 작은 조각(안테나 끝 등)은 빼고, 몸에 붙은 큰 덩어리만 팔로 본다
    e = inp0["edges"]
    m_ = arm[e[:, 0]] & arm[e[:, 1]]
    _, lab = connected_components(coo_matrix((np.ones(m_.sum()), (e[m_, 0], e[m_, 1])), shape=(len(Wr),) * 2),
                                  directed=False)
    sizes = np.bincount(lab[arm])
    keep_lab = np.where(sizes > 0.1 * sizes.max())[0]
    arm &= np.isin(lab, keep_lab)
    body = ~arm
    if args.body in names:
        body &= (pa == names.index(args.body)) & (pc > args.conf)
    # 말랑한 정도 w: 팔 1, 몸체 0 (--elbow 를 주면 팔꿈치 앞뒤에서 0 → 1)
    arms = []
    if args.elbow == "none":
        wraw = arm.astype(np.float64)
    else:
        arms = find_arms(Wr, arm, body, e, "auto" if args.elbow == "auto" else float(args.elbow), args.elbow_blend)
        wraw = np.zeros(len(Wr))
        for a in arms:
            wraw[a["idx"]] = a["w"]
    soft = wraw >= 0.5                                                # 지표용 '팔'
    wsoft = wraw if args.rigid_body else graph_blend(wraw, e, args.blend_hops)
    z = Wr[:, 2]
    pin = np.where(z < z.min() + args.pin_h * np.ptp(z))[0].astype(np.int32)
    body_rigid = np.where(wraw <= 0)[0].astype(np.int32)               # --rigid-body 1: 팔이 아닌 전부를 받침에
    print(f"[shake] {args.soft} {arm.sum():,} / {len(Wr):,} Gaussians in {len(keep_lab)} pieces | soft {int(soft.sum()):,}, "
          f"blended {int(((wsoft > 0.02) & (wsoft < 0.98)).sum()):,} | pinned {len(pin):,}", flush=True)
    for a in arms:
        print(f"[shake] arm {a['side']}: {len(a['idx']):,} Gaussians, elbow at {a['elbow_frac']:.0%} of shoulder->fingertip "
              f"(bend {a['bend_deg']} deg), soft {int((a['w'] >= 0.5).sum()):,}", flush=True)

    src = max([os.path.join(RUNTIME, "xpbd_dll", n) for n in ("xpbd_isaac.dll", "xpbd_isaac_next.dll")
               if os.path.exists(os.path.join(RUNTIME, "xpbd_dll", n))], key=os.path.getmtime)
    sims = []
    for k, mode in enumerate(("uniform", "semantic")):
        inp = {kk: (v.copy() if isinstance(v, np.ndarray) else v) for kk, v in inp0.items()}
        if mode == "uniform":
            inp["stiff"] = np.ascontiguousarray((inp["stiff"] * args.uniform_stiff).astype(np.float32))
        else:
            we = 0.5 * (wsoft[e[:, 0]] + wsoft[e[:, 1]])              # 몸체 강성 → 팔 강성 으로 이어진다
            k_body = 1.0 if args.rigid_body else args.body_stiff
            inp["stiff"] = np.ascontiguousarray((inp["stiff"] * k_body ** (1 - we) * args.soft_stiff ** we)
                                                .astype(np.float32))
        dll = os.path.join(out, f"xpbd_{k}.dll")
        if not os.path.exists(dll) or os.path.getmtime(dll) < os.path.getmtime(src):
            shutil.copy2(src, dll)
        sim = XPBD(dll)
        sim.create(inp)
        sim.set_solver(iters=16, dt=1 / 60, under_relax=0.6, vel_damping=args.damping, dist_compliance=args.compliance)
        sim.set_constraints(distance=True, shape=False, angle=False, volume=True)
        sim.set_volume(compliance=1e-6, ring_k=3, max_members=2048, leader_min_hop=2)
        sim.set_object_shape(args.uniform_shape if mode == "uniform" else
                             args.semantic_shape if args.rigid_body else args.body_shape)
        sim.set_object_shape_gpu(True)
        sim.set_ground(False, (0, 0, 1), 0.0, gravity=0.0)
        sim.step()
        sim.reset()
        if mode == "semantic" and not args.rigid_body:
            if hasattr(sim, "set_particle_weights"):             # 형상 유지도 몸체 → 팔로 이어지게 (6 단계)
                lev = np.round(wsoft * 5) / 5
                for j, v in enumerate(np.unique(lev)):
                    ids_v = np.where(lev == v)[0].astype(np.int32)
                    sim.set_particle_weights(ids_v, inv_mass_scale=1.0, fit_weight=1.0 + (args.soft_fit - 1.0) * v,
                                             shape_stiffness=args.body_shape + (args.soft_shape - args.body_shape) * v,
                                             append=j > 0)
            else:
                print("[shake] WARNING this runtime has no set_particle_weights: shape retention is --body-shape "
                      "everywhere, only edge stiffness differs", flush=True)
        sims.append(sim)

    P0 = inp0["pos"].astype(np.float64)
    ex_src = Rm.T @ np.array([1.0, 0, 0]) / s                   # 월드 x 1 m → 원본 좌표
    offs = [np.array([-args.gap / 2, 0, 0]), np.array([args.gap / 2, 0, 0])]
    cam = Cam(vec(args.eye), vec(args.look), W, H, args.fovy)
    font = load_font(26)
    titles = args.titles.split(",")
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
                dev = np.linalg.norm(P - (P0 + base), axis=1) * s * 100   # 받침을 따라간 이동을 뺀 흔들림
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
    st = {k_: getattr(args, k_) for k_ in ("asset", "elbow", "compliance", "soft_stiff", "soft_shape", "soft_fit",
                                            "body_stiff", "body_shape", "blend_hops", "uniform_stiff", "uniform_shape",
                                            "semantic_shape", "rigid_body", "elbow_blend", "pin_h", "amp",
                                            "freq", "shake_s", "seconds", "damping")}
    arms_st = [{"side": a["side"], "gaussians": int(len(a["idx"])), "soft_gaussians": int((a["w"] >= 0.5).sum()), "elbow_frac": a["elbow_frac"], "bend_deg": a["bend_deg"],
                "shoulder_world": np.round(a["shoulder"], 4).tolist(), "elbow_world": np.round(a["elbow"], 4).tolist()}
               for a in arms]
    st.update({"soft": args.soft if args.elbow == "none" else f"{args.soft} beyond the elbow", "arm_gaussians": int(arm.sum()), "soft_gaussians": int(soft.sum()),
               "arms": arms_st, "wobble_rms_cm": rms, "wobble_peak_cm": peak})
    json.dump(st, open(os.path.join(out, f"{args.tag}.json"), "w"), indent=2, ensure_ascii=False)
    print("[shake]", json.dumps({"rms": rms}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
