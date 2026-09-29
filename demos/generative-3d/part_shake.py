"""부위를 아는 XPBD 흔들기 비교 — 왼쪽: 기존 그래프 · 물성 하나 / 오른쪽: part_id 를 솔버에 넘긴 한 덩어리 연체.

  C:\\anaconda\\anaconda3\\envs\\trellis\\python.exe part_shake.py --tag robot_part

두 쪽 모두 발(아래 --pin-h)만 받침에 고정하고 받침을 좌우로 흔든 뒤 멈춘다. 강체 고정 없음, 같은 모양 · 그래프 · 솔버 DLL.
오른쪽만 다른 것 (part_physics.py 와 같은 비공개 런타임 API):
  · 부피 클러스터를 부위 조각 안에서만 만든다 (set_part_ids restrict_volume)
  · 형상 유지를 부위 조각마다 따로 계산하고 세기도 조각마다 (set_part_ids shape_groups · set_part_shape)
  · 간선 강성을 부위별로 주되, 경계에서 그래프 이웃 평균으로 몇 가우시안에 걸쳐 이어지게 (--blend-hops)
부위 조각 = 같은 부위끼리 이어진 덩어리. 솔버는 (그래프 연결 성분, id) 로 형상 그룹을 나누는데 로봇 전체가 한 연결 성분이라
'arm' 하나로 넘기면 두 팔이 한 강체 맞춤에 묶인다 → 조각 번호(왼팔 · 오른팔 따로)를 넘긴다. 작은 조각은 이웃 조각에 붙인다.
--groups soft (기본): 말랑한 부위 조각만 따로 형상 그룹이고, 나머지 몸(머리 · 몸통 · 다리)은 한 그룹.
  몸의 부위마다 따로 강체로 되돌리면(--groups all) 몸통이 다리 위에서 통째로 미끄러져 허리가 끊겨 보였다
  (로봇, 경계 간선 11.6% 가 1.5 배 넘게 늘어남). 몸은 한 덩어리로 되돌아가야 부위 사이가 이어진다.
지표 (json):
  wobble_rms_cm   부위별 흔들림 — 받침 이동을 뺀, 쉬는 자세에서 벗어난 거리의 시간 RMS (semantic_shake.py 와 같은 정의)
  shape_error_cm  부위별 모양 망가짐 — 조각마다 강체 맞춤 뒤 남는 RMS 의 시간 RMS · 최대 (semantic_drop.rigid_rms)
  boundary        부위 경계 간선의 늘어남 (길이 / 원래 길이) 95 퍼센타일 · 최대 · 1.5 배 넘은 비율의 최대
  soft_rel_body_cm 말랑한 부위가 몸에 대해 움직인 거리 — 몸(말랑하지 않은 가우시안)의 강체 운동을 빼고 잰 평균의 시간 RMS · 최대
                  ('팔만 덜렁' 의 지표. wobble 은 몸 전체의 흔들림이 섞인다)
  step_ms         스텝 평균 시간 (호스트에서 잰 값)
--no-part-shape / --no-part-volume / --no-edge-ramp 로 오른쪽 요소를 하나씩 끈 비교를 따로 만든다.
출력 out/part_shake/<tag>.mp4 · _mid.png · .json
"""
import argparse
import glob
import json
import math
import os
import shutil
import sys
import time

import numpy as np
from PIL import Image, ImageDraw
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from edit_demo import Cam, render, quat_to_mat, mat_to_quat  # noqa: E402
from edit_demo import RUNTIME, load_font  # noqa: E402
from prepare_splat import read_ply  # noqa: E402
from semantic_drop import rigid_rms  # noqa: E402
from xpbd import XPBD, load_inputs  # noqa: E402

SH_C0 = 0.28209479177387814


def part_pieces(part, edges, min_size=200, rounds=20):
    """같은 부위 간선으로 이어진 덩어리 → 조각 번호 (0..P-1), 조각의 부위 (P,).
    min_size 보다 작은 조각은 가장 많이 맞닿은 큰 조각에 붙인다 (분류 잡음 · 가는 부위)."""
    part = np.asarray(part, np.int64)
    n = len(part)
    a, b = edges[:, 0].astype(np.int64), edges[:, 1].astype(np.int64)
    same = part[a] == part[b]
    _, lab = connected_components(coo_matrix((np.ones(same.sum()), (a[same], b[same])), shape=(n, n)), directed=False)
    for _ in range(rounds):
        small = np.bincount(lab, minlength=lab.max() + 1)[lab] < min_size
        m1, m2 = small[a] & ~small[b], small[b] & ~small[a]
        src = np.r_[lab[a[m1]], lab[b[m2]]]
        dst = np.r_[lab[b[m1]], lab[a[m2]]]
        if len(src) == 0:
            break
        pairs, cnt = np.unique(np.stack([src, dst], 1), axis=0, return_counts=True)
        best = {}
        for (s_, d_), c in zip(pairs, cnt):
            if c > best.get(s_, (0, -1))[0]:
                best[s_] = (c, d_)
        remap = np.arange(lab.max() + 1)
        for s_, (_, d_) in best.items():
            remap[s_] = d_
        lab = remap[lab]
    uniq, piece = np.unique(lab, return_inverse=True)
    piece_part = np.array([np.bincount(part[piece == k]).argmax() for k in range(len(uniq))])
    return piece.astype(np.int32), piece_part


def graph_blend(w, edges, hops):
    """간선 이웃 평균을 hops 번 — 부위 경계에서 물성이 몇 가우시안에 걸쳐 이어지게 (경계에서 먼 곳은 그대로)."""
    n = len(w)
    i, j = edges[:, 0].astype(np.int64), edges[:, 1].astype(np.int64)
    A = coo_matrix((np.ones(2 * len(i)), (np.r_[i, j], np.r_[j, i])), shape=(n, n)).tocsr()
    deg = np.maximum(np.asarray(A.sum(1)).ravel(), 1)
    w = w.astype(np.float64)
    for _ in range(hops):
        w = 0.5 * w + 0.5 * (A @ w) / deg
    return w


def newest_part_dll():
    c = sorted(glob.glob(os.path.join(RUNTIME, "xpbd_dll", "xpbd_isaac_part*.dll")), key=os.path.getmtime)
    return c[-1] if c else os.path.join(RUNTIME, "xpbd_dll", "xpbd_isaac_part.dll")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="gen_robot_sem_d100k")
    ap.add_argument("--parts", default=os.path.join(HERE, "out", "robot_sem", "parts3d.npz"))
    ap.add_argument("--dll", default="", help="부위를 아는 런타임 DLL (기본: xpbd_dll/xpbd_isaac_part*.dll 중 최신)")
    ap.add_argument("--soft", default="arm", help="말랑하게 할 부위 (쉼표로 여럿)")
    ap.add_argument("--compliance", type=float, default=5e-5, help="거리 제약 전역 compliance")
    ap.add_argument("--uniform-stiff", type=float, default=0.3, help="왼쪽: 온몸 간선 강성 배율")
    ap.add_argument("--uniform-shape", type=float, default=0.1, help="왼쪽: 물체 형상 유지")
    ap.add_argument("--body-stiff", type=float, default=0.6, help="오른쪽: 몸체 간선 강성 배율")
    ap.add_argument("--body-shape", type=float, default=0.15, help="오른쪽: 몸체 조각 형상 유지")
    ap.add_argument("--soft-stiff", type=float, default=0.2, help="오른쪽: 말랑한 부위 간선 강성 배율")
    ap.add_argument("--soft-shape", type=float, default=0.05, help="오른쪽: 말랑한 부위 조각 형상 유지")
    ap.add_argument("--blend-hops", type=int, default=6, help="오른쪽: 경계에서 간선 강성을 섞는 횟수")
    ap.add_argument("--min-piece", type=int, default=200, help="이보다 작은 부위 조각은 이웃 조각에 붙인다")
    ap.add_argument("--groups", default="soft", choices=["soft", "all"],
                    help="soft = 말랑한 부위 조각만 따로 형상 그룹, 몸은 한 그룹 · all = 모든 부위 조각이 따로")
    ap.add_argument("--no-part-shape", action="store_true", help="비교용: 오른쪽 형상 유지를 물체 하나로")
    ap.add_argument("--no-part-volume", action="store_true", help="비교용: 오른쪽 부피 클러스터 부위 제한 끔")
    ap.add_argument("--no-edge-ramp", action="store_true", help="비교용: 오른쪽 간선 강성도 왼쪽과 같게")
    ap.add_argument("--pin-h", type=float, default=0.33, help="받침에 붙이는 높이 (다리)")
    ap.add_argument("--amp", type=float, default=0.02, help="흔들기 폭 [m]")
    ap.add_argument("--axis", default="x", choices=["x", "y", "z"],
                    help="흔드는 방향 (월드). x = 좌우 (옆으로 뻗은 팔에는 길이 방향이라 팔이 돌지 않는다) · z = 위아래")
    ap.add_argument("--freq", type=float, default=2.0, help="[Hz]")
    ap.add_argument("--shake-s", type=float, default=2.4)
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--damping", type=float, default=0.002)
    ap.add_argument("--titles", default="기존 그래프 · 물성 하나,부위를 아는 솔버 · 팔 말랑 + 몸 덜 말랑")
    ap.add_argument("--gap", type=float, default=0.42)
    ap.add_argument("--res", default="1600x800")
    ap.add_argument("--eye", default="0,-1.1,0.34")
    ap.add_argument("--look", default="0,0,0.15")
    ap.add_argument("--fovy", type=float, default=27)
    ap.add_argument("--tag", default="robot_part")
    args = ap.parse_args()
    vec = lambda t: np.array([float(v) for v in t.split(",")])  # noqa: E731
    W, H = (int(v) for v in args.res.split("x"))
    out = os.path.join(HERE, "out", "part_shake")
    os.makedirs(out, exist_ok=True)
    src = args.dll or newest_part_dll()
    if not os.path.isfile(src):
        raise FileNotFoundError(f"부위를 아는 DLL 이 없다: {src} (--dll 로 지정)")

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
    part = pz["asset_part"].astype(np.int64)
    if len(part) != len(Wr):
        raise ValueError(f"asset_part {len(part):,} 개 ≠ {args.asset} {len(Wr):,} 개 — 같은 에셋으로 만든 parts3d.npz 여야 한다")
    e = inp0["edges"]
    soft_ids = [names.index(n) for n in args.soft.split(",") if n in names]
    piece, piece_part = part_pieces(part, e, args.min_piece)
    part_clean = piece_part[piece]                              # 작은 조각을 붙인 뒤의 부위 (지표 · 강성에 같은 값)
    soft = np.isin(part_clean, soft_ids)
    P0 = inp0["pos"].astype(np.float64)
    z = Wr[:, 2]
    pin = np.where(z < z.min() + args.pin_h * np.ptp(z))[0].astype(np.int32)
    pieces = [{"piece": k, "part": names[piece_part[k]], "gaussians": int((piece == k).sum())} for k in range(len(piece_part))]
    print(f"[part_shake] {len(pieces)} part pieces: " + ", ".join(f"{p['part']}:{p['gaussians']:,}" for p in pieces)
          + f" | soft ({args.soft}) {soft.sum():,} | pinned {len(pin):,} | dll {os.path.basename(src)}", flush=True)
    cross = part_clean[e[:, 0]] != part_clean[e[:, 1]]           # 부위 경계 간선 (지표용)
    pa_, pb_ = np.sort(np.stack([part_clean[e[cross, 0]], part_clean[e[cross, 1]]], 1), 1).T
    pair_of = [f"{names[a_]}-{names[b_]}" for a_, b_ in zip(pa_, pb_)]
    pair_names = sorted(set(pair_of))
    pair_idx = np.array([pair_names.index(x) for x in pair_of], np.int64)
    soft_piece = np.isin(piece_part, soft_ids)
    if args.groups == "soft":                                    # 몸 = id 0 하나, 말랑한 조각 = 1, 2, …
        gid = np.zeros(len(piece_part), np.int32)
        gid[soft_piece] = np.arange(1, soft_piece.sum() + 1)
        group_ids = gid[piece]
        group_shape = np.r_[args.body_shape, np.full(soft_piece.sum(), args.soft_shape)].astype(np.float32)
    else:
        group_ids = piece
        group_shape = np.where(soft_piece, args.soft_shape, args.body_shape).astype(np.float32)
    body = ~soft

    def soft_rel_body(P):
        """말랑한 가우시안이 몸의 강체 운동을 뺀 뒤 쉬는 자세에서 벗어난 평균 거리 (원본 단위)."""
        a0, a = P0[body], P[body]
        c0, c = a0.mean(0), a.mean(0)
        U, _, Vt = np.linalg.svd((a0 - c0).T @ (a - c))
        R = (U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt).T
        return float(np.mean(np.linalg.norm(P[soft] - ((P0[soft] - c0) @ R.T + c), axis=1)))

    modes = ("uniform", "part")
    sims = []
    for k, mode in enumerate(modes):
        inp = {kk: (v.copy() if isinstance(v, np.ndarray) else v) for kk, v in inp0.items()}
        if mode == "uniform" or args.no_edge_ramp:
            inp["stiff"] = np.ascontiguousarray((inp["stiff"] * args.uniform_stiff).astype(np.float32))
        else:
            wv = graph_blend(soft.astype(np.float64), e, args.blend_hops)
            we = 0.5 * (wv[e[:, 0]] + wv[e[:, 1]])
            inp["stiff"] = np.ascontiguousarray((inp["stiff"] * args.body_stiff ** (1 - we) * args.soft_stiff ** we)
                                                .astype(np.float32))
        dll = os.path.join(out, f"xpbd_part_{k}.dll")             # DLL 전역 상태 → 시뮬레이션마다 복사본
        if not os.path.exists(dll) or os.path.getmtime(dll) < os.path.getmtime(src):
            shutil.copy2(src, dll)
        sim = XPBD(dll)
        if not getattr(sim, "has_part_constraints", False):
            raise RuntimeError(f"{src} 에 부위 API (set_part_ids · set_part_shape · part_stats) 가 없다")
        sim.create(inp)
        sim.set_solver(iters=16, dt=1 / 60, under_relax=0.6, vel_damping=args.damping, dist_compliance=args.compliance)
        sim.set_constraints(distance=True, shape=False, angle=False, volume=True)
        sim.set_volume(compliance=1e-6, ring_k=3, max_members=2048, leader_min_hop=2)
        sim.set_object_shape(args.uniform_shape if mode == "uniform" else args.body_shape)
        sim.set_object_shape_gpu(True)
        if mode == "part":
            sim.set_part_ids(np.ascontiguousarray(group_ids.astype(np.int32)), restrict_volume=not args.no_part_volume,
                             shape_groups=not args.no_part_shape)
            sim.set_part_shape(np.ascontiguousarray(group_shape))
        sim.set_ground(False, (0, 0, 1), 0.0, gravity=0.0)
        sim.step()
        sim.reset()
        sims.append(sim)
    part_stats = sims[1].part_stats()

    ex_src = Rm.T @ np.eye(3)["xyz".index(args.axis)] / s       # 월드 축 1 m → 원본 좌표
    offs = [np.array([-args.gap / 2, 0, 0]), np.array([args.gap / 2, 0, 0])]
    cam = Cam(vec(args.eye), vec(args.look), W, H, args.fovy)
    font = load_font(26)
    titles = args.titles.split(",")
    rest = inp0["rest"].astype(np.float64)
    wob = {m: {n: [] for n in names} for m in modes}
    shp = {m: {n: [] for n in names} for m in modes}
    bnd = {m: {"p95": [], "max": [], "over_1p5": [], "pairs": {n: [] for n in pair_names}} for m in modes}
    rel = {m: [] for m in modes}
    step_s = {m: 0.0 for m in modes}
    frames = []
    n_steps = int(args.seconds * 60)
    import imageio
    for k in range(n_steps):
        t = k / 60
        amp = args.amp * math.sin(2 * math.pi * args.freq * t) if t < args.shake_s else 0.0
        amp *= min(1.0, t / 0.2)                                  # 부드럽게 시작
        base = amp * ex_src
        for sim, mode in zip(sims, modes):
            sim.set_attached(pin, np.ascontiguousarray((P0[pin] + base).astype(np.float32)), 1.0)
            t0 = time.perf_counter()
            sim.step()
            step_s[mode] += time.perf_counter() - t0
        if k % 2:
            continue
        Ps, Qs, Ss, Os, Cs = [], [], [], [], []
        for sim, off, mode in zip(sims, offs, modes):
            P = sim.positions().astype(np.float64)
            dev = np.linalg.norm(P - (P0 + base), axis=1) * s * 100
            for kk, n in enumerate(names):
                mm = part_clean == kk
                if mm.sum() > 50:
                    wob[mode][n].append(float(np.mean(dev[mm])))
                    errs = [rigid_rms(P[piece == p], P0[piece == p]) * s * 100
                            for p in np.nonzero(piece_part == kk)[0] if (piece == p).sum() >= 4]
                    if errs:
                        shp[mode][n].append(float(np.mean(errs)))
            if soft.any() and body.any():
                rel[mode].append(soft_rel_body(P) * s * 100)
            if cross.any():
                r = np.linalg.norm(P[e[cross, 0]] - P[e[cross, 1]], axis=1) / np.maximum(rest[cross], 1e-12)
                bnd[mode]["p95"].append(float(np.percentile(r, 95)))
                bnd[mode]["max"].append(float(r.max()))
                bnd[mode]["over_1p5"].append(float(np.mean(r > 1.5)))
                for q_, n_ in enumerate(pair_names):                  # 어느 경계가 늘어나나 (팔-몸통 · 몸통-다리 …)
                    bnd[mode]["pairs"][n_].append(float(np.mean(r[pair_idx == q_] > 1.5)))
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
    rms = lambda v: round(float(np.sqrt(np.mean(np.square(v)))), 3)  # noqa: E731
    st = {k_: getattr(args, k_) for k_ in ("asset", "soft", "axis", "compliance", "uniform_stiff", "uniform_shape", "body_stiff",
                                            "body_shape", "soft_stiff", "soft_shape", "blend_hops", "min_piece",
                                            "groups", "no_part_shape", "no_part_volume", "no_edge_ramp", "pin_h", "amp", "freq",
                                            "shake_s", "seconds", "damping")}
    st.update({
        "dll": os.path.basename(src), "pieces": pieces, "boundary_edges": int(cross.sum()),
        "part_stats": {"cross_label_volume_clusters": int(part_stats[0]), "fallback_groups": int(part_stats[1]),
                       "shape_groups": int(part_stats[2])},
        "wobble_rms_cm": {m: {n: rms(v) for n, v in c.items() if v} for m, c in wob.items()},
        "soft_rel_body_cm": {m: {"rms": rms(v), "max": round(float(np.max(v)), 3)} for m, v in rel.items() if v},
        "wobble_peak_cm": {m: {n: round(float(np.max(v)), 3) for n, v in c.items() if v} for m, c in wob.items()},
        "shape_error_cm": {m: {n: {"rms": rms(v), "max": round(float(np.max(v)), 3)} for n, v in c.items() if v}
                           for m, c in shp.items()},
        "boundary": {m: {"p95_max": round(max(b["p95"]), 3), "max": round(max(b["max"]), 3),
                         "over_1p5_max": round(max(b["over_1p5"]), 5),
                         "over_1p5_max_by_pair": {n: round(max(v), 5) for n, v in b["pairs"].items() if v}}
                     for m, b in bnd.items() if b["p95"]},
        "step_ms": {m: round(1000 * v / max(n_steps, 1), 2) for m, v in step_s.items()},
    })
    json.dump(st, open(os.path.join(out, f"{args.tag}.json"), "w"), indent=2, ensure_ascii=False)
    print("[part_shake]", json.dumps({k_: st[k_] for k_ in ("part_stats", "wobble_rms_cm", "soft_rel_body_cm", "boundary", "step_ms")},
                                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
