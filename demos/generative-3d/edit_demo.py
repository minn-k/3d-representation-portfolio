"""생성된 3DGS 에셋을 APG 그래프 + XPBD 로 '잡아당겨' 편집하고, 별도 엔진 없이 3DGS 래스터라이저로 렌더한다.

  C:\\anaconda\\anaconda3\\envs\\trellis\\python.exe edit_demo.py --name gen_bear --grab-dir 0,1,0.3 [--dist 0.06]

render_cov_pull.py(기존 오프라인 파이프라인) 와 같은 절차: 아래 --pin-h 를 고정, --grab-dir 방향 끝 조각(--grab-r 안)을 잡아
그 방향으로 --dist [m] 천천히 끈 뒤 풀어 둔다. 세 가지 렌더:
  rest        당기기 전
  posonly     위치만 갱신 (가우시안 방향·크기는 그대로)
  posshape    위치 + 방향 + 크기 갱신 (Σ' = F Σ₀ Fᵀ — 이웃 변위로 추정한 국소 F)
출력: genai/out/<name>/edit_*.png, edit_compare.mp4 (왼쪽 위치만 / 오른쪽 Σ' 갱신), edit_stats.json
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gs_utils import SH_C0, load_font, mat_to_quat, quat_to_mat, runtime_root  # noqa: E402,F401  (다른 스크립트가 여기서 가져간다)

ROOT = os.environ.get("APG_ROOT", os.path.dirname(HERE))       # 3DGS 작업 폴더 (output_1/, apg_runtime/, SIBR_viewers/)
RUNTIME = runtime_root()                                        # APG-GS 준비 스크립트 · XPBD DLL (APG_RUNTIME_ROOT)
sys.path.insert(0, RUNTIME)
from prepare_splat import read_ply  # noqa: E402
from xpbd import XPBD, load_inputs  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer  # noqa: E402

class Cam:
    def __init__(self, eye, at, W, H, fovy_deg):
        eye, at = np.asarray(eye, float), np.asarray(at, float)
        f = at - eye
        f /= np.linalg.norm(f)
        r = np.cross(f, [0, 0, 1.0])
        r /= np.linalg.norm(r)
        d = np.cross(f, r)                                   # 카메라 y = 아래 (COLMAP/3DGS 규약)
        Rc = np.stack([r, d, f], 0)                          # world -> camera
        V = np.eye(4)
        V[:3, :3] = Rc
        V[:3, 3] = -Rc @ eye
        self.W, self.H = W, H
        self.tany = math.tan(math.radians(fovy_deg) / 2)
        self.tanx = self.tany * W / H
        n, fa = 0.01, 100.0
        P = np.zeros((4, 4))
        P[0, 0], P[1, 1] = 1 / self.tanx, 1 / self.tany
        P[2, 2], P[2, 3], P[3, 2] = fa / (fa - n), -fa * n / (fa - n), 1.0
        self.view = torch.tensor(V.T, dtype=torch.float32, device="cuda")
        self.proj = torch.tensor((P @ V).T, dtype=torch.float32, device="cuda")
        self.center = torch.tensor(eye, dtype=torch.float32, device="cuda")


def render(cam, pos, quat, scale, opacity, color, bg=(1.0, 1.0, 1.0)):
    t = lambda a: torch.tensor(np.ascontiguousarray(a), dtype=torch.float32, device="cuda")  # noqa: E731
    s = GaussianRasterizationSettings(image_height=cam.H, image_width=cam.W, tanfovx=cam.tanx, tanfovy=cam.tany,
                                      bg=torch.tensor(bg, dtype=torch.float32, device="cuda"), scale_modifier=1.0,
                                      viewmatrix=cam.view, projmatrix=cam.proj, sh_degree=0, campos=cam.center,
                                      prefiltered=False, debug=False, antialiasing=True)
    m = t(pos)
    out = GaussianRasterizer(s)(means3D=m, means2D=torch.zeros_like(m), opacities=t(opacity[:, None]),
                                colors_precomp=t(color), scales=t(scale), rotations=t(quat))
    img = out[0].clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy()
    return (img * 255 + 0.5).astype(np.uint8)


def label(img, text):
    from PIL import Image, ImageDraw
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    font = load_font(max(22, img.shape[0] // 17))
    dr.text((16, 12), text, fill=(30, 30, 30), font=font)
    return np.asarray(im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="apg_runtime/<name> (예: gen_bear)")
    ap.add_argument("--grab-dir", default="0,1,0.3", help="이 방향으로 가장 끝에 있는 조각을 잡는다 (월드, z 위)")
    ap.add_argument("--pull", default="", help="당기는 방향 (기본: grab-dir)")
    ap.add_argument("--grab-r", type=float, default=0.03)
    ap.add_argument("--pin-h", type=float, default=0.35)
    ap.add_argument("--pin-far", type=float, default=0.0, help="잡은 곳에서 이 거리[m]보다 먼 가우시안도 고정 (국소 편집)")
    ap.add_argument("--dist", type=float, default=0.06)
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--settle", type=int, default=60)
    ap.add_argument("--every", type=int, default=3, help="영상 프레임 간격 (스텝)")
    ap.add_argument("--eye", default="", help="카메라 x,y,z (기본: 잡는 쪽 옆 비스듬히)")
    ap.add_argument("--fovy", type=float, default=35.0)
    ap.add_argument("--focus", type=float, default=0.0, help="카메라가 바라보는 점을 물체 중심에서 잡은 곳 쪽으로 이 비율만큼")
    ap.add_argument("--res", default="720x720")
    ap.add_argument("--out", default="")
    ap.add_argument("--parts", default="", help="lift_parts.py 의 parts3d.npz — 부위 인식 그래프")
    ap.add_argument("--cross-scale", type=float, default=1.0, help="부위 경계를 넘는 간선 강성 배율 (1 = 기하 그래프 그대로)")
    ap.add_argument("--grab-part", default="", help="이 부위(parts3d 이름) 안에서만 잡을 곳을 찾는다")
    ap.add_argument("--near-r", type=float, default=0.025, help="누수 지표: 잡은 부위에서 이 거리[m] 안의 다른 부위")
    ap.add_argument("--cross-keep", type=float, default=1.0,
                    help="부위 경계를 넘는 간선 중 남길 비율 (1 = 기하 그래프 그대로). 형상·부피 클러스터도 이 연결로 만들어진다")
    ap.add_argument("--conf", type=float, default=0.5, help="이 신뢰도 이상인 가우시안끼리만 경계 판정")
    ap.add_argument("--joint", default="random", choices=["random", "overlap"],
                    help="남길 경계 간선 고르기: random (포트폴리오 수치) · overlap (부위 쌍마다 Bhattacharyya 겹침이 가장 큰 것)")
    ap.add_argument("--dist-compliance", type=float, default=0.0,
                    help="거리 제약 compliance. 0 이면 완전 강체라 간선 강성(--cross-scale)이 효과가 없다 (α̃ = compliance/강성/dt²)")
    ap.add_argument("--tag", default="", help="출력 파일 이름 꼬리표")
    ap.add_argument("--single", default="", help="이 글자를 붙인 Σ′ 렌더만 영상으로 (비교 영상 이어 붙이기용)")
    ap.add_argument("--views", action="store_true", help="당기지 않고 네 방향(+x +y -x -y) rest 만 찍는다 (잡을 곳 고르기)")
    args = ap.parse_args()
    vec = lambda t: np.array([float(v) for v in t.split(",")])  # noqa: E731
    W_, H_ = (int(v) for v in args.res.split("x"))
    d = os.path.join(RUNTIME, args.name)
    out = args.out or os.path.join(HERE, "out", args.name.replace("gen_", "", 1))
    os.makedirs(out, exist_ok=True)

    inp = load_inputs(d, args.name)
    part_st = {}
    if args.parts:
        from part_graph import apply_part_graph
        pz = np.load(args.parts)
        pa, pc = pz["asset_part"].astype(int), pz["asset_conf"]
        pnames = [str(n) for n in pz["names"]]
        if len(pa) != len(inp["pos"]):
            raise SystemExit(f"parts3d 의 에셋({len(pa):,})이 {args.name}({len(inp['pos']):,})와 다르다 — "
                             "lift_parts.py --asset 을 이 에셋으로 다시 돌릴 것")
        inp, part_st = apply_part_graph(inp, pa, pc, args.conf, args.cross_scale, args.cross_keep, args.joint)   # 관절만 남김
        print(f"[edit] parts {pnames}: cross-part edges {part_st['cross_part_edge_ratio']:.2%} -> stiffness "
              f"x{args.cross_scale}, keep {args.cross_keep:.0%} ({args.joint})", flush=True)
    _, data = read_ply(os.path.join(d, f"{args.name}_crop.ply"))
    color = np.clip(SH_C0 * np.stack([data["f_dc_0"], data["f_dc_1"], data["f_dc_2"]], 1) + 0.5, 0, 1)
    xf = json.load(open(os.path.join(d, f"{args.name}_transform.json"), encoding="utf-8"))
    s, Rm, tw = float(xf["scale"]), np.array(xf["rotation_matrix"], float), np.array(xf["translate"], float)
    to_w = lambda P: s * (P.astype(np.float64) @ Rm.T) + tw  # noqa: E731
    Wr = to_w(inp["pos"])
    R0 = quat_to_mat(inp["rots"].astype(np.float64))
    q_rest_w = mat_to_quat(Rm[None] @ R0)
    sc_rest_w = inp["scales"] * s
    op = inp["opacity"]

    c = Wr.mean(0)
    lo, hi = Wr.min(0), Wr.max(0)
    g_dir = vec(args.grab_dir)
    g_dir /= np.linalg.norm(g_dir)
    proj = (Wr - c) @ g_dir
    cand = np.ones(len(Wr), bool)
    if args.grab_part:
        cand = pa == pnames.index(args.grab_part)
        proj = np.where(cand, proj, -np.inf)
    g0 = Wr[np.argsort(-proj)[:200]].mean(0)                       # 끝 200개 평균 = 잡는 점
    grab = np.where((np.linalg.norm(Wr - g0, axis=1) < args.grab_r) & cand)[0].astype(np.int32)
    z = Wr[:, 2]
    pinm = z < z.min() + args.pin_h * np.ptp(z)
    if args.pin_far > 0:
        pinm |= np.linalg.norm(Wr - g0, axis=1) > args.pin_far
    pin = np.where(pinm)[0].astype(np.int32)
    pin = np.setdiff1d(pin, grab).astype(np.int32)
    u = vec(args.pull) if args.pull else g_dir.copy()
    u /= np.linalg.norm(u)

    size = np.linalg.norm(hi - lo)
    if args.views:
        from PIL import Image
        ims = []
        for dx, dy in ((1, 0), (0, 1), (-1, 0), (0, -1)):
            cv = Cam(c + 1.4 * size * np.array([dx, dy, 0.35]), c, 400, 400, args.fovy)
            ims.append(label(render(cv, Wr, q_rest_w, sc_rest_w, op, color), f"eye {dx:+d},{dy:+d}"))
        Image.fromarray(np.concatenate(ims, 1)).save(os.path.join(out, "views.png"))
        print(f"[edit] views -> {out} | bbox {np.round(lo, 3).tolist()} .. {np.round(hi, 3).tolist()}")
        return
    if args.eye:
        eye = vec(args.eye)
    else:
        side = np.cross([0, 0, 1.0], g_dir)
        side = side / np.linalg.norm(side) if np.linalg.norm(side) > 1e-6 else np.array([1.0, 0, 0])
        eye = c + 1.6 * size * (0.85 * side + 0.35 * g_dir) / np.linalg.norm(0.85 * side + 0.35 * g_dir) \
            + np.array([0, 0, 0.45 * size])
    at = (1 - args.focus) * c + args.focus * g0 + 0.5 * args.dist * u
    cam = Cam(eye, at, W_, H_, args.fovy)

    img_rest = render(cam, Wr, q_rest_w, sc_rest_w, op, color)
    from PIL import Image
    Image.fromarray(img_rest).save(os.path.join(out, "edit_rest.png"))

    sim = XPBD()
    sim.create(inp)
    sim.set_solver(iters=20, dt=1 / 60, under_relax=0.6, vel_damping=0.1, dist_compliance=args.dist_compliance)
    sim.set_constraints(distance=True, shape=True, angle=False, volume=True)
    sim.set_volume(compliance=1e-6, ring_k=3, max_members=2048, leader_min_hop=2)
    sim.set_object_shape(0.0)
    sim.set_ground(False, (0, 0, 1), 0.0, gravity=0.0)
    sim.step()
    sim.reset()
    P0 = inp["pos"].astype(np.float64)
    idx = np.concatenate([pin, grab])
    print(f"[edit] {args.name}: {len(Wr):,} Gaussians | pinned {len(pin):,} | grabbed {len(grab):,} at "
          f"{np.round(g0, 3).tolist()} | pull {args.dist * 100:.1f} cm along {np.round(u, 2).tolist()}", flush=True)

    frames = []
    t_sim = t_shape = 0.0
    total = args.steps + args.settle
    for k in range(total):
        f = min(1.0, (k + 1) / args.steps)
        f = f * f * (3 - 2 * f)
        Pg = P0[grab] + (Rm.T @ (u * args.dist * f)) / s
        sim.set_attached(idx, np.concatenate([P0[pin], Pg]).astype(np.float32), 1.0)
        t0 = time.perf_counter()
        sim.step()
        t_sim += time.perf_counter() - t0
        if k % args.every == 0 or k == total - 1:
            P = sim.positions()
            t0 = time.perf_counter()
            sc, q, kk = sim.shapes(1e-2)
            t_shape += time.perf_counter() - t0
            Pw = to_w(P)
            a = render(cam, Pw, q_rest_w, sc_rest_w, op, color)
            qw = mat_to_quat(Rm[None] @ quat_to_mat(q.astype(np.float64)))
            b = render(cam, Pw, qw, sc * s, op, color)
            frames.append(label(b, args.single) if args.single else
                          np.concatenate([label(a, "위치만 갱신"), label(b, "위치 + Σ' = FΣ₀Fᵀ")], axis=1))
    n_shape_calls = len(frames)
    P = sim.positions()
    sc, q, kk = sim.shapes(1e-2)
    e = inp["edges"]
    r_ = np.linalg.norm(P[e[:, 0]] - P[e[:, 1]], axis=1) / np.maximum(inp["rest"], 1e-12)
    Pw = to_w(P)
    img_pos = render(cam, Pw, q_rest_w, sc_rest_w, op, color)
    qw = mat_to_quat(Rm[None] @ quat_to_mat(q.astype(np.float64)))
    img_shape = render(cam, Pw, qw, sc * s, op, color)
    Image.fromarray(img_pos).save(os.path.join(out, f"edit_posonly{args.tag}.png"))
    Image.fromarray(img_shape).save(os.path.join(out, f"edit_posshape{args.tag}.png"))
    if args.parts:                                   # 부위별 변위 [cm] — 잡지 않은 부위가 얼마나 끌려왔나 (누수)
        disp = np.linalg.norm(P.astype(np.float64) - P0, axis=1) * s * 100
        free = np.ones(len(P), bool)
        free[pin] = False
        free[grab] = False
        part_st["disp_cm_by_part"] = {n: round(float(disp[(pa == k) & free].mean()), 3) for k, n in enumerate(pnames)
                                      if np.any((pa == k) & free)}
        gp_ = int(np.bincount(pa[grab]).argmax())
        part_st["grab_part"] = pnames[gp_]
        gpos = Wr[pa == gp_]
        dn, _ = cKDTree(gpos).query(Wr)
        near = (pa != gp_) & (dn < args.near_r) & free
        part_st["near_other_parts"] = int(near.sum())
        part_st["disp_cm_near_other_parts"] = round(float(disp[near].mean()), 4) if near.any() else 0.0
        part_st["disp_cm_grab_part"] = round(float(disp[(pa == gp_) & free].mean()), 4)
    diff = float(np.abs(img_pos.astype(np.float32) - img_shape.astype(np.float32)).mean())
    sim.destroy()

    import imageio
    imageio.mimsave(os.path.join(out, f"edit_compare{args.tag}.mp4"), frames + [frames[-1]] * 30, fps=30, quality=8,
                    macro_block_size=8)
    st = {"name": args.name, "gaussians": int(len(Wr)), "edges": int(len(e)), "pinned": int(len(pin)),
          "grabbed": int(len(grab)), "pull_cm": args.dist * 100, "steps": total,
          "xpbd_ms_per_step": round(1000 * t_sim / total, 2),
          "shape_update_ms": round(1000 * t_shape / n_shape_calls, 2),
          "deformed_gaussians": int(kk), "edges_over_1p5x": float(np.mean(r_ > 1.5)),
          "edges_over_2x": float(np.mean(r_ > 2)), "mean_abs_pixel_diff_pos_vs_shape": round(diff, 2), "dist_compliance": args.dist_compliance, **part_st}
    json.dump(st, open(os.path.join(out, f"edit_stats{args.tag}.json"), "w"), indent=2)
    print("[edit]", json.dumps(st), flush=True)


if __name__ == "__main__":
    main()
