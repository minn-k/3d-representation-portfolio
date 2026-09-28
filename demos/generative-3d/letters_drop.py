"""TRELLIS 티저 연출 따라 하기 + APG-GS 물리: 생성된 글자 T R E L L I S 가 차례로 떨어져 바닥에 부딪혀 눌리고 튕긴다.

  C:\\anaconda\\anaconda3\\envs\\trellis\\python.exe letters_drop.py [--preview] [--fps 30] [--res 1920x1080]

- 글자마다 XPBD 인스턴스 하나: DLL 은 전역 상태 하나라 글자 수만큼 DLL 파일을 복사해 따로 불러온다 (글자끼리는 충돌 없음 —
  원본 영상처럼 떨어져 놓인다).
- 물리 좌표 = 각 글자의 원본 3DGS 좌표. 바닥은 월드 z = -DROP (보이지 않는 흰 바닥), 렌더 때 +DROP 해서 바닥을 z = 0 으로.
- 렌더: 3DGS 래스터라이저, 흰 배경. 글자 줄 배치(x 오프셋 · 정면 yaw)는 렌더 변환일 뿐 물리에 영향 없음 (수직축 회전).
- 카메라: ① 정면 고정 — 글자가 위에서 차례로 떨어진다 ② 뒤로 물러나며 비스듬히 ③ 글자 사이를 낮게 지나간다.
출력: genai/out/letters/letters_drop.mp4, letters_drop.json (설정·물리 통계)
"""
import argparse
import json
import math
import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("APG_ROOT", os.path.dirname(HERE))       # 3DGS 작업 폴더 (output_1/, apg_runtime/, SIBR_viewers/)
RUNTIME = os.environ.get("APG_RUNTIME_ROOT", os.path.join(ROOT, "apg_runtime"))   # APG-GS 준비 스크립트 · XPBD DLL
sys.path.insert(0, RUNTIME)
sys.path.insert(0, HERE)
from prepare_splat import read_ply  # noqa: E402
from xpbd import XPBD, load_inputs, world_ground  # noqa: E402
from edit_demo import Cam, render, quat_to_mat, mat_to_quat  # noqa: E402

LETTERS = ["T", "R", "E", "L", "L2", "I", "S"]
VAR = "d50k"
DROP = 0.6            # 떨어지는 높이 [m] (글자 높이 0.3 m) — 화면 위에서 들어온다
GAP_S = 0.33          # 글자 사이 출발 간격 [s] (원본 영상 약 0.3~0.5 s)
GRAVITY = 9.81
OBJECT_SHAPE = 0.6    # 물체 단위 형상 유지 (낮을수록 무르게 눌린다)
RESTITUTION = 0.5     # 입자 단위 반발
FRICTION = 0.6
ITERS = 16
DAMPING = 0.01        # 속도 감쇠 [/step]
SPACING = 0.035       # 글자 사이 틈 [m]
SH_C0 = 0.28209479177387814


def rotz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def smooth(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--res", default="1920x1080")
    ap.add_argument("--seconds", type=float, default=13.0)
    ap.add_argument("--preview", action="store_true", help="저해상도 · 몇 프레임만 (배치 · 방향 확인)")
    ap.add_argument("--tag", default="", help="출력 파일 이름 꼬리표")
    ap.add_argument("--probe", action="store_true", help="렌더 없이 물리만 6 s — 글자마다 높이 비율(서 있으면 ~1) 출력")
    for k, v in (("drop", DROP), ("gravity", GRAVITY), ("shape", OBJECT_SHAPE), ("restitution", RESTITUTION),
                 ("damping", 0.0), ("friction", FRICTION)):
        ap.add_argument(f"--{k.replace('_', '-')}", type=float, default=None)
    ap.add_argument("--yaw", default="", help="글자별 정면 yaw [deg] 덮어쓰기 T,R,E,L,L2,I,S")
    args = ap.parse_args()
    g_ = globals()
    for k, name in (("drop", "DROP"), ("gravity", "GRAVITY"), ("shape", "OBJECT_SHAPE"), ("restitution", "RESTITUTION"),
                    ("damping", "DAMPING"), ("friction", "FRICTION")):
        if getattr(args, k) is not None:
            g_[name] = getattr(args, k)
    W, H = (int(v) for v in args.res.split("x"))
    if args.preview:
        W, H = W // 2, H // 2
    out = os.path.join(HERE, "out", "letters")
    os.makedirs(out, exist_ok=True)

    # DLL 복사본 (글자마다 독립 전역 상태)
    dll_root = os.path.join(RUNTIME, "xpbd_dll")
    candidates = [os.path.join(dll_root, n) for n in os.listdir(dll_root) if n.lower().endswith(".dll")]
    if not candidates:
        raise FileNotFoundError(f"No XPBD DLL found in {dll_root}")
    src = max(candidates, key=os.path.getmtime)
    dll_dir = os.path.join(HERE, "out", "letters", "dll")
    os.makedirs(dll_dir, exist_ok=True)

    L = []
    for k, name in enumerate(LETTERS):
        nm = f"gen_letter_{name}_{VAR}"
        d = os.path.join(RUNTIME, nm)
        inp = load_inputs(d, nm)
        _, data = read_ply(os.path.join(d, f"{nm}_crop.ply"))
        col = np.clip(SH_C0 * np.stack([data["f_dc_0"], data["f_dc_1"], data["f_dc_2"]], 1) + 0.5, 0, 1)
        xf = json.load(open(os.path.join(d, f"{nm}_transform.json"), encoding="utf-8"))
        s, Rm, tw = float(xf["scale"]), np.array(xf["rotation_matrix"], float), np.array(xf["translate"], float)
        up, h, g, _ = world_ground(d, nm, ground_z=-DROP, gravity=GRAVITY)
        Wr = s * (inp["pos"].astype(np.float64) @ Rm.T) + tw
        # 정면 yaw: 글자 판의 얇은 수평 축을 y 로 (PCA). 부호(앞/뒤)는 --yaw 로 고친다
        xy = Wr[:, :2] - Wr[:, :2].mean(0)
        ev, V = np.linalg.eigh(xy.T @ xy)
        thin = V[:, 0]
        yaw = -math.atan2(thin[0], thin[1]) + math.pi     # PCA 부호로는 뒷면이 보였다 → 180°
        L.append(dict(name=name, nm=nm, inp=inp, col=col, s=s, Rm=Rm, tw=tw, up=up, h=h, g=g, yaw=yaw,
                      Hs=float(np.ptp(inp["pos"].astype(np.float64) @ up))))
    if args.yaw:
        for l, v in zip(L, args.yaw.split(",")):
            if v:
                l["yaw"] = math.radians(float(v))
    # 줄 배치: 회전 뒤 x 폭으로 차례로
    x = 0.0
    for l in L:
        Rz = rotz(l["yaw"])
        P = (l["s"] * (l["inp"]["pos"].astype(np.float64) @ l["Rm"].T) + l["tw"]) @ Rz.T
        lo, hi = np.percentile(P, 1, 0), np.percentile(P, 99, 0)
        l["Rz"] = Rz
        l["off"] = np.array([x - lo[0], -0.5 * (lo[1] + hi[1]), 0.0])
        x += (hi[0] - lo[0]) + SPACING
    row_w = x - SPACING
    for l in L:
        l["off"][0] -= 0.5 * row_w

    for k, l in enumerate(L):
        dll = os.path.join(dll_dir, f"xpbd_{k}.dll")
        if not os.path.exists(dll) or os.path.getmtime(dll) < os.path.getmtime(src):
            shutil.copy2(src, dll)
        sim = XPBD(dll)
        sim.create(l["inp"])
        sim.set_solver(iters=ITERS, dt=1 / 60, under_relax=0.6, vel_damping=DAMPING)
        sim.set_constraints(distance=True, shape=False, angle=False, volume=True)
        sim.set_volume(compliance=1e-6, ring_k=3, max_members=2048, leader_min_hop=2)
        sim.set_object_shape(OBJECT_SHAPE)
        sim.set_object_shape_gpu(True)
        sim.set_ground(False, tuple(l["up"]), 0.0, gravity=0.0)
        sim.step()
        sim.reset()
        sim.set_ground(True, tuple(l["up"]), l["h"], friction=FRICTION, restitution=RESTITUTION,
                       contact_radius=0.003 * l["Hs"], contact_slop=0.002 * l["Hs"], gravity=l["g"])
        l["sim"] = sim
        l["start"] = 0.25 + k * GAP_S
        l["P"] = l["inp"]["pos"].copy()
        l["sc"], l["q"] = l["inp"]["scales"].copy(), l["inp"]["rots"].copy()
        l["landed"] = None
        l["min_h"] = 1e9

    def world(l, P, sc, q):
        M = l["Rz"] @ l["Rm"]
        Pw = l["s"] * (P.astype(np.float64) @ M.T) + l["Rz"] @ l["tw"] + l["off"] + np.array([0, 0, DROP])
        qw = mat_to_quat(M[None] @ quat_to_mat(q.astype(np.float64)))
        return Pw, qw, sc * l["s"]

    # 카메라 경로 (원본 영상 순서): 정면 → 물러나며 비스듬히 → 낮게 지나가기
    zc = 0.15

    def camera(t):
        if t < 4.2:                                   # ① 정면 고정 (글자 줄 전체가 화면 폭의 ~90%)
            d = 0.56 * row_w / (math.tan(math.radians(20)) * W / H)
            return np.array([0, -d, zc + 0.45]), np.array([0, 0, 0.08]), 40.0     # 조금 위에서: 누운 글자도 보이게
        if t < 6.5:                                   # ② 물러나며 왼쪽 위로 돌아 비스듬히
            u = smooth((t - 4.2) / 2.3)
            d0 = 0.56 * row_w / (math.tan(math.radians(20)) * W / H)
            a = math.radians(-90 - 35 * u)
            d = d0 * (1 + 0.6 * u)
            return np.array([d * math.cos(a), d * math.sin(a), zc + 0.45 + 0.6 * u]), np.array([0, 0, 0.08]), 40.0
        u = smooth((t - 6.5) / (args.seconds - 6.5))  # ③ 글자 앞을 낮고 가깝게 왼쪽 → 오른쪽으로
        xx = -0.45 * row_w + 0.8 * row_w * u         # 줄 안쪽에서 끝난다 (1.1 배는 끝 몇 초가 빈 화면이었다)
        eye = np.array([xx - 0.2, -0.5, 0.38 + 0.05 * math.sin(3 * u)])
        at = np.array([xx + 0.1, 0.0, 0.05])
        return eye, at, 45.0

    def advance(l):
        l["sim"].step()

    if args.probe:
        for step in range(int(6.0 * 60)):
            for l in L:
                if step / 60 >= l["start"]:
                    advance(l)
        res = {}
        for l in L:
            P = l["sim"].positions().astype(np.float64)
            h0 = np.ptp(l["inp"]["pos"].astype(np.float64) @ l["up"])
            res[l["name"]] = round(float(np.percentile(P @ l["up"], 99) - np.percentile(P @ l["up"], 1)) / h0, 2)
            l["sim"].destroy()
        print("[probe]", json.dumps({"drop": DROP, "g": GRAVITY, "shape": OBJECT_SHAPE, "e": RESTITUTION, "damp": DAMPING,
                                     "mu": FRICTION}), json.dumps(res),
              "upright", sum(v > 0.75 for v in res.values()), flush=True)
        return

    frames = []
    n_frames = int(args.seconds * args.fps) if not args.preview else 4
    steps_per_frame = max(1, round(60 / args.fps))
    t_phys = t_rend = 0.0
    times = [i / args.fps for i in range(n_frames)] if not args.preview else [1.6, 3.8, 5.6, 9.0]
    step_count = 0
    import imageio
    for fi, t in enumerate(times):
        # 물리: 60 Hz 로 t 까지
        while step_count / 60 < t:
            tt = step_count / 60
            t0 = time.perf_counter()
            for l in L:
                if tt >= l["start"]:
                    advance(l)
                    l["dirty"] = True
            t_phys += time.perf_counter() - t0
            step_count += 1
        t0 = time.perf_counter()
        Ps, Qs, Ss, Os, Cs = [], [], [], [], []
        for l in L:
            if t < l["start"]:                        # 출발 전에는 그리지 않는다 (화면 위 밖에서 들어온다)
                continue
            if l.get("dirty"):
                l["P"] = l["sim"].positions()
                l["sc"], l["q"], _ = l["sim"].shapes(1e-2)
                l["dirty"] = False
                hgt = float(np.percentile(l["P"].astype(np.float64) @ l["up"], 0.5))
                l["min_h"] = min(l["min_h"], hgt)
            Pw, qw, sw = world(l, l["P"], l["sc"], l["q"])
            Ps.append(Pw), Qs.append(qw), Ss.append(sw), Os.append(l["inp"]["opacity"]), Cs.append(l["col"])
        eye, at, fov = camera(t)
        cam = Cam(eye, at, W, H, fov)
        if Ps:
            img = render(cam, np.concatenate(Ps), np.concatenate(Qs), np.concatenate(Ss), np.concatenate(Os),
                         np.concatenate(Cs))
        else:                                         # 아직 출발한 글자가 없다
            img = np.full((H, W, 3), 255, np.uint8)
        t_rend += time.perf_counter() - t0
        frames.append(img)
        if args.preview:
            imageio.imwrite(os.path.join(out, f"preview_{fi}.png"), img)
        elif fi % 30 == 0:
            print(f"[letters] frame {fi}/{n_frames}", flush=True)
    if not args.preview:
        imageio.mimsave(os.path.join(out, f"letters_drop{args.tag}.mp4"), frames, fps=args.fps, quality=8, macro_block_size=8)
    st = {"letters": LETTERS, "variant": VAR, "gaussians_per_letter": [int(len(l["P"])) for l in L],
          "drop_m": DROP, "gap_s": GAP_S, "gravity": GRAVITY, "object_shape": OBJECT_SHAPE, "restitution": RESTITUTION,
          "friction": FRICTION, "iters": ITERS, "damping": DAMPING, "physics_hz": 60, "fps": args.fps, "frames": len(frames),
          "physics_s": round(t_phys, 1), "render_s": round(t_rend, 1),
          "yaw_deg": [round(math.degrees(l["yaw"]), 1) for l in L]}
    for l in L:
        l["sim"].destroy()
    json.dump(st, open(os.path.join(out, f"letters_drop{args.tag}.json" if not args.preview else "preview.json"), "w"), indent=2)
    print("[letters]", json.dumps(st), flush=True)


if __name__ == "__main__":
    main()
