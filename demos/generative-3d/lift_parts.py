"""생성 과정의 신호로 2D 부위 이름을 3D 가우시안에 옮긴다.

  python lift_parts.py --name robot_sem --asset gen_robot_sem_d100k

1 투표 (attention): 입력 이미지 패치(37×37)마다 부위 비율 M (parts2d.npz). SLat 트랜스포머 토큰의 앞쪽 블록 cross-attention
  (cls·register 제외, 토큰 평균 대비 '특별히 더 본' 패치로 바꿔 정규화) 으로 S0 = A_rel · M.
2 전파 (DiT 특징): 토큰마다 블록 6·12 특징(정규화해 이어 붙임) 의 k-NN 그래프 (공간적으로도 가까운 것만) 위에서
  Y ← α·W·Y + (1−α)·S0 반복. 입력 사진에 안 보이던 뒷면 · 2D 에서 비어 있던 곳이 이웃 특징을 따라 채워진다.
3 가우시안: 원본 가우시안 i → 복셀 i//32 → 토큰. 편집용 에셋(개수 줄인 판)은 위치로 가장 가까운 복셀.
출력 out/<name>/parts3d.npz (tok_prob · tok_prob_attn · asset_part · asset_conf), parts3d_grid.png, parts3d_stats.json
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sem_viz import load_gs, vox_to_tok, kmeans, T_SAVE, N_PATCH  # noqa: E402
from edit_demo import Cam, render, RUNTIME  # noqa: E402

PART_COLORS = np.array([[0.75, 0.75, 0.75], [0.90, 0.30, 0.25], [0.25, 0.55, 0.90], [0.30, 0.75, 0.35],
                        [0.95, 0.75, 0.20], [0.65, 0.40, 0.85], [0.20, 0.80, 0.80], [0.95, 0.50, 0.70]])


def patch_fractions(label, K):
    H, W = label.shape
    ph, pw = H / N_PATCH, W / N_PATCH
    M = np.zeros((N_PATCH * N_PATCH, K + 1), np.float32)
    for r in range(N_PATCH):
        for c in range(N_PATCH):
            blk = label[int(r * ph):int((r + 1) * ph), int(c * pw):int((c + 1) * pw)]
            M[r * N_PATCH + c] = np.bincount(blk.ravel(), minlength=K + 1)[:K + 1] / max(blk.size, 1)
    return M[:, 1:]                       # 부위만 (배경·미분류 제외)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--asset", default="", help="<APG runtime>/<asset> — 편집용 에셋(크롭 PLY)에 부위를 붙인다")
    ap.add_argument("--attn-blocks", default="4,8,12")
    ap.add_argument("--feat-blocks", default="6,12")
    ap.add_argument("--knn", type=int, default=12)
    ap.add_argument("--alpha", type=float, default=0.9)
    ap.add_argument("--iters", type=int, default=40)
    args = ap.parse_args()
    out = os.path.join(HERE, "out", args.name)
    sem = dict(np.load(os.path.join(out, "sem.npz")))
    p2 = np.load(os.path.join(out, "parts2d.npz"))
    names = [str(n) for n in p2["names"]][1:]
    K = len(names)
    M = patch_fractions(p2["label"], K)

    # 1 attention 투표
    blocks = [int(b) for b in args.attn_blocks.split(",")]
    A = sum(sem[f"attn_{b}"].astype(np.float32)[:, 5:] for b in blocks) / len(blocks)
    A = A / np.maximum(A.sum(1, keepdims=True), 1e-9)
    Ar = A / np.maximum(A.mean(0, keepdims=True), 1e-9)
    Ar = np.maximum(Ar - 1.0, 0)                           # 평균보다 더 본 몫만
    Ar = Ar / np.maximum(Ar.sum(1, keepdims=True), 1e-9)
    S0 = Ar @ M
    S0 = S0 / np.maximum(S0.sum(1, keepdims=True), 1e-9)

    # 2 DiT 특징 그래프 전파
    tc = sem["tok_coords"].astype(np.float32)
    F = np.concatenate([(lambda f: (f - f.mean(0)) / (f.std(0) + 1e-6))(sem[f"feat_{b}"].astype(np.float32))
                        for b in args.feat_blocks.split(",")], 1)
    F /= np.linalg.norm(F, axis=1, keepdims=True)
    tree = cKDTree(tc)
    nb = tree.query_ball_point(tc, r=4.0)                   # 공간 반경 4 토큰(= 8 복셀) 안에서
    rows, cols, vals = [], [], []
    for i, js in enumerate(nb):
        js = np.array([j for j in js if j != i])
        if len(js) == 0:
            continue
        sim = F[js] @ F[i]
        top = js[np.argsort(-sim)[:args.knn]]
        w = np.exp((F[top] @ F[i] - 1) / 0.1)
        rows += [i] * len(top)
        cols += list(top)
        vals += list(w)
    from scipy.sparse import csr_matrix
    Wm = csr_matrix((vals, (rows, cols)), shape=(len(tc), len(tc)))
    Wm = Wm + Wm.T
    d = np.asarray(Wm.sum(1)).ravel()
    Wn = csr_matrix(Wm.multiply(1 / np.maximum(d, 1e-9)[:, None]))
    Y = S0.copy()
    for _ in range(args.iters):
        Y = args.alpha * (Wn @ Y) + (1 - args.alpha) * S0
    Y = Y / np.maximum(Y.sum(1, keepdims=True), 1e-9)
    tok_part, tok_conf = Y.argmax(1), Y.max(1)

    # 기준선: 좌표 k-means (같은 부위 수), 특징 k-means (이름 없음)
    base_xyz = kmeans(tc, K)
    base_feat = kmeans(F, K)

    # 3 가우시안 (원본 전체 — 그림용)
    P, q, S, op, col = load_gs(out)
    v2t = vox_to_tok(sem)
    g2t = v2t[np.arange(len(P)) // 32]
    st = {"parts": names, "tokens": int(len(tc)),
          "token_share_attn": {n: float(np.mean(S0.argmax(1) == k)) for k, n in enumerate(names)},
          "token_share_prop": {n: float(np.mean(tok_part == k)) for k, n in enumerate(names)},
          "changed_by_propagation": float(np.mean(S0.argmax(1) != tok_part)),
          "conf_median": float(np.median(tok_conf))}

    rowsimg = [("original", col),
               ("① attention 투표만", PART_COLORS[1 + S0.argmax(1)][g2t]),
               ("① + ② DiT 특징 전파", PART_COLORS[1 + tok_part][g2t]),
               ("기준선: DiT 특징 k-means (이름 없음)", PART_COLORS[1 + base_feat][g2t]),
               ("기준선: 좌표 k-means", PART_COLORS[1 + base_xyz][g2t])]
    c = P.mean(0)
    size = np.linalg.norm(P.max(0) - P.min(0))
    cams = [Cam(c + 1.3 * size * np.array([np.cos(a), np.sin(a), 0.3]), c, 256, 256, 35)
            for a in np.radians([-90, 0, 90, 180])]
    font = ImageFont.truetype("C:/Windows/Fonts/malgunbd.ttf", 16)
    tiles = []
    for nm, cc in rowsimg:
        im = Image.fromarray(np.concatenate([render(cm, P, q, S, op, cc) for cm in cams], 1))
        ImageDraw.Draw(im).text((6, 4), nm, fill=(20, 20, 20), font=font)
        tiles.append(np.asarray(im))
    leg = Image.new("RGB", (1024, 30), (255, 255, 255))
    dr = ImageDraw.Draw(leg)
    for k, n in enumerate(names):
        dr.rectangle((10 + k * 140, 8, 28 + k * 140, 24), fill=tuple(int(v * 255) for v in PART_COLORS[1 + k]))
        dr.text((34 + k * 140, 6), n, fill=(20, 20, 20), font=font)
    tiles.append(np.asarray(leg))
    Image.fromarray(np.concatenate(tiles, 0)).save(os.path.join(out, "parts3d_grid.png"))

    res = {"names": np.array(names), "tok_prob": Y.astype(np.float32), "tok_prob_attn": S0.astype(np.float32),
           "gs_part_full": tok_part[g2t].astype(np.int8)}
    if args.asset:
        from prepare_splat import read_ply
        _, data = read_ply(os.path.join(RUNTIME, args.asset, f"{args.asset}_crop.ply"))
        Pa = np.stack([data["x"], data["y"], data["z"]], 1).astype(np.float64) @ T_SAVE   # PLY → 내부
        vc = (sem["slat_coords"].astype(np.float64) + 0.5) / 64 - 0.5
        _, vi = cKDTree(vc).query(Pa)
        at = v2t[vi]
        res["asset_part"] = tok_part[at].astype(np.int8)
        res["asset_conf"] = tok_conf[at].astype(np.float32)
        res["asset_prob"] = Y[at].astype(np.float32)
        st["asset"] = args.asset
        st["asset_share"] = {n: float(np.mean(res["asset_part"] == k)) for k, n in enumerate(names)}
    np.savez_compressed(os.path.join(out, "parts3d.npz"), **res)
    json.dump(st, open(os.path.join(out, "parts3d_stats.json"), "w"), indent=2, ensure_ascii=False)
    print("[lift]", json.dumps(st, ensure_ascii=False))


if __name__ == "__main__":
    main()
