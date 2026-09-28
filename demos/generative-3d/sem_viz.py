"""gen3d_sem.py 가 저장한 의미 신호를 가우시안 색으로 칠해 비교한다 (TRELLIS 내부 좌표, z 위).

  python sem_viz.py --name robot_sem

행: 원래 색 | attention 이 가리키는 입력 이미지 색 | DiT 특징 PCA (블록별) | SLat 8차원 PCA | 좌표 K-means (기준선)
열: 4 방향. 출력 out/<name>/sem_grid.png
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gs_utils import SH_C0, T_SAVE, load_font, load_gs  # noqa: E402,F401
from parts_core import N_PATCH, kmeans, vox_to_tok  # noqa: E402,F401


def patch_attn(sem, blocks):
    """토큰 × 1369 패치 attention (cls·register 5개 빼고 다시 정규화), 블록 평균."""
    a = sum(sem[f"attn_{b}"].astype(np.float32)[:, 5:] for b in blocks) / len(blocks)
    return a / np.maximum(a.sum(1, keepdims=True), 1e-9)


def pca_rgb(F, k=3):
    X = F - F.mean(0)
    U, s, Vt = np.linalg.svd(X[np.random.default_rng(0).choice(len(X), min(len(X), 20000), replace=False)],
                             full_matrices=False)
    Y = X @ Vt[:k].T
    lo, hi = np.percentile(Y, 2, 0), np.percentile(Y, 98, 0)
    return np.clip((Y - lo) / np.maximum(hi - lo, 1e-9), 0, 1)


PALETTE = np.array([[0.90, 0.30, 0.25], [0.25, 0.55, 0.90], [0.30, 0.75, 0.35], [0.95, 0.75, 0.20],
                    [0.65, 0.40, 0.85], [0.20, 0.80, 0.80], [0.95, 0.50, 0.70], [0.55, 0.55, 0.55]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--k", type=int, default=6)
    args = ap.parse_args()
    from edit_demo import Cam, render                            # CUDA 래스터라이저 (그릴 때만)
    out = os.path.join(HERE, "out", args.name)
    sem = dict(np.load(os.path.join(out, "sem.npz")))
    P, q, S, op, col = load_gs(out)
    v2t = vox_to_tok(sem)
    g2v = np.arange(len(P)) // 32
    img = np.asarray(Image.open(os.path.join(out, "cond.png")).convert("RGB").resize((N_PATCH, N_PATCH), Image.BOX),
                     np.float32).reshape(-1, 3) / 255

    rows = [("original", col)]
    for blocks in ([4, 8], [12, 16], [20, 23]):
        A = patch_attn(sem, blocks)
        rows.append((f"attn→image b{blocks}", (A @ img)[v2t][g2v]))
    for b in (6, 12, 18, 23):
        rows.append((f"DiT feat b{b} PCA", pca_rgb(sem[f"feat_{b}"].astype(np.float32))[v2t][g2v]))
    rows.append(("SLat 8d PCA", pca_rgb(sem["slat_feats"])[g2v]))
    vc = sem["slat_coords"].astype(np.float32)
    rows.append((f"xyz k-means k{args.k}", PALETTE[kmeans(vc, args.k)][g2v]))

    c = P.mean(0)
    size = np.linalg.norm(P.max(0) - P.min(0))
    cams = [Cam(c + 1.3 * size * np.array([np.cos(a), np.sin(a), 0.3]), c, 256, 256, 35)
            for a in np.radians([-90, 0, 90, 180])]
    font = load_font(16)
    tiles = []
    for name, cc in rows:
        r = np.concatenate([render(cm, P, q, S, op, cc) for cm in cams], 1)
        im = Image.fromarray(r)
        ImageDraw.Draw(im).text((6, 4), name, fill=(20, 20, 20), font=font)
        tiles.append(np.asarray(im))
    Image.fromarray(np.concatenate(tiles, 0)).save(os.path.join(out, "sem_grid.png"))
    print("[viz] rows", [r[0] for r in rows])


if __name__ == "__main__":
    main()
