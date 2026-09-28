"""생성 과정의 신호로 2D 부위 이름을 3D 가우시안에 옮긴다 (계산은 parts_core.py, 원리는 그 파일 머리말).

  python lift_parts.py --name bear_sem --asset gen_bear_sem_d100k            # 기본: --mode proj
  python lift_parts.py --name bear_sem --asset gen_bear_sem_d100k --mode attn   # 예전 방식 (비교 · 재현용)

입력 out/<name>/: sem.npz (gen3d_sem.py) · parts2d.npz (seg2d.py) · cond.png · gaussian.ply
  mode proj  ① attention 투표 → 입력 카메라 추정(attention affine + 실루엣) → 보이는 복셀만 2D 라벨 투영 →
             DiT 특징 26-이웃 그래프 전파(보이는 토큰 고정) → 작은 조각 정리 → 보이는 복셀은 복셀 해상도로
  mode attn  ① attention 투표 → ② 반경 4 토큰 특징 k-NN 그래프 전파 (예전 그대로)
  원본 가우시안 i → 복셀 i // 32 (TRELLIS 디코더는 복셀마다 32개를 복셀 순서로 낸다). 편집용 에셋(개수를 줄인 판)은
  위치로 가장 가까운 복셀.
출력 out/<name>/:
  parts3d.npz        names · tok_prob · tok_prob_attn · vox_part · vox_conf · gs_part_full (원본 가우시안마다)
                     (--asset) asset_part · asset_conf · asset_prob  ← edit_demo / semantic_* / export_parts.py 가 읽는다
  parts3d_stats.json 카메라 (실루엣 IoU · 색 상관) · 가시성 · 부위 비율
  camera_fit.png     왼쪽 2D 부위 · 가운데 추정 카메라로 그린 '보이는 복셀' 의 3D 부위 · 오른쪽 실루엣 비교 (파랑 = 사진만,
                     빨강 = 복셀만) — 카메라가 맞는지 눈으로 확인
  parts3d_grid.png   (CUDA 래스터라이저가 있을 때) 행: 원래 색 · ① attention · 예전 ①+② · 투영 투표 · 최종
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import parts_core as pc  # noqa: E402
from gs_utils import T_SAVE, load_font, load_gs, runtime_root  # noqa: E402

PART_COLORS = pc.PART_COLORS          # parts_video.py 등이 여기서 가져간다


def overlay_2d(cond, label):
    ov = cond.copy()
    m = label > 0
    ov[m] = 0.45 * ov[m] + 0.55 * pc.part_colors(label[m] - 1)
    return ov


def camera_fit_image(cond, label, alpha, names, res, sem):
    """[2D 부위 | 추정 카메라로 그린 보이는 복셀의 3D 부위 | 실루엣 비교] (0..1 RGB)."""
    cam = res["camera"]
    Xv = pc.voxel_centers(sem)
    u, z, den = cam.project(Xv)
    rad = 0.5 * 1.25 * cam.s / pc.VOX_RES / den
    img3, cover = pc.splat_colors(u, z, rad, pc.part_colors(res["vox_part"]), label.shape)
    mid = 0.35 * cond + 0.65
    mid[cover] = 0.25 * cond[cover] + 0.75 * img3[cover]
    sil = np.ones(cond.shape)
    sil[alpha & cover] = [0.55, 0.55, 0.55]
    sil[alpha & ~cover] = [0.20, 0.45, 0.95]
    sil[~alpha & cover] = [0.95, 0.30, 0.25]
    return np.concatenate([overlay_2d(cond, label), mid, sil], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--asset", default="", help="<APG runtime>/<asset> — 편집용 에셋(크롭 PLY)에 부위를 붙인다")
    ap.add_argument("--mode", default="proj", choices=["proj", "attn"],
                    help="proj = 카메라 추정 + 가시성 투영 + 고정 전파 (기본) · attn = 예전 attention + 특징 전파")
    ap.add_argument("--attn-blocks", default="4,8,12")
    ap.add_argument("--feat-blocks", default="6,12")
    ap.add_argument("--knn", type=int, default=12, help="attn 모드 그래프: 반경 안 특징 k-NN")
    ap.add_argument("--alpha", type=float, default=0.9, help="전파 세기 (Y ← α·W·Y + (1−α)·P0)")
    ap.add_argument("--iters", type=int, default=0, help="0 = 모드 기본값 (attn 40, proj 120)")
    ap.add_argument("--radius", type=float, default=0.0, help="0 = 모드 기본값 (attn 4 토큰, proj 1.8 = 맞닿은 26-이웃)")
    ap.add_argument("--erode", type=int, default=2, help="2D 부위 마스크 침식 [px] — 경계 픽셀은 투영 투표에서 뺀다")
    ap.add_argument("--gate-iou", type=float, default=0.7, help="추정 카메라 실루엣 IoU 가 이보다 낮으면 attn 모드로")
    ap.add_argument("--gate-color", type=float, default=0.2, help="보이는 복셀 색 · 사진 색 상관이 이보다 낮으면 attn 모드로")
    ap.add_argument("--no-cleanup", action="store_true", help="작은 조각 정리를 끈다")
    ap.add_argument("--vox-refine", type=float, default=0.5, help="보이는 복셀에 자기 투영 라벨을 섞는 비율 (0 = 끔)")
    ap.add_argument("--no-render", action="store_true", help="parts3d_grid.png 를 건너뛴다 (CUDA 래스터라이저 불필요)")
    ap.add_argument("--baselines", action="store_true", help="그림에 좌표 · 특징 k-means 기준선 행도 넣는다")
    args = ap.parse_args()
    out = os.path.join(HERE, "out", args.name)
    sem = dict(np.load(os.path.join(out, "sem.npz")))
    p2 = np.load(os.path.join(out, "parts2d.npz"))
    names = [str(n) for n in p2["names"]][1:]
    K = len(names)
    label = p2["label"].astype(np.int64)
    alpha = p2["alpha"].astype(bool) if "alpha" in p2.files else label > 0
    cond = np.asarray(Image.open(os.path.join(out, "cond.png")).convert("RGB"), np.float64) / 255
    P, q, S, op, col = load_gs(out)
    V = len(sem["slat_coords"])
    if len(P) != 32 * V:
        raise ValueError(f"gaussian.ply {len(P):,} 개 ≠ 복셀 {V:,} × 32 — gen3d_sem.py 가 쓴 원본 PLY 여야 한다")
    g2v = np.arange(len(P)) // 32
    blocks = dict(attn_blocks=[int(b) for b in args.attn_blocks.split(",")],
                  feat_blocks=[int(b) for b in args.feat_blocks.split(",")], knn=args.knn, alpha=args.alpha)

    res, st = pc.lift(sem, label, names, mode=args.mode, alpha_mask=alpha, cond_rgb=cond,
                      vox_col=pc.voxel_colors(col, op, V), iters=args.iters or None, radius=args.radius or None,
                      erode=args.erode, gate_iou=args.gate_iou, gate_color=args.gate_color,
                      cleanup=not args.no_cleanup, vox_refine=args.vox_refine, **blocks)
    legacy = res if args.mode == "attn" else pc.lift(sem, label, names, mode="attn", log=lambda *a: None, **blocks)[0]
    st["parts"] = names
    st["token_share_prop"] = st["token_share"]                    # 예전 키 이름
    if args.mode == "proj":
        st["changed_vs_legacy_voxels"] = float(np.mean(res["vox_part"] != legacy["vox_part"]))

    if "camera" in res:
        im = camera_fit_image(cond, label, alpha, names, res, sem)
        im = Image.fromarray((np.clip(im, 0, 1) * 255).astype(np.uint8))
        dr = ImageDraw.Draw(im)
        font = load_font(16)
        W = label.shape[1]
        cs = st["camera"]
        for i, t in enumerate(("2D 부위 (seg2d)", "추정 카메라 · 보이는 복셀의 3D 부위",
                               f"실루엣 IoU {cs['silhouette_iou']:.3f} · 색 상관 {cs.get('color_corr_visible', 0):.2f}")):
            dr.text((i * W + 6, 4), t, fill=(20, 20, 20), font=font)
        for k, n in enumerate(names):
            dr.rectangle((6 + k * 110, label.shape[0] - 24, 22 + k * 110, label.shape[0] - 8),
                         fill=tuple(int(v * 255) for v in pc.part_colors([k])[0]))
            dr.text((26 + k * 110, label.shape[0] - 27), n, fill=(20, 20, 20), font=font)
        im.save(os.path.join(out, "camera_fit.png"))

    save = {"names": np.array(names), "tok_prob": res["tok_prob"], "tok_prob_attn": res["tok_prob_attn"],
            "vox_part": res["vox_part"].astype(np.int8), "vox_conf": res["vox_conf"],
            "gs_part_full": res["vox_part"][g2v].astype(np.int8), "mode": np.array(args.mode)}
    if "vox_visible" in res:
        save["vox_visible"] = res["vox_visible"]
        save["vox_proj_label"] = res["vox_proj_label"]
    if args.asset:
        from plyfile import PlyData
        v = PlyData.read(os.path.join(runtime_root(), args.asset, f"{args.asset}_crop.ply"))["vertex"].data
        Pa = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64) @ T_SAVE      # PLY → 내부
        _, vi = cKDTree(pc.voxel_centers(sem)).query(Pa)
        save["asset_part"] = res["vox_part"][vi].astype(np.int8)
        save["asset_conf"] = res["vox_conf"][vi].astype(np.float32)
        save["asset_prob"] = res["vox_prob"][vi].astype(np.float32)
        st["asset"] = args.asset
        st["asset_share"] = {n: float(np.mean(save["asset_part"] == k)) for k, n in enumerate(names)}
    np.savez_compressed(os.path.join(out, "parts3d.npz"), **save)
    json.dump(st, open(os.path.join(out, "parts3d_stats.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False,
              default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print("[lift]", json.dumps({k: st[k] for k in ("mode", "token_share", "conf_median") if k in st}, ensure_ascii=False),
          flush=True)

    if args.no_render:
        return
    try:
        from edit_demo import Cam, render
    except Exception as e:                                       # CUDA 래스터라이저 · APG 런타임이 없는 환경
        print(f"[lift] parts3d_grid.png skipped ({type(e).__name__}: {e})")
        return
    v2t = res["v2t"]
    rows = [("original", col),
            ("① attention 투표만", pc.part_colors(res["tok_prob_attn"].argmax(1))[v2t][g2v]),
            ("예전: ① + ② 반경 4 토큰 특징 전파", pc.part_colors(legacy["vox_part"])[g2v])]
    if "vox_proj_label" in res:
        rows += [("투영 투표 (입력 사진에 보이는 복셀만, 회색 = 모름)", pc.part_colors(res["vox_proj_label"])[g2v]),
                 ("최종: 투영 + 고정 전파 + 조각 정리", pc.part_colors(res["vox_part"])[g2v])]
    if args.baselines:
        tc = sem["tok_coords"].astype(np.float32)
        rows += [("기준선: DiT 특징 k-means (이름 없음)",
                  pc.part_colors(pc.kmeans(pc.dit_features(sem, blocks["feat_blocks"]), K))[v2t][g2v]),
                 ("기준선: 좌표 k-means", pc.part_colors(pc.kmeans(tc, K))[v2t][g2v])]
    c = P.mean(0)
    size = np.linalg.norm(P.max(0) - P.min(0))
    cams = [Cam(c + 1.3 * size * np.array([np.cos(a), np.sin(a), 0.3]), c, 256, 256, 35)
            for a in np.radians([-90, 0, 90, 180])]
    font = load_font(16)
    tiles = []
    for nm, cc in rows:
        im = Image.fromarray(np.concatenate([render(cm, P, q, S, op, cc) for cm in cams], 1))
        ImageDraw.Draw(im).text((6, 4), nm, fill=(20, 20, 20), font=font)
        tiles.append(np.asarray(im))
    leg = Image.new("RGB", (1024, 30), (255, 255, 255))
    dr = ImageDraw.Draw(leg)
    for k, n in enumerate(names):
        dr.rectangle((10 + k * 140, 8, 28 + k * 140, 24), fill=tuple(int(v * 255) for v in pc.part_colors([k])[0]))
        dr.text((34 + k * 140, 6), n, fill=(20, 20, 20), font=font)
    tiles.append(np.asarray(leg))
    Image.fromarray(np.concatenate(tiles, 0)).save(os.path.join(out, "parts3d_grid.png"))


if __name__ == "__main__":
    main()
