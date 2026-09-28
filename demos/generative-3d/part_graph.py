"""부위 인식 그래프: SIBR 에서 만든 APG 그래프(Bhattacharyya 겹침 점수 → 간선 강성)에 가우시안 part_id 를 반영한다.

  python part_graph.py --asset gen_bear_sem_d100k --parts out/bear_sem/parts3d.npz --cross-keep 0.1 [--joint overlap]
                       [--materials materials.json]

'경계 간선' = 서로 다른 부위를 잇는 간선 (양 끝 가우시안의 부위 신뢰도가 모두 --conf 초과).
  --cross-scale  경계 간선 강성 배율 (1 = 그대로). 다만 형상 · 부피 클러스터도 그래프 연결로 만들어지므로 강성만 낮추면 거의
                 효과가 없고, 경계 간선을 모두 끊으면 부위가 떨어져 나간다 → 관절만 남긴다:
  --cross-keep   경계 간선 중 남길 비율
  --joint random   무작위로 남긴다 (edit_demo 의 예전 방식 · 포트폴리오 수치는 이것으로 쟀다, 기본값)
          overlap  부위 쌍마다 Bhattacharyya 거리가 가장 작은 (= 두 타원체가 가장 크게 겹친) 경계 간선을 남긴다.
                   관절이 '맞닿기만 한 곳' 이 아니라 표면이 실제로 이어진 곳에 생긴다
  --materials    부위별 간선 강성 배율 JSON, 예: {"arm": {"stiffness": 0.2}, "body": {"stiffness": 1.0}}
                 부위 안 간선 = 그 부위 값, 경계 간선 = 두 부위 값 중 작은 것
출력 <runtime>/<asset>/<asset>_graph_parts.npz — <asset>_graph.npz 와 같은 CSR 형식(offset · count · idx · dist · stiff ·
     pos_rest) + part_id · part_conf. edit_demo.py / semantic_*.py 는 같은 함수(apply_part_graph)를 메모리에서 쓴다.

D_B(i, j) = ⅛ dᵀ Σ̄⁻¹ d + ½ ln( det Σ̄ / √(det Σᵢ det Σⱼ) ),  Σ̄ = ½(Σᵢ + Σⱼ),  Σ = R diag(s²) Rᵀ
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gs_utils import quat_to_mat, runtime_root  # noqa: E402


def covariances(scales, rots, eps_rel=1e-3):
    """Σ = R diag(s²) Rᵀ (+ 아주 작은 등방 항 — 납작한 가우시안의 det → 0 로 log 가 튀지 않게)."""
    R = quat_to_mat(np.asarray(rots, np.float64))
    s = np.asarray(scales, np.float64)
    C = np.einsum("nij,nj,nkj->nik", R, s * s, R)
    eps = (eps_rel * np.median(s)) ** 2
    return C + eps * np.eye(3)[None]


def bhattacharyya(pos, cov, e):
    """간선 (M,2) 마다 D_B. pos (N,3) · cov (N,3,3), 단위 무관 (배율에 불변)."""
    d = pos[e[:, 1]] - pos[e[:, 0]]
    Ci, Cj = cov[e[:, 0]], cov[e[:, 1]]
    Cm = 0.5 * (Ci + Cj)
    t1 = 0.125 * np.einsum("mi,mi->m", d, np.linalg.solve(Cm, d[..., None])[..., 0])
    t2 = 0.5 * (np.linalg.slogdet(Cm)[1] - 0.5 * (np.linalg.slogdet(Ci)[1] + np.linalg.slogdet(Cj)[1]))
    return t1 + t2


def cross_edges(edges, part, conf, conf_thr):
    e0, e1 = edges[:, 0], edges[:, 1]
    return (part[e0] != part[e1]) & (conf[e0] > conf_thr) & (conf[e1] > conf_thr)


def part_stiffness(edges, part, names, materials):
    """부위별 강성 배율 표 → 간선 배율 (부위 안 = 그 부위 값, 경계 = 작은 쪽)."""
    k = np.ones(len(names))
    for n, m in (materials or {}).items():
        if n in names:
            k[names.index(n)] = float(m.get("stiffness", 1.0))
    return np.minimum(k[part[edges[:, 0]]], k[part[edges[:, 1]]])


def apply_part_graph(inp, part, conf, conf_thr=0.5, cross_scale=1.0, cross_keep=1.0, joint="random", seed=0,
                     min_joint=8, names=None, materials=None):
    """XPBD 입력 dict (load_inputs 결과: pos · scales · rots · edges · rest · stiff) 의 복사본을 부위에 맞게 바꾼다.
    → (새 dict, 통계). joint="random" 은 예전 edit_demo 와 같은 결과 (같은 난수 호출)."""
    part = np.asarray(part, np.int64)
    conf = np.asarray(conf, np.float64)
    out = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in inp.items()}
    e = inp["edges"]
    cross = cross_edges(e, part, conf, conf_thr)
    stiff = inp["stiff"] * np.where(cross, cross_scale, 1.0)
    if materials:
        stiff = stiff * part_stiffness(e, part, list(names), materials)
    keep = np.ones(len(e), bool)
    st = {"cross_part_edge_ratio": float(cross.mean()), "cross_scale": cross_scale, "conf": conf_thr,
          "cross_keep": cross_keep, "joint": joint}
    if cross_keep < 1.0:
        if joint == "random":
            keep = ~cross | (np.random.default_rng(seed).random(len(cross)) < cross_keep)
        elif joint == "overlap":
            keep = ~cross
            ci = np.nonzero(cross)[0]
            db = bhattacharyya(inp["pos"].astype(np.float64), covariances(inp["scales"], inp["rots"]), e[ci])
            pa, pb = part[e[ci, 0]], part[e[ci, 1]]
            pair = np.minimum(pa, pb) * 1000 + np.maximum(pa, pb)
            joints = {}
            for p in np.unique(pair):
                m = np.nonzero(pair == p)[0]
                n_keep = min(len(m), max(min_joint, int(np.ceil(cross_keep * len(m)))))
                sel = m[np.argsort(db[m])[:n_keep]]
                keep[ci[sel]] = True
                joints[f"{int(p // 1000)}-{int(p % 1000)}"] = {"cross_edges": int(len(m)), "kept": int(n_keep),
                                                               "db_kept_max": float(db[sel].max())}
            st["joints"] = joints
        else:
            raise ValueError(f"joint = random | overlap (got {joint!r})")
    out["edges"] = np.ascontiguousarray(e[keep])
    out["rest"] = np.ascontiguousarray(inp["rest"][keep])
    out["stiff"] = np.ascontiguousarray(stiff[keep].astype(np.float32))
    st["edges_after"] = int(keep.sum())
    return out, st


def write_graph_npz(path, g, part, conf, keep_pairs, stiff_pairs):
    """<asset>_graph.npz (양방향 CSR) 를 남길 무향 간선 목록으로 걸러 같은 형식으로 쓴다."""
    N = len(g["count"])
    owner = np.repeat(np.arange(N, dtype=np.int64), g["count"])
    idx = g["idx"].astype(np.int64)
    valid = idx >= 0
    a, b = np.minimum(owner, idx), np.maximum(owner, idx)
    key = a * N + b
    kk = keep_pairs[:, 0].astype(np.int64) * N + keep_pairs[:, 1].astype(np.int64)
    order = np.argsort(kk)
    kk, ks = kk[order], stiff_pairs[order]
    if len(kk) == 0:
        raise ValueError("남길 간선이 없다")
    pos = np.clip(np.searchsorted(kk, key), 0, len(kk) - 1)
    hit = valid & (kk[pos] == key)
    count = np.bincount(owner[hit], minlength=N).astype(np.int32)
    offset = np.zeros(N, np.int32)
    offset[1:] = np.cumsum(count)[:-1]
    extra = {k: g[k] for k in g.files if k not in ("offset", "count", "idx", "dist", "stiff")}
    np.savez_compressed(path, offset=offset, count=count, idx=idx[hit].astype(np.int32),
                        dist=g["dist"][hit].astype(np.float32), stiff=ks[pos[hit]].astype(np.float32),
                        part_id=np.asarray(part, np.int8), part_conf=np.asarray(conf, np.float32), **extra)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", required=True, help="<APG runtime>/<asset> (예: gen_bear_sem_d100k)")
    ap.add_argument("--parts", required=True, help="lift_parts.py --asset 로 만든 parts3d.npz")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--cross-scale", type=float, default=1.0)
    ap.add_argument("--cross-keep", type=float, default=0.1)
    ap.add_argument("--joint", default="random", choices=["random", "overlap"])
    ap.add_argument("--materials", default="", help="부위별 강성 배율 JSON 파일")
    args = ap.parse_args()
    d = os.path.join(runtime_root(), args.asset)
    sys.path.insert(0, runtime_root())
    from xpbd import load_inputs                                        # APG 런타임 (PLY · 그래프 읽기)
    inp = load_inputs(d, args.asset)
    pz = np.load(args.parts)
    names = [str(n) for n in pz["names"]]
    part, conf = pz["asset_part"].astype(np.int64), pz["asset_conf"]
    if len(part) != len(inp["pos"]):
        raise SystemExit(f"parts3d 의 에셋({len(part):,})이 {args.asset}({len(inp['pos']):,})와 다르다 — "
                         "lift_parts.py --asset 을 이 에셋으로 다시 돌릴 것")
    mats = json.load(open(args.materials, encoding="utf-8")) if args.materials else None
    new, st = apply_part_graph(inp, part, conf, args.conf, args.cross_scale, args.cross_keep, args.joint,
                               names=names, materials=mats)
    g = np.load(os.path.join(d, f"{args.asset}_graph.npz"))
    path = os.path.join(d, f"{args.asset}_graph_parts.npz")
    write_graph_npz(path, g, part, conf, new["edges"], new["stiff"])
    st.update({"asset": args.asset, "parts": names, "edges_before": int(len(inp["edges"])), "materials": mats,
               "part_counts": {n: int((part == k).sum()) for k, n in enumerate(names)}})
    json.dump(st, open(path.replace(".npz", ".json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("[part_graph]", json.dumps({k: st[k] for k in ("cross_part_edge_ratio", "edges_before", "edges_after")}),
          "->", path, flush=True)


if __name__ == "__main__":
    main()
