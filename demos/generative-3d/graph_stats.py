"""isaac_demo/gen_<name>/gen_<name>_graph.npz → genai/out/<name>/graph.json (노드·간선·연결 성분·차수).

  python graph_stats.py bear plant robot
"""
import json
import os
import sys

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("APG_ROOT", os.path.dirname(HERE))       # 3DGS 작업 폴더 (output_1/, isaac_demo/, SIBR_viewers/)
DEMO = os.environ.get("APG_ISAAC_DEMO", os.path.join(ROOT, "isaac_demo"))   # APG-GS 준비 스크립트 · XPBD DLL

for name in sys.argv[1:]:
    g = np.load(os.path.join(DEMO, f"gen_{name}", f"gen_{name}_graph.npz"))
    cnt, idx = g["count"], g["idx"]
    n = len(cnt)
    owner = np.repeat(np.arange(n), cnt)
    A = coo_matrix((np.ones(len(idx), np.int8), (owner, idx)), shape=(n, n))
    k, lab = connected_components(A, directed=False)
    largest = int(np.bincount(lab).max())
    st = {"nodes": int(n), "undirected_edges": int(len(idx) // 2), "components": int(k),
          "largest_pct": 100.0 * largest / n, "degree_median": float(np.median(cnt)),
          "degree_max": int(cnt.max())}
    json.dump(st, open(os.path.join(HERE, "out", name, "graph.json"), "w"), indent=2)
    print(name, st)
