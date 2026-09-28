"""생성된 3DGS(output_1/gen_<name>) 를 기존 APG 파이프라인에 넣는다 — isaac_demo/prepare_dataset.py 의 1~4 단계.

  python prepare_gen.py bear [--skip-sibr]      (anaconda 또는 trellis env — numpy, scipy)

1 크롭 상자 + 그래프 가중치(face 와 같은 case C)  2 크롭 PLY·Isaac 변환  3 SIBR 뷰어 무인 실행으로 APG 그래프 내보내기
4 뷰어 순서 → USD 순서 그래프.  5 단계(USD, Isaac 파이썬)는 --usd 를 줄 때만 (Isaac GUI 가 켜져 있으면 돌리지 말 것).
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("APG_ROOT", os.path.dirname(HERE))       # 3DGS 작업 폴더 (output_1/, isaac_demo/, SIBR_viewers/)
DEMO = os.environ.get("APG_ISAAC_DEMO", os.path.join(ROOT, "isaac_demo"))   # APG-GS 준비 스크립트 · XPBD DLL
sys.path.insert(0, DEMO)

import prepare_dataset as pd  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--skip-sibr", action="store_true")
    ap.add_argument("--usd", action="store_true")
    ap.add_argument("--target-height", type=float, default=0.3)
    args = ap.parse_args()
    name = f"gen_{args.name}"
    model = os.path.join(ROOT, "output_1", name)
    ply = os.path.join(model, "point_cloud", "iteration_1", "point_cloud.ply")
    out = os.path.join(DEMO, name)
    os.makedirs(out, exist_ok=True)

    lo, hi, n_act, n_in = pd.crop_box(ply, pct=0.0, pad=0.05)   # 생성물엔 floater 가 없다 → 전부 남긴다
    cfg = {"experiment_name": f"isaac_{name}", "dataset_name": name, "crop_min": lo.tolist(), "crop_max": hi.tolist(),
           **pd.GRAPH_WEIGHTS}
    cfg_path = os.path.join(out, f"{name}_apg.json")
    json.dump(cfg, open(cfg_path, "w", encoding="utf-8"), indent=2)
    print(f"[gen] crop keeps {n_in:,} of {n_act:,} active Gaussians", flush=True)

    code, log = pd.run([sys.executable, os.path.join(DEMO, "prepare_splat.py"), "--ply", ply, "--crop-json", cfg_path,
                        "--cameras", os.path.join(model, "cameras.json"), "--out-dir", out, "--name", name,
                        "--target-height", str(args.target_height), "--yaw-deg", "0"])
    print("\n".join(ln for ln in log.splitlines() if ln.startswith("[prepare]")), flush=True)
    if code != 0:
        raise RuntimeError(log[-2000:])

    graph_bin = os.path.join(out, f"{name}_graph_sibr.bin")
    if not args.skip_sibr:
        auto = os.path.join(out, f"{name}_autorun.txt")
        log_path = os.path.join(out, f"{name}_autorun.log").replace("\\", "/")
        with open(auto, "w", encoding="utf-8") as f:
            f.write(f"set log={log_path} exit=1\n")
            f.write(f"export graph path={graph_bin.replace(chr(92), '/')}\n")
        if os.path.exists(graph_bin):
            os.remove(graph_bin)
        t0 = time.time()
        code, log = pd.run([pd.SIBR, "-m", model, "--iteration", "1", "--config", cfg_path],
                           env=dict(os.environ, APG_AUTORUN=auto), timeout=900)
        ok = os.path.exists(graph_bin)
        print(f"[gen] SIBR exited {code} after {time.time() - t0:.0f} s, graph {'written' if ok else 'MISSING'}",
              flush=True)
        if not ok:
            raise RuntimeError(log[-3000:])

    code, log = pd.run([sys.executable, os.path.join(DEMO, "import_graph.py"), "--dir", out, "--name", name])
    print("\n".join(log.splitlines()[-8:]), flush=True)
    if code != 0:
        raise RuntimeError(log[-2000:])

    if args.usd:
        code, log = pd.run(["cmd", "/c", pd.ISAAC_PY, os.path.join(DEMO, "build_usd.py"), "--dir", out, "--name", name],
                           timeout=1200)
        print("\n".join([ln for ln in log.splitlines() if "[build]" in ln][-6:]), flush=True)
        if code != 0:
            raise RuntimeError(log[-3000:])


if __name__ == "__main__":
    main()
