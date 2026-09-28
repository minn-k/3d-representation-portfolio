"""이미지 한 장 → TRELLIS → 3D Gaussian (.ply) + 턴테이블 영상.

  C:\\anaconda\\anaconda3\\envs\\trellis\\python.exe gen3d.py --image in.png --name bear [--seed 1]

출력: genai/out/<name>/  gaussian.ply (3DGS 표준 필드), turntable.mp4, cond.png (배경 제거된 입력), stats.json
TRELLIS 원본은 xformers/flash-attn 을 요구하지만 이 PC(torch 2.5.1+cu118, Windows)에는 휠이 없어
genai/shims/xformers (torch SDPA) 로 대신한다.
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "shims"))
sys.path.insert(0, os.path.join(HERE, "TRELLIS"))
os.environ.setdefault("ATTN_BACKEND", "xformers")
os.environ.setdefault("SPCONV_ALGO", "native")

import imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from trellis.pipelines import TrellisImageTo3DPipeline  # noqa: E402
from trellis.utils import render_utils  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--ss-steps", type=int, default=25)
    ap.add_argument("--slat-steps", type=int, default=25)
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()

    out = os.path.join(HERE, "out", args.name)
    os.makedirs(out, exist_ok=True)

    t0 = time.time()
    pipe = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
    pipe.cuda()
    t_load = time.time() - t0

    image = Image.open(args.image)
    cond = pipe.preprocess_image(image)
    cond.save(os.path.join(out, "cond.png"))

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t1 = time.time()
    outputs = pipe.run(
        cond, seed=args.seed, preprocess_image=False, formats=["gaussian"],
        sparse_structure_sampler_params={"steps": args.ss_steps, "cfg_strength": 7.5},
        slat_sampler_params={"steps": args.slat_steps, "cfg_strength": 3.0},
    )
    torch.cuda.synchronize()
    t_gen = time.time() - t1
    peak = torch.cuda.max_memory_allocated() / 2**30

    g = outputs["gaussian"][0]
    g.save_ply(os.path.join(out, "gaussian.ply"))
    n = int(g.get_xyz.shape[0])

    if not args.no_video:
        frames = render_utils.render_video(g, resolution=512, num_frames=120, bg_color=(1, 1, 1))["color"]
        imageio.mimsave(os.path.join(out, "turntable.mp4"), frames, fps=30, quality=8)
        imageio.imwrite(os.path.join(out, "preview.png"), frames[0])

    stats = {"name": args.name, "image": os.path.abspath(args.image), "seed": args.seed,
             "gaussians": n, "load_s": round(t_load, 1), "generate_s": round(t_gen, 1),
             "peak_vram_gb": round(peak, 2), "ss_steps": args.ss_steps, "slat_steps": args.slat_steps,
             "gpu": torch.cuda.get_device_name(0)}
    json.dump(stats, open(os.path.join(out, "stats.json"), "w"), indent=2)
    print("[gen3d]", json.dumps(stats))


if __name__ == "__main__":
    main()
