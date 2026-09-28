"""텍스트 → 이미지 (SDXL-Turbo, 1~4 스텝). 3D 생성(gen3d.py) 입력용: 물체 하나, 흰 배경, 전신.

  C:\\anaconda\\anaconda3\\envs\\trellis\\python.exe gen2d.py --name bear --prompt "a plush teddy bear" --seeds 0-3

출력: genai/out/<name>/text2img_s<seed>.png, text2img.json (프롬프트·시간)
"""
import argparse
import json
import os
import time

import torch
from diffusers import AutoPipelineForText2Image

HERE = os.path.dirname(os.path.abspath(__file__))
STYLE = ", single object, full body, centered, 3/4 view, plain white background, soft studio lighting, high detail"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--seeds", default="0-3")
    ap.add_argument("--steps", type=int, default=2)
    args = ap.parse_args()
    a, b = (int(v) for v in args.seeds.split("-")) if "-" in args.seeds else (int(args.seeds),) * 2
    out = os.path.join(HERE, "out", args.name)
    os.makedirs(out, exist_ok=True)

    pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16,
                                                     variant="fp16")
    pipe.enable_model_cpu_offload()   # 다른 GPU 작업이 VRAM 을 쓰고 있을 때도 돌게 (모듈을 쓸 때만 GPU 로)
    prompt = args.prompt + STYLE
    log = {"model": "stabilityai/sdxl-turbo", "prompt": prompt, "steps": args.steps, "images": []}
    for seed in range(a, b + 1):
        g = torch.Generator("cuda").manual_seed(seed)
        torch.cuda.synchronize()
        t0 = time.time()
        img = pipe(prompt=prompt, num_inference_steps=args.steps, guidance_scale=0.0, generator=g,
                   width=512, height=512).images[0]
        torch.cuda.synchronize()
        dt = time.time() - t0
        p = os.path.join(out, f"text2img_s{seed}.png")
        img.save(p)
        log["images"].append({"seed": seed, "path": p, "seconds": round(dt, 2)})
        print(f"[gen2d] {p} {dt:.2f}s", flush=True)
    json.dump(log, open(os.path.join(out, "text2img.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
