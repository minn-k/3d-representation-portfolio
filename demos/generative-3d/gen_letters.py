"""TRELLIS 티저 글자(T R E L I S) 를 같은 스타일로: 글꼴 밑그림(돌출 · 격자 무늬) → SDXL-Turbo image-to-image.

  C:\\anaconda\\anaconda3\\envs\\trellis\\python.exe gen_letters.py --letters REL IS --seeds 0-3 [--strength 0.62]

T 는 TRELLIS 저장소 예제(assets/example_image/T.png)를 그대로 쓴다. 출력: out/letter_<X>/init.png, i2i_s<seed>.png
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from diffusers import AutoPipelineForImage2Image
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPT = ("isometric 3D render of the capital letter {L} built from a wooden garden trellis lattice, "
          "brown wood slats, green climbing ivy vines and leaves hanging, small grass tuft at the base, "
          "single object, centered, plain white background, soft studio lighting, high detail")


def letter_init(L, size=512):
    font = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 380)
    mask = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(mask)
    bb = d.textbbox((0, 0), L, font=font)
    x = (size - (bb[2] - bb[0])) // 2 - bb[0] - 20
    y = (size - (bb[3] - bb[1])) // 2 - bb[1] - 10
    d.text((x, y), L, fill=255, font=font)
    # 비스듬한 시점: 약간 기울이고(shear) 뒤로 돌출
    mask = mask.transform(mask.size, Image.AFFINE, (1, 0.18, -40, -0.12, 1, 30), resample=Image.BICUBIC)
    m = np.asarray(mask).astype(np.float32) / 255
    img = np.ones((size, size, 3), np.float32)
    depth = np.zeros_like(m)
    for k in range(18, 0, -1):                              # 돌출: 오른쪽 위로 밀린 복사본을 어둡게
        sh = np.roll(np.roll(m, -k, 0), k, 1)
        depth = np.maximum(depth, sh)
    side = np.clip(depth - m, 0, 1)[..., None]
    img = img * (1 - side) + side * np.array([0.33, 0.2, 0.11])
    # 앞면: 나무색 + 격자 (가로·세로 살)
    yy, xx = np.mgrid[0:size, 0:size]
    grid = ((xx % 34) < 7) | ((yy % 34) < 7)
    wood = np.where(grid[..., None], np.array([0.55, 0.34, 0.18]), np.array([0.93, 0.93, 0.9]))
    edge = np.asarray(Image.fromarray((m * 255).astype(np.uint8)).filter(ImageFilter.FIND_EDGES)).astype(np.float32) / 255
    wood = np.where((edge > 0.2)[..., None], np.array([0.45, 0.27, 0.14]), wood)
    front = m[..., None]
    img = img * (1 - front) + front * wood
    # 덩굴 느낌: 가장자리 근처 초록 점
    rng = np.random.default_rng(ord(L))
    ys, xs = np.where(edge > 0.2)
    pick = rng.choice(len(xs), size=min(160, len(xs)), replace=False)
    im = Image.fromarray((img * 255).astype(np.uint8))
    dr = ImageDraw.Draw(im)
    for i in pick:
        r = rng.integers(4, 11)
        g = rng.integers(120, 190)
        oy = rng.integers(0, 14)
        dr.ellipse((xs[i] - r, ys[i] - r + oy, xs[i] + r, ys[i] + r + oy), fill=(40, g, 40))
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--letters", nargs="+", default=["R", "E", "L", "I", "S"])
    ap.add_argument("--seeds", default="0-3")
    ap.add_argument("--strength", type=float, default=0.62)
    ap.add_argument("--steps", type=int, default=4)
    args = ap.parse_args()
    a, b = (int(v) for v in args.seeds.split("-"))
    pipe = AutoPipelineForImage2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16,
                                                      variant="fp16")
    pipe.enable_model_cpu_offload()
    for L in "".join(args.letters):
        out = os.path.join(HERE, "out", f"letter_{L}")
        os.makedirs(out, exist_ok=True)
        init = letter_init(L)
        init.save(os.path.join(out, "init.png"))
        log = {"model": "stabilityai/sdxl-turbo (img2img)", "prompt": PROMPT.format(L=L), "strength": args.strength,
               "steps": args.steps, "images": []}
        for s in range(a, b + 1):
            t0 = time.time()
            img = pipe(PROMPT.format(L=L), image=init, strength=args.strength, num_inference_steps=args.steps,
                       guidance_scale=0.0, generator=torch.Generator("cuda").manual_seed(s)).images[0]
            p = os.path.join(out, f"i2i_s{s}.png")
            img.save(p)
            log["images"].append({"seed": s, "path": p, "seconds": round(time.time() - t0, 2)})
        json.dump(log, open(os.path.join(out, "text2img.json"), "w"), indent=2)
        print(f"[letters] {L} done", flush=True)


if __name__ == "__main__":
    main()
