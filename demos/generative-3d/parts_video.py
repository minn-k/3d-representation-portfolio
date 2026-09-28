"""부위 라벨(lift_parts.py) 을 가우시안 색으로 칠한 턴테이블 영상: 왼쪽 원래 색 · 오른쪽 부위 색.

  python parts_video.py --name robot_sem
출력 out/<name>/parts_turntable.mp4, parts_front.png
"""
import argparse
import math
import os
import sys

import imageio
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gs_utils import load_font, load_gs  # noqa: E402
from parts_core import PART_COLORS  # noqa: E402
from edit_demo import Cam, render  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--res", type=int, default=480)
    ap.add_argument("--start-deg", type=float, default=-90.0, help="첫 프레임 방향 (정면)")
    args = ap.parse_args()
    out = os.path.join(HERE, "out", args.name)
    P, q, S, op, col = load_gs(out)
    pz = np.load(os.path.join(out, "parts3d.npz"))
    names = [str(n) for n in pz["names"]]
    pcol = PART_COLORS[1 + pz["gs_part_full"].astype(int)]
    c = P.mean(0)
    size = np.linalg.norm(P.max(0) - P.min(0))
    font = load_font(18)
    frames = []
    for i in range(args.frames):
        a = math.radians(args.start_deg + 360.0 * i / args.frames)
        cam = Cam(c + 1.3 * size * np.array([math.cos(a), math.sin(a), 0.3]), c, args.res, args.res, 35)
        fr = np.concatenate([render(cam, P, q, S, op, col), render(cam, P, q, S, op, pcol)], 1)
        im = Image.fromarray(fr)
        dr = ImageDraw.Draw(im)
        for k, n in enumerate(names):
            x = args.res + 12 + (k % 3) * 150
            y = 10 + (k // 3) * 26
            dr.rectangle((x, y + 3, x + 16, y + 19), fill=tuple(int(v * 255) for v in PART_COLORS[1 + k]))
            dr.text((x + 22, y), n, fill=(30, 30, 30), font=font)
        frames.append(np.asarray(im))
    imageio.mimsave(os.path.join(out, "parts_turntable.mp4"), frames, fps=30, quality=8, macro_block_size=8)
    Image.fromarray(frames[0]).save(os.path.join(out, "parts_front.png"))
    print("[parts_video] done", len(frames))


if __name__ == "__main__":
    main()
