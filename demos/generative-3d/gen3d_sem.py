"""이미지 → TRELLIS 3D Gaussians, 생성 도중의 '의미 신호' 를 복셀마다 함께 저장한다.

  C:\\anaconda\\anaconda3\\envs\\trellis\\python.exe gen3d_sem.py --image out/robot/text2img_s3.png --name robot_sem

SLat 생성 트랜스포머(SLatFlowModel, 24 블록, 토큰 = 64³ 복셀을 2³ 로 묶은 32³ 격자)에 hook 을 건다.
  ① cross-attention: 토큰 → 입력 이미지 DINOv2 토큰(cls 1 + register 4 + 37×37 패치) 가중치. 헤드 평균,
     조건 있는 호출만 (CFG 음성 조건은 0 이라 제외), 지정한 스텝 범위 평균. 블록별로 따로.
  ② DiT 중간층 특징: 지정한 블록 출력 (1024 차원), 같은 스텝 범위 평균.
  ③ 최종 SLat (64³ 복셀, 8 차원) — Gaussian 디코더 입력.
출력 out/<name>/: gaussian.ply, cond.png, sem.npz (tok_coords, attn_<b>, feat_<b>, slat_coords, slat_feats, gs_xyz_internal),
                  stats.json, turntable.mp4
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

import trellis.modules.sparse.attention.modules as sp_attn_mod  # noqa: E402
from trellis.pipelines import TrellisImageTo3DPipeline  # noqa: E402
from trellis.utils import render_utils  # noqa: E402

ATTN_BLOCKS = [4, 8, 12, 16, 20, 23]
FEAT_BLOCKS = [6, 12, 18, 23]


class Recorder:
    def __init__(self, steps):
        self.steps = set(steps)       # 기록할 샘플링 스텝 (0 부터)
        self.step = -1                # 조건 있는 호출마다 +1
        self.pos = False              # 지금 호출이 조건 있는 호출인가
        self.block = -1
        self.active = False           # SLat 단계에서만
        self.attn = {}
        self.feat = {}
        self.n = {}
        self.tok_coords = None

    def rec(self):
        return self.active and self.pos and self.step in self.steps


R = None
_orig_sdpa = sp_attn_mod.sparse_scaled_dot_product_attention


def sdpa_hook(*args, **kwargs):
    out = _orig_sdpa(*args, **kwargs)
    if R is not None and R.rec() and len(args) == 2 and R.block in ATTN_BLOCKS:
        q, kv = args                               # q: SparseTensor [T,H,C], kv: dense [1,L,2,H,C]
        if isinstance(kv, torch.Tensor):
            qf = q.feats.float()                   # [T,H,C]
            k = kv[0, :, 0].float()                # [L,H,C]
            scale = qf.shape[-1] ** -0.5
            acc = None
            for h in range(qf.shape[1]):           # 헤드마다 (메모리 절약)
                a = torch.softmax((qf[:, h] @ k[:, h].T) * scale, dim=-1)
                acc = a if acc is None else acc + a
            acc /= qf.shape[1]
            key = R.block
            R.attn[key] = acc if key not in R.attn else R.attn[key] + acc
            R.n[("a", key)] = R.n.get(("a", key), 0) + 1
    return out


sp_attn_mod.sparse_scaled_dot_product_attention = sdpa_hook


def main():
    global R
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--rec-from", type=int, default=8, help="이 스텝부터")
    ap.add_argument("--rec-to", type=int, default=20, help="이 스텝까지 기록 (포함)")
    args = ap.parse_args()
    out = os.path.join(HERE, "out", args.name)
    os.makedirs(out, exist_ok=True)

    pipe = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
    pipe.cuda()
    flow = pipe.models["slat_flow_model"]
    R = Recorder(range(args.rec_from, args.rec_to + 1))

    def model_pre(mod, a, kw):
        cond = a[2] if len(a) > 2 else kw.get("cond")
        R.pos = bool(cond is not None and cond.abs().sum() > 0)
        if R.pos:
            R.step += 1
    flow.register_forward_pre_hook(model_pre, with_kwargs=True)

    for i, blk in enumerate(flow.blocks):
        def pre(mod, a, i=i):
            R.block = i
        def post(mod, a, o, i=i):
            if R.rec() and i in FEAT_BLOCKS:
                f = o.feats.float()
                R.feat[i] = f if i not in R.feat else R.feat[i] + f
                R.n[("f", i)] = R.n.get(("f", i), 0) + 1
                if R.tok_coords is None:
                    R.tok_coords = o.coords[:, 1:].int().cpu().numpy()
        blk.register_forward_pre_hook(pre)
        blk.register_forward_hook(post)

    orig_sample_slat = pipe.sample_slat

    def sample_slat(*a, **kw):
        R.active = True
        try:
            return orig_sample_slat(*a, **kw)
        finally:
            R.active = False
    pipe.sample_slat = sample_slat

    image = Image.open(args.image)
    cond_img = pipe.preprocess_image(image)
    cond_img.save(os.path.join(out, "cond.png"))
    torch.manual_seed(args.seed)
    t0 = time.time()
    cond = pipe.get_cond([cond_img])
    torch.manual_seed(args.seed)
    coords = pipe.sample_sparse_structure(cond, 1, {"steps": args.steps, "cfg_strength": 7.5})
    slat = pipe.sample_slat(cond, coords, {"steps": args.steps, "cfg_strength": 3.0})
    outs = pipe.decode_slat(slat, ["gaussian"])
    dt = time.time() - t0
    g = outs["gaussian"][0]
    g.save_ply(os.path.join(out, "gaussian.ply"))

    sem = {"tok_coords": R.tok_coords.astype(np.int16),
           "slat_coords": slat.coords[:, 1:].int().cpu().numpy().astype(np.int16),
           "slat_feats": slat.feats.float().cpu().numpy().astype(np.float32),
           "gs_xyz_internal": g.get_xyz.detach().float().cpu().numpy().astype(np.float32)}
    for b, a in R.attn.items():
        sem[f"attn_{b}"] = (a / R.n[("a", b)]).cpu().numpy().astype(np.float16)
    for b, f in R.feat.items():
        sem[f"feat_{b}"] = (f / R.n[("f", b)]).cpu().numpy().astype(np.float16)
    np.savez_compressed(os.path.join(out, "sem.npz"), **sem)

    frames = render_utils.render_video(g, resolution=512, num_frames=120, bg_color=(1, 1, 1))["color"]
    imageio.mimsave(os.path.join(out, "turntable.mp4"), frames, fps=30, quality=8)
    st = {"name": args.name, "image": os.path.abspath(args.image), "seed": args.seed, "gaussians": int(g.get_xyz.shape[0]),
          "voxels": int(slat.coords.shape[0]), "tokens": int(len(R.tok_coords)), "generate_s": round(dt, 1),
          "recorded_steps": [args.rec_from, args.rec_to], "attn_blocks": ATTN_BLOCKS, "feat_blocks": FEAT_BLOCKS,
          "attn_calls": {str(k[1]): v for k, v in R.n.items() if k[0] == "a"}}
    json.dump(st, open(os.path.join(out, "stats.json"), "w"), indent=2)
    print("[sem]", json.dumps(st), flush=True)


if __name__ == "__main__":
    main()
