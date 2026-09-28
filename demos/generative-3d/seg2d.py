"""입력 이미지(TRELLIS cond.png) 의 부위를 이름으로 나눈다: Grounding DINO (텍스트 → 상자) + SAM (상자 → 마스크).

  python seg2d.py --name robot_sem --parts "head,antenna,arm=arm|hand,torso,leg=leg|foot"
  (부위=문구1|문구2 — 물리에 쓸 부위 하나에 여러 검출 문구)

겹치는 마스크는 작은(구체적인) 것이 이긴다 (예: hand 가 arm 위). 배경(알파 0)은 0.
출력 out/<name>/parts2d.npz (label HxW int, names), parts2d.png (색 오버레이), parts2d.json (검출 상자·점수)
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, SamModel, SamProcessor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gs_utils import load_font  # noqa: E402

PALETTE = np.array([[255, 255, 255], [230, 76, 64], [64, 140, 230], [77, 191, 89], [242, 191, 51], [166, 102, 217],
                    [51, 204, 204], [242, 128, 179], [140, 140, 140], [120, 80, 40]], np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--parts", required=True)
    ap.add_argument("--box-thr", type=float, default=0.25)
    ap.add_argument("--text-thr", type=float, default=0.2)
    args = ap.parse_args()
    out = os.path.join(HERE, "out", args.name)
    specs = [p.strip() for p in args.parts.split(",") if p.strip()]
    names = [p.split("=")[0] for p in specs]
    phrases = [(i + 1, ph) for i, p in enumerate(specs) for ph in (p.split("=")[1].split("|") if "=" in p else [p])]
    src = Image.open(os.path.join(out, "cond.png"))
    if src.mode == "RGBA":
        alpha = np.asarray(src)[..., 3] > 127
    else:                                           # 알파 없음: 네 모서리 배경색과 다른 픽셀 = 물체
        a = np.asarray(src.convert("RGB")).astype(np.float32)
        bg = np.median(np.concatenate([a[:4, :4].reshape(-1, 3), a[-4:, -4:].reshape(-1, 3),
                                       a[:4, -4:].reshape(-1, 3), a[-4:, :4].reshape(-1, 3)]), 0)
        alpha = np.abs(a - bg).max(-1) > 12
    rgb = np.asarray(src.convert("RGB")).copy()
    rgb[~alpha] = 255
    img = Image.fromarray(rgb)
    W, H = img.size

    dev = "cuda"
    gp = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
    gm = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny").to(dev)
    dets = []
    for pid, pn in phrases:                          # 문구마다 따로 물어본다 (한 문장에 섞으면 라벨이 엉킨다)
        inputs = gp(images=img, text=f"{pn}.", return_tensors="pt").to(dev)
        with torch.no_grad():
            o = gm(**inputs)
        try:
            r = gp.post_process_grounded_object_detection(o, inputs.input_ids, threshold=args.box_thr,
                                                          text_threshold=args.text_thr, target_sizes=[(H, W)])[0]
        except TypeError:
            r = gp.post_process_grounded_object_detection(o, inputs.input_ids, box_threshold=args.box_thr,
                                                          text_threshold=args.text_thr, target_sizes=[(H, W)])[0]
        for b, s in zip(r["boxes"].cpu().numpy(), r["scores"].cpu().numpy()):
            area = (b[2] - b[0]) * (b[3] - b[1]) / (W * H)
            if area > 0.6:                          # 물체 전체를 부위로 잡은 상자는 버린다
                continue
            dets.append({"part": names[pid - 1], "phrase": pn, "pid": pid, "box": [float(v) for v in b], "score": float(s)})

    def iou(a, b):
        ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = ix * iy
        return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)
    # 다른 부위끼리 상자가 크게 겹치면 점수 높은 쪽만 (예: 'foot' 이 손 상자에도 걸림)
    dets.sort(key=lambda d: -d["score"])
    kept = []
    for d in dets:
        if all(not (k["pid"] != d["pid"] and iou(k["box"], d["box"]) > 0.5) for k in kept):
            kept.append(d)
    dets = kept
    del gm
    torch.cuda.empty_cache()

    sp = SamProcessor.from_pretrained("facebook/sam-vit-base")
    sm = SamModel.from_pretrained("facebook/sam-vit-base").to(dev)
    masks = []
    if dets:
        inputs = sp(img, input_boxes=[[d["box"] for d in dets]], return_tensors="pt").to(dev)
        with torch.no_grad():
            o = sm(**inputs, multimask_output=False)
        m = sp.image_processor.post_process_masks(o.pred_masks.cpu(), inputs["original_sizes"].cpu(),
                                                  inputs["reshaped_input_sizes"].cpu())[0][:, 0].numpy()
        masks = [mm & alpha for mm in m]
    label = np.zeros((H, W), np.int32)
    order = np.argsort([-mm.sum() for mm in masks])     # 큰 것부터 칠하고 작은 것이 덮는다
    for i in order:
        if masks[i].sum() == 0:
            continue
        label[masks[i]] = dets[i]["pid"]
        dets[i]["pixels"] = int(masks[i].sum())
    unl = alpha & (label == 0)
    np.savez_compressed(os.path.join(out, "parts2d.npz"), label=label, names=np.array(["background"] + names),
                        alpha=alpha)
    ov = np.asarray(img).copy()
    col = PALETTE[label % len(PALETTE)]
    ov[alpha] = (0.45 * ov[alpha] + 0.55 * col[alpha]).astype(np.uint8)
    im = Image.fromarray(ov)
    dr = ImageDraw.Draw(im)
    font = load_font(16)
    for d in dets:
        b = d["box"]
        dr.rectangle(b, outline=tuple(int(v) for v in PALETTE[d["pid"] % len(PALETTE)]), width=2)
        dr.text((b[0] + 3, b[1] + 2), f"{d['part']} {d['score']:.2f}", fill=(20, 20, 20), font=font)
    im.save(os.path.join(out, "parts2d.png"))
    json.dump({"names": names, "dets": dets, "unlabeled_object_pixels": int(unl.sum()), "object_pixels": int(alpha.sum())},
              open(os.path.join(out, "parts2d.json"), "w"), indent=2)
    print("[seg2d]", {n: int((label == i + 1).sum()) for i, n in enumerate(names)}, "unlabeled", int(unl.sum()),
          "/", int(alpha.sum()))


if __name__ == "__main__":
    main()
