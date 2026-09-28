# Prompt → Editable 3D Gaussian Asset

Text or an image goes in, and an **editable, physically simulated 3D Gaussian asset** comes out. The generative
models are used as released (no training). This demo connects their output representation to the APG-GS
structure graph, a CUDA XPBD solver, and the Σ′ = FΣ₀Fᵀ covariance update.

텍스트 → 이미지 → 3D Gaussians → 개수 조절 → 구조 그래프 → 편집 · 물리. 생성 모델은 공개 가중치를 그대로 쓰고,
생성된 표현을 APG-GS 편집 엔진에 연결하는 부분과 측정이 이 데모의 작업이다.

| Step | Script | What it does |
|---|---|---|
| 2D generation | `gen2d.py` | SDXL-Turbo, 2 steps, 512² (text → image) |
| 2D generation (letters) | `gen_letters.py` | Font silhouette with extrusion → SDXL-Turbo image-to-image in the TRELLIS teaser style |
| 3D generation | `gen3d.py` | TRELLIS-image-large, Gaussian output only |
| Reduction | `distill.py` | Keep top-N Gaussians by importance, then re-fit them to renders of the full asset (Adam, L1, 3000 iterations) |
| Format | `to_model.py` | TRELLIS PLY → 3DGS training-output layout (SH padding, up axis, opacity cleanup, optional top-N) |
| Structure graph | `prepare_gen.py`, `graph_stats.py` | APG-GS graph via the SIBR viewer (unattended run), viewer→USD order mapping, graph statistics |
| Editing | `edit_demo.py` | Pin + pull with XPBD; renders position-only vs. position + Σ′ side by side |
| Letter drop | `letters_drop.py` | Seven generated letters dropped one by one, one XPBD instance each, camera path following the TRELLIS teaser |
| Pipelines | `run_asset.sh`, `run_variants.sh`, `run_letters.sh` | End-to-end batches used for the portfolio |

## Results (from `results/*/*.json`)

Measured on one RTX 4070 SUPER (12 GB).

- **3D generation:** 19–25 s per asset, 5.7–5.9 GB peak VRAM; the three demo assets have 340k–800k Gaussians.
- **Reduction:** TRELLIS emits a fixed 32 Gaussians per active voxel, so the count does not follow object complexity.
  Keeping 50k–100k and re-fitting keeps object-pixel PSNR at 33–49 dB against the full asset. Pruning alone gives 14–24 dB.
  The XPBD step drops from 49–97 ms to 9–19 ms.
- **Editing:** after pulling, 0–0.06% of graph edges are stretched beyond 2×. Larger (re-fitted) Gaussians make the
  covariance update matter more: the mean pixel difference between position-only and Σ′ renders grows as the count drops.
- **Letter drop:** no stabilization is applied. The letters are 3–9 cm thick, 30 cm tall soft slabs; they land, squash,
  bounce and fall over as the solver dictates.

## Honest limits

- The reduction is a post-process; generating a task-appropriate count directly is future work.
- The graph is built from geometry (covariance overlap) only; part/material awareness is future work.
- Single-image generation hallucinates unseen sides (the robot's back is blurry).

## Running

Requires the APG-GS work tree (not in this repository): `isaac_demo/` (`xpbd.py`, `prepare_splat.py`,
`import_graph.py`, `prepare_dataset.py`, XPBD CUDA DLL) and the APG-GS SIBR viewer build. Point to it with
`APG_ROOT` (default: the parent of this folder) or `APG_ISAAC_DEMO`.

Environment used: Windows, Python 3.10, torch 2.5.1+cu118, `spconv-cu118`, `diffusers`, `transformers`, `rembg`,
`utils3d`, and the graphdeco `diff-gaussian-rasterization`.

TRELLIS (commit in `patches/TRELLIS_COMMIT.txt`) needs two patches on this setup:

- `patches/trellis_windows_gaussian_only.patch`: open3d / nvdiffrast become optional imports (mesh and text paths
  only), and the Gaussian renderer accepts the graphdeco rasterizer.
- `patches/flexicubes_optional_kaolin.patch`: kaolin becomes optional.

No xformers wheel matches torch 2.5.1+cu118 on Windows. `shims/xformers` implements the two calls TRELLIS uses
(`memory_efficient_attention`, `BlockDiagonalMask`) with PyTorch SDPA, grouping equal-length blocks. `gen3d.py`
puts it on `sys.path`.

```bash
python gen2d.py --name bear --prompt "a cute plush teddy bear toy sitting" --seeds 0-3
bash run_asset.sh bear out/bear/text2img_s3.png "--grab-dir 1,0,1.3 --pull 1,-0.2,0.4 --dist 0.06"
python distill.py --name bear --keep 100000
```

Model weights, generated splats, graphs and renders stay outside the repository (see the root `.gitignore`).
Third-party models: SDXL-Turbo (Stability AI), TRELLIS (Microsoft), DINOv2 (Meta), used under their licenses.
