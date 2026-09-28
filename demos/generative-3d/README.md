# Generative 3D → Editable Gaussian Assets

Scripts, TRELLIS patches, and measured results behind the portfolio page. Installation and a quick start are in the
[root README](../../README.md). This page lists what each stage does, what it writes, and what was measured.

## Stages

| Stage | Script | Output (`out/<name>/` unless noted) |
|---|---|---|
| Text → image | `gen2d.py` (SDXL-Turbo, 2 steps, 512²) | `text2img_s<seed>.png`, `text2img.json` |
| Image → 3D Gaussians | `gen3d.py` (TRELLIS-image-large, 25 + 25 steps) | `gaussian.ply`, `turntable.mp4`, `cond.png`, `stats.json` |
| … with generation signals | `gen3d_sem.py` | the above + `sem.npz` (attention, DiT features, SLat) |
| Count reduction | `distill.py` (importance selection + re-fit, Adam, L1, 3000 its) | `distill_<N>k.ply`, `distill_<N>k.json` |
| 3DGS model folder | `to_model.py` (SH 3, opacity ≥ 0.02, `-y` up, orbit cameras) | `$APG_ROOT/output_1/gen_<name>/` |
| APG graph | `prepare_gen.py` (crop → SIBR headless graph export → PLY order), `graph_stats.py` | `$APG_RUNTIME_ROOT/gen_<name>/`, `graph.json` |
| Editing | `edit_demo.py` (pin bottom, pull a tip, XPBD, Σ′ = FΣ₀Fᵀ) | `edit_compare.mp4`, `edit_*.png`, `edit_stats.json` |
| 2D parts | `seg2d.py` (Grounding DINO boxes → SAM masks), `dl_seg.py` | `parts2d.npz`, `parts2d.png`, `parts2d.json` |
| 3D parts | `lift_parts.py` → `parts_core.py` | `parts3d.npz`, `parts3d_stats.json`, `camera_fit.png`, `parts3d_grid.png` |
| Part outputs | `export_parts.py`, `part_graph.py`, `parts_video.py` | `<asset>_parts.ply/.json`, `<asset>_part_id.u8`, `<asset>_graph_parts.npz`, SIBR folders |
| Part physics | `semantic_pose.py`, `semantic_shake.py`, `semantic_drop.py` | `out/semantic_*/` |
| Letter opener | `gen_letters.py`, `run_letters.sh`, `letters_drop.py` | `out/letters/letters_drop.mp4` |

`run_asset.sh`, `run_variants.sh` and `run_letters.sh` chain the stages (Git Bash; set `PY` to your interpreter).

## Pipeline notes

1. SDXL-Turbo makes a single object on a white background (`gen2d.py` adds the style suffix).
2. TRELLIS removes the background, crops, and generates. Gaussians come out 32 per active 64³ voxel, in voxel order.
3. The re-fit keeps the N most important Gaussians (opacity × the two largest axes) and re-optimises them against
   renders of the full asset from random views. Prune-only loses 18–25 dB of object PSNR; the re-fit recovers it.
4. `to_model.py` writes a standard 3DGS model folder (SH degree 3 with zero higher bands, normalised quaternions,
   orbit cameras around the `-y` up axis). `prepare_gen.py` crops, runs the SIBR viewer headless to export the APG
   graph (Bhattacharyya overlap + k-NN), and reorders it to PLY order.
5. `edit_demo.py` renders the same deformation twice: positions only and positions + covariance (Σ′ = FΣ₀Fᵀ from a
   local deformation gradient).

## Semantic parts from generation internals

`gen3d_sem.py` hooks TRELLIS' structured-latent transformer (24 blocks on 32³ tokens = 2³ voxels each) during
sampling, for the conditional branch only and steps 8–20. It records:

- **Cross-attention** of each token to the DINOv2 tokens of the input image (1 class + 4 register + 37 × 37 patches,
  head-averaged, blocks 4, 8, 12, 16, 20, 23).
- **DiT features**, the block outputs of blocks 6, 12, 18 and 23 (1024-dim).
- The final **SLat** (8-dim per voxel) and the Gaussian positions.

`seg2d.py` names parts on the input image; `name=phrase1|phrase2` maps several detection phrases to one part, and
overlapping masks go to the smaller mask. `lift_parts.py` moves those names onto the Gaussians. The method is in
`parts_core.py`:

| Step | `--mode proj` (default) | `--mode attn` (earlier, reproducible) |
|---|---|---|
| Evidence | attention vote + **camera estimate + visibility-aware projection** of the 2D label map | attention vote |
| Camera | starts: affine fit token centres → attention centroids (Huber IRLS) and a global azimuth × elevation × perspective search; each refined on the silhouette with a footprint-aware Chamfer cost; kept by silhouette IoU + colour correlation; gated by IoU ≥ 0.7 and colour ≥ 0.2 | none |
| Propagation | touching tokens (26-neighbourhood), DiT-feature weights, visible confident tokens clamped, 120 its | radius 4 tokens, top-12 feature neighbours, 40 its |
| Cleanup | small same-part fragments take their neighbours' part; visible voxels sharpened to 64³ | none |
| Hidden side | shell filled into a solid; each unlabeled voxel climbs the depth field to a part core and takes the seeds' part that shares the core and is reached through thick interior (thin necks are costly); voxel-level fragment cleanup | none |

Gaussian `i` → voxel `i // 32`; the re-fit asset takes the nearest voxel. `parts3d.npz` keeps the field names used by
the other scripts (`asset_part`, `asset_conf`, `asset_prob`, `gs_part_full`) and adds per-voxel labels.

**Why the teddy bear's arm mixed** under the attention-only version:

1. Cross-attention retrieves features rather than finding correspondences. On uniform fur, arm tokens also look at
   head and leg patches, which shows up as speckle in the attention-only row of `parts3d_grid.png`.
2. The vote has no visibility. A pixel's 2D label belongs to the front surface, but the inner arm, the flank behind
   it and the thigh under the paw vote with the same patches.
3. The propagation graph (radius 4 tokens = 8 voxels, about 1/8 of the object) jumps the arm–body–leg gaps. Its fur
   features are nearly identical, so noise spread along the inner arm.
4. 13.6% of the bear's object pixels had no 2D label (flanks, chest).

The projection version answers 1–3 directly: visible voxels read their own pixel. The first real run on the bear
fixed the front, but surface propagation from the visible seeds let the arm spread over the hidden sides and back of
the body. The arm's seeds are wide, while the body's are only the belly. The hidden side is therefore now decided by
volume: the back of the body belongs to the same interior as the belly, not to the arm pressed against it. On a
synthetic teddy bear with known parts (`tests/test_parts_core.py`):

| | visible voxels | hidden voxels | arm | body | all voxels |
|---|---|---|---|---|---|
| attention-only | 0.849 | 0.730 | 0.914 | 0.003 | 0.766 |
| projection + clamped propagation | 0.975 | 0.766 | 0.964 | 0.247 | 0.829 |
| + volumetric hidden side | **0.977** | **0.846** | **0.984** | **0.484** | **0.886** |

The camera is recovered to 0.7 px (silhouette IoU 0.991). The body is still the weakest because only the belly is
labeled in 2D; with its visible sides labeled as well, the hidden body reaches about 0.82. Check `camera_fit.png` after
each run.

On the real bear (`results/bear_sem/parts3d_stats_try1.json`, `_try2.json`, `parts3d_stats.json`) the estimated camera
reaches silhouette IoU 0.894 (0.298 from attention alone) with colour correlation 0.41. On the robot, the attention
start alone got stuck (IoU 0.69, colour 0.08), so the labels fell back to the 1st method. The global search was added
for this; on the synthetic bear it recovers the camera to 0.7 px even when the attention is replaced by noise. The share of Gaussians labeled
body goes from 11.6% (1st) to 15.0% (2nd) to 25.2% (3rd) as the hidden sides and back return to the body. There is no
ground truth, so the part turntables are the check. A little arm label remains on the back of the head.

### Using the labels

- **Part-aware graph** (`edit_demo.py --parts`, `part_graph.py`). A cross-part edge joins two confident Gaussians of
  different parts. Lowering its stiffness alone changes nothing, because shape-matching and volume clusters follow
  graph connectivity; removing all such edges detaches the part. So only a joint is kept: a random fraction
  (`--joint random`, used for the numbers below) or, per part pair, the most overlapping edges (`--joint overlap`,
  smallest Bhattacharyya distance). `--materials` scales stiffness per part.
- **Part-level posing** (`semantic_pose.py`). "The right arm" is selected by name, the shoulder joint is found where
  arm and torso Gaussians touch, and the whole arm is rotated about it (60°). XPBD moves the rest.
- **Per-part materials** (`semantic_shake.py`). Only the arm beyond the elbow (forearm and hand) is a soft XPBD
  body; the upper arm, shoulder and everything else move rigidly with the shaken base. The elbow is found from the
  arm's shape: the arm is ordered by distance along its graph from the shoulder, and the elbow is where it bends
  farthest from the shoulder–fingertip line (`--elbow` sets it by hand). The forearm rests lowered 30° about the
  elbow (`--droop-deg`), so it hangs a little and swings while keeping its shape. Gravity is not used: with this
  solver it stretches the soft arms into poles.
- **Export** (`export_parts.py`). `part_id` as a PLY property, raw bytes and JSON, plus SIBR model folders coloured by
  part or with chosen parts hidden. The SIBR viewer has no per-part toggles or per-part physics UI yet.

### Measured

- Bear arm pull (`edit_demo.py --parts … --cross-keep 0.1`, random joints): touching parts are dragged about 11% less,
  and edges stretched beyond 2× drop about 33%. The robot, whose arm attaches at one shoulder, changes little.
- Part-level posing (robot, 60°): arm shape error 0.52 cm (drag the hand) vs 0 (rotate the part); torso drag 0.15 vs
  0.60 cm.
- Per-part materials (robot; feet on a base shaken ±4 cm, 2 Hz, 1.6 s). Geometry graph with one material (edge
  stiffness 0.3, object shape 0.1): head / torso / arm wobble 2.20 / 1.35 / 2.18 cm RMS. Part-aware (rigid body, arm
  stiffness 0.2, shape 0.08): 0 / 0 / 3.05 cm. Which parts are rigid or soft was chosen by hand.
- `semantic_drop.py` (per-part shape stiffness during a drop) was not convincing with the current solver, whose shape
  matching is one object-level rigid fit; per-part rigid fitting in the solver is the next step.

## Letter opener

`letters_drop.py` drops the seven generated letters (T R E L L I S, 50k Gaussians each, one XPBD instance per letter)
and follows the TRELLIS teaser camera: front, pull back, then a low pass along the row. From 7 s, during the low pass,
each letter's invisible floor pulses up 4 cm in 0.07 s and drops back in 0.12 s, once every 0.9 s. This is the same
`set_ground(height)` as the floor-height control of the viewer. The pulses run left to right with 0.12 s between
letters. Object shape stiffness is 0.45 and damping 0.006, so the letters pop and wobble like jelly. A point-mass check
of the pulse gives a 6 cm hop with 0.18 s in the air. `--no-bounce --shape 0.6 --damping 0.01` reproduces the earlier
video; `--probe --probe-seconds 13` reports how many letters stay upright.

## Limitations

- Single-image 3D generation must infer unseen surfaces, so the back side may be less stable.
- Count reduction is post-processing; generation-time control over representation size remains future work.
- Part prompts are chosen per object and thin parts can be missed; the part lifting depends on the estimated camera.
- The graph builder and the extended XPBD runtime are not part of this repository (root README, Installation step 5).
