# Generative 3D → Editable Gaussian Assets

Scripts, TRELLIS patches, and measured results behind the portfolio page. Installation and a quick start are in the
[root README](../../README.md). This page lists what each stage does, what it writes, and what was measured.

## Stages

| Stage | Script | Output (`out/<name>/` unless noted) |
|---|---|---|
| Text → image | `gen2d.py` (SDXL-Turbo, 2 steps, 512²) | `text2img_s<seed>.png`, `text2img.json` |
| Image → 3D Gaussians | `gen3d.py` (TRELLIS-image-large, 25 + 25 steps) | `gaussian.ply`, `turntable.mp4`, `cond.png`, `stats.json` |
| … with generation signals | `gen3d_sem.py` | the above + `sem.npz` (attention, DiT features, SLat); `--dec-feat` additionally writes `dec_feat.npz` |
| Count reduction | `distill.py` (importance selection + re-fit, Adam, L1, 3000 its) | `distill_<N>k.ply`, `distill_<N>k.json` |
| 3DGS model folder | `to_model.py` (SH 3, opacity ≥ 0.02, `-y` up, orbit cameras) | `$APG_ROOT/output_1/gen_<name>/` |
| APG graph | `prepare_gen.py` (crop → SIBR headless graph export → PLY order), `graph_stats.py` | `$APG_RUNTIME_ROOT/gen_<name>/`, `graph.json` |
| Editing | `edit_demo.py` (pin bottom, pull a tip, XPBD, Σ′ = FΣ₀Fᵀ) | `edit_compare.mp4`, `edit_*.png`, `edit_stats.json` |
| 2D parts | `seg2d.py` (Grounding DINO boxes → SAM masks), `dl_seg.py` | `parts2d.npz`, `parts2d.png`, `parts2d.json` |
| 3D parts | `lift_parts.py` → `parts_core.py` | `parts3d.npz`, `parts3d_stats.json`, `camera_fit.png`, `parts3d_grid.png`; optional `--vox-feat dec --diff-png` also writes `parts3d_vox_feat_diff.png` |
| Part outputs | `export_parts.py`, `part_graph.py`, `parts_video.py` | `<asset>_parts.ply/.json`, `<asset>_part_id.u8`, `<asset>_graph_parts.npz`, SIBR folders |
| Part physics | public demos: `semantic_pose.py`, `semantic_shake.py`, `semantic_drop.py`; optional private-runtime driver and comparison: `part_physics.py`, `part_shake.py` | `out/semantic_*/`; a deformed PLY or SIBR model folder on request |
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

### Optional decoder-feature boundary refinement

This is deliberately a refinement, not a source of named semantic labels. `dec_features.py` replays the saved SLat
through TRELLIS' Gaussian decoder and records the feature immediately before `out_layer`. It first checks that the
saved voxel order is unchanged and that the re-decoded Gaussian coordinates agree with the saved output. In the local
bear and robot validation runs, this captured one 768-D feature per active voxel, 32 generated Gaussians per voxel,
and `xyz_max_abs_error = 0.0`.

With `lift_parts.py --mode proj --vox-feat dec`, those features are deterministically reduced to 64-D PCA descriptors
and used only on the 26-neighbour voxel graph and against the named, visible 2D seeds. The visible seeds remain fixed;
the feature is not clustered into names and does not claim that TRELLIS directly predicts `head`, `arm`, or `leg`.
The refinement is skipped when camera fitting falls back, and it never changes the older `--mode attn` path.

On the synthetic teddy test, this optional pass must improve boundary accuracy by at least two percentage points while
not reducing total or visible accuracy. The latest local runs changed 305 / 24,842 bear voxels (28 visible, 277 hidden)
and 252 / 20,498 robot voxels (6 visible, 246 hidden). Real assets have no 3D ground truth, so those counts are not an
accuracy claim.

The pass has two parts: propagation on the 26-neighbour voxel graph, and a weak (20%) vote for the named part whose
visible seeds have the most similar mean feature. On the synthetic teddy (overall 88.57% → 90.26%, boundary 76.25% →
79.95%) nearly all of the gain comes from the vote: the graph step alone gives 88.74% / 76.58%, and the same pass run
on the existing DiT token features copied to their 2³ voxels gives 90.22% / 79.89%. So the synthetic test supports the
new step, not a benefit of the decoder feature's finer resolution. On the real bear, the changed voxels include the
back-of-head streak that the 3rd method still labeled arm; they now read head. `--vox-feat dit` fixes the same streak
(492 changed voxels, including a band at the back of the neck, vs 305 with the decoder feature), so on real data too
the fix comes from the vote; the decoder feature only makes the change more local.

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
reaches silhouette IoU 0.894 (0.298 from attention alone) with colour correlation 0.41. The share of Gaussians labeled
body goes from 11.6% (1st) to 15.0% (2nd) to 25.2% (3rd) as the hidden sides and back return to the body. There is no
ground truth, so the part turntables are the check. A little arm label remains on the back of the head.

The earlier robot initialization from attention alone got stuck (silhouette IoU 0.69, colour 0.08). The current robot
run uses the global camera search instead (`az0`, elevation -10°, perspective 0.25): silhouette IoU 0.853 and visible
colour correlation 0.438, with no attention fallback. On the synthetic bear the same search recovers the camera to
0.7 px even when the attention is replaced by noise.

### Using the labels

- **Part-aware graph** (`edit_demo.py --parts`, `part_graph.py`). A cross-part edge joins two confident Gaussians of
  different parts. Lowering its stiffness alone changes nothing, because shape-matching and volume clusters follow
  graph connectivity; removing all such edges detaches the part. So only a joint is kept: a random fraction
  (`--joint random`, used for the numbers below) or, per part pair, the most overlapping edges (`--joint overlap`,
  smallest Bhattacharyya distance). `--materials` scales stiffness per part.
- **Part-level posing** (`semantic_pose.py`). "The right arm" is selected by name, the shoulder joint is found where
  arm and torso Gaussians touch, and the whole arm is rotated about it (60°). XPBD moves the rest.
- **Per-part materials** (`semantic_shake.py`). Gaussians that are not arm move rigidly with the shaken base; the arms
  are a soft XPBD body.
- **Part-bounded XPBD, private runtime** (`part_physics.py`). It verifies that `asset_part` and the graph use the same
  Gaussian order, then passes IDs through a C ABI. Volume clusters cannot span different known labels, and object shape
  matching is split by `(graph component, part)`. Unresolved IDs use neighbouring labels where possible; remaining
  ones are explicitly reported as fallback groups. `--part-stiffness head=0.8,arm=0.2` sets only per-part shape-match
  strength. `--out-ply` and `--out-model` write a static result for SIBR; they do not add an interactive per-part UI.
  Because the whole robot is one graph component, `(graph component, part)` puts both arms into one rigid fit.
- **Part-aware shake comparison, private runtime** (`part_shake.py`). The same shake as `semantic_shake.py`, but both
  robots are one soft body fixed only at the feet. The right one passes *part pieces* (same-part connected pieces, so
  the left and right arm are separate; pieces under 200 Gaussians join their neighbour) to `set_part_ids`, gives the
  soft pieces weaker shape matching (0.05 vs 0.15) and blends edge stiffness from body (0.6) to arm (0.2) over six
  graph hops at the boundary. Only the soft pieces get their own shape group (`--groups soft`); head, torso and legs
  share one. With a group per body part (`--groups all`) the torso slid as a block over the pinned legs and the waist
  looked cut: 11.6% of boundary edges stretched beyond 1.5× (1.6% for the uniform robot). It records per-part wobble, arm motion relative to the body's rigid motion, per-piece shape error, boundary
  edge stretch and step time;
  `--no-part-shape`, `--no-part-volume` and `--no-edge-ramp` switch the parts off one at a time.
  `tests/runtime_regression.py` checks that the part-aware DLL matches the previous DLL bit for bit when no part IDs
  are set.
- **Export** (`export_parts.py`). `part_id` as a PLY property, raw bytes and JSON, plus SIBR model folders coloured by
  part or with chosen parts hidden. The SIBR viewer has no per-part toggles or per-part physics UI yet.

### Run the optional local physics path

The following is a Windows PowerShell example for the local bear labels. It is intentionally opt-in and does not
replace any published video or asset. `--drive-part` is only a small, repeatable validation action: it holds the named
part at an offset for a few solver steps, then exports the static result.

```powershell
$env:APG_RUNTIME_ROOT = "C:\gaussian-splatting\isaac_demo"
$py = "C:\anaconda\anaconda3\envs\trellis\python.exe"
cd C:\public-repos\3d-representation-portfolio\demos\generative-3d

& $py .\part_physics.py `
  --asset gen_bear_sem_d100k `
  --parts C:\gaussian-splatting\genai\out\bear_sem\parts3d.npz `
  --dll C:\gaussian-splatting\isaac_demo\xpbd_dll\xpbd_isaac_part_v2.dll `
  --steps 2 --drive-part arm --drive-offset "0.08,0,0.03" --drive-steps 2 `
  --out-ply C:\gaussian-splatting\genai\out\bear_sem\part_physics_bear_v2.ply `
  --out-model C:\gaussian-splatting\output_1\gen_bear_sem_part_physics_v2
```

Open that generated model with the local SIBR viewer:

```powershell
cd C:\gaussian-splatting\SIBR_viewers\install\bin
.\SIBR_gaussianViewer_app.exe `
  --model-path C:\gaussian-splatting\output_1\gen_bear_sem_part_physics_v2 `
  --config C:\gaussian-splatting\output_1\gen_bear_sem_part_physics_v2\viewer_config.json `
  --iteration 1 --device 0 --no_interop
```

For an A/B comparison against ordinary object-level shape matching, add `--no-part-shape`. To disable the
part-bounded volume topology too, add `--no-part-volume`.

### Measured

- Bear arm pull (`edit_demo.py --parts … --cross-keep 0.1`, random joints): touching parts are dragged about 11% less,
  and edges stretched beyond 2× drop about 33%. The robot, whose arm attaches at one shoulder, changes little.
- Part-level posing (robot, 60°): arm shape error 0.52 cm (drag the hand) vs 0 (rotate the part); torso drag 0.15 vs
  0.60 cm.
- Per-part materials (robot; feet on a base shaken ±4 cm, 2 Hz, 1.6 s). Geometry graph with one material (edge
  stiffness 0.3, object shape 0.1): head / torso / arm wobble 2.20 / 1.35 / 2.18 cm RMS. Part-aware (rigid body, arm
  stiffness 0.2, shape 0.08): 0 / 0 / 3.05 cm. Which parts are rigid or soft was chosen by hand.
- Local part-aware XPBD validation on the 99,999-Gaussian bear formed five shape groups and reported zero
  cross-label volume clusters and zero unresolved fallback groups. Holding 18,344 arm Gaussians 8 cm aside produces a
  different result from ordinary object-level shape matching (mean position difference 0.0090 in source units).
  This confirms the constraint path is active; it is not yet a stability, speed, or visual-quality benchmark.
- The public `semantic_drop.py` remains an object-level demo. The private runtime now has opt-in `(component, part)`
  shape matching and part-bounded volume clusters, but drop-scenario quality still needs a separate evaluation.

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
- `part_physics.py` is included as an execution driver, but its graph builder and modified XPBD runtime/DLL remain in
  the private local runtime; cloning this repository alone cannot run that path.
- The current part-aware solver uses labels for volume-cluster topology and shape-match grouping only. It does not infer
  `rigid/deformable/fixed` types, set mass or damping by part, or automatically remove all cross-part graph edges.
