# Generative 3D → Editable Gaussian Assets

This demo connects public pretrained image and 3D generation models to the APG-GS editing representation.

## Pipeline

1. Generate a reference image with SDXL-Turbo.
2. Convert the image to 3D Gaussians with TRELLIS.
3. Reduce Gaussian count with importance selection and render-guided re-optimization.
4. Build the APG-GS structure graph and apply CUDA XPBD deformation.
5. Update Gaussian covariance with `Σ′ = FΣ₀Fᵀ` during deformation.

The repository includes scripts, patches, and measured JSON results. It does not include pretrained weights, generated PLY assets, or the local APG runtime. Set `APG_RUNTIME_ROOT` to an APG-GS runtime checkout before running the preparation and editing scripts.

## Limitations

- Single-image 3D generation must infer unseen surfaces, so the back side may be less stable.
- Count reduction is post-processing; generation-time control over representation size remains future work.

## Semantic parts from generation internals

`gen3d_sem.py` records, during TRELLIS sampling, each 3D token's cross-attention to the input-image patches and the
intermediate DiT features. `seg2d.py` names parts on the input image (Grounding DINO + SAM). `lift_parts.py` votes
those names into 3D through early-block attention, then propagates them on a DiT-feature k-NN graph, and maps
Gaussian `i` → voxel `i // 32` → token. `edit_demo.py --parts … --cross-keep 0.1` builds a part-aware graph by
removing 90% of cross-part edges.

- Early-block attention localizes parts, including sides unseen in the input; the DiT-feature propagation cleans
  noisy labels (bear). xyz k-means and DiT-feature k-means baselines do not produce named parts.
- Lowering edge stiffness alone has no effect because shape-matching and volume clusters follow graph
  connectivity. Removing all cross-part edges detaches parts, so joints are needed.
- Bear arm pull: touching parts are dragged about 11% less and edges stretched beyond 2× drop about 33%.
  The robot, whose arm attaches at one shoulder, shows almost no change.
- Part prompts were chosen per object; thin parts can be missed.

Part-level editing (`semantic_pose.py`): with `part_id`, "the right arm" is selected by name, the shoulder joint is found
where arm and torso Gaussians touch, and the whole arm is rotated about it (60°); XPBD moves the rest. The geometry-only
baseline can only drag the hand. Arm shape error 0.52 cm (geometry) vs 0 (part), torso drag 0.15 vs 0.60 cm.
Per-part materials (`semantic_drop.py`: rigid body, soft arms) were not convincing with the current solver, whose shape
matching is a single object-level rigid fit; per-part rigid fitting in the solver is the next step.
