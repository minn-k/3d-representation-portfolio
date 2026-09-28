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
