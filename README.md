# 3D Representation Portfolio

A public portfolio workspace for interactive and editable 3D representation research.

**Portfolio site: https://minn-k.github.io/3d-representation-portfolio/**

## Focus

- Structure-aware 3D Gaussian Splatting
- Geometry-aware deformation and covariance updates
- CUDA implementation for interactive rendering
- OpenUSD and 3D-engine integration
- Generative 3D → editable, simulated Gaussian assets ([demos/generative-3d](demos/generative-3d))

## Related technical repositories

- [APG-GS ChainMail](https://github.com/minn-k/apg-gs-chainmail): graph construction, ChainMail deformation, and Gaussian covariance updates in a SIBR viewer overlay.
- [3DGS–Isaac Sim XPBD](https://github.com/minn-k/3dgs-isaac-sim-xpbd): OpenUSD, Isaac Sim, CUDA XPBD, and two-arm interaction integration.

## Repository layout

~~~text
site/       source for the public portfolio site
demos/      self-contained interactive and generative 3D demonstrations
~~~

## Data policy

This repository does not track private scans, face data, trained models, generated splat assets, compiled binaries, or large intermediate artifacts. Published media must be reproducible or cleared for public sharing.
