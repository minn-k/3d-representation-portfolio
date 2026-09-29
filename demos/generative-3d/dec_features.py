"""Capture the Gaussian decoder feature attached to every generated voxel.

The decoder feature is not a part classifier.  It is used later only as a
local appearance/shape similarity signal while named parts still originate
from the input-image segmentation.

Usage:
  python dec_features.py --name bear_sem

Input:  out/<name>/sem.npz, produced by gen3d_sem.py
Output: out/<name>/dec_feat.npz
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "shims"))
sys.path.insert(0, os.path.join(HERE, "TRELLIS"))
os.environ.setdefault("ATTN_BACKEND", "xformers")
os.environ.setdefault("SPCONV_ALGO", "native")
# TRELLIS imports rembg → pymatting → numba even though this utility only
# decodes an already saved SLat.  Give numba a project-local writable cache so
# the import also works from a restricted shell or a fresh Windows profile.
_CACHE_DIR = os.path.join(HERE, "out", ".numba-cache")
os.makedirs(_CACHE_DIR, exist_ok=True)
os.environ.setdefault("NUMBA_CACHE_DIR", _CACHE_DIR)
os.environ.setdefault("TEMP", _CACHE_DIR)
os.environ.setdefault("TMP", _CACHE_DIR)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from trellis import models  # noqa: E402
from trellis.modules import sparse as sp  # noqa: E402


DECODER_CHECKPOINT = "microsoft/TRELLIS-image-large/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16"


def trellis_revision():
    """Best-effort TRELLIS revision, included only as output provenance."""
    root = os.path.join(HERE, "TRELLIS")
    try:
        return subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def make_slat(sem, decoder):
    """Rebuild the exact single-batch SparseTensor saved by gen3d_sem.py."""
    coords3 = np.asarray(sem["slat_coords"], np.int32)
    feats = np.asarray(sem["slat_feats"], np.float32)
    if coords3.ndim != 2 or coords3.shape[1] != 3:
        raise ValueError(f"slat_coords must be Vx3, got {coords3.shape}")
    if feats.ndim != 2 or feats.shape[0] != len(coords3):
        raise ValueError(f"slat_feats must be VxC matching coordinates, got {feats.shape}")
    try:
        param = next(decoder.parameters())
    except StopIteration as exc:
        raise RuntimeError("Gaussian decoder has no parameters") from exc
    coords4 = np.concatenate([np.zeros((len(coords3), 1), np.int32), coords3], axis=1)
    return sp.SparseTensor(
        feats=torch.from_numpy(feats).to(device=param.device, dtype=param.dtype).contiguous(),
        coords=torch.from_numpy(coords4).to(device=param.device, dtype=torch.int32).contiguous(),
    )


def capture_decoder_features(decoder, slat):
    """Decode once and capture the SparseTensor immediately before out_layer.

    The hook is read-only.  It deliberately copies both coordinates and
    features so the caller can validate that decoder order still matches the
    saved SLat order.
    """
    captured = {}

    def pre_out_layer(_module, args):
        if not args:
            raise RuntimeError("Gaussian decoder out_layer received no input")
        h = args[0]
        captured["coords"] = h.coords.detach().cpu().numpy().copy()
        captured["feat"] = h.feats.detach().float().cpu().numpy().copy()

    handle = decoder.out_layer.register_forward_pre_hook(pre_out_layer)
    try:
        with torch.no_grad():
            decoded = decoder(slat)
    finally:
        handle.remove()
    if "coords" not in captured or "feat" not in captured:
        raise RuntimeError("Decoder out_layer hook did not run")
    if len(decoded) != 1:
        raise ValueError(f"Expected one decoded Gaussian batch, got {len(decoded)}")
    return captured["coords"], captured["feat"], decoded[0]


def validate_capture(sem, decoder, dec_coords4, dec_feat, gaussian, max_xyz_error=1e-3):
    """Reject a feature file if voxel/Gaussian correspondence changed."""
    expected_coords = np.asarray(sem["slat_coords"], np.int32)
    if dec_coords4.shape != (len(expected_coords), 4):
        raise ValueError(f"Decoder feature coordinates have shape {dec_coords4.shape}, expected {(len(expected_coords), 4)}")
    if not np.array_equal(dec_coords4[:, 0], np.zeros(len(expected_coords), np.int32)):
        raise ValueError("Decoder feature batch coordinate is not the expected single batch")
    if not np.array_equal(dec_coords4[:, 1:].astype(np.int32), expected_coords):
        raise ValueError("Decoder changed voxel coordinate order; refusing to save a mismatched feature file")
    n_per_voxel = int(decoder.rep_config["num_gaussians"])
    xyz = gaussian.get_xyz.detach().float().cpu().numpy()
    expected_xyz = np.asarray(sem["gs_xyz_internal"], np.float32)
    if len(xyz) != len(expected_coords) * n_per_voxel:
        raise ValueError(f"Decoder produced {len(xyz):,} Gaussians, expected {len(expected_coords) * n_per_voxel:,}")
    if xyz.shape != expected_xyz.shape:
        raise ValueError(f"Decoded xyz shape {xyz.shape} does not match saved xyz {expected_xyz.shape}")
    xyz_error = float(np.max(np.abs(xyz - expected_xyz))) if len(xyz) else 0.0
    if not np.isfinite(xyz_error) or xyz_error >= max_xyz_error:
        raise ValueError(f"Decoded Gaussian positions differ from sem.npz by {xyz_error:.6g} (limit {max_xyz_error:.6g})")
    return {
        "trellis_revision": trellis_revision(),
        "voxels": int(len(expected_coords)),
        "feature_dim": int(dec_feat.shape[1]),
        "num_gaussians": n_per_voxel,
        "decoded_gaussians": int(len(xyz)),
        "xyz_max_abs_error": xyz_error,
    }


def extract_features(sem, decoder, max_xyz_error=1e-3):
    """Re-decode a saved SLat and return validated decoder voxel features."""
    slat = make_slat(sem, decoder)
    dec_coords4, dec_feat, gaussian = capture_decoder_features(decoder, slat)
    meta = validate_capture(sem, decoder, dec_coords4, dec_feat, gaussian, max_xyz_error=max_xyz_error)
    return dec_coords4[:, 1:].astype(np.int16), dec_feat.astype(np.float16), meta


def save_features(path, dec_coords, dec_feat, meta):
    np.savez_compressed(path, dec_coords=dec_coords, dec_feat=dec_feat,
                        meta=np.array(json.dumps(meta, sort_keys=True)))


def load_decoder():
    """Load only the 3D Gaussian decoder, not DINO or either flow model.

    Re-extracting a saved feature must not initialize the image encoder: no
    image is involved and it would unnecessarily load several gigabytes.
    ``models.from_pretrained`` resolves the same cached checkpoint used by the
    full pipeline.
    """
    decoder = models.from_pretrained(DECODER_CHECKPOINT)
    decoder.cuda()
    decoder.eval()
    return decoder


def main():
    ap = argparse.ArgumentParser(description="Capture per-voxel features immediately before TRELLIS Gaussian out_layer")
    ap.add_argument("--name", required=True, help="out/<name>/sem.npz to decode")
    ap.add_argument("--max-xyz-error", type=float, default=1e-3)
    args = ap.parse_args()
    out = os.path.join(HERE, "out", args.name)
    sem_path = os.path.join(out, "sem.npz")
    if not os.path.isfile(sem_path):
        raise FileNotFoundError(f"Missing {sem_path}; run gen3d_sem.py first")
    sem = dict(np.load(sem_path))
    required = {"slat_coords", "slat_feats", "gs_xyz_internal"}
    missing = sorted(required.difference(sem))
    if missing:
        raise ValueError(f"sem.npz is missing {', '.join(missing)}; regenerate it with gen3d_sem.py")

    decoder = load_decoder()
    dec_coords, dec_feat, meta = extract_features(sem, decoder, max_xyz_error=args.max_xyz_error)
    path = os.path.join(out, "dec_feat.npz")
    save_features(path, dec_coords, dec_feat, meta)
    print("[dec-feat]", json.dumps({**meta, "path": path}), flush=True)


if __name__ == "__main__":
    main()
