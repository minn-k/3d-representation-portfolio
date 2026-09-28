"""공용 도구 (GPU · APG 런타임 없이 import 된다): 쿼터니언 ↔ 회전행렬, TRELLIS gaussian.ply 읽기, 글꼴.

TRELLIS 내부 좌표는 z 위이고, save_ply 는 (x, -z, y) 로 바꿔 쓴다 (PLY 위쪽 = -y). T_SAVE 가 그 변환이다.
"""
import os

import numpy as np

SH_C0 = 0.28209479177387814
HERE = os.path.dirname(os.path.abspath(__file__))


def runtime_root():
    """APG 런타임 폴더 (prepare_splat.py · xpbd.py · XPBD DLL · gen_<name>/ 에셋). APG_RUNTIME_ROOT > APG_ROOT/apg_runtime."""
    root = os.environ.get("APG_ROOT", os.path.dirname(HERE))
    return os.environ.get("APG_RUNTIME_ROOT", os.path.join(root, "apg_runtime"))


T_SAVE = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)     # TRELLIS save_ply: PLY = T · 내부


def quat_to_mat(q):  # w,x,y,z (N,4) -> (N,3,3)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], axis=1).reshape(-1, 3, 3)


def mat_to_quat(R):  # (N,3,3) -> w,x,y,z
    t = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    q = np.zeros((len(R), 4))
    m = t > 0
    s = np.sqrt(np.maximum(t[m] + 1, 1e-12)) * 2
    q[m] = np.stack([0.25 * s, (R[m, 2, 1] - R[m, 1, 2]) / s, (R[m, 0, 2] - R[m, 2, 0]) / s,
                     (R[m, 1, 0] - R[m, 0, 1]) / s], 1)
    for i, (a, b, c) in enumerate(((0, 1, 2), (1, 2, 0), (2, 0, 1))):
        mm = (~m) & (R[:, a, a] >= R[:, b, b]) & (R[:, a, a] >= R[:, c, c])
        m |= mm
        s = np.sqrt(np.maximum(1 + R[mm, a, a] - R[mm, b, b] - R[mm, c, c], 1e-12)) * 2
        qq = np.zeros((mm.sum(), 4))
        qq[:, 0] = (R[mm, c, b] - R[mm, b, c]) / s
        qq[:, 1 + a] = 0.25 * s
        qq[:, 1 + b] = (R[mm, b, a] + R[mm, a, b]) / s
        qq[:, 1 + c] = (R[mm, c, a] + R[mm, a, c]) / s
        q[mm] = qq
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def load_gs(out):
    """out/<name>/gaussian.ply (TRELLIS 원본, 복셀 순서 · 복셀당 32개) → 내부 좌표 (z 위)의
    위치 P, 회전 q (w,x,y,z), 크기 S, 불투명도 op, 색 col."""
    from plyfile import PlyData
    v = PlyData.read(os.path.join(out, "gaussian.ply"))["vertex"].data
    P = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64) @ T_SAVE      # 행벡터: 내부 = Tᵀ·PLY
    q = np.stack([v[f"rot_{i}"] for i in range(4)], 1).astype(np.float64)
    q = mat_to_quat(T_SAVE.T[None] @ quat_to_mat(q / np.linalg.norm(q, axis=1, keepdims=True)))
    S = np.exp(np.stack([v[f"scale_{i}"] for i in range(3)], 1))
    op = 1 / (1 + np.exp(-v["opacity"].astype(np.float64)))
    col = np.clip(SH_C0 * np.stack([v[f"f_dc_{i}"] for i in range(3)], 1) + 0.5, 0, 1)
    return P, q, S, op, col


_FONTS = ("C:/Windows/Fonts/malgunbd.ttf", "C:/Windows/Fonts/malgun.ttf",
          "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
          "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
          "/System/Library/Fonts/AppleSDGothicNeo.ttc")


def load_font(size):
    """한글이 되는 글꼴 (Windows 맑은 고딕 → Linux/macOS 대체 → PIL 기본)."""
    from PIL import ImageFont
    for p in _FONTS:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()
