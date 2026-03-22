"""Test image sources and timing for 64x64x3."""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import time


def make_scene_image(size: int = 64) -> np.ndarray:
    """Generate a colorful synthetic scene with geometric shapes."""
    img = Image.new("RGB", (size, size), (135, 206, 235))  # sky blue
    draw = ImageDraw.Draw(img)
    s = size

    # Ground
    draw.rectangle([0, int(0.62 * s), s, s], fill=(34, 139, 34))

    # Sun
    sx, sy, sr = int(0.7 * s), int(0.1 * s), int(0.1 * s)
    draw.ellipse([sx - sr, sy - sr, sx + sr, sy + sr], fill=(255, 215, 0))

    # House body
    hx1, hy1, hx2, hy2 = int(0.3 * s), int(0.4 * s), int(0.65 * s), int(0.7 * s)
    draw.rectangle([hx1, hy1, hx2, hy2], fill=(139, 69, 19))
    # Roof
    mid_x = (hx1 + hx2) // 2
    draw.polygon([(hx1 - 5, hy1), (mid_x, int(0.25 * s)), (hx2 + 5, hy1)], fill=(178, 34, 34))
    # Window
    wx1, wy1 = int(0.42 * s), int(0.48 * s)
    wx2, wy2 = int(0.53 * s), int(0.58 * s)
    draw.rectangle([wx1, wy1, wx2, wy2], fill=(173, 216, 230))
    # Door
    dx1, dy1 = int(0.35 * s), int(0.55 * s)
    dx2, dy2 = int(0.43 * s), int(0.7 * s)
    draw.rectangle([dx1, dy1, dx2, dy2], fill=(160, 82, 45))

    # Tree
    tx1, ty1, tx2, ty2 = int(0.08 * s), int(0.35 * s), int(0.18 * s), int(0.7 * s)
    draw.rectangle([tx1, ty1, tx2, ty2], fill=(101, 67, 33))
    tr = int(0.12 * s)
    tc = (tx1 + tx2) // 2
    draw.ellipse([tc - tr, ty1 - tr, tc + tr, ty1 + int(0.3 * tr)], fill=(0, 128, 0))

    # Cloud
    cx, cy = int(0.35 * s), int(0.08 * s)
    cr = int(0.06 * s)
    draw.ellipse([cx - cr, cy - cr, cx + cr, cy + cr], fill=(255, 255, 255))
    draw.ellipse([cx - 2 * cr, cy - cr // 2, cx, cy + cr // 2], fill=(255, 255, 255))
    draw.ellipse([cx, cy - cr // 2, cx + 2 * cr, cy + cr // 2], fill=(255, 255, 255))

    # Slight blur for smoother look
    img = img.filter(ImageFilter.GaussianBlur(0.5))

    return np.asarray(img, dtype=np.float64) / 255.0


# Test the synthetic image
Y = make_scene_image(64)
print(f"Synthetic scene: shape={Y.shape}, range=[{Y.min():.3f}, {Y.max():.3f}]")

# Build dictionaries and test timing
import pywt


def build_dct_dictionary(n):
    i = np.arange(n).reshape(-1, 1)
    j = np.arange(n).reshape(1, -1)
    D = np.cos(np.pi * (2 * i + 1) * j / (2 * n))
    norms = np.linalg.norm(D, axis=0, keepdims=True)
    norms[norms == 0] = 1.0
    return D / norms


def build_wavelet_packet_dictionary(n, wavelet="sym4", level=3):
    wp = pywt.WaveletPacket(
        data=np.zeros(n), wavelet=wavelet, maxlevel=level, mode="periodization"
    )
    nodes = [node.path for node in wp.get_level(level, order="freq")]
    n_coeffs = len(wp[nodes[0]].data)
    atoms = []
    for path in nodes:
        for k in range(n_coeffs):
            wp2 = pywt.WaveletPacket(
                data=None, wavelet=wavelet, maxlevel=level, mode="periodization"
            )
            for p in nodes:
                wp2[p] = np.zeros(n_coeffs)
            wp2[path] = np.zeros(n_coeffs)
            wp2[path].data[k] = 1.0
            atom = wp2.reconstruct(update=False)[:n]
            atoms.append(atom)
    W = np.column_stack(atoms)
    norms = np.linalg.norm(W, axis=0, keepdims=True)
    norms[norms == 0] = 1.0
    return W / norms


def build_dct_wavelet_dictionary(n, wavelet="sym4", level=3):
    D_dct = build_dct_dictionary(n)
    D_wav = build_wavelet_packet_dictionary(n, wavelet=wavelet, level=level)
    D = np.hstack([D_dct, D_wav])
    norms = np.linalg.norm(D, axis=0, keepdims=True)
    norms[norms == 0] = 1.0
    return D / norms


from tensor_ml import TLARS

n = 64
D1 = build_dct_wavelet_dictionary(n)
D2 = build_dct_wavelet_dictionary(n)
D3 = np.eye(3)
total = D1.shape[1] * D2.shape[1] * D3.shape[1]
print(f"\nD1={D1.shape}, D2={D2.shape}, D3={D3.shape}")
print(f"Total atoms: {total}")
print(f"5% = {int(0.05 * total)}, 10% = {int(0.10 * total)}")

# Test timing at various iteration counts
for n_iter in [100, 500, 1000, 2000, 3000, 5000]:
    model = TLARS(iterations=n_iter, tolerance=1e-15, l0_mode=True)
    t0 = time.perf_counter()
    model.fit([D1, D2, D3], Y)
    t1 = time.perf_counter()
    pct = len(model.active_columns_) / total * 100
    print(
        f"  iters={n_iter:>5d} -> active={len(model.active_columns_):>5d} "
        f"({pct:.1f}%), time={t1-t0:.2f}s"
    )
