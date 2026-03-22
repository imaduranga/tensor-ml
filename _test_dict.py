"""Test DCT + Wavelet Packet dictionary construction and 3D T-LARS."""
import numpy as np
import pywt
import time


def build_dct_dictionary(n):
    """DCT-II dictionary, column-normalised."""
    i = np.arange(n).reshape(-1, 1)
    j = np.arange(n).reshape(1, -1)
    D = np.cos(np.pi * (2 * i + 1) * j / (2 * n))
    norms = np.linalg.norm(D, axis=0, keepdims=True)
    norms[norms == 0] = 1.0
    return D / norms


def build_wavelet_packet_dictionary(n, wavelet="sym4", level=3):
    """Build wavelet packet dictionary matrix (like MATLAB wmpdictionary with wpsym4)."""
    wp = pywt.WaveletPacket(
        data=np.zeros(n), wavelet=wavelet, maxlevel=level, mode="periodization"
    )
    nodes = [node.path for node in wp.get_level(level, order="freq")]
    node0 = wp[nodes[0]]
    n_coeffs = len(node0.data)

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
    """Concatenated DCT + Wavelet Packet dictionary."""
    D_dct = build_dct_dictionary(n)
    D_wav = build_wavelet_packet_dictionary(n, wavelet=wavelet, level=level)
    D = np.hstack([D_dct, D_wav])
    norms = np.linalg.norm(D, axis=0, keepdims=True)
    norms[norms == 0] = 1.0
    return D / norms


# Test n=32
n = 32
D_dct = build_dct_dictionary(n)
D_wav = build_wavelet_packet_dictionary(n, "sym4", 3)
D_combined = build_dct_wavelet_dictionary(n, "sym4", 3)
print(f"n={n}: DCT={D_dct.shape}, Wavelet={D_wav.shape}, Combined={D_combined.shape}")
print(f"  Overcompleteness: {D_combined.shape[1]/D_combined.shape[0]:.2f}x")

# Test n=128
n = 128
t0 = time.perf_counter()
D_combined_128 = build_dct_wavelet_dictionary(n, "sym4", 3)
t1 = time.perf_counter()
print(f"\nn={n}: Combined={D_combined_128.shape}, OC={D_combined_128.shape[1]/D_combined_128.shape[0]:.2f}x, build time={t1-t0:.2f}s")

# 3D T-LARS test
from tensor_ml import TLARS

n = 32
D1 = build_dct_wavelet_dictionary(n, "sym4", 3)
D2 = build_dct_wavelet_dictionary(n, "sym4", 3)
D3 = np.eye(3)

print(f"\n3D: D1={D1.shape}, D2={D2.shape}, D3={D3.shape}")
print(f"Total Kronecker atoms: {D1.shape[1] * D2.shape[1] * D3.shape[1]}")

Y = np.random.randn(n, n, 3) * 0.1
model = TLARS(tolerance=0.1, iterations=50, l0_mode=True)
t0 = time.perf_counter()
model.fit(factor_matrices=[D1, D2, D3], Y=Y)
t1 = time.perf_counter()
print(f"Fit: n_iter={model.n_iter_}, active={len(model.active_columns_)}, time={t1-t0:.2f}s")
Y_hat = model.predict([D1, D2, D3])
print(f"Predict: shape={np.array(Y_hat).shape}")
r2 = model.score([D1, D2, D3], Y)
print(f"R²={r2:.4f}")

# Now test with 128x128x3 to gauge timing
print("\n--- 128x128x3 timing test ---")
n = 128
D1_128 = build_dct_wavelet_dictionary(n, "sym4", 3)
D2_128 = build_dct_wavelet_dictionary(n, "sym4", 3)
D3_3 = np.eye(3)
print(f"D1={D1_128.shape}, D2={D2_128.shape}, D3={D3_3.shape}")
print(f"Total atoms: {D1_128.shape[1] * D2_128.shape[1] * D3_3.shape[1]}")

Y_128 = np.random.randn(n, n, 3) * 0.1
model128 = TLARS(tolerance=0.5, iterations=20, l0_mode=True)
t0 = time.perf_counter()
model128.fit(factor_matrices=[D1_128, D2_128, D3_3], Y=Y_128)
t1 = time.perf_counter()
print(f"Fit: n_iter={model128.n_iter_}, active={len(model128.active_columns_)}, time={t1-t0:.2f}s")
