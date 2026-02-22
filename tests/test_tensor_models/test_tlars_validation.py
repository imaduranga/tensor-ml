"""
TLARS Validation Tests
======================
These tests validate the TLARS algorithm by replicating the verification
logic from the MATLAB RunTLARSFn function:

  1. Run TLARS on a tensor Y with dictionary factor matrices.
  2. Normalize dictionaries: D_n = normc(D_n) for each mode.
  3. Reconstruct: Ax = kroneckerMatrixPartialVectorProduct(D_Cell_Array, Active_Columns, {}, x, false, ...).
  4. Compute residual: y = normc(vec(Y)), r = y - Ax.
  5. Verify: norm(r) < tolerance.

Additional validations include predict/score consistency, coefficient
sparsity, and both L0 and L1 mode correctness.

Run with ``pytest -s`` (or ``-rP``) to see the displayed output.
"""

import time
import numpy as np
import pytest
from tensor_ml.tensor_models.multilinear.tlars import TLARS
import tensor_ml.tensorops._tensor_products as tp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normc(D: np.ndarray) -> np.ndarray:
    """Column-normalise a matrix (equivalent to MATLAB normc)."""
    norms = np.linalg.norm(D, axis=0, keepdims=True)
    norms[norms == 0] = 1.0
    return D / norms


def _create_synthetic_data(tensor_shape, dict_sizes, seed=42):
    """
    Create a synthetic tensor Y and factor (dictionary) matrices.

    Parameters
    ----------
    tensor_shape : tuple
        Shape of the target tensor Y.
    dict_sizes : list[int]
        Number of dictionary atoms per mode.
    seed : int
        Random seed.

    Returns
    -------
    Y : np.ndarray
        Target tensor.
    factor_matrices : list[np.ndarray]
        Dictionary matrices, one per mode.
    """
    rng = np.random.RandomState(seed)
    factor_matrices = [rng.randn(tensor_shape[n], dict_sizes[n]) for n in range(len(tensor_shape))]

    # Build a sparse coefficient tensor and construct Y = full_multilinear_product(X, D)
    # so that a ground-truth sparse representation exists.
    core_shape = tuple(dict_sizes)
    X_sparse = np.zeros(core_shape)
    n_nonzero = max(3, int(0.01 * np.prod(core_shape)))
    flat_indices = rng.choice(int(np.prod(core_shape)), size=n_nonzero, replace=False)
    np.put(X_sparse, flat_indices, rng.randn(n_nonzero))

    # Normalize dictionaries for construction
    D_normed = [_normc(D) for D in factor_matrices]
    Y = tp.full_multilinear_product(X_sparse, D_normed, use_transpose=False)

    return Y, factor_matrices


def _validate_residual_like_matlab(model: TLARS, factor_matrices, Y, tolerance):
    """
    Replicate the MATLAB RunTLARSFn validation:
      D_n = normc(D_n)
      Ax = kroneckerMatrixPartialVectorProduct(D, Active_Columns, {}, x)
      y  = normc(vec(Y))
      r  = y - Ax
      assert norm(r) < tolerance
    """
    # Step 1: Normalize dictionaries (same as MATLAB normc)
    D_normed = [_normc(D.copy()) for D in factor_matrices]
    core_tensor_shape = [D.shape[1] for D in D_normed]

    # Step 2: Retrieve learned coefficients and active columns
    active_columns = model.active_columns_
    coefs = model.coef_

    # Step 3: Reconstruct Ax via the Kronecker partial vector product
    Ax = tp.kronecker_matrix_vector_product(
        factor_matrices=D_normed,
        x=coefs,
        tensor_shape=core_tensor_shape,
        active_columns=active_columns,
        active_indices=None,
        use_transpose=False,
    )

    # Step 4: Normalise Y and vectorise (column-major, matching MATLAB vec)
    y = tp.vectorize(Y)
    y = y / np.linalg.norm(y)

    # Step 5: Compute residual
    r = y - Ax
    norm_r = np.linalg.norm(r)

    return norm_r


def _run_tlars_validation(
    test_name: str,
    tensor_shape: tuple,
    dict_sizes: list,
    tolerance: float,
    l0_mode: bool,
    seed: int,
    iterations: int = 10000,
    active_coefficients: int = 1000,
    backend: str = "numpy",
    factor_matrices_override=None,
    Y_override=None,
):
    """
    Run TLARS and display results in the style of MATLAB RunTLARSFn.

    This mirrors the MATLAB workflow:
      1. Print configuration
      2. Run TLARS
      3. Normalize dictionaries, reconstruct, compute residual
      4. Display: norm(r), iterations, active columns, sparsity, timing
    """
    lp = "L0" if l0_mode else "L1"
    product = "Kronecker"

    # ── Header ──────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  {test_name}")
    print(f"{'='*70}")
    print(f"Running TLARS {product} {lp} until norm of the residual reaches {tolerance}")
    print(f"  Tensor shape        : {tensor_shape}")
    print(f"  Dictionary sizes    : {dict_sizes}")
    print(f"  Backend             : {backend}")
    print(f"  Max iterations      : {iterations}")
    print(f"  Active coeff. limit : {active_coefficients}")

    # ── Data ────────────────────────────────────────────────────────
    if Y_override is not None and factor_matrices_override is not None:
        Y = Y_override
        factor_matrices = factor_matrices_override
    else:
        Y, factor_matrices = _create_synthetic_data(tensor_shape, dict_sizes, seed=seed)

    total_columns = int(np.prod(dict_sizes))
    total_elements = int(np.prod(tensor_shape))
    print(f"  Total dict. columns : {total_columns}")
    print(f"  Tensor elements     : {total_elements}")

    # ── Fit ─────────────────────────────────────────────────────────
    model = TLARS(
        backend=backend,
        tolerance=tolerance,
        l0_mode=l0_mode,
        iterations=iterations,
        active_coefficients=active_coefficients,
    )

    t_start = time.perf_counter()
    model.fit(factor_matrices, Y)
    t_elapsed = time.perf_counter() - t_start

    # ── MATLAB-style residual validation ────────────────────────────
    norm_r = _validate_residual_like_matlab(model, factor_matrices, Y, tolerance)

    n_active = len(model.active_columns_)
    sparsity = 1.0 - n_active / total_columns
    converged = norm_r < tolerance

    # ── Display results (like MATLAB fprintf) ───────────────────────
    print(f"\n  TLARS Completed.")
    print(f"  Norm of the Residual = {norm_r:.6g}")
    print(f"  Tolerance            = {tolerance}")
    print(f"  Converged            = {converged}")
    print(f"  Iterations           = {model.n_iter_}")
    print(f"  Active columns       = {n_active} / {total_columns}")
    print(f"  Sparsity             = {sparsity:.4%}")
    print(f"  Elapsed time         = {t_elapsed:.3f} s")

    # Residual norm history (first / last few entries)
    norms = model.norm_r_
    if len(norms) > 6:
        hist_str = (
            f"[{norms[0]:.6f}, {norms[1]:.6f}, ..., "
            f"{norms[-2]:.6f}, {norms[-1]:.6f}]  ({len(norms)} entries)"
        )
    else:
        hist_str = str([f"{v:.6f}" for v in norms])
    print(f"  Residual history     = {hist_str}")
    print(f"{'─'*70}")

    return model, norm_r, converged


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTLARSValidation:
    """
    End-to-end validation of the TLARS algorithm, mirroring the MATLAB
    RunTLARSFn residual norm check.  All tests display results via print().
    Run with ``pytest -s`` to see the output.
    """

    # -- 2-D tensor (matrix case) ------------------------------------------

    def test_2d_l1_residual_below_tolerance(self):
        """TLARS L1 on a 2-D tensor should converge below the tolerance."""
        tolerance = 0.10
        _, norm_r, _ = _run_tlars_validation(
            test_name="2-D Tensor – L1 Mode",
            tensor_shape=(8, 8), dict_sizes=[12, 12],
            tolerance=tolerance, l0_mode=False, seed=0,
            iterations=5000, active_coefficients=500,
        )
        assert norm_r < tolerance, (
            f"2-D L1: residual norm {norm_r:.6f} exceeds tolerance {tolerance}"
        )

    def test_2d_l0_residual_below_tolerance(self):
        """TLARS L0 on a 2-D tensor should converge below the tolerance."""
        tolerance = 0.10
        _, norm_r, _ = _run_tlars_validation(
            test_name="2-D Tensor – L0 Mode",
            tensor_shape=(8, 8), dict_sizes=[12, 12],
            tolerance=tolerance, l0_mode=True, seed=1,
            iterations=5000, active_coefficients=500,
        )
        assert norm_r < tolerance, (
            f"2-D L0: residual norm {norm_r:.6f} exceeds tolerance {tolerance}"
        )

    # -- 3-D tensor ---------------------------------------------------------

    def test_3d_l1_residual_below_tolerance(self):
        """TLARS L1 on a 3-D tensor should converge below the tolerance."""
        tolerance = 0.10
        _, norm_r, _ = _run_tlars_validation(
            test_name="3-D Tensor – L1 Mode",
            tensor_shape=(6, 6, 6), dict_sizes=[8, 8, 8],
            tolerance=tolerance, l0_mode=False, seed=10,
            iterations=10000, active_coefficients=1000,
        )
        assert norm_r < tolerance, (
            f"3-D L1: residual norm {norm_r:.6f} exceeds tolerance {tolerance}"
        )

    def test_3d_l0_residual_below_tolerance(self):
        """TLARS L0 on a 3-D tensor should converge below the tolerance."""
        tolerance = 0.10
        _, norm_r, _ = _run_tlars_validation(
            test_name="3-D Tensor – L0 Mode",
            tensor_shape=(6, 6, 6), dict_sizes=[8, 8, 8],
            tolerance=tolerance, l0_mode=True, seed=11,
            iterations=10000, active_coefficients=1000,
        )
        assert norm_r < tolerance, (
            f"3-D L0: residual norm {norm_r:.6f} exceeds tolerance {tolerance}"
        )

    # -- Predict / Score consistency ----------------------------------------

    def test_predict_reconstructs_Y(self):
        """
        predict() should reconstruct Y to within the tolerance,
        consistent with the residual norm from fit().
        """
        tolerance = 0.10
        Y, D = _create_synthetic_data(tensor_shape=(6, 6, 6), dict_sizes=[8, 8, 8], seed=20)

        model, norm_r, _ = _run_tlars_validation(
            test_name="3-D Tensor – Predict Reconstruction",
            tensor_shape=(6, 6, 6), dict_sizes=[8, 8, 8],
            tolerance=tolerance, l0_mode=False, seed=20,
            iterations=10000, active_coefficients=1000,
            factor_matrices_override=D, Y_override=Y,
        )

        Y_pred = model.predict(D)
        rel_error = np.linalg.norm(Y - Y_pred) / np.linalg.norm(Y)
        print(f"  Predict relative error = {rel_error:.6g}")
        print(f"  Predict shape match    = {Y_pred.shape == Y.shape}")
        print(f"{'─'*70}")

        assert rel_error < tolerance, (
            f"predict relative error {rel_error:.6f} exceeds tolerance {tolerance}"
        )
        assert Y_pred.shape == Y.shape, "predict shape mismatch"

    def test_score_is_high(self):
        """R² score should be close to 1 for a well-fitted sparse model."""
        tolerance = 0.10
        Y, D = _create_synthetic_data(tensor_shape=(6, 6, 6), dict_sizes=[8, 8, 8], seed=21)

        model, _, _ = _run_tlars_validation(
            test_name="3-D Tensor – R² Score",
            tensor_shape=(6, 6, 6), dict_sizes=[8, 8, 8],
            tolerance=tolerance, l0_mode=False, seed=21,
            iterations=10000, active_coefficients=1000,
            factor_matrices_override=D, Y_override=Y,
        )

        score = model.score(D, Y)
        print(f"  R² Score               = {score:.6f}")
        print(f"{'─'*70}")

        assert isinstance(score, float)
        assert score > 0.5, f"R² score {score:.4f} is unexpectedly low"

    # -- Residual norm tracking ---------------------------------------------

    def test_residual_norm_decreases(self):
        """The tracked residual norm should be monotonically non-increasing."""
        tolerance = 0.10
        model, _, _ = _run_tlars_validation(
            test_name="3-D Tensor – Residual Monotonicity",
            tensor_shape=(6, 6, 6), dict_sizes=[8, 8, 8],
            tolerance=tolerance, l0_mode=False, seed=30,
            iterations=5000, active_coefficients=500,
        )

        norms = model.norm_r_
        assert len(norms) > 1, "Expected at least 2 residual norm entries"
        eps = 1e-10
        monotonic = True
        for i in range(1, len(norms)):
            if norms[i] > norms[i - 1] + eps:
                monotonic = False
                print(f"  WARNING: Residual norm increased at step {i}: "
                      f"{norms[i-1]:.8f} -> {norms[i]:.8f}")
        print(f"  Monotonically decreasing = {monotonic}")
        print(f"{'─'*70}")

        for i in range(1, len(norms)):
            assert norms[i] <= norms[i - 1] + eps, (
                f"Residual norm increased at iteration {i}: "
                f"{norms[i-1]:.8f} -> {norms[i]:.8f}"
            )

    # -- Coefficient sparsity -----------------------------------------------

    def test_coefficients_are_sparse(self):
        """
        The number of active coefficients should be much smaller
        than the total number of dictionary columns.
        """
        tolerance = 0.10
        model, _, _ = _run_tlars_validation(
            test_name="3-D Tensor – Coefficient Sparsity",
            tensor_shape=(6, 6, 6), dict_sizes=[8, 8, 8],
            tolerance=tolerance, l0_mode=False, seed=40,
            iterations=10000, active_coefficients=1000,
        )

        total_columns = 8 ** 3
        n_active = len(model.active_columns_)
        coef_length_match = len(model.coef_) == n_active

        print(f"  Active columns         = {n_active}")
        print(f"  Total columns          = {total_columns}")
        print(f"  coef_ length matches   = {coef_length_match}")
        print(f"{'─'*70}")

        assert n_active < total_columns, (
            f"Active columns ({n_active}) not fewer than total ({total_columns})"
        )
        assert n_active > 0, "No active columns found"
        assert coef_length_match, "coef_ length does not match active_columns_"

    # -- Orthogonal dictionaries (identity-like) ----------------------------

    def test_orthogonal_dictionaries(self):
        """
        With orthogonal (identity) dictionaries, TLARS should converge
        quickly since the Gramian is the identity.
        """
        tolerance = 0.05
        shape = (5, 5, 5)
        D = [np.eye(s) for s in shape]
        rng = np.random.RandomState(50)
        Y = rng.randn(*shape)

        _, norm_r, _ = _run_tlars_validation(
            test_name="3-D Tensor – Orthogonal (Identity) Dictionaries",
            tensor_shape=shape, dict_sizes=[5, 5, 5],
            tolerance=tolerance, l0_mode=False, seed=50,
            iterations=50000, active_coefficients=5000,
            factor_matrices_override=D, Y_override=Y,
        )
        assert norm_r < tolerance, (
            f"Orthogonal dict: residual norm {norm_r:.6f} exceeds tolerance {tolerance}"
        )

    # -- 4-D tensor ---------------------------------------------------------

    def test_4d_tensor(self):
        """TLARS should work on 4-D tensors as well."""
        tolerance = 0.15
        Y, D = _create_synthetic_data(tensor_shape=(4, 4, 4, 4), dict_sizes=[5, 5, 5, 5], seed=60)

        model, norm_r, _ = _run_tlars_validation(
            test_name="4-D Tensor – L1 Mode",
            tensor_shape=(4, 4, 4, 4), dict_sizes=[5, 5, 5, 5],
            tolerance=tolerance, l0_mode=False, seed=60,
            iterations=10000, active_coefficients=1000,
            factor_matrices_override=D, Y_override=Y,
        )

        pred_shape_ok = model.predict(D).shape == Y.shape
        print(f"  Predict shape match    = {pred_shape_ok}")
        print(f"{'─'*70}")

        assert norm_r < tolerance, (
            f"4-D L1: residual norm {norm_r:.6f} exceeds tolerance {tolerance}"
        )
        assert pred_shape_ok

    # -- Tight tolerance convergence ----------------------------------------

    def test_tight_tolerance(self):
        """
        With enough iterations, TLARS should reach a very small residual
        on data that has an exact sparse representation.
        """
        tolerance = 0.01
        _, norm_r, _ = _run_tlars_validation(
            test_name="3-D Tensor – Tight Tolerance (0.01)",
            tensor_shape=(6, 6, 6), dict_sizes=[8, 8, 8],
            tolerance=tolerance, l0_mode=False, seed=70,
            iterations=50000, active_coefficients=5000,
        )
        assert norm_r < tolerance, (
            f"Tight tolerance: residual norm {norm_r:.6f} exceeds tolerance {tolerance}"
        )

    # -- Torch backend (if available) ---------------------------------------

    @pytest.mark.skipif(
        not _torch_available(),
        reason="PyTorch not installed",
    )
    def test_3d_torch_residual_below_tolerance(self):
        """TLARS with torch backend should also pass the residual validation."""
        import torch

        tolerance = 0.10
        Y_np, D_np = _create_synthetic_data(tensor_shape=(6, 6, 6), dict_sizes=[8, 8, 8], seed=80)

        Y_t = torch.from_numpy(Y_np).double()
        D_t = [torch.from_numpy(d).double() for d in D_np]

        print(f"\n{'='*70}")
        print(f"  3-D Tensor – Torch Backend")
        print(f"{'='*70}")
        print(f"Running TLARS Kronecker L1 until norm of the residual reaches {tolerance}")
        print(f"  Tensor shape        : (6, 6, 6)")
        print(f"  Dictionary sizes    : [8, 8, 8]")
        print(f"  Backend             : torch (cpu)")

        model = TLARS(backend='torch', device='cpu', tolerance=tolerance,
                      l0_mode=False, iterations=10000, active_coefficients=1000)
        t_start = time.perf_counter()
        model.fit(D_t, Y_t)
        t_elapsed = time.perf_counter() - t_start

        # Validate using numpy conversion
        active_cols = model.active_columns_
        coefs = model.coef_
        if hasattr(active_cols, 'cpu'):
            active_cols = active_cols.cpu().numpy()
        if hasattr(coefs, 'cpu'):
            coefs = coefs.cpu().numpy()

        D_normed = [_normc(d.copy()) for d in D_np]
        core_tensor_shape = [d.shape[1] for d in D_normed]

        Ax = tp.kronecker_matrix_vector_product(
            factor_matrices=D_normed, x=coefs,
            tensor_shape=core_tensor_shape, active_columns=active_cols,
            active_indices=None, use_transpose=False,
        )
        y = tp.vectorize(Y_np)
        y = y / np.linalg.norm(y)
        r = y - Ax
        norm_r = np.linalg.norm(r)
        converged = norm_r < tolerance

        n_active = len(model.active_columns_)
        print(f"\n  TLARS Completed.")
        print(f"  Norm of the Residual = {norm_r:.6g}")
        print(f"  Tolerance            = {tolerance}")
        print(f"  Converged            = {converged}")
        print(f"  Iterations           = {model.n_iter_}")
        print(f"  Active columns       = {n_active} / 512")
        print(f"  Elapsed time         = {t_elapsed:.3f} s")
        print(f"{'─'*70}")

        assert norm_r < tolerance, (
            f"3-D torch: residual norm {norm_r:.6f} exceeds tolerance {tolerance}"
        )
