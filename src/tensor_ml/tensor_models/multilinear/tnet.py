"""Tensor Elastic NET (T-NET) algorithm.

Provides the :class:`TNET` sparse tensor regression model and its
associated :class:`TNETConfig` parameter schema.

T-NET extends T-LARS with an L2 regularisation term (``lambda2``) added to
the Gramian in the direction-vector update, yielding a combined L1+L2
(Elastic Net) penalty. The output coefficients are rescaled by
``sqrt(1 + lambda2)`` following the standard Elastic Net normalisation.

Reference
---------
Wickramasingha, I. — *Tensor Elastic NET* (NAMSP 2025).
MATLAB source: ``src/tensor_ml/matlab/TNET/TNET.m``
"""

from typing import Optional, Union, Any, Dict
import logging
import math
import numpy as np
from pydantic import Field
from tensor_ml.tensor_models.multilinear.tlars import TLARS, TLARSConfig
from tensor_ml.enums import BackendType
from tensor_ml.exceptions import NotFittedError, ValidationError, ShapeMismatchError

__all__ = ["TNET", "TNETConfig"]

logger = logging.getLogger(__name__)


class TNETConfig(TLARSConfig):
    """Validated configuration for the T-NET algorithm.

    Extends :class:`TLARSConfig` with the Elastic Net L2 regularisation
    coefficient ``lambda2``.
    """

    lambda2: float = Field(
        0.1,
        gt=0,
        description="L2 regularisation coefficient for the Elastic Net penalty.",
    )


class TNET(TLARS):
    """Tensor Elastic NET (T-NET).

    Solves sparse tensor recovery problems with a combined L1+L2 (Elastic Net)
    penalty. Extends :class:`TLARS` by incorporating an L2 regularisation term
    ``lambda2`` into the Gramian inverse update of the direction-vector step.

    The two algorithmic differences from T-LARS are:

    1. **Direction vector**: the diagonal of the active-column Gramian is
       shifted by ``lambda2`` before inversion — equivalent to the
       ``getDirectionVectorEN`` update in the MATLAB reference:

       .. math::
           \\alpha = \\frac{1}{g_{NN} + \\lambda_2 + g_{1:N-1}^T b_{1:N-1}}

       and the direction vector is scaled by ``(1 + lambda2)``.

    2. **Output rescaling**: coefficients are multiplied by
       ``sqrt(1 + lambda2)`` at the end of fitting, matching the standard
       Elastic Net normalisation.

    Parameters
    ----------
    lambda2 : float, default=0.1
        L2 regularisation coefficient.  Must be > 0.
    tolerance : float, default=0.075
        Residual norm stopping threshold.
    l0_mode : bool, default=False
        If ``True``, use greedy L0 selection (no column removal).
    mask_type : str, default='KP'
        ``'KP'`` (Kronecker Product) or ``'KR'`` (Khatri-Rao).
    debug_mode : bool, default=False
        Emit per-iteration DEBUG-level log messages.
    active_coefficients : int, default=1_000_000
        Maximum number of active (non-zero) coefficients.
    iterations : int, default=1_000_000
        Maximum number of iterations.
    precision_factor : int, default=5
        Multiplier for machine epsilon.
    show_progress : bool, default=False
        Display a ``tqdm`` progress bar during fitting.
    backend : str or BackendType, optional
        Backend to use.  Inferred from data if ``None``.
    device : str, optional
        Device hint (e.g. ``'cpu'``, ``'cuda'``).

    Attributes
    ----------
    config : TNETConfig
        Validated parameter configuration.
    coef_tensor_ : array-like or None
        Coefficient tensor after fitting (rescaled by ``sqrt(1 + lambda2)``).
    active_columns_ : array-like or None
        Indices of active dictionary columns after fitting.
    coef_ : array-like or None
        Coefficient vector for active columns (rescaled).
    norm_r_ : list[float] or None
        Residual norm history across iterations.
    n_iter_ : int
        Number of iterations run.
    """

    def __init__(self, **kwargs: Any) -> None:
        # Build a TNETConfig — pulls lambda2 out, passes rest to TLARSConfig
        config = TNETConfig(**kwargs)
        # Initialise TLARS with all shared params (excluding lambda2)
        tlars_params = {k: v for k, v in config.model_dump().items() if k != 'lambda2'}
        super().__init__(**tlars_params)
        # Replace the TLARSConfig set by TLARS.__init__ with the full TNETConfig
        self.config = config

    def __repr__(self) -> str:
        params = ", ".join(
            f"{k}={v!r}"
            for k, v in self.config.model_dump().items()
            if v != TNETConfig.model_fields[k].default
        )
        return f"TNET({params})" if params else "TNET()"

    # ------------------------------------------------------------------
    # Parameter introspection (scikit-learn-style)
    # ------------------------------------------------------------------

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return self.config.model_dump()

    def set_params(self, **params: Any) -> 'TNET':
        self.config = self.config.model_copy(update=params)
        if 'backend' in params:
            self.backend = self.config.backend
        if 'device' in params:
            self._device_hint = self.config.device
        self.precision = self.config.precision_factor * np.finfo(float).eps
        return self

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        factor_matrices: list,
        Y: Any,
        coef_tensor: Any = None,
    ) -> 'TNET':
        """Fit the T-NET model to tensor data.

        Runs the T-LARS/LASSO path with an additional L2 regularisation term
        ``lambda2`` in the Gramian inverse update, then rescales the output
        coefficients by ``sqrt(1 + lambda2)``.

        Parameters
        ----------
        factor_matrices : list
            List of factor (dictionary) matrices for each tensor mode.
        Y : array-like
            Target tensor to approximate.
        coef_tensor : array-like, optional
            Optional initial coefficient tensor for warm start.

        Returns
        -------
        self : TNET
            Returns the fitted model instance.
        """
        if not isinstance(factor_matrices, list) or len(factor_matrices) == 0:
            raise ValidationError("factor_matrices must be a non-empty list of matrices")
        if Y is None:
            raise ValidationError("Y (target tensor) must not be None")

        lambda2 = self.config.lambda2
        tolerance = self.config.tolerance
        l0_mode = self.config.l0_mode
        mask_type = self.config.mask_type
        debug_mode = self.config.debug_mode
        active_coefficients = self.config.active_coefficients
        iterations = self.config.iterations
        precision_factor = self.config.precision_factor

        self._resolve_backend(Y)

        Y = self.normalize_input(Y)
        factor_matrices = [self.normalize_input(D) for D in factor_matrices]

        order = Y.ndim
        if len(factor_matrices) != order:
            raise ShapeMismatchError(
                f"Number of factor matrices ({len(factor_matrices)}) must match "
                f"the number of tensor modes ({order})."
            )

        tensor_shape = Y.shape
        core_tensor_shape = []
        gramians = []

        precision = precision_factor * np.finfo(float).eps
        precision_order = round(abs(np.log10(precision)))
        is_orthogonal = True

        for n in range(order):
            D = factor_matrices[n]
            Dn = self.ops.normalize(D)
            factor_matrices[n] = Dn
            G_n = self.ops.gramian(Dn)
            G_n = self.tp.tround(G_n, precision_order)
            gramians.append(G_n)
            core_tensor_shape.append(Dn.shape[1])

            if is_orthogonal:
                G_check = self.tp.tround(G_n, 10)
                I_n = self.ops.eye(G_n.shape[0])
                if not self.ops.allclose(G_check, I_n):
                    is_orthogonal = False

        total_column_count = int(np.prod(core_tensor_shape))

        logger.debug(
            "TNET setup: backend=%s, tensor_shape=%s, core_tensor_shape=%s, "
            "total_columns=%d, orthogonal=%s, l0_mode=%s, mask_type=%s, lambda2=%.4g",
            self.backend, tensor_shape, core_tensor_shape,
            total_column_count, is_orthogonal, l0_mode, mask_type, lambda2,
        )

        tensor_norm = self.ops.norm(Y)
        Y = Y / tensor_norm
        Y_vec = self.ops.flatten(Y)
        r = self.ops.copy(Y_vec)
        norm_r = [float(self.ops.to_scalar(self.ops.norm(r)))]

        # Mask type logic (KR/KP)
        column_mask_indices = []
        if mask_type == 'KR':
            if all(x == core_tensor_shape[0] for x in core_tensor_shape):
                tensor_indices = tuple(1 for _ in range(order))
                stride = self.tp.get_vector_index(tensor_indices, core_tensor_shape)
                kr_columns = list(range(0, total_column_count, stride))
                column_mask_indices = [i for i in range(total_column_count) if i not in kr_columns]
            else:
                raise ShapeMismatchError(
                    "Column dimensions of the dictionary matrices should be equal for Khatri-Rao Product."
                )

        C = self.tp.full_multilinear_product(Y, factor_matrices, use_transpose=True)
        c = self.ops.flatten(C)
        del C
        if column_mask_indices:
            c[column_mask_indices] = 0
        c = self.tp.tround(c, precision_order)

        lambda_value = float(self.ops.to_scalar(self.ops.max(self.ops.abs(c))))
        changed_dict_column_index = int(self.ops.argmax(self.ops.abs(c)))

        add_column_flag = True
        active_columns = self.ops.asarray([changed_dict_column_index])
        coef_ = self.ops.zeros(1)
        changed_active_column_index = 0

        active_factor_column_indices = [[] for _ in range(order)]
        col_indices = self.tp.get_kronecker_factor_column_indices(changed_dict_column_index, core_tensor_shape)
        for n in range(order):
            active_factor_column_indices[n] = sorted(set(active_factor_column_indices[n]) | {int(col_indices[n])})

        # ── Elastic Net: initialise GInv with lambda2 on diagonal ──
        # For the first active column: G_active = g_11 + lambda2, so GInv = 1/(g_11 + lambda2)
        g_11 = float(self.tp.get_gramian(gramians, active_columns, core_tensor_shape)[0, 0])
        GInv = self.ops.ones((1, 1)) / (g_11 + lambda2)

        if coef_tensor is not None:
            coef_tensor_flat = self.ops.flatten(self.ops.asarray(coef_tensor))
            nonzero_indices = self.ops.nonzero(coef_tensor_flat)
            if len(nonzero_indices) > 0:
                active_columns = nonzero_indices
                coef_ = coef_tensor_flat[nonzero_indices]

                active_factor_column_indices = [set() for _ in range(order)]
                for col in active_columns:
                    ci = self.tp.get_kronecker_factor_column_indices(int(col), core_tensor_shape)
                    for n in range(order):
                        active_factor_column_indices[n].add(int(ci[n]))
                active_factor_column_indices = [sorted(s) for s in active_factor_column_indices]

                # Warm-start GInv: add lambda2*I to the active Gramian
                GI = self.tp.get_gramian(gramians, active_columns, core_tensor_shape)
                N_active = GI.shape[0]
                GI = GI + lambda2 * self.ops.eye(N_active)
                GInv = self.ops.pinv(GI)

        n_iter = 0
        iter_range = range(int(iterations))

        _pbar = None
        if self.config.show_progress:
            try:
                from tqdm import tqdm
                iter_range = tqdm(iter_range, desc="T-NET", unit="iter", leave=True)
                _pbar = iter_range
            except ImportError:
                logger.warning("tqdm not installed — progress bar disabled.")

        for t in iter_range:
            n_iter = t + 1
            n_active = len(active_columns)

            if debug_mode:
                logger.debug("iter %d: n_active=%d, lambda=%.6g", n_iter, n_active, lambda_value)

            zI = self.ops.sign(c[active_columns])

            if n_active > 1 and not is_orthogonal:
                # ── Elastic Net direction vector ──────────────────────────────
                # Uses (G + lambda2*I)^{-1} via getDirectionVectorEN
                dI, GInv = self.tp.get_direction_vector_en(
                    GInv=GInv, zI=zI, gramians=gramians, lambda2=lambda2,
                    active_columns=active_columns,
                    add_column_flag=add_column_flag,
                    changed_dict_column_index=changed_dict_column_index,
                    changed_active_column_index=changed_active_column_index,
                    tensor_shape=core_tensor_shape,
                    precision_order=precision_order,
                )

                if self.ops.has_nan(dI):
                    logger.debug("iter %d: NaN in direction vector, recomputing via pinv", n_iter)
                    GI = self.tp.get_gramian(gramians, active_columns, core_tensor_shape)
                    N_active = GI.shape[0]
                    GI = GI + lambda2 * self.ops.eye(N_active)
                    GInv = self.ops.pinv(GI)
                    dI = (1.0 + lambda2) * (GInv @ zI)
            elif is_orthogonal:
                dI = (1.0 + lambda2) * zI
            else:
                g_11 = float(self.tp.get_gramian(gramians, active_columns, core_tensor_shape)[0, 0])
                GInv = self.ops.ones((1, 1)) / (g_11 + lambda2)
                dI = (1.0 + lambda2) * zI

            v = self.tp.kronecker_matrix_vector_product(
                factor_matrices=gramians, x=dI,
                tensor_shape=core_tensor_shape, active_columns=active_columns,
                active_indices=active_factor_column_indices,
                use_transpose=False,
            )
            if column_mask_indices:
                v[column_mask_indices] = 0
            v = self.tp.tround(v, precision_order)
            v[active_columns] = self.ops.sign(v[active_columns])

            changed_dict_column_index = -1
            delta = -1.0
            add_column_flag = False

            with np.errstate(divide='ignore', invalid='ignore'):
                delta_plus_1 = (lambda_value - c) / (1 - v)
            delta_plus_1[active_columns] = self.ops.inf
            if column_mask_indices:
                delta_plus_1[column_mask_indices] = self.ops.inf
            delta_plus_1[delta_plus_1 <= precision] = self.ops.inf
            min_idx1 = self.ops.argmin(delta_plus_1)
            min_delta_plus_1 = delta_plus_1[min_idx1]

            with np.errstate(divide='ignore', invalid='ignore'):
                delta_plus_2 = (lambda_value + c) / (1 + v)
            delta_plus_2[active_columns] = self.ops.inf
            if column_mask_indices:
                delta_plus_2[column_mask_indices] = self.ops.inf
            delta_plus_2[delta_plus_2 <= precision] = self.ops.inf
            min_idx2 = self.ops.argmin(delta_plus_2)
            min_delta_plus_2 = delta_plus_2[min_idx2]

            if min_delta_plus_1 < min_delta_plus_2:
                changed_dict_column_index = int(min_idx1)
                delta = float(min_delta_plus_1)
                add_column_flag = True
            else:
                changed_dict_column_index = int(min_idx2)
                delta = float(min_delta_plus_2)
                add_column_flag = True

            if not l0_mode:
                delta_minus = -coef_ / dI
                delta_minus[delta_minus <= precision] = self.ops.inf
                col_idx3 = self.ops.argmin(delta_minus)
                min_delta_minus = float(delta_minus[col_idx3])
                min_idx3 = active_columns[col_idx3]

                if n_active > 1 and min_delta_minus < delta:
                    changed_dict_column_index = int(min_idx3)
                    delta = min_delta_minus
                    add_column_flag = False

            delta = float(self.tp.tround(np.array([delta]), precision_order)[0])

            if lambda_value < delta or lambda_value < 0 or delta < 0:
                if debug_mode:
                    logger.debug(
                        "iter %d: stopping — lambda=%.6g, delta=%.6g",
                        n_iter, lambda_value, delta,
                    )
                break

            coef_ = coef_ + delta * dI
            lambda_value = float(self.tp.tround(np.array([lambda_value - delta]), precision_order)[0])
            c = c - delta * v
            c[active_columns] = lambda_value * self.ops.sign(c[active_columns])

            ad = self.tp.kronecker_matrix_vector_product(
                factor_matrices=factor_matrices, x=dI,
                tensor_shape=core_tensor_shape, active_columns=active_columns,
                active_indices=active_factor_column_indices,
                use_transpose=False,
            )
            r = r - delta * ad
            nr = float(self.ops.to_scalar(self.ops.norm(r)))
            norm_r.append(nr)

            if _pbar is not None:
                _pbar.set_postfix({"||r||": f"{nr:.4g}", "active": n_active}, refresh=False)

            if nr < tolerance or n_active >= active_coefficients:
                if debug_mode:
                    reason = "tolerance" if nr < tolerance else "max active coefficients"
                    logger.debug(
                        "iter %d: stopping — %s (||r||=%.6g, n_active=%d)",
                        n_iter, reason, nr, n_active,
                    )
                break

            if add_column_flag:
                if debug_mode:
                    logger.debug("iter %d: adding column %d", n_iter, changed_dict_column_index)
                active_columns = self.ops.concatenate([active_columns, self.ops.asarray([changed_dict_column_index])])
                coef_ = self.ops.concatenate([coef_, self.ops.zeros(1)])
                changed_active_column_index = self.ops.numel(coef_) - 1
            else:
                if debug_mode:
                    logger.debug("iter %d: removing column %d", n_iter, changed_dict_column_index)
                changed_active_column_index = self.ops.find_index(active_columns, changed_dict_column_index)
                coef_ = self.ops.concatenate([coef_[:changed_active_column_index], coef_[changed_active_column_index + 1:]])
                active_columns = self.ops.concatenate([active_columns[:changed_active_column_index], active_columns[changed_active_column_index + 1:]])

            col_indices = self.tp.get_kronecker_factor_column_indices(changed_dict_column_index, core_tensor_shape)
            for n in range(order):
                active_factor_column_indices[n] = sorted(set(active_factor_column_indices[n]) | {int(col_indices[n])})

        # ── Elastic Net output rescaling: x = sqrt(1 + lambda2) * x ──
        en_scale = math.sqrt(1.0 + lambda2)
        coef_ = coef_ * en_scale

        if _pbar is not None:
            _pbar.close()

        self.tensor_norm_ = tensor_norm
        coef_tensor_result = self.tp.tensorize(x=coef_, tensor_shape=core_tensor_shape, active_elements=active_columns)
        self.coef_tensor_ = coef_tensor_result
        self.active_columns_ = active_columns
        self.coef_ = coef_
        self.norm_r_ = norm_r
        self.n_iter_ = n_iter

        logger.debug(
            "TNET finished: iterations=%d, final ||r||=%.6g, n_active=%d",
            n_iter, norm_r[-1], len(active_columns),
        )

        return self

    def predict(
        self,
        X: Any,
        **kwargs: Any,
    ) -> Any:
        """Predict the target tensor using the learned coefficients.

        Parameters
        ----------
        X : list
            List of factor (dictionary) matrices for each tensor mode.

        Returns
        -------
        Y_pred : array-like
            The predicted tensor.
        """
        if self.coef_tensor_ is None:
            raise NotFittedError("Model is not fitted yet. Call 'fit' before 'predict'.")
        normalized_matrices = []
        for D in X:
            D_normed = self.normalize_input(D)
            D_final = self.ops.normalize(D_normed)
            normalized_matrices.append(D_final)
        Y_pred = self.tp.full_multilinear_product(self.coef_tensor_, normalized_matrices, use_transpose=False)
        Y_pred = Y_pred * self.tensor_norm_
        return Y_pred
