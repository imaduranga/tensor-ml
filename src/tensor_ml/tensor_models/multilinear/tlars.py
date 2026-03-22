"""Tensor Least Angle Regression and Selection (T-LARS) algorithm.

Provides the :class:`TLARS` sparse tensor regression model and its
associated :class:`TLARSConfig` parameter schema.
"""

from typing import Optional, Union, Any, Dict
import logging
import numpy as np
from pydantic import BaseModel, Field, field_validator
from tensor_ml.tensor_models.multilinear.multilinear_model import MultilinearModel
from tensor_ml.enums import BackendType
from tensor_ml.exceptions import NotFittedError, ValidationError, ShapeMismatchError

__all__ = ["TLARS", "TLARSConfig"]

logger = logging.getLogger(__name__)


class TLARSConfig(BaseModel):
    """Validated configuration for the TLARS algorithm.

    All numeric fields are validated to be positive.  ``mask_type`` must be
    one of ``'KP'`` (Kronecker Product) or ``'KR'`` (Khatri-Rao).
    """

    tolerance: float = Field(0.075, gt=0, description="Residual norm stopping threshold.")
    l0_mode: bool = Field(default=False, description="If True, use greedy L0 selection (no column removal).")
    mask_type: str = Field(default='KP', description="Column selection type: 'KP' or 'KR'.")
    debug_mode: bool = Field(default=False, description="If True, emit per-iteration DEBUG-level log messages.")
    active_coefficients: int = Field(int(1e6), gt=0, description="Maximum number of active (non-zero) coefficients.")
    iterations: int = Field(int(1e6), gt=0, description="Maximum number of LARS iterations.")
    precision_factor: int = Field(5, gt=0, description="Multiplier for machine epsilon used as numerical precision.")
    show_progress: bool = Field(default=False, description="If True, display a tqdm progress bar during fitting (requires tqdm).")
    backend: Optional[Union[str, BackendType]] = Field(default=None, description="Backend to use ('numpy', 'torch', etc.). Inferred from data if None.")
    device: Optional[str] = Field(default=None, description="Device hint for the backend (e.g., 'cpu', 'cuda').")

    @field_validator('mask_type')
    @classmethod
    def check_mask_type(cls, v: str) -> str:
        if v not in ('KP', 'KR'):
            raise ValueError("mask_type must be 'KP' or 'KR'")
        return v


class TLARS(MultilinearModel):
    """Tensor Least Angle Regression and Selection (T-LARS).

    Solves sparse tensor recovery problems by iteratively selecting columns
    from a Kronecker-structured dictionary via the LARS/LASSO path.
    Supports both L0 (greedy) and L1 (LASSO) modes.

    Parameters are validated via :class:`TLARSConfig` (Pydantic) and
    accessible through ``self.config``.

    Parameters
    ----------
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
        Maximum number of LARS iterations.
    precision_factor : int, default=5
        Multiplier for machine epsilon.
    show_progress : bool, default=False
        Display a ``tqdm`` progress bar during fitting (requires ``tqdm``).
    backend : str or BackendType, optional
        Backend to use.  Inferred from data if ``None``.
    device : str, optional
        Device hint (e.g. ``'cpu'``, ``'cuda'``).

    Attributes
    ----------
    config : TLARSConfig
        Validated parameter configuration.
    coef_tensor_ : array-like or None
        Coefficient tensor after fitting.
    active_columns_ : array-like or None
        Indices of active dictionary columns after fitting.
    coef_ : array-like or None
        Coefficient vector for active columns.
    norm_r_ : list[float] or None
        Residual norm history across iterations.
    n_iter_ : int
        Number of iterations run.
    """

    def __init__(self, **kwargs: Any) -> None:
        config = TLARSConfig(**kwargs)
        super().__init__(backend=config.backend, device=config.device)
        self.config = config
        self.coef_tensor_ = None
        self.active_columns_ = None
        self.coef_ = None
        self.norm_r_ = None
        self.n_iter_ = 0
        self.tensor_norm_ = None
        self.precision = self.config.precision_factor * np.finfo(float).eps

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v!r}" for k, v in self.config.model_dump().items()
                          if v != TLARSConfig.model_fields[k].default)
        return f"TLARS({params})" if params else "TLARS()"

    # ------------------------------------------------------------------
    # Parameter introspection (scikit-learn-style)
    # ------------------------------------------------------------------

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, return parameters for contained sub-objects.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        return self.config.model_dump()

    def set_params(self, **params: Any) -> 'TLARS':
        """Set the parameters of this estimator.

        Parameters
        ----------
        **params
            Estimator parameters.

        Returns
        -------
        self : TLARS
        """
        self.config = self.config.model_copy(update=params)
        if 'backend' in params:
            self.backend = self.config.backend
        if 'device' in params:
            self._device_hint = self.config.device
        self.precision = self.config.precision_factor * np.finfo(float).eps
        return self

    # ------------------------------------------------------------------
    # Backend / device management
    # ------------------------------------------------------------------

    def to(self, backend: Optional[str] = None, device: Optional[str] = None) -> 'TLARS':
        """Move the model to a different backend or device.

        Parameters
        ----------
        backend : str, optional
            Backend name (e.g. ``'numpy'``, ``'torch'``).
        device : str, optional
            Device hint (e.g. ``'cpu'``, ``'cuda'``).

        Returns
        -------
        self : TLARS
            The model instance (for chaining).
        """
        if backend is not None:
            self.config = self.config.model_copy(update={'backend': backend})
            if isinstance(backend, BackendType):
                self.backend = backend
            else:
                self.backend = BackendType(str(backend).lower())
            self._setup_ops()
        if device is not None:
            self.config = self.config.model_copy(update={'device': device})
            self._device_hint = device
            self._setup_ops()
        return self

    def cpu(self) -> 'TLARS':
        """Move the model to CPU.

        Returns
        -------
        self : TLARS
        """
        return self.to(device='cpu')

    def cuda(self) -> 'TLARS':
        """Move the model to CUDA.

        Returns
        -------
        self : TLARS
        """
        return self.to(backend='torch', device='cuda')

    # ------------------------------------------------------------------
    # Fit / predict
    # ------------------------------------------------------------------

    def fit(
        self,
        factor_matrices: list,
        Y: Any,
        coef_tensor: Any = None,
    ) -> 'TLARS':
        """Fit the TLARS model to tensor data.

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
        self : TLARS
            Returns the fitted model instance.
        """
        if not isinstance(factor_matrices, list) or len(factor_matrices) == 0:
            raise ValidationError("factor_matrices must be a non-empty list of matrices")
        if Y is None:
            raise ValidationError("Y (target tensor) must not be None")

        # Unpack config for readability
        tolerance = self.config.tolerance
        l0_mode = self.config.l0_mode
        mask_type = self.config.mask_type
        debug_mode = self.config.debug_mode
        active_coefficients = self.config.active_coefficients
        iterations = self.config.iterations
        precision_factor = self.config.precision_factor

        # Resolve backend from data if not set explicitly
        self._resolve_backend(Y)

        # Normalize input (also handles device placement)
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

        # Precision settings
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

            # Check orthogonality
            if is_orthogonal:
                G_check = self.tp.tround(G_n, 10)
                I_n = self.ops.eye(G_n.shape[0])
                if not self.ops.allclose(G_check, I_n):
                    is_orthogonal = False

        total_column_count = int(np.prod(core_tensor_shape))

        logger.debug(
            "TLARS setup: backend=%s, tensor_shape=%s, core_tensor_shape=%s, "
            "total_columns=%d, orthogonal=%s, l0_mode=%s, mask_type=%s",
            self.backend, tensor_shape, core_tensor_shape,
            total_column_count, is_orthogonal, l0_mode, mask_type,
        )

        # Normalize Y
        tensor_norm = self.ops.norm(Y)
        Y = Y / tensor_norm
        Y_vec = self.ops.flatten(Y)
        r = self.ops.copy(Y_vec)
        norm_r = [float(self.ops.to_scalar(self.ops.norm(r)))]

        logger.debug(
            "TLARS normalisation: tensor_norm=%.6g, initial ||r||=%.6g",
            float(tensor_norm), norm_r[0],
        )

        # Mask type logic (KR/KP)
        column_mask_indices = []
        if mask_type == 'KR':
            if all(x == core_tensor_shape[0] for x in core_tensor_shape):
                tensor_indices = tuple(1 for _ in range(order))
                stride = self.tp.get_vector_index(tensor_indices, core_tensor_shape)
                kr_columns = list(range(0, total_column_count, stride))
                column_mask_indices = [i for i in range(total_column_count) if i not in kr_columns]
            else:
                raise ShapeMismatchError("Column dimensions of the dictionary matrices should be equal for Khatri-Rao Product.")

        # Initial correlation
        C = self.tp.full_multilinear_product(Y, factor_matrices, use_transpose=True)
        c = self.ops.flatten(C)
        del C
        if column_mask_indices:
            c[column_mask_indices] = 0
        c = self.tp.tround(c, precision_order)

        lambda_value = float(self.ops.to_scalar(self.ops.max(self.ops.abs(c))))
        changed_dict_column_index = int(self.ops.argmax(self.ops.abs(c)))

        # Initial active set
        add_column_flag = True
        active_columns = self.ops.asarray([changed_dict_column_index])
        coef_ = self.ops.zeros(1)
        changed_active_column_index = 0

        # Track active factor column indices for sub-tensor optimization
        active_factor_column_indices = [[] for _ in range(order)]
        col_indices = self.tp.get_kronecker_factor_column_indices(changed_dict_column_index, core_tensor_shape)
        for n in range(order):
            active_factor_column_indices[n] = sorted(set(active_factor_column_indices[n]) | {int(col_indices[n])})

        GInv = self.ops.ones((1, 1))

        if coef_tensor is not None:
            coef_tensor_flat = self.ops.flatten(self.ops.asarray(coef_tensor))
            nonzero_indices = self.ops.nonzero(coef_tensor_flat)
            if len(nonzero_indices) > 0:
                active_columns = nonzero_indices
                coef_ = coef_tensor_flat[nonzero_indices]

                # Rebuild active_factor_column_indices from ALL active columns
                active_factor_column_indices = [set() for _ in range(order)]
                for col in active_columns:
                    ci = self.tp.get_kronecker_factor_column_indices(int(col), core_tensor_shape)
                    for n in range(order):
                        active_factor_column_indices[n].add(int(ci[n]))
                active_factor_column_indices = [sorted(s) for s in active_factor_column_indices]

                # Recompute GInv from full Gramian for warm start
                GI = self.tp.get_gramian(gramians, active_columns, core_tensor_shape)
                GInv = self.ops.pinv(GI)

        n_iter = 0
        iter_range = range(int(iterations))

        # Optional tqdm progress bar
        _pbar = None
        if self.config.show_progress:
            try:
                from tqdm import tqdm
                iter_range = tqdm(iter_range, desc="T-LARS", unit="iter", leave=True)
                _pbar = iter_range
            except ImportError:
                logger.warning("tqdm not installed — progress bar disabled. Install with: pip install tqdm")

        for t in iter_range:
            n_iter = t + 1
            n_active = len(active_columns)

            if debug_mode:
                logger.debug(
                    "iter %d: n_active=%d, lambda=%.6g", n_iter, n_active, lambda_value,
                )

            zI = self.ops.sign(c[active_columns])

            if n_active > 1 and not is_orthogonal:
                dI, GInv = self.tp.get_direction_vector(
                    GInv=GInv, zI=zI, gramians=gramians, active_columns=active_columns,
                    add_column_flag=add_column_flag,
                    changed_dict_column_index=changed_dict_column_index,
                    changed_active_column_index=changed_active_column_index,
                    tensor_shape=core_tensor_shape, precision_order=precision_order,
                )

                # Fallback if direction vector has NaN values
                if self.ops.has_nan(dI):
                    logger.debug("iter %d: NaN in direction vector, recomputing via pinv", n_iter)
                    GI = self.tp.get_gramian(gramians, active_columns, core_tensor_shape)
                    GInv = self.ops.pinv(GI)
                    dI = GInv @ zI
            elif is_orthogonal:
                dI = zI
            else:
                GInv = self.ops.ones((1, 1))
                dI = zI

            # Compute equicorrelation vector v = G @ A @ dI
            v = self.tp.kronecker_matrix_vector_product(
                factor_matrices=gramians, x=dI,
                tensor_shape=core_tensor_shape, active_columns=active_columns,
                active_indices=active_factor_column_indices,
                use_transpose=False,
            )
            if column_mask_indices:
                v[column_mask_indices] = 0
            v = self.tp.tround(v, precision_order)
            v[active_columns] = self.ops.sign(v[active_columns])  # Enforce equicorrelation

            # Calculate delta_plus
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

            # Calculate delta_minus for L1 minimization
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

            # Check stopping conditions
            if lambda_value < delta or lambda_value < 0 or delta < 0:
                if debug_mode:
                    logger.debug(
                        "iter %d: stopping — lambda=%.6g, delta=%.6g",
                        n_iter, lambda_value, delta,
                    )
                break

            # Update solution
            coef_ = coef_ + delta * dI
            lambda_value = float(self.tp.tround(np.array([lambda_value - delta]), precision_order)[0])
            c = c - delta * v
            c[active_columns] = lambda_value * self.ops.sign(c[active_columns])

            # Update residual
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

            # Stopping criteria
            if nr < tolerance or n_active >= active_coefficients:
                if debug_mode:
                    reason = "tolerance" if nr < tolerance else "max active coefficients"
                    logger.debug(
                        "iter %d: stopping — %s (||r||=%.6g, n_active=%d)",
                        n_iter, reason, nr, n_active,
                    )
                break

            # Add or remove column from the active set
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

            # Update active factor column indices
            col_indices = self.tp.get_kronecker_factor_column_indices(changed_dict_column_index, core_tensor_shape)
            for n in range(order):
                active_factor_column_indices[n] = sorted(set(active_factor_column_indices[n]) | {int(col_indices[n])})

        # Store results
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
            "TLARS finished: iterations=%d, final ||r||=%.6g, n_active=%d",
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

