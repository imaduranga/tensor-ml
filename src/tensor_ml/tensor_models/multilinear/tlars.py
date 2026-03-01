"""Tensor Least Angle Regression and Selection (T-LARS) algorithm."""

from typing import Optional, Union, Any
import logging
import numpy as np
from tensor_ml.tensor_models.multilinear.multilinear_model import MultilinearModel
from tensor_ml.enums import BackendType

__all__ = ["TLARS"]

logger = logging.getLogger(__name__)


class TLARS(MultilinearModel):
    """Tensor Least Angle Regression and Selection (T-LARS).

    Solves sparse tensor recovery problems by iteratively selecting columns
    from a Kronecker-structured dictionary via the LARS/LASSO path.
    Supports both L0 (greedy) and L1 (LASSO) modes.

    Parameters
    ----------
    tolerance : float, default=0.075
        Residual norm stopping threshold.
    l0_mode : bool, default=False
        If ``True``, use greedy L0 selection (no column removal).
        If ``False``, use L1 (LASSO) with column add/remove.
    mask_type : str, default='KP'
        Column selection type: ``'KP'`` (Kronecker Product) or ``'KR'``
        (Khatri-Rao, restricts to diagonal Kronecker columns).
    debug_mode : bool, default=False
        When ``True``, emit per-iteration ``DEBUG``-level log messages
        via the ``tensor_ml.tensor_models.multilinear.tlars`` logger.
    active_coefficients : int, default=1_000_000
        Maximum number of active (non-zero) coefficients.
    iterations : int, default=1_000_000
        Maximum number of LARS iterations.
    precision_factor : int, default=5
        Multiplier for machine epsilon used as numerical precision.
    backend : str | BackendType, optional
        Backend to use. Inferred from data if ``None``.
    device : str | torch.device, optional
        Device hint for the PyTorch backend.
    """

    def __init__(
        self,
        tolerance: float = 0.075,
        l0_mode: bool = False,
        mask_type: str = 'KP',
        debug_mode: bool = False,
        active_coefficients: int = int(1e6),
        iterations: int = int(1e6),
        precision_factor: int = 5,
        backend: Optional[Union[str, BackendType]] = None,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(backend=backend, device=device)
        self.tolerance = tolerance
        self.l0_mode = l0_mode
        self.mask_type = mask_type
        self.debug_mode = debug_mode
        self.active_coefficients = active_coefficients
        self.iterations = iterations
        self.precision_factor = precision_factor
        self.coef_tensor_ = None  # Solution tensor
        self.active_columns_ = None
        self.coef_ = None  # Solution vector
        self.norm_r_ = None
        self.precision = self.precision_factor * np.finfo(float).eps

    def fit(
        self,
        factor_matrices: list,
        Y: Any,
        coef_tensor: Any = None,
    ) -> 'TLARS':
        """
        Fits the TLARS model to the provided tensor data.

        Parameters
        ----------
        factor_matrices : List
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
        # Resolve backend from data if not set explicitly
        self._resolve_backend(Y)

        # Normalize input (also handles device placement)
        Y = self.normalize_input(Y)
        factor_matrices = [self.normalize_input(D) for D in factor_matrices]

        order = Y.ndim
        if len(factor_matrices) != order:
            raise ValueError(
                f"Number of factor matrices ({len(factor_matrices)}) must match "
                f"the number of tensor modes ({order})."
            )

        tensor_shape = Y.shape
        core_tensor_shape = []
        gramians = []

        # Precision settings
        precision = self.precision_factor * np.finfo(float).eps
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
            total_column_count, is_orthogonal, self.l0_mode, self.mask_type,
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
        if self.mask_type == 'KR':
            if all(x == core_tensor_shape[0] for x in core_tensor_shape):
                tensor_indices = tuple(1 for _ in range(order))
                stride = self.tp.get_vector_index(tensor_indices, core_tensor_shape)
                kr_columns = list(range(0, total_column_count, stride))
                column_mask_indices = [i for i in range(total_column_count) if i not in kr_columns]
            else:
                raise ValueError("Column dimensions of the dictionary matrices should be equal for Khatri-Rao Product.")

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
        for t in range(int(self.iterations)):
            n_iter = t + 1
            n_active = len(active_columns)

            if self.debug_mode:
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
            if not self.l0_mode:
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
                if self.debug_mode:
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

            # Stopping criteria
            if nr < self.tolerance or n_active >= self.active_coefficients:
                if self.debug_mode:
                    reason = "tolerance" if nr < self.tolerance else "max active coefficients"
                    logger.debug(
                        "iter %d: stopping — %s (||r||=%.6g, n_active=%d)",
                        n_iter, reason, nr, n_active,
                    )
                break

            # Add or remove column from the active set
            if add_column_flag:
                if self.debug_mode:
                    logger.debug("iter %d: adding column %d", n_iter, changed_dict_column_index)
                active_columns = self.ops.concatenate([active_columns, self.ops.asarray([changed_dict_column_index])])
                coef_ = self.ops.concatenate([coef_, self.ops.zeros(1)])
                changed_active_column_index = self.ops.numel(coef_) - 1
            else:
                if self.debug_mode:
                    logger.debug("iter %d: removing column %d", n_iter, changed_dict_column_index)
                changed_active_column_index = self.ops.find_index(active_columns, changed_dict_column_index)
                coef_ = self.ops.concatenate([coef_[:changed_active_column_index], coef_[changed_active_column_index + 1:]])
                active_columns = self.ops.concatenate([active_columns[:changed_active_column_index], active_columns[changed_active_column_index + 1:]])

            # Update active factor column indices
            col_indices = self.tp.get_kronecker_factor_column_indices(changed_dict_column_index, core_tensor_shape)
            for n in range(order):
                active_factor_column_indices[n] = sorted(set(active_factor_column_indices[n]) | {int(col_indices[n])})

        # Store results
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
        """
        Predicts the target tensor using the learned coefficients and provided factor matrices.

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
            raise ValueError("Model is not fitted yet. Call 'fit' before 'predict'.")
        normalized_matrices = []
        for D in X:
            D_normed = self.normalize_input(D)
            D_final = self.ops.normalize(D_normed)
            normalized_matrices.append(D_final)
        Y_pred = self.tp.full_multilinear_product(self.coef_tensor_, normalized_matrices, use_transpose=False)
        Y_pred = Y_pred * self.tensor_norm_
        return Y_pred

