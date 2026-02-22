from typing import Optional, List, Union
import numpy as np
from tensor_ml.tensor_models.multilinear.multilinear_model import MultilinearModel
from tensor_ml.enums import BackendType
import tensor_ml.tensorops._tensor_products as tp


class TLARS(MultilinearModel):
    def __init__(
        self,
        tolerance: float = 0.075,
        l0_mode: bool = False,
        mask_type: str = 'KP',
        debug_mode: bool = False,
        path: str = '',
        active_coefficients: int = int(1e6),
        iterations: int = int(1e6),
        precision_factor: int = 5,
        plot_frequency: int = 100,
        backend: Optional[Union[str, BackendType]] = None,
        device: Optional[Union[str, object]] = "cuda",
    ):
        super().__init__(backend=backend, device=device)
        self.tolerance = tolerance
        self.l0_mode = l0_mode
        self.mask_type = mask_type
        self.debug_mode = debug_mode
        self.path = path
        self.active_coefficients = active_coefficients
        self.iterations = iterations
        self.precision_factor = precision_factor
        self.plot_frequency = plot_frequency
        self.coef_tensor_ = None  # Solution tensor
        self.active_columns_ = None
        self.coef_ = None  # Solution vector
        self.norm_r_ = None
        self.precision = self.precision_factor * np.finfo(float).eps

    def fit(
        self,
        factor_matrices: List[Union[np.ndarray, object]],
        Y: Union[np.ndarray, object],
        coef_tensor: Optional[Union[np.ndarray, object]] = None,
    ):
        """
        Fits the TLARS model to the provided tensor data.

        Parameters
        ----------
        factor_matrices : List[Union[np.ndarray, object]]
            List of factor (dictionary) matrices for each tensor mode. Each matrix should have shape (mode_dim, dict_size).
        Y : Union[np.ndarray, object]
            Target tensor to approximate, as a NumPy array or backend-specific tensor.
        coef_tensor : Optional[Union[np.ndarray, object]], default=None
            Optional initial coefficient tensor. If provided, nonzero entries are used to initialize the active set.

        Returns
        -------
        self : TLARS
            Returns the fitted model instance.
        """
        # Normalize input
        Y = self.normalize_input(Y)
        factor_matrices = [self.normalize_input(D) for D in factor_matrices]
        Y = self.ops.to_device(Y, self.device)
        factor_matrices = [self.ops.to_device(D, self.device) for D in factor_matrices]

        order = Y.ndim
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
            G_n = tp.tround(G_n, precision_order)
            gramians.append(G_n)
            core_tensor_shape.append(Dn.shape[1])

            # Check orthogonality
            if is_orthogonal:
                G_check = tp.tround(G_n, 10)
                I_n = np.eye(G_n.shape[0])
                if self.backend == BackendType.TORCH:
                    G_check = G_check.cpu().numpy()
                else:
                    G_check = np.asarray(G_check)
                if not np.allclose(G_check, np.round(I_n, 10)):
                    is_orthogonal = False

        total_column_count = int(np.prod(core_tensor_shape))

        # Normalize Y
        tensor_norm = self.ops.norm(Y)
        Y = Y / tensor_norm
        Y_vec = self.ops.flatten(Y)
        r = Y_vec.copy() if self.backend == BackendType.NUMPY else Y_vec.clone()
        norm_r = [float(self.ops.norm(r).item()) if self.backend == BackendType.TORCH else float(self.ops.norm(r))]

        # Mask type logic (KR/KP)
        column_mask_indices = []
        if self.mask_type == 'KR':
            if all(x == core_tensor_shape[0] for x in core_tensor_shape):
                tensor_indices = tuple(1 for _ in range(order))  # 0-based second element
                stride = tp.get_vector_index(tensor_indices, core_tensor_shape)
                kr_columns = list(range(0, total_column_count, stride))
                column_mask_indices = [i for i in range(total_column_count) if i not in kr_columns]
            else:
                raise ValueError("Column dimensions of the dictionary matrices should be equal for Khatri-Rao Product.")

        # Initial correlation
        C = tp.full_multilinear_product(Y, factor_matrices, use_transpose=True)
        c = self.ops.flatten(C)
        del C
        if column_mask_indices:
            c[column_mask_indices] = 0
        c = tp.tround(c, precision_order)

        if self.backend == BackendType.TORCH:
            lambda_value = float(self.ops.abs(c).max().item())
        else:
            lambda_value = float(self.ops.abs(c).max())
        changed_dict_column_index = int(self.ops.argmax(self.ops.abs(c)))

        # Initial active set
        add_column_flag = True
        active_columns = self.ops.asarray([changed_dict_column_index])
        coef_ = self.ops.zeros(1)
        changed_active_column_index = 0  # 0-indexed

        # Track active factor column indices for sub-tensor optimization
        active_factor_column_indices = [[] for _ in range(order)]
        col_indices = tp.get_kronecker_factor_column_indices(changed_dict_column_index, core_tensor_shape)
        for n in range(order):
            active_factor_column_indices[n] = sorted(set(active_factor_column_indices[n]) | {int(col_indices[n])})

        GInv = self.ops.ones((1, 1))

        if coef_tensor is not None:
            coef_tensor_flat = self.ops.flatten(self.ops.asarray(coef_tensor))
            nonzero_indices = self.ops.nonzero(coef_tensor_flat)
            if len(nonzero_indices) > 0:
                active_columns = nonzero_indices
                coef_ = coef_tensor_flat[nonzero_indices]
                # Recompute GInv from full Gramian for warm start
                GI = tp.get_gramian(gramians, active_columns, core_tensor_shape, device=self.device)
                if self.backend == BackendType.NUMPY:
                    GInv = np.linalg.pinv(np.asarray(GI))
                else:
                    import torch as _torch
                    GInv = _torch.linalg.pinv(GI)

        n_iter = 0
        for t in range(int(self.iterations)):
            n_iter = t + 1
            n_active = len(active_columns)
            zI = self.ops.sign(c[active_columns])

            if n_active > 1 and not is_orthogonal:
                dI, GInv = tp.get_direction_vector(
                    GInv=GInv, zI=zI, G=gramians, active_columns=active_columns,
                    add_column_flag=add_column_flag,
                    changed_dict_column_index=changed_dict_column_index,
                    changed_active_column_index=changed_active_column_index,
                    tensor_shape=core_tensor_shape, precision_order=precision_order,
                    device=self.device
                )

                # Fallback if direction vector has NaN values
                has_nan = bool(np.any(np.isnan(dI))) if self.backend == BackendType.NUMPY else bool(dI.isnan().any())
                if has_nan:
                    GI = tp.get_gramian(gramians, active_columns, core_tensor_shape, device=self.device)
                    if self.backend == BackendType.NUMPY:
                        GInv = np.linalg.pinv(np.asarray(GI))
                    else:
                        import torch as _torch
                        GInv = _torch.linalg.pinv(GI)
                    dI = GInv @ zI
            elif is_orthogonal:
                dI = zI
            else:
                GInv = self.ops.ones((1, 1))
                dI = zI

            # Compute equicorrelation vector v = G @ A @ dI
            v = tp.kronecker_matrix_vector_product(
                factor_matrices=gramians, x=dI,
                tensor_shape=core_tensor_shape, active_columns=active_columns,
                active_indices=active_factor_column_indices,
                use_transpose=False, device=self.device
            )
            if column_mask_indices:
                v[column_mask_indices] = 0
            v = tp.tround(v, precision_order)
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

            delta = float(tp.tround(np.array([delta]), precision_order)[0])

            # Check stopping conditions
            if lambda_value < delta or lambda_value < 0 or delta < 0:
                break

            # Update solution
            coef_ = coef_ + delta * dI
            lambda_value = float(tp.tround(np.array([lambda_value - delta]), precision_order)[0])
            c = c - delta * v
            c[active_columns] = lambda_value * self.ops.sign(c[active_columns])  # Enforce equicorrelation

            # Update residual
            ad = tp.kronecker_matrix_vector_product(
                factor_matrices=factor_matrices, x=dI,
                tensor_shape=core_tensor_shape, active_columns=active_columns,
                active_indices=active_factor_column_indices,
                use_transpose=False, device=self.device
            )
            r = r - delta * ad
            nr = float(self.ops.norm(r).item()) if self.backend == BackendType.TORCH else float(self.ops.norm(r))
            norm_r.append(nr)

            # Stopping criteria
            if nr < self.tolerance or n_active >= self.active_coefficients:
                break

            # Add or remove column from the active set
            if add_column_flag:
                active_columns = self.ops.concatenate([active_columns, self.ops.asarray([changed_dict_column_index])])
                coef_ = self.ops.concatenate([coef_, self.ops.zeros(1)])
                changed_active_column_index = len(coef_) - 1 if self.backend == BackendType.NUMPY else int(coef_.size(0)) - 1
            else:
                if self.backend == BackendType.NUMPY:
                    changed_active_column_index = int(np.where(active_columns == changed_dict_column_index)[0][0])
                else:
                    changed_active_column_index = int((active_columns == changed_dict_column_index).nonzero()[0])
                coef_ = self.ops.concatenate([coef_[:changed_active_column_index], coef_[changed_active_column_index + 1:]])
                active_columns = self.ops.concatenate([active_columns[:changed_active_column_index], active_columns[changed_active_column_index + 1:]])

            # Update active factor column indices
            col_indices = tp.get_kronecker_factor_column_indices(changed_dict_column_index, core_tensor_shape)
            for n in range(order):
                active_factor_column_indices[n] = sorted(set(active_factor_column_indices[n]) | {int(col_indices[n])})

        # Store results
        self.tensor_norm_ = tensor_norm
        coef_tensor_result = tp.tensorize(x=coef_, tensor_shape=core_tensor_shape, active_elements=active_columns, device=self.device)
        self.coef_tensor_ = coef_tensor_result
        self.active_columns_ = active_columns
        self.coef_ = coef_
        self.norm_r_ = norm_r
        self.n_iter_ = n_iter
        return self

    def predict(
        self,
        factor_matrices: List[Union[np.ndarray, object]],
    ) -> Union[np.ndarray, object]:
        """
        Predicts the target tensor using the learned coefficients and provided factor matrices.

        Parameters
        ----------
        factor_matrices : List[Union[np.ndarray, object]]
            List of factor (dictionary) matrices for each tensor mode. Each matrix should have shape (mode_dim, dict_size).

        Returns
        -------
        Y_pred : Union[np.ndarray, object]
            The predicted tensor, as a NumPy array or backend-specific tensor.
        """
        if self.coef_tensor_ is None:
            raise ValueError("Model is not fitted yet. Call 'fit' before 'predict'.")
        # Ensure factor_matrices are on the correct device and normalized
        normalized_matrices = []
        for D in factor_matrices:
            D_normed = self.normalize_input(D)
            D_device = self.ops.to_device(D_normed, self.device)
            D_final = self.ops.normalize(D_device)
            normalized_matrices.append(D_final)
        # Use the learned coef_tensor_ to reconstruct the tensor
        Y_pred = tp.full_multilinear_product(self.coef_tensor_, normalized_matrices, use_transpose=False)
        # Rescale by the stored normalization factor
        Y_pred = Y_pred * self.tensor_norm_
        return Y_pred

