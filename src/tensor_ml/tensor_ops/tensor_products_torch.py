"""PyTorch implementation of :class:`TensorProductsBase`."""

import numpy as np
from string import ascii_lowercase as letters

from tensor_ml.tensor_ops.tensor_products_base import TensorProductsBase


# ---------------------------------------------------------------------------
# Fortran-order helpers (package-private)
# ---------------------------------------------------------------------------

def _torch_flatten_fortran(x):
    """Flatten a torch tensor in Fortran (column-major) order, matching MATLAB's X(:)."""
    if x.ndim <= 1:
        return x.contiguous().flatten()
    return x.permute(*reversed(range(x.ndim))).contiguous().flatten()


def _torch_reshape_fortran(x, shape):
    """Reshape a 1D torch tensor to given shape in Fortran (column-major) order."""
    shape = list(shape)
    if len(shape) <= 1:
        return x.reshape(*shape) if shape else x
    return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape)))).contiguous()


# ---------------------------------------------------------------------------
# Concrete class
# ---------------------------------------------------------------------------

class TorchTensorProducts(TensorProductsBase):
    """PyTorch backend for tensor-product operations."""

    def __init__(self, device=None):
        import torch
        self.torch = torch
        self.device = device

    # ── Products ───────────────────────────────────────────────────

    def kronecker_product(self, matrices):
        torch = self.torch
        if not matrices:
            raise ValueError("The list of matrices is empty.")
        matrices = [torch.tensor(m) if not isinstance(m, torch.Tensor) else m for m in matrices]
        reversed_matrices = matrices[::-1]
        result = reversed_matrices[0]
        for matrix in reversed_matrices[1:]:
            result = torch.kron(result, matrix)
        return result

    def khatri_rao_product(self, matrices):
        torch = self.torch
        if not matrices:
            raise ValueError("The list of matrices is empty.")
        matrices = [torch.tensor(m) if not isinstance(m, torch.Tensor) else m for m in matrices]
        num_columns = matrices[0].shape[1]
        for m in matrices:
            if m.shape[1] != num_columns:
                raise ValueError(
                    f"All matrices must have the same number of columns, "
                    f"expected {num_columns} but got {m.shape[1]}."
                )
        result = torch.hstack([
            self.kronecker_product([m[:, i] for m in matrices]).reshape(-1, 1)
            for i in range(num_columns)
        ])
        return result

    def tensor_product(self, matrices):
        torch = self.torch
        if not matrices:
            raise ValueError("The list of matrices is empty.")
        matrices = [torch.tensor(m) if not isinstance(m, torch.Tensor) else m for m in matrices]
        reversed_matrices = matrices[::-1]
        result = reversed_matrices[0]
        for matrix in reversed_matrices[1:]:
            result = torch.tensordot(result, matrix, dims=0)
        return result

    def hadamard_product(self, tensors):
        torch = self.torch
        if not tensors:
            raise ValueError("The list of tensors is empty.")
        tensors = [torch.tensor(t) if not isinstance(t, torch.Tensor) else t for t in tensors]
        result = tensors[0].clone()
        for t in tensors[1:]:
            result = result * t
        return result

    # ── Kronecker helpers ──────────────────────────────────────────

    def get_kronecker_matrix_column(self, factor_matrices, column_indices):
        selected_columns = [fm[:, col_idx] for fm, col_idx in zip(factor_matrices, column_indices)]
        return self.kronecker_product(selected_columns)

    def full_multilinear_product(self, X, factor_matrices, use_transpose=False):
        torch = self.torch
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X)
        factor_matrices = [torch.tensor(m) if not isinstance(m, torch.Tensor) else m for m in factor_matrices]
        # Ensure all factor matrices match X's dtype
        factor_matrices = [fm.to(dtype=X.dtype) for fm in factor_matrices]

        order = X.ndim
        Y = X.clone()
        for n in range(order):
            if use_transpose:
                oper = letters[:order][n] + "z"
            else:
                oper = "z" + letters[:order][n]
            op = letters[:order] + "," + oper + "->" + letters[:order][:n] + 'z' + letters[:order][n + 1:]
            Y = torch.einsum(op, Y, factor_matrices[n])
        return Y

    def kronecker_matrix_vector_product(self, factor_matrices, x, tensor_shape,
                                         active_columns, active_indices=None,
                                         use_transpose=False):
        torch = self.torch
        device = x.device

        X = torch.zeros(np.prod(tensor_shape), dtype=torch.double, device=device)
        X[active_columns] = x.to(dtype=torch.double)
        X = _torch_reshape_fortran(X, tensor_shape)

        factor_matrices = [fm.to(dtype=torch.double) for fm in factor_matrices]

        sub_tensor = active_indices is not None and len(factor_matrices) > 1
        if sub_tensor:
            sub_factors = []
            for i in range(len(factor_matrices)):
                idx = active_indices[i]
                if isinstance(idx, (int, np.integer)):
                    idx = [idx]
                idx_tensor = torch.tensor(idx, device=device) if not isinstance(idx, torch.Tensor) else idx.to(device)
                if use_transpose:
                    sub_factors.append(factor_matrices[i][idx_tensor, :])
                else:
                    sub_factors.append(factor_matrices[i][:, idx_tensor])
                X = torch.index_select(X, i, idx_tensor)
            factor_matrices = sub_factors

        Y = self.full_multilinear_product(X, factor_matrices, use_transpose)
        return _torch_flatten_fortran(Y)

    # ── Vectorize / tensorize ──────────────────────────────────────

    def tensorize(self, x, tensor_shape, active_elements):
        torch = self.torch
        device = x.device
        X = torch.zeros(np.prod(tensor_shape), dtype=torch.double, device=device)
        X[active_elements] = x
        return _torch_reshape_fortran(X, tensor_shape)

    def vectorize(self, X):
        return _torch_flatten_fortran(X)

    # ── Gramian / direction vector ─────────────────────────────────

    def get_gramian(self, gramians, active_columns, tensor_shape):
        torch = self.torch
        device = gramians[0].device
        GI = torch.zeros(len(active_columns), len(active_columns), dtype=torch.double, device=device)
        for i in range(len(active_columns)):
            indices = np.unravel_index(active_columns[i].item(), tensor_shape, order='F')
            gk = self.get_kronecker_matrix_column(gramians, indices)
            gk = gk[active_columns]
            GI[:, i] = torch.tensor(gk, device=device)
        return GI

    def get_direction_vector(self, GInv, zI, gramians, active_columns,
                              add_column_flag, changed_dict_column_index,
                              changed_active_column_index, tensor_shape,
                              precision_order=10):
        torch = self.torch
        device = zI.device
        N = len(active_columns)

        if add_column_flag:
            old_N = N - 1
            indices = np.unravel_index(int(changed_dict_column_index), tensor_shape, order='F')
            ga = self.get_kronecker_matrix_column(gramians, indices)
            ga = ga[active_columns].to(dtype=torch.float64, device=device)

            b = torch.zeros(N, dtype=torch.float64, device=device)
            b[-1] = 1.0
            GInv = GInv.to(dtype=torch.float64)
            if old_N > 0:
                b[:old_N] -= GInv[:old_N, :old_N] @ ga[:old_N]

            schur_complement = ga[N - 1] + (torch.dot(ga[:old_N], b[:old_N]) if old_N > 0 else 0.0)
            alpha = 1.0 / schur_complement

            new_GInv = torch.zeros((N, N), dtype=torch.float64, device=device)
            if old_N > 0:
                new_GInv[:old_N, :old_N] = GInv[:old_N, :old_N]
            new_GInv += alpha * torch.outer(b, b)
            GInv = new_GInv
        else:
            old_N = N + 1
            k = int(changed_active_column_index)

            alpha = GInv[k, k].clone()
            ab = GInv[:old_N, k].clone()

            keep = [i for i in range(old_N) if i != k]
            keep_t = torch.tensor(keep, device=device)
            GInv = GInv[keep_t][:, keep_t].clone()
            ab = ab[keep_t].clone()
            GInv -= (1.0 / alpha) * torch.outer(ab, ab)

        zI = zI.to(dtype=GInv.dtype)
        dI = GInv @ zI
        dI = self.tround(dI, precision_order)
        return dI, GInv

    # ── Rounding ───────────────────────────────────────────────────

    def tround(self, tensor, precision_order=0):
        torch = self.torch
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.tensor(tensor)
        return torch.round(tensor * 10 ** precision_order) * 10 ** -precision_order
