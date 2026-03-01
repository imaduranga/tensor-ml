"""NumPy implementation of :class:`TensorProductsBase`."""

import numpy as np
from string import ascii_lowercase as letters

from tensor_ml.tensor_ops.tensor_products_base import TensorProductsBase


class NumpyTensorProducts(TensorProductsBase):
    """NumPy backend for tensor-product operations."""

    # ── Products ───────────────────────────────────────────────────

    def kronecker_product(self, matrices: list[np.ndarray]) -> np.ndarray:
        if not matrices:
            raise ValueError("The list of matrices is empty.")
        matrices = [np.array(m) if not isinstance(m, np.ndarray) else m for m in matrices]
        reversed_matrices = matrices[::-1]
        result = reversed_matrices[0]
        for matrix in reversed_matrices[1:]:
            result = np.kron(result, matrix)
        return result

    def khatri_rao_product(self, matrices: list[np.ndarray]) -> np.ndarray:
        if not matrices:
            raise ValueError("The list of matrices is empty.")
        matrices = [np.array(m) if not isinstance(m, np.ndarray) else m for m in matrices]
        num_columns = matrices[0].shape[1]
        for m in matrices:
            if m.shape[1] != num_columns:
                raise ValueError(
                    f"All matrices must have the same number of columns, "
                    f"expected {num_columns} but got {m.shape[1]}."
                )
        result = np.hstack([
            self.kronecker_product([m[:, i] for m in matrices]).reshape(-1, 1)
            for i in range(num_columns)
        ])
        return result

    def tensor_product(self, matrices: list[np.ndarray]) -> np.ndarray:
        if not matrices:
            raise ValueError("The list of matrices is empty.")
        matrices = [np.array(m) if not isinstance(m, np.ndarray) else m for m in matrices]
        reversed_matrices = matrices[::-1]
        result = reversed_matrices[0]
        for matrix in reversed_matrices[1:]:
            result = np.tensordot(result, matrix, axes=0)
        return result

    def hadamard_product(self, tensors: list) -> np.ndarray:
        if not tensors:
            raise ValueError("The list of tensors is empty.")
        tensors = [np.array(t) if not isinstance(t, np.ndarray) else t for t in tensors]
        result = tensors[0].copy()
        for t in tensors[1:]:
            result = result * t
        return result

    # ── Kronecker helpers ──────────────────────────────────────────

    def get_kronecker_matrix_column(self, factor_matrices: list[np.ndarray],
                                     column_indices: list[int]) -> np.ndarray:
        selected_columns = [fm[:, col_idx] for fm, col_idx in zip(factor_matrices, column_indices)]
        return self.kronecker_product(selected_columns)

    def full_multilinear_product(self, X: np.ndarray, factor_matrices: list,
                                  use_transpose: bool = False) -> np.ndarray:
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        factor_matrices = [np.array(m) if not isinstance(m, np.ndarray) else m for m in factor_matrices]

        order = X.ndim
        Y = X.copy()
        for n in range(order):
            if use_transpose:
                oper = letters[:order][n] + "z"
            else:
                oper = "z" + letters[:order][n]
            op = letters[:order] + "," + oper + "->" + letters[:order][:n] + 'z' + letters[:order][n + 1:]
            Y = np.einsum(op, Y, factor_matrices[n])
        return Y

    def kronecker_matrix_vector_product(self, factor_matrices, x, tensor_shape,
                                         active_columns, active_indices=None,
                                         use_transpose=False):
        X = np.zeros(np.prod(tensor_shape), dtype=np.double)
        X[active_columns] = x
        X = X.reshape(tensor_shape, order='F')

        sub_tensor = active_indices is not None and len(factor_matrices) > 1
        if sub_tensor:
            sub_factors = []
            idx_arrays = []
            for i in range(len(factor_matrices)):
                idx = active_indices[i]
                if isinstance(idx, (int, np.integer)):
                    idx = np.array([idx])
                else:
                    idx = np.asarray(idx)
                idx_arrays.append(idx)
                if use_transpose:
                    sub_factors.append(factor_matrices[i][idx, :])
                else:
                    sub_factors.append(factor_matrices[i][:, idx])
            factor_matrices = sub_factors
            X = X[np.ix_(*idx_arrays)]

        Y = self.full_multilinear_product(X, factor_matrices, use_transpose)
        return Y.flatten(order='F')

    # ── Vectorize / tensorize ──────────────────────────────────────

    def tensorize(self, x, tensor_shape, active_elements):
        X = np.zeros(np.prod(tensor_shape), dtype=np.double)
        X[active_elements] = x
        return X.reshape(tensor_shape, order='F')

    def vectorize(self, X):
        return X.flatten(order='F')

    # ── Gramian / direction vector ─────────────────────────────────

    def get_gramian(self, gramians, active_columns, tensor_shape):
        GI = np.zeros((len(active_columns), len(active_columns)), dtype=np.double)
        for i in range(len(active_columns)):
            indices = np.unravel_index(active_columns[i], tensor_shape, order='F')
            gk = self.get_kronecker_matrix_column(gramians, indices)
            gk = gk[active_columns]
            GI[:, i] = gk
        return GI

    def get_direction_vector(self, GInv, zI, gramians, active_columns,
                              add_column_flag, changed_dict_column_index,
                              changed_active_column_index, tensor_shape,
                              precision_order=10):
        N = len(active_columns)

        if add_column_flag:
            old_N = N - 1
            indices = np.unravel_index(int(changed_dict_column_index), tensor_shape, order='F')
            ga = self.get_kronecker_matrix_column(gramians, indices).astype(np.float64)
            ga = ga[active_columns]

            b = np.zeros(N, dtype=np.float64)
            b[-1] = 1.0
            if old_N > 0:
                b[:old_N] -= GInv[:old_N, :old_N] @ ga[:old_N]

            schur_complement = ga[N - 1] + (np.dot(ga[:old_N], b[:old_N]) if old_N > 0 else 0.0)
            alpha = 1.0 / schur_complement

            new_GInv = np.zeros((N, N), dtype=np.float64)
            if old_N > 0:
                new_GInv[:old_N, :old_N] = GInv[:old_N, :old_N]
            new_GInv += alpha * np.outer(b, b)
            GInv = new_GInv
        else:
            old_N = N + 1
            k = int(changed_active_column_index)

            alpha = GInv[k, k]
            ab = GInv[:old_N, k].copy()

            GInv = np.delete(np.delete(GInv[:old_N, :old_N], k, axis=0), k, axis=1)
            ab = np.delete(ab, k)
            GInv -= (1.0 / alpha) * np.outer(ab, ab)

        dI = GInv @ zI
        dI = self.tround(dI, precision_order)
        return dI, GInv

    # ── Rounding ───────────────────────────────────────────────────

    def tround(self, tensor, precision_order=0):
        if not isinstance(tensor, np.ndarray):
            tensor = np.array(tensor)
        return np.round(tensor * 10 ** precision_order) * 10 ** -precision_order
