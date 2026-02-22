from typing import List

import numpy as np
import torch
from string import ascii_lowercase as letters


def _torch_flatten_fortran(x: torch.Tensor) -> torch.Tensor:
    """Flatten a torch tensor in Fortran (column-major) order, matching MATLAB's X(:)."""
    if x.ndim <= 1:
        return x.contiguous().flatten()
    return x.permute(*reversed(range(x.ndim))).contiguous().flatten()


def _torch_reshape_fortran(x: torch.Tensor, shape) -> torch.Tensor:
    """Reshape a 1D torch tensor to given shape in Fortran (column-major) order, matching MATLAB's reshape."""
    shape = list(shape)
    if len(shape) <= 1:
        return x.reshape(*shape) if shape else x
    return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape)))).contiguous()


def _kronecker_product(matrices: List[torch.Tensor]) -> torch.Tensor:
    """
    Compute the Kronecker product of a list of Torch tensors from N to 1.
    A = (A^(N) ⊗ A^(N−1) ⊗· · ·⊗A^(2) ⊗ A^(1) )

    :param matrices: List of matrices as torch.Tensor.
    :return: Kronecker product of all matrices as torch.Tensor.
    """

    if not matrices:
        raise ValueError("The list of matrices is empty.")

    matrices = [torch.tensor(matrix) if not isinstance(matrix, torch.Tensor) else matrix for matrix in matrices]
    matrices.reverse()  # Reverse the list of matrices
    result = matrices[0]
    for matrix in matrices[1:]:
        result = torch.kron(result, matrix)
    return result

def _khatri_rao_product(matrices: List[torch.Tensor]) -> torch.Tensor:
    """
    Compute the Khatri-Rao product of a list of Torch tensors from N to 1.

    :param matrices: List of matrices as torch.Tensor.
    :return: Khatri-Rao product of all matrices as torch.Tensor.
    """
    if not matrices:
        raise ValueError("The list of matrices is empty.")

    matrices = [torch.tensor(matrix) if not isinstance(matrix, torch.Tensor) else matrix for matrix in matrices]
    num_columns = matrices[0].shape[1]
    for matrix in matrices:
        assert matrix.shape[1] == num_columns, "All matrices must have the same number of columns."

    result = torch.hstack([_kronecker_product([matrix[:, i] for matrix in matrices]).reshape(-1, 1) for i in range(num_columns)])

    return result


def _tensor_product(matrices: List[torch.Tensor]) -> torch.Tensor:
    """
    Compute the tensor product of a list of Torch tensors from N to 1.

    :param matrices: List of matrices as torch.Tensor.
    :return: Tensor product of all matrices as torch.Tensor.
    """
    if not matrices:
        raise ValueError("The list of matrices is empty.")

    matrices = [torch.tensor(matrix) if not isinstance(matrix, torch.Tensor) else matrix for matrix in matrices]
    matrices.reverse()  # Reverse the list of matrices
    result = matrices[0]
    for matrix in matrices[1:]:
        result = torch.tensordot(result, matrix, dims=0)
    return result


def _get_kronecker_matrix_column(factor_matrices: List[torch.Tensor], column_indices: List[int]) -> torch.Tensor:
    """
    This function returns a column of the Kronecker matrix (Kronecker Product of the Factor Matrices) when the
    factor matrices and column indices of each factor matrix is given.

    :param factor_matrices: List of factor matrices as torch tensors.
    :param column_indices: This field requires a list of column indices of factor matrix columns to be used in calculating the
                    column of the Kronecker matrix.
    :return: kronecker_column : This is the column of the Kronecker matrix corresponding to the column indices
    """

    # Extract the specified columns from each factor matrix
    selected_columns = [fm[:, col_idx] for fm, col_idx in zip(factor_matrices, column_indices)]

    # Compute the Kronecker product of the selected columns
    kronecker_column = _kronecker_product(selected_columns)

    return kronecker_column


def _full_multilinear_product(X: torch.Tensor, factor_matrices: list, use_transpose: bool = False) -> torch.Tensor:
    """
    Full multilinear product of the tensor X with factor matrices.
    If use_transpose is set to True, then use the transpose of each factor matrix when calculating the
    full multilinear product.

    :param X: Core Tensor as a torch.Tensor.
    :param factor_matrices: List of factor matrices as torch tensors.
    :param use_transpose: If True transpose each factor matrix before calculating the multilinear product.
    :return: Y  Resultant tensor of the full multilinear product of the tensor X with factor matrices
    """

    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X)

    factor_matrices = [torch.tensor(matrix) if not isinstance(matrix, torch.Tensor) else matrix for matrix in factor_matrices]

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


def _kronecker_matrix_vector_product(
    factor_matrices: List[torch.Tensor],
    x: torch.Tensor,
    tensor_shape: List[int],
    active_columns: List[int],
    active_indices: List = None,
    use_transpose: bool = False,
    device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """
    Calculates the product between a Kronecker matrix A and a vector x (y = Ax) using full multilinear product.
    Columns of matrix A are obtained by Kronecker product of respective columns in the factor matrices.
    If matrix B is the Kronecker product of all factor matrices, columns of A are a subset of columns of B.
    If use_transpose is True, computes y = A'x.

    Uses Fortran (column-major) order for vectorization/tensorization to match the mathematical convention.

    :param factor_matrices: List of factor matrices as torch tensors.
    :param x: The vector that is going to be multiplied with the Kronecker matrix.
    :param tensor_shape: Shape of the core tensor X.
    :param active_columns: Active indices of the factor matrices to be used in the multiplications (Fortran-order linear indices).
    :param active_indices: List of per-factor active column indices for sub-tensor optimization.
                          If None, uses full factor matrices (no sub-tensor optimization).
    :param use_transpose: If True transpose each factor matrix before calculating the multilinear product.
    :param device: The device "CPU" or "GPU" to be used in the calculations.
    :return: y : Result vector of the Kronecker matrix vector product.
    """

    X = torch.zeros(np.prod(tensor_shape), dtype=torch.double, device=device)
    X[active_columns] = x
    X = _torch_reshape_fortran(X, tensor_shape)

    # Ensure factor matrices are double precision to match X
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

    Y: torch.Tensor = _full_multilinear_product(X, factor_matrices, use_transpose)
    y: torch.Tensor = _torch_flatten_fortran(Y)
    return y


def _tensorize(x: torch.Tensor,
              tensor_shape: List[int],
              active_elements: List[int],
              device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """
    Tensorize a vector based on active elements.

    :param x: The vector with tensor elements.
    :param tensor_shape: Shape of the core tensor X.
    :param active_elements: Active elements of the Tensor X corresponds to elements of x.
    :param device: The device "CPU" or "GPU" to be used in the calculations.
    :return: Return the tensor X
    """

    X = torch.zeros(np.prod(tensor_shape), dtype=torch.double, device=device)
    X[active_elements] = x
    X = _torch_reshape_fortran(X, tensor_shape)
    return X


def _vectorize(X: torch.Tensor) -> torch.Tensor:
    """
    Vectorize a tensor from dimension N to 1.
    Uses Fortran (column-major) order to match the mathematical convention vec(X).

    :param X: The tensor to be vectorized.
    :return: Return the vector x
    """
    return _torch_flatten_fortran(X)


def _get_gramian(G: List[torch.Tensor],
                active_columns: List[int],
                tensor_shape: List[int],
                device: torch.device) -> torch.Tensor:
    """
    Obtain a Gram matrix by selecting a subset of columns from a kronecker matrix given by kronecker product of
    matrices in the Kron_Cell_Array.

    :param G: List of factor matrices as torch tensors.
    :param active_columns: Active indices of the columns.
    :param tensor_shape: Shape of the tensor.
    :param device: The device "CPU" or "GPU" to be used in the calculations.
    :return: The Gramian Matrix.
    """

    GI = torch.zeros(len(active_columns), len(active_columns), dtype=torch.double, device=device)

    for i in range(len(active_columns)):
        indices = np.unravel_index(active_columns[i].item(), tensor_shape, order='F')
        gk = _get_kronecker_matrix_column(G, indices)
        gk = gk[active_columns]
        GI[:, i] = torch.tensor(gk, device=device)

    return GI


def _tround(tensor: torch.Tensor, precision_order: int = 0) -> torch.Tensor:
    """
    Round tensor elements to a specific precision.

    :param tensor: Tensor to be rounded.
    :param precision_order: Required precision.
    :return: Rounded tensor.
    """
    return torch.round(tensor * 10 ** precision_order) * 10 ** -precision_order


def _get_direction_vector(GInv: torch.Tensor, zI: torch.Tensor, gramians: List[torch.Tensor],
                         active_columns: torch.Tensor, add_column_flag: bool,
                         changed_dict_column_index: int, changed_active_column_index: int,
                         tensor_shape: List[int], precision_order: int,
                         device: torch.device = torch.device("cpu")) -> tuple:
    """
    Torch implementation of get_direction_vector.
    Update the inverse Gramian using the Schur complement formula and compute dI = GInv @ zI.

    :param GInv: Current inverse Gramian matrix (2D torch tensor).
    :param zI: Sign vector of correlations at active columns.
    :param gramians: List of per-mode Gram matrices.
    :param active_columns: Current active column indices.
    :param add_column_flag: True if a column was added, False if removed.
    :param changed_dict_column_index: Linear index of the changed dictionary column.
    :param changed_active_column_index: Position (0-indexed) of the changed column in active_columns.
    :param tensor_shape: Shape of the core tensor.
    :param precision_order: Precision order for rounding.
    :param device: Torch device.
    :return: (dI, GInv) - direction vector and updated inverse Gramian.
    """
    N = len(active_columns)

    if add_column_flag:
        old_N = N - 1

        indices = np.unravel_index(int(changed_dict_column_index), tensor_shape, order='F')
        ga = _get_kronecker_matrix_column(gramians, indices)
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
    dI = _tround(dI, precision_order)

    return dI, GInv