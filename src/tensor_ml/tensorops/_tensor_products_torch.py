from typing import List

import numpy as np
import torch
from string import ascii_lowercase as letters


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
        Y = torch.einsum(op, Y, factor_matrices[-n - 1])

    return Y


def _kronecker_matrix_vector_product(factor_matrices: List[torch.Tensor],
                                    x: torch.Tensor,
                                    tensor_shape: List[int],
                                    active_columns: List[int],
                                    active_indices: List[int] = None,
                                    use_transpose: bool = False,
                                    device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """
    kronecker_matrix_vector_product function calculates the product
    between a matrix A and a vector x (y = Ax) using full multilinear product.
    Columns of matrix A can be obtained by kronecker product of respective columns in the kron_cell_array.
    If matrix B is the kronecker product of all factor matrices in Factor_Matrices, columns of A is a subset of columns of B.
    If transpose = True, y = A'x is calculated

    :param factor_matrices: List of factor matrices as torch tensors.
    :param x: The vector that is going to be multiplied with the Kronecker matrix.
    :param tensor_shape: Shape of the core tensor X.
    :param active_columns: Active indices of the factor matrices to be used in the multiplications.
    :param active_indices: List of tensor columns (column indices of x as a core tensor X) to be used in the multiplication.
    :param use_transpose: If True transpose each factor matrix before calculating the multilinear product.
    :param device: The device "CPU" or "GPU" to be used in the calculations.
    :return: y : Result vector of the Kronecker matrix vector product.
    """

    X: torch.Tensor = torch.zeros(np.prod(tensor_shape), dtype=torch.double, device=device)
    X[active_columns] = x
    X = X.reshape(tensor_shape)

    if use_transpose:
        factor_matrices = [factor_matrices[i][active_indices[-i - 1], :] for i in range(len(factor_matrices))]
    else:
        factor_matrices = [factor_matrices[i][:, active_indices[-i - 1]] for i in range(len(factor_matrices))]

    # for i in range(len(factor_matrices)):
    #     X = torch.index_select(X, i, torch.tensor(active_indices[i], device=device))

    Y: torch.Tensor = _full_multilinear_product(X, factor_matrices, use_transpose)
    y: torch.Tensor = Y.flatten()
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
    X = X.reshape(tensor_shape)
    return X


def _vectorize(X: torch.Tensor) -> torch.Tensor:
    """
    Vectorize a tensor from dimension N to 1.

    :param X: The tensor to be vectorized.
    :return: Return the vector x
    """
    return X.flatten()


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
        indices = np.unravel_index(active_columns[i].item(), tensor_shape)
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