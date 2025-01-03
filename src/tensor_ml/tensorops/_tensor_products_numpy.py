from typing import List
import numpy as np
from string import ascii_lowercase as letters


def _kronecker_product(matrices: List[np.ndarray]) -> np.ndarray:
    """
    Compute the Kronecker product of a list of NumPy ND arrays from N to 1.
    A = (A^(N) ⊗ A^(N−1) ⊗· · ·⊗A^(2) ⊗ A^(1) )

    :param matrices: List of matrices as np.ndarray.
    :return: Kronecker product of all matrices as np.ndarray.
    """

    if not matrices:
        raise ValueError("The list of matrices is empty.")

    matrices = [np.array(matrix) if not isinstance(matrix, np.ndarray) else matrix for matrix in matrices]
    matrices.reverse()  # Reverse the list of matrices
    result = matrices[0]
    for matrix in matrices[1:]:
        result = np.kron(result, matrix)
    return result


def _khatri_rao_product(matrices: List[np.ndarray]) -> np.ndarray:
    """
    Compute the Khatri-Rao product of a list of NumPy arrays from N to 1.

    :param matrices: List of matrices as np.ndarray.
    :return: Khatri-Rao product of all matrices as np.ndarray.
    """
    if not matrices:
        raise ValueError("The list of matrices is empty.")

    matrices = [np.array(matrix) if not isinstance(matrix, np.ndarray) else matrix for matrix in matrices]
    num_columns = matrices[0].shape[1]
    for matrix in matrices:
        assert matrix.shape[1] == num_columns, "All matrices must have the same number of columns."

    result = np.hstack([_kronecker_product([matrix[:, i] for matrix in matrices]).reshape(-1, 1) for i in range(num_columns)])

    return result


def _tensor_product(matrices: List[np.ndarray]) -> np.ndarray:
    """
    Compute the tensor product of a list of NumPy arrays from N to 1.

    :param matrices: List of matrices as np.ndarray.
    :return: Tensor product of all matrices as np.ndarray.
    """
    if not matrices:
        raise ValueError("The list of matrices is empty.")

    matrices = [np.array(matrix) if not isinstance(matrix, np.ndarray) else matrix for matrix in matrices]
    matrices.reverse()  # Reverse the list of matrices
    result = matrices[0]
    for matrix in matrices[1:]:
        result = np.tensordot(result, matrix, axes=0)
    return result


def _get_kronecker_matrix_column(factor_matrices: List[np.ndarray], column_indices: List[int]) -> np.ndarray:
    """
    This function returns a column of the Kronecker matrix (Kronecker Product of the Factor Matrices) when the
    factor matrices and column indices of each factor matrix is given.

    :param factor_matrices: List of factor matrices as numpy array.
    :param column_indices: This field requires a list of column indices of factor matrix columns to be used in calculating the
                    column of the Kronecker matrix.
    :return: kronecker_column : This is the column of the Kronecker matrix corresponding to the column indices
    """

    # Extract the specified columns from each factor matrix
    selected_columns = [fm[:, col_idx] for fm, col_idx in zip(factor_matrices, column_indices)]

    # Compute the Kronecker product of the selected columns
    kronecker_column = _kronecker_product(selected_columns)

    return kronecker_column


def _full_multilinear_product(X: np.ndarray, factor_matrices: list, use_transpose: bool = False) -> np.ndarray:
    """
    Full multilinear product of the tensor X with factor matrices.
    If use_transpose is set to True, then use the transpose of each factor matrix when calculating the
    full multilinear product.

    :param X: Core Tensor as an N-D Array.
    :param factor_matrices: List of factor matrices as numpy arrays.
    :param use_transpose: If True transpose each factor matrix before calculating the multilinear product.
    :return: Y  Resultant tensor of the full multilinear product of the tensor X with factor matrices
    """

    if not isinstance(X, np.ndarray):
        X = np.array(X)

    factor_matrices = [np.array(matrix) if not isinstance(matrix, np.ndarray) else matrix for matrix in factor_matrices]

    order = X.ndim
    Y = X.copy()

    for n in range(order):
        if use_transpose:
            oper = letters[:order][n] + "z"
        else:
            oper = "z" + letters[:order][n]

        op = letters[:order] + "," + oper + "->" + letters[:order][:n] + 'z' + letters[:order][n + 1:]
        Y = np.einsum(op, Y, factor_matrices[-n - 1])

    return Y


def _kronecker_matrix_vector_product(factor_matrices: List[np.ndarray],
                                    x: np.ndarray,
                                    tensor_shape: List[int],
                                    active_columns: List[int],
                                    active_indices: List[int],
                                    use_transpose: bool = False) -> np.ndarray:
    """
    kronecker_matrix_vector_product function calculates the product
    between a matrix A and a vector x (y = Ax) using full multilinear product.
    Columns of matrix A can be obtained by kronecker product of respective columns in the kron_cell_array.
    If matrix B is the kronecker product of all factor matrices in Factor_Matrices, columns of A is a subset of columns of B.
    If transpose = True, y = A'x is calculated

    :param factor_matrices: List of factor matrices as numpy arrays.
    :param x: The vector that is going to be multiplied with the Kronecker matrix.
    :param tensor_shape: Shape of the core tensor X.
    :param active_columns: Active indices of the factor matrices to be used in the multiplications.
    :param active_indices: List of tensor columns (column indices of x as a core tensor X) to be used in the multiplication.
    :param use_transpose: If True transpose each factor matrix before calculating the multilinear product.
    :return: y : Result vector of the Kronecker matrix vector product.
    """

    X: np.ndarray = np.zeros(np.prod(tensor_shape))
    X[active_columns] = x
    X = X.reshape(tensor_shape)

    if use_transpose:
        factor_matrices = [factor_matrices[i][active_indices[-i - 1], :] for i in range(len(factor_matrices))]
    else:
        factor_matrices = [factor_matrices[i][:, active_indices[-i - 1]] for i in range(len(factor_matrices))]

    for i in range(len(factor_matrices)):
        X = np.take(X, active_indices[i], axis=i)

    Y: np.ndarray = _full_multilinear_product(X, factor_matrices, use_transpose)
    y: np.ndarray = Y.flatten()
    return y


def _tensorize(x: np.ndarray,
              tensor_shape: list,
              active_elements: list) -> np.ndarray:
    """
    Tensorize a vector based on active elements.

    :param x: The vector with tensor elements.
    :param tensor_shape: Shape of the core tensor X.
    :param active_elements: Active elements of the Tensor X corresponds to elements of x.
    :return: Return the tensor X
    """

    X = np.zeros(np.prod(tensor_shape), dtype=np.double)
    X[active_elements] = x
    X = X.reshape(tensor_shape)
    return X


def _vectorize(X: np.ndarray) -> np.ndarray:
    """
    Vectorize a tensor from dimension N to 1.

    :param X: The tensor to be vectorized.
    :return: Return the vector x
    """
    return X.flatten()


def _get_gramian(G: List[np.ndarray],
                active_columns: List[int],
                tensor_shape: List[int]) -> np.ndarray:
    """
    Obtain a Gram matrix by selecting a subset of columns from a kronecker matrix given by kronecker product of
    matrices in the Kron_Cell_Array.

    :param G: List of factor matrices as numpy arrays.
    :param active_columns: Active indices of the columns.
    :param tensor_shape: Shape of the tensor.
    :return: The Gramian Matrix.
    """

    GI = np.zeros((len(active_columns), len(active_columns)), dtype=np.double)

    for i in range(len(active_columns)):
        indices = np.unravel_index(active_columns[i], tensor_shape)
        gk = _get_kronecker_matrix_column(G, indices)
        gk = gk[active_columns]
        GI[:, i] = gk

    return GI


def _tround(tensor: np.ndarray, precision_order: int = 0) -> np.ndarray:
    """
    Round array elements to a specific precision.

    :param tensor: ndarray to be rounded.
    :param precision_order: Required precision.
    :return: Rounded ndarray.
    """
    if not isinstance(tensor, np.ndarray):
        tensor = np.array(tensor)
    return np.round(tensor * 10 ** precision_order) * 10 ** -precision_order