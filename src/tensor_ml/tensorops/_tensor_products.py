import torch
import numpy as np
from typing import List, Union

import tensor_ml.tensorops._tensor_products_numpy as npt
import tensor_ml.tensorops._tensor_products_torch as tpt


def kronecker_product(factor_matrices: List[Union[np.ndarray, torch.Tensor]]) -> Union[np.ndarray, torch.Tensor]:
    """
    Compute the Kronecker product of a list of factor matrices from N to 1.
    A = (A^(N) ⊗ A^(N−1) ⊗· · ·⊗A^(2) ⊗ A^(1) )

    :param factor_matrices: List of factor matrices as numpy array or torch tensor.
    :return: Kronecker product of all factor matrices.
    """

    if not factor_matrices:
        raise ValueError("The list of factor_matrices is empty.")

    # Compute the Kronecker product using the appropriate method
    if isinstance(factor_matrices[0], torch.Tensor):
        return tpt._kronecker_product(factor_matrices)
    else:
        return npt._kronecker_product(factor_matrices)


def khatri_rao_product(matrices: List[Union[np.ndarray, torch.Tensor]]) -> Union[np.ndarray, torch.Tensor]:
    """
    Compute the Khatri-Rao product of a list of matrices from N to 1.

    :param matrices: List of matrices (torch.Tensor or numpy.ndarray).
    :return: Khatri-Rao product of all matrices (torch.Tensor or np.ndarray).
    """

    if not matrices:
        raise ValueError("The list of matrices is empty.")

    if isinstance(matrices[0], torch.Tensor):
        matrices = [torch.tensor(matrix) if not isinstance(matrix, torch.Tensor) else matrix for matrix in matrices]
        return tpt._khatri_rao_product(matrices)
    else:
        matrices = [np.array(matrix) if not isinstance(matrix, np.ndarray) else matrix for matrix in matrices]
        return npt._khatri_rao_product(matrices)


def tensor_product(matrices: List[Union[np.ndarray, torch.Tensor]]) -> Union[np.ndarray, torch.Tensor]:
    """
    Compute the tensor product of a list of matrices from N to 1.

    :param matrices: List of matrices (torch.Tensor or numpy.ndarray).
    :return: Tensor product of all matrices (torch.Tensor or np.ndarray).
    """

    if not matrices:
        raise ValueError("The list of matrices is empty.")

    if isinstance(matrices[0], torch.Tensor):
        matrices = [torch.tensor(matrix) if not isinstance(matrix, torch.Tensor) else matrix for matrix in matrices]
        return tpt._tensor_product(matrices)
    else:
        matrices = [np.array(matrix) if not isinstance(matrix, np.ndarray) else matrix for matrix in matrices]
        return npt._tensor_product(matrices)


def hadamard_product(tensors: List[Union[np.ndarray, torch.Tensor]]) -> Union[np.ndarray, torch.Tensor]:
    """
    Compute the Hadamard product of a list of tensors.

    :param tensors: List of tensors (torch.Tensor or numpy.ndarray).
    :return: Hadamard product of all tensors in the list (torch.Tensor or np.ndarray).
    """

    if not tensors:
        raise ValueError("The list of tensors is empty.")

    if isinstance(tensors[0], torch.Tensor):
        tensors = [torch.tensor(tensor) if not isinstance(tensor, torch.Tensor) else tensor for tensor in tensors]
    else:
        tensors = [np.array(tensor) if not isinstance(tensor, np.ndarray) else tensor for tensor in tensors]

    result = tensors[0]
    for tensor in tensors[1:]:
        result = result * tensor
    return result


def get_kronecker_matrix_column(factor_matrices: List[Union[np.ndarray, torch.Tensor]], column_indices: List[int]) -> Union[np.ndarray, torch.Tensor]:
    """
    This function returns a column of the Kronecker matrix (Kronecker Product of the Factor Matrices) when the
    factor matrices and column indices of each factor matrix is given.

    :param factor_matrices: List of factor matrices as numpy array or torch tensor.
    :param column_indices: This field requires a list of column indices of factor matrix columns to be used in calculating the
                    column of the Kronecker matrix.
    :return: kronecker_column : This is the column of the Kronecker matrix corresponding to the column indices
    """

    if not factor_matrices:
        raise ValueError("The list of factor_matrices is empty.")
    if not column_indices:
        raise ValueError("The list of column_indices is empty.")

    if isinstance(factor_matrices[0], torch.Tensor):
        return tpt._get_kronecker_matrix_column(factor_matrices, column_indices)
    else:
        return npt._get_kronecker_matrix_column(factor_matrices, column_indices)


def full_multilinear_product(X: Union[np.ndarray, torch.Tensor], factor_matrices: list, use_transpose: bool = False) -> Union[np.ndarray, torch.Tensor]:
    """
    Full multilinear product of the tensor X with factor matrices.
    If use_transpose is set to True, then use the transpose of each factor matrix when calculating the
    full multilinear product.

    :param X: Core Tensor as an N-D Array or torch.Tensor.
    :param factor_matrices: List of factor matrices as numpy arrays or torch tensors.
    :param use_transpose: If True transpose each factor matrix before calculating the multilinear product.
    :return: Y  Resultant tensor of the full multilinear product of the tensor X with factor matrices
    """

    if X is None:
        raise ValueError("The input tensor X is empty or None.")

    if isinstance(X, torch.Tensor):
        return tpt._full_multilinear_product(X, factor_matrices, use_transpose)
    else:
        return npt._full_multilinear_product(X, factor_matrices, use_transpose)


def kronecker_matrix_vector_product(factor_matrices: List[Union[np.ndarray, torch.Tensor]],
                                    x: Union[np.ndarray, torch.Tensor],
                                    tensor_shape: List[int],
                                    active_columns: List[int],
                                    active_indices: List[int],
                                    use_transpose: bool = False,
                                    device: torch.device = torch.device("cpu")) -> Union[np.ndarray, torch.Tensor]:
    """
    kronecker_matrix_vector_product function calculates the product
    between a matrix A and a vector x (y = Ax) using full multilinear product.
    Columns of matrix A can be obtained by kronecker product of respective columns in the factor_matrices.
    If matrix B is the kronecker product of all factor matrices in Factor_Matrices, columns of A is a subset of columns of B.
    If transpose = True, y = A'x is calculated

    :param factor_matrices: List of factor matrices as numpy arrays or torch tensors.
    :param x: The vector that is going to be multiplied with the Kronecker matrix.
    :param tensor_shape: Shape of the core tensor X.
    :param active_columns: Active indices of the factor matrices to be used in the multiplications.
    :param active_indices: List of tensor columns (column indices of x as a core tensor X) to be used in the multiplication.
    :param use_transpose: If True transpose each factor matrix before calculating the multilinear product.
    :param device: The device "CPU" or "GPU" to be used in the calculations.
    :return: y : Result vector of the Kronecker matrix vector product.
    """

    if isinstance(x, torch.Tensor):
        return tpt._kronecker_matrix_vector_product(factor_matrices, x, tensor_shape, active_columns, active_indices, use_transpose, device)
    else:
        return npt._kronecker_matrix_vector_product(factor_matrices, x, tensor_shape, active_columns, active_indices, use_transpose)


def tensorize(x: Union[np.ndarray, torch.Tensor],
              tensor_shape: List[int],
              active_elements: List[int],
              device: torch.device = torch.device("cpu")) -> Union[np.ndarray, torch.Tensor]:
    """
    Tensorize a vector based on active elements.

    :param x: The vector with tensor elements.
    :param tensor_shape: Shape of the core tensor X.
    :param active_elements: Active elements of the Tensor X corresponds to elements of x.
    :param device: The device "CPU" or "GPU" to be used in the calculations.
    :return: Return the tensor X
    """
    if x is None:
        raise ValueError("The input vector x is None.")

    if isinstance(x, torch.Tensor):
        return tpt._tensorize(x, tensor_shape, active_elements, device)
    else:
        return npt._tensorize(x, tensor_shape, active_elements)


def vectorize(X: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    Vectorize a tensor from dimension N to 1.

    :param X: The tensor to be vectorized.
    :return: Return the vector x
    """
    return X.flatten()


def get_reverse_shape(tensor_shape):
    """
    Calculate the reversed shape of the tensor(tensor shapes are reversed).
    :param tensor_shape:
    :return: Return the reversed shape of the tensor.
    """
    reverse_shape = list(tensor_shape)
    reverse_shape.reverse()
    return reverse_shape


def get_vector_index(order, tensor_indices, tensor_shape):
    """
    Get vector index of a tensor element.

    :param order: Order of the tensor
    :param tensor_indices: Tensor indices
    :param tensor_shape: Shape of the tensor
    :return: Vector index of the tensor element.
    """
    vector_index = tensor_indices[0]
    m = 1
    for i in range(1, order):
        m = m * tensor_shape[i - 1]
        vector_index = vector_index + (tensor_shape[i] - 1) * m

    return vector_index


def get_gramian(G: List[Union[np.ndarray, torch.Tensor]],
                active_columns: List[int],
                tensor_shape: List[int],
                device: Union[torch.device, str] = 'cpu') -> Union[np.ndarray, torch.Tensor]:
    """
    Obtain a Gram matrix by selecting a subset of columns from a kronecker matrix given by kronecker product of
    matrices in the Kron_Cell_Array.

    :param G: List of factor matrices as numpy arrays or torch tensors.
    :param active_columns: Active indices of the columns.
    :param tensor_shape: Shape of the tensor.
    :param device: The device "CPU" or "GPU" to be used in the calculations (only applicable for torch).
    :return: The Gramian Matrix.
    """

    if G is None:
        raise ValueError("The input list G is None.")

    if isinstance(G[0], torch.Tensor):
        return tpt._get_gramian(G, active_columns, tensor_shape, device)
    else:
        return npt._get_gramian(G, active_columns, tensor_shape)


def tround(tensor: Union[np.ndarray, torch.Tensor], precision_order: int = 0) -> Union[np.ndarray, torch.Tensor]:
    """
    Round tensor elements to a specific precision.

    :param tensor: Tensor to be rounded.
    :param precision_order: Required precision.
    :return: Rounded tensor.
    """
    if tensor is None:
        raise ValueError("The input tensor is None.")

    if isinstance(tensor, torch.Tensor):
        return tpt._tround(tensor, precision_order)
    else:
        return npt._tround(tensor, precision_order)