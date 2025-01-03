import torch
import numpy as np
from string import ascii_lowercase as letters
from typing import List, Union

from tensor_ml.tensorops._tensor_products_numpy import _kronecker_product_numpy, _khatri_rao_product_numpy, \
    _tensor_product_numpy
from tensor_ml.tensorops._tensor_products_torch import _kronecker_product_torch, _khatri_rao_product_torch, \
    _tensor_product_torch


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
        return _kronecker_product_torch(factor_matrices)
    else:
        return _kronecker_product_numpy(factor_matrices)


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
        return _khatri_rao_product_torch(matrices)
    else:
        matrices = [np.array(matrix) if not isinstance(matrix, np.ndarray) else matrix for matrix in matrices]
        return _khatri_rao_product_numpy(matrices)


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
        return _tensor_product_torch(matrices)
    else:
        matrices = [np.array(matrix) if not isinstance(matrix, np.ndarray) else matrix for matrix in matrices]
        return _tensor_product_numpy(matrices)


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


def get_kronecker_matrix_column(factor_matrices: list, column_indices: list):
    """
    This function returns a column of the Kronecker matrix (Kronecker Product of the Factor Matrices) when the
    factor matrices and column indices of each factor matrix is given.

    :param factor_matrices: List of factor matrices as numpy array or torch tensor.
    :param column_indices: This field requires a list of column indices of factor matrix columns to be used in calculating the
                    column of the Kronecker matrix.
    :return:  kronecker_column : This is the column of the Kronecker matrix corresponding to the column indices
    """

    # Convert factor matrices to torch tensors if they are not already
    factor_matrices = [torch.tensor(fm) if isinstance(fm, np.ndarray) else fm for fm in factor_matrices]

    kronecker_column = factor_matrices[0][:, column_indices[-1]]
    for n in range(1, len(factor_matrices)):
        vec = factor_matrices[n][:, column_indices[-n - 1]]
        kronecker_column = torch.mm(vec.unsqueeze(1), kronecker_column.unsqueeze(1).t())
        kronecker_column = kronecker_column.flatten()
    return kronecker_column


def full_multilinear_product(X: np.ndarray, factor_matrices: list, use_transpose: bool = False):
    """
     Full multilinear product of the tensor X with factor matrices.
        If use_transpose is set to True, then use the transpose of each factor matrix when calculating the
        full multilinear product.

    :param X: Core Tensor as an N-D Array.
    :param factor_matrices: List of factor matrices as numpy arrays.
    :param use_transpose: If True transpose each facto matrix before calculating the multilinear product.
    :return: Y  Resultant tensor of the full multilinear product of the tensor X with factor matrices
    """

    order = X.ndim
    Y = X.copy()

    for n in range(order):

        if use_transpose:
            oper = letters[:order][n] + "z"
        else:
            oper = "z" + letters[:order][n]

        op = letters[:order] + "," + oper + "->" + letters[:order][:n] + 'z' + letters[:order][n + 1:]
        Y = torch.einsum(op, Y, factor_matrices[-n - 1])

    return Y


def kronecker_matrix_vector_product(factor_matrices: list,
                                    x,
                                    tensor_shape: list,
                                    active_columns: list,
                                    active_indices: list,
                                    use_transpose: bool = False,
                                    device: torch.device = torch.device("cpu")):
    """
    kronecker_matrix_vector_product function calculates the product
    between a matrix A and a vector x (y = Ax) using full multilinear product.
    Columns of matrix A can be obtained by kronecker product of respective columns in the kron_cell_array.
    If matrix B is the kronecker product of all factor matrices in Factor_Matrices, columns of A is a subset of columns of B.
    If transpose = True, y = A'x is calculated

    :param factor_matrices: List of factor matrices as numpy arrays.
    :param x: The vector that is going to be multiplied with the Kronecker matrix.
    :param tensor_shape: List of tensor columns (column indices of x as a core tensor X) to be used in the multiplication.
    :param active_columns: Active indices of the factor matrices to be used in the multiplications.
    :param active_indices: Shape of the core tensor X.
    :param use_transpose: If True transpose each factor matrix before calculating the multilinear product.
    :param device: The device "CPU" or "GPU" to be used in the calculations.
    :return: y : Result vector of the Kronecker matrix vector product.
    """

    X = torch.zeros(np.prod(tensor_shape), dtype=torch.double, device=device)
    X[active_columns] = x
    X = X.reshape(tensor_shape)

    if use_transpose:
        factor_matrices = [factor_matrices[i][active_indices[-i - 1], :] for i in range(len(factor_matrices))]
    else:
        factor_matrices = [factor_matrices[i][:, active_indices[-i - 1]] for i in range(len(factor_matrices))]

    for i in range(len(factor_matrices)):
        X = torch.index_select(X, i, torch.tensor(active_indices[i], device=device))

    Y = full_multilinear_product(X, factor_matrices, use_transpose)
    y = Y.flatten()
    return y


def kronecker_matrix_partial_vector_product(factor_matrices: list,
                                            x,
                                            tensor_shape: list,
                                            active_columns: list,
                                            active_indices: list,
                                            use_transpose: bool = False,
                                            device: torch.device = torch.device("cpu")):
    """
    kroneckerMatrixPartialVectorProduct function calculates the product
    between a matrix A and a vector x (y = Ax) using full multilinear product.
    Columns of matrix A can be obtained by kronecker product of respective columns in the kron_cell_array.
    If matrix B is the kronecker product of all factor matrices in Factor_Matrices, columns of A is a subset of columns of B.
    If transpose = True, y = A'x is calculated

    :param factor_matrices: List of factor matrices as numpy arrays.
    :param x: The vector that is going to be multiplied with the Kronecker matrix.
    :param tensor_shape: List of tensor columns (column indices of x as a core tensor X) to be used in the multiplication.
    :param active_columns: Active indices of the factor matrices to be used in the multiplications.
    :param active_indices: Shape of the core tensor X.
    :param use_transpose: If True transpose each factor matrix before calculating the multilinear product.
    :param device: The device "CPU" or "GPU" to be used in the calculations.
    :return: y : Result vector of the Kronecker matrix vector product.
    """

    X = torch.zeros(np.prod(tensor_shape), dtype=torch.double, device=device)
    X[active_columns] = x
    X = X.reshape(tensor_shape)

    if use_transpose:
        factor_matrices = [factor_matrices[i][active_indices[-i - 1], :] for i in range(len(factor_matrices))]
    else:
        factor_matrices = [factor_matrices[i][:, active_indices[-i - 1]] for i in range(len(factor_matrices))]

    for i in range(len(factor_matrices)):
        X = torch.index_select(X, i, torch.tensor(active_indices[i], device=device))

    Y = full_multilinear_product(X, factor_matrices, use_transpose)
    y = Y.flatten()
    return y


def tensorize(x,
              tensor_shape,
              active_elements,
              device: torch.device = torch.device("cpu")):
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


def get_reverse_shape(tensor_shape):
    """
    Calculate the reversed shape of the tensor(Torch tensor shapes are reversed).
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


def get_gramian(G, active_columns, tensor_shape, device):
    """
    Obtain a Gram matrix by selecting a subset of columns from a kronecker matrix given by kronecker product of
    matrices in the Kron_Cell_Array

    :param G:
    :param active_columns:
    :param tensor_shape:
    :param device:
    :return: The Gamian Matrix
    """

    GI = torch.zeros(len(active_columns), len(active_columns), dtype=torch.double, device=device)

    for i in range(len(active_columns)):
        indices = np.unravel_index(active_columns[i].item(), tensor_shape)
        gk = get_kronecker_matrix_column(G, indices)
        gk = gk[active_columns]
        GI[:, i] = torch.tensor(gk, device=device)

    return GI


def tround(tensor, precision_order=0):
    """
    Round tensor elements to a specific precision.

    :param tensor: Tensor to be rounded.
    :param precision_order: Require Precision
    :return: Return the rounded tensor
    """
    """round tensor to the given decimal places"""
    return torch.round(tensor * 10 ** precision_order) * 10 ** -precision_order
