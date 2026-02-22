import torch
import numpy as np
from typing import List, Union, Optional
from tensor_ml.enums import BackendType
from tensor_ml.utils import infer_backend
import tensor_ml.tensorops._tensor_products_numpy as npt
import tensor_ml.tensorops._tensor_products_torch as tpt


def kronecker_product(factor_matrices: List[Union[np.ndarray, torch.Tensor]], backend: Optional[BackendType] = None) -> Union[np.ndarray, torch.Tensor]:
    """
    Compute the Kronecker product of a list of factor matrices from N to 1.
    A = (A^(N) ⊗ A^(N−1) ⊗· · ·⊗A^(2) ⊗ A^(1) )

    :param factor_matrices: List of factor matrices as numpy array or torch tensor.
    :param backend: BackendType (optional, inferred if not provided)
    :return: Kronecker product of all factor matrices.
    """

    if not factor_matrices:
        raise ValueError("The list of factor_matrices is empty.")
    backend = infer_backend(factor_matrices, backend)
    if backend == BackendType.TORCH:
        return tpt._kronecker_product([torch.as_tensor(m) for m in factor_matrices])
    elif backend == BackendType.NUMPY:
        return npt._kronecker_product([np.asarray(m) for m in factor_matrices])
    else:
        raise ValueError(f"Unsupported backend: {backend}")


def khatri_rao_product(matrices: List[Union[np.ndarray, torch.Tensor]], backend: Optional[BackendType] = None) -> Union[np.ndarray, torch.Tensor]:
    """
    Compute the Khatri-Rao product of a list of matrices from N to 1.

    :param matrices: List of matrices (torch.Tensor or numpy.ndarray).
    :param backend: BackendType (optional, inferred if not provided)
    :return: Khatri-Rao product of all matrices (torch.Tensor or np.ndarray).
    """

    if not matrices:
        raise ValueError("The list of matrices is empty.")
    backend = infer_backend(matrices, backend)
    if backend == BackendType.TORCH:
        return tpt._khatri_rao_product([torch.as_tensor(m) for m in matrices])
    elif backend == BackendType.NUMPY:
        return npt._khatri_rao_product([np.asarray(m) for m in matrices])
    else:
        raise ValueError(f"Unsupported backend: {backend}")


def tensor_product(matrices: List[Union[np.ndarray, torch.Tensor]], backend: Optional[BackendType] = None) -> Union[np.ndarray, torch.Tensor]:
    """
    Compute the tensor product of a list of matrices from N to 1.

    :param matrices: List of matrices (torch.Tensor or numpy.ndarray).
    :param backend: BackendType (optional, inferred if not provided)
    :return: Tensor product of all matrices (torch.Tensor or np.ndarray).
    """

    if not matrices:
        raise ValueError("The list of matrices is empty.")
    backend = infer_backend(matrices, backend)
    if backend == BackendType.TORCH:
        return tpt._tensor_product([torch.as_tensor(m) for m in matrices])
    elif backend == BackendType.NUMPY:
        return npt._tensor_product([np.asarray(m) for m in matrices])
    else:
        raise ValueError(f"Unsupported backend: {backend}")


def hadamard_product(tensors: List[Union[np.ndarray, torch.Tensor]], backend: Optional[BackendType] = None) -> Union[np.ndarray, torch.Tensor]:
    """
    Compute the Hadamard product of a list of tensors.

    :param tensors: List of tensors (torch.Tensor or numpy.ndarray).
    :param backend: BackendType (optional, inferred if not provided)
    :return: Hadamard product of all tensors in the list (torch.Tensor or np.ndarray).
    """

    if not tensors:
        raise ValueError("The list of tensors is empty.")
    backend = infer_backend(tensors, backend)
    if backend == BackendType.TORCH:
        tensors = [torch.as_tensor(t) for t in tensors]
    elif backend == BackendType.NUMPY:
        tensors = [np.asarray(t) for t in tensors]
    else:
        raise ValueError(f"Unsupported backend: {backend}")
    result = tensors[0]
    for tensor in tensors[1:]:
        result = result * tensor
    return result


def get_kronecker_matrix_column(factor_matrices: List[Union[np.ndarray, torch.Tensor]], column_indices: List[int], backend: Optional[BackendType] = None) -> Union[np.ndarray, torch.Tensor]:
    """
    This function returns a column of the Kronecker matrix (Kronecker Product of the Factor Matrices) when the
    factor matrices and column indices of each factor matrix is given.
    :param factor_matrices: List of factor matrices as numpy array or torch tensor.
    :param column_indices: List of column indices of factor matrix columns to be used in calculating the
                          column of the Kronecker matrix.
    :param backend: BackendType (optional, inferred if not provided)
    :return: kronecker_column : This is the column of the Kronecker matrix corresponding to the column indices
    """
    if not factor_matrices:
        raise ValueError("The list of factor_matrices is empty.")
    if not column_indices:
        raise ValueError("The list of column_indices is empty.")
    backend = infer_backend(factor_matrices, backend)
    if backend == BackendType.TORCH:
        return tpt._get_kronecker_matrix_column([torch.as_tensor(m) for m in factor_matrices], column_indices)
    elif backend == BackendType.NUMPY:
        return npt._get_kronecker_matrix_column([np.asarray(m) for m in factor_matrices], column_indices)
    else:
        raise ValueError(f"Unsupported backend: {backend}")


def full_multilinear_product(X: Union[np.ndarray, torch.Tensor], factor_matrices: list, use_transpose: bool = False, backend: Optional[BackendType] = None) -> Union[np.ndarray, torch.Tensor]:
    """
    Full multilinear product of the tensor X with factor matrices.
    If use_transpose is set to True, then use the transpose of each factor matrix when calculating the
    full multilinear product.
    :param X: Core Tensor as an N-D Array or torch.Tensor.
    :param factor_matrices: List of factor matrices as numpy arrays or torch tensors.
    :param use_transpose: If True transpose each factor matrix before calculating the multilinear product.
    :param backend: BackendType (optional, inferred if not provided)
    :return: Y  Resultant tensor of the full multilinear product of the tensor X with factor matrices
    """
    if X is None:
        raise ValueError("The input tensor X is empty or None.")
    backend = infer_backend(X, backend)
    if backend == BackendType.TORCH:
        return tpt._full_multilinear_product(torch.as_tensor(X), [torch.as_tensor(m) for m in factor_matrices], use_transpose)
    elif backend == BackendType.NUMPY:
        return npt._full_multilinear_product(np.asarray(X), [np.asarray(m) for m in factor_matrices], use_transpose)
    else:
        raise ValueError(f"Unsupported backend: {backend}")


def kronecker_matrix_vector_product(factor_matrices: List[Union[np.ndarray, torch.Tensor]],
                                    x: Union[np.ndarray, torch.Tensor],
                                    tensor_shape: List[int],
                                    active_columns: List[int],
                                    active_indices: List = None,
                                    use_transpose: bool = False,
                                    device: torch.device = torch.device("cpu"),
                                    backend: Optional[BackendType] = None) -> Union[np.ndarray, torch.Tensor]:
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
    :param backend: BackendType (optional, inferred if not provided)
    :return: y : Result vector of the Kronecker matrix vector product.
    """
    backend = infer_backend(x, backend)
    if backend == BackendType.TORCH:
        return tpt._kronecker_matrix_vector_product([torch.as_tensor(m) for m in factor_matrices], torch.as_tensor(x), tensor_shape, active_columns, active_indices, use_transpose, device)
    elif backend == BackendType.NUMPY:
        return npt._kronecker_matrix_vector_product([np.asarray(m) for m in factor_matrices], np.asarray(x), tensor_shape, active_columns, active_indices, use_transpose)
    else:
        raise ValueError(f"Unsupported backend: {backend}")


def tensorize(x: Union[np.ndarray, torch.Tensor],
              tensor_shape: List[int],
              active_elements: List[int],
              device: torch.device = torch.device("cpu"),
              backend: Optional[BackendType] = None) -> Union[np.ndarray, torch.Tensor]:
    """
    Tensorize a vector based on active elements.
    :param x: The vector with tensor elements.
    :param tensor_shape: Shape of the core tensor X.
    :param active_elements: Active elements of the Tensor X corresponds to elements of x.
    :param device: The device "CPU" or "GPU" to be used in the calculations.
    :param backend: BackendType (optional, inferred if not provided)
    :return: Return the tensor X
    """
    if x is None:
        raise ValueError("The input vector x is None.")
    backend = infer_backend(x, backend)
    if backend == BackendType.TORCH:
        return tpt._tensorize(torch.as_tensor(x), tensor_shape, active_elements, device)
    elif backend == BackendType.NUMPY:
        return npt._tensorize(np.asarray(x), tensor_shape, active_elements)
    else:
        raise ValueError(f"Unsupported backend: {backend}")


def vectorize(X: Union[np.ndarray, torch.Tensor], backend: Optional[BackendType] = None) -> Union[np.ndarray, torch.Tensor]:
    """
    Vectorize a tensor from dimension N to 1.
    Uses Fortran (column-major) order to match the mathematical convention vec(X).
    :param X: The tensor to be vectorized.
    :param backend: BackendType (optional, inferred if not provided)
    :return: Return the vector x
    """
    backend = infer_backend(X, backend)
    if backend == BackendType.TORCH:
        return tpt._vectorize(torch.as_tensor(X))
    elif backend == BackendType.NUMPY:
        return npt._vectorize(np.asarray(X))
    else:
        raise ValueError(f"Unsupported backend: {backend}")


def get_gramian(G: List[Union[np.ndarray, torch.Tensor]],
                active_columns: List[int],
                tensor_shape: List[int],
                device: Union[torch.device, str] = 'cpu',
                backend: Optional[BackendType] = None) -> Union[np.ndarray, torch.Tensor]:
    """
    Obtain a Gram matrix by selecting a subset of columns from a kronecker matrix given by kronecker product of
    matrices in the Kron_Cell_Array.
    :param G: List of factor matrices as numpy arrays or torch tensors.
    :param active_columns: Active indices of the columns.
    :param tensor_shape: Shape of the tensor.
    :param device: The device "CPU" or "GPU" to be used in the calculations (only applicable for torch).
    :param backend: BackendType (optional, inferred if not provided)
    :return: The Gramian Matrix.
    """
    if G is None:
        raise ValueError("The input list G is None.")
    backend = infer_backend(G, backend)
    if backend == BackendType.TORCH:
        return tpt._get_gramian([torch.as_tensor(m) for m in G], active_columns, tensor_shape, device)
    elif backend == BackendType.NUMPY:
        return npt._get_gramian([np.asarray(m) for m in G], active_columns, tensor_shape)
    else:
        raise ValueError(f"Unsupported backend: {backend}")


def tround(tensor, precision_order: int = 0, backend=None):
    """
    Round tensor elements to a specific precision.

    :param tensor: Tensor or scalar to be rounded.
    :param precision_order: Required precision order.
    :param backend: BackendType (optional, inferred if not provided).
    :return: Rounded tensor.
    """
    if isinstance(tensor, (int, float)):
        tensor = np.array(tensor)
    backend = infer_backend(tensor, backend)
    if backend == BackendType.TORCH:
        return tpt._tround(torch.as_tensor(tensor), precision_order)
    elif backend == BackendType.NUMPY:
        return npt._tround(np.asarray(tensor), precision_order)
    else:
        raise ValueError(f"Unsupported backend: {backend}")


def get_vector_index(tensor_indices, tensor_shape):
    """
    Get the linear (Fortran-order) index from tensor subscript indices.

    :param tensor_indices: Tuple of per-mode indices (0-based).
    :param tensor_shape: Shape of the tensor.
    :return: Linear index (0-based, Fortran order).
    """
    return int(np.ravel_multi_index(tensor_indices, tensor_shape, order='F'))


def get_kronecker_factor_column_indices(column_index, tensor_shape):
    """
    Decompose a linear (Fortran-order) index into per-factor column indices.

    :param column_index: Linear index (0-based, Fortran order).
    :param tensor_shape: Shape of the core tensor.
    :return: Tuple of per-factor column indices (0-based).
    """
    return np.unravel_index(column_index, tensor_shape, order='F')


def get_direction_vector(GInv, zI, G, active_columns, add_column_flag,
                        changed_dict_column_index, changed_active_column_index,
                        tensor_shape, precision_order=10, device='cpu', backend=None):
    """
    Update the inverse Gramian using the Schur complement formula and compute
    the direction vector dI = GInv @ zI.

    :param GInv: Current inverse Gramian matrix.
    :param zI: Sign vector of correlations at active columns.
    :param G: List of per-mode Gram matrices.
    :param active_columns: Current active column indices.
    :param add_column_flag: True if a column was added, False if removed.
    :param changed_dict_column_index: Linear index of the changed column.
    :param changed_active_column_index: Position (0-indexed) of the changed column in active_columns.
    :param tensor_shape: Shape of the core tensor.
    :param precision_order: Precision order for rounding.
    :param device: Device for torch computations.
    :param backend: BackendType (optional, inferred from zI).
    :return: (dI, GInv) - direction vector and updated inverse Gramian.
    """
    backend = infer_backend(zI, backend)
    if backend == BackendType.TORCH:
        return tpt._get_direction_vector(
            GInv, zI, [torch.as_tensor(m) for m in G], active_columns,
            add_column_flag, changed_dict_column_index, changed_active_column_index,
            tensor_shape, precision_order, device)
    elif backend == BackendType.NUMPY:
        return npt._get_direction_vector(
            GInv, zI, [np.asarray(m) for m in G], active_columns,
            add_column_flag, changed_dict_column_index, changed_active_column_index,
            tensor_shape, precision_order)
    else:
        raise ValueError(f"Unsupported backend: {backend}")


def tround(tensor: Union[np.ndarray, torch.Tensor], precision_order: int = 0, backend: Optional[BackendType] = None) -> Union[np.ndarray, torch.Tensor]:
    """
    Round tensor elements to a specific precision.
    :param tensor: Tensor to be rounded.
    :param precision_order: Required precision.
    :param backend: BackendType (optional, inferred if not provided)
    :return: Rounded tensor.
    """
    if tensor is None:
        raise ValueError("The input tensor is None.")
    backend = infer_backend(tensor, backend)
    if backend == BackendType.TORCH:
        return tpt._tround(torch.as_tensor(tensor), precision_order)
    elif backend == BackendType.NUMPY:
        return npt._tround(np.asarray(tensor), precision_order)
    else:
        raise ValueError(f"Unsupported backend: {backend}")
