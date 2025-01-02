from typing import List
import numpy as np


def _kronecker_product_numpy(matrices: List[np.ndarray]) -> np.ndarray:
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


def _khatri_rao_product_numpy(matrices: List[np.ndarray]) -> np.ndarray:
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

    result = np.hstack([_kronecker_product_numpy([matrix[:, i] for matrix in matrices]).reshape(-1, 1) for i in range(num_columns)])

    return result


def _tensor_product_numpy(matrices: List[np.ndarray]) -> np.ndarray:
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