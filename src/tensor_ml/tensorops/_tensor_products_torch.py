from typing import List
import torch


def _kronecker_product_torch(matrices: List[torch.Tensor]) -> torch.Tensor:
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

def _khatri_rao_product_torch(matrices: List[torch.Tensor]) -> torch.Tensor:
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

    result = torch.hstack([_kronecker_product_torch([matrix[:, i] for matrix in matrices]).reshape(-1, 1) for i in range(num_columns)])

    return result


def _tensor_product_torch(matrices: List[torch.Tensor]) -> torch.Tensor:
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