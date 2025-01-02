import numpy as np
import torch
from tensor_ml.tensorops._tensor_products import kronecker_product, khatri_rao_product, hadamard_product


class TestTensorProducts:

    def setup_method(self):
        self.factor_matrices_np = [
            np.array([[1, 2, 3], [4, 5, 6]]),
            np.array([[7, 8, 9], [10, 11, 12]]),
            np.array([[13, 14, 15], [16, 17, 18]])
        ]

    def test_kronecker_product(self):
        # Test with NumPy
        result_np = kronecker_product(self.factor_matrices_np)
        assert isinstance(result_np, np.ndarray)
        assert result_np.shape == (8, 27)
        expected_column_0_np = np.array([91, 364, 130, 520, 112, 448, 160, 640])
        assert np.allclose(result_np[:, 0],
                           expected_column_0_np), "NumPy: Column 0 values do not match the expected values"

        # Convert factor matrices to Torch
        factor_matrices_torch = [torch.tensor(matrix, dtype=torch.double) for matrix in self.factor_matrices_np]

        # Test with Torch
        result_torch = kronecker_product(factor_matrices_torch)
        assert isinstance(result_torch, torch.Tensor)
        assert result_torch.shape == (8, 27)
        expected_column_0_torch = torch.tensor([91, 364, 130, 520, 112, 448, 160, 640], dtype=torch.double)
        assert torch.allclose(result_torch[:, 0],
                              expected_column_0_torch), "Torch: Column 0 values do not match the expected values"

    def test_khatri_rao_product(self):
        # Test with NumPy
        result_np = khatri_rao_product(self.factor_matrices_np)
        assert isinstance(result_np, np.ndarray)
        assert result_np.shape == (8, 3)
        expected_column_0_np = np.array([91, 364, 130, 520, 112, 448, 160, 640])
        assert np.allclose(result_np[:, 0],
                           expected_column_0_np), "NumPy: Column 0 values do not match the expected values"

        # Convert factor matrices to Torch
        factor_matrices_torch = [torch.tensor(matrix, dtype=torch.double) for matrix in self.factor_matrices_np]

        # Test with Torch
        result_torch = khatri_rao_product(factor_matrices_torch)
        assert isinstance(result_torch, torch.Tensor)
        assert result_torch.shape == (8, 3)
        expected_column_0_torch = torch.tensor([91, 364, 130, 520, 112, 448, 160, 640], dtype=torch.double)
        assert torch.allclose(result_torch[:, 0],
                              expected_column_0_torch), "Torch: Column 0 values do not match the expected values"


def test_hadamard_product():
    # Test with NumPy arrays
    tensors_np = [
        np.array([[1, 2], [3, 4]]),
        np.array([[5, 6], [7, 8]]),
        np.array([[9, 10], [11, 12]])
    ]
    result_np = hadamard_product(tensors_np)
    expected_result_np = np.array([[45, 120], [231, 384]])
    assert np.array_equal(result_np, expected_result_np), f"Expected {expected_result_np}, but got {result_np}"

    # Test with PyTorch tensors
    tensors_torch = [
        torch.tensor([[1, 2], [3, 4]], dtype=torch.double),
        torch.tensor([[5, 6], [7, 8]], dtype=torch.double),
        torch.tensor([[9, 10], [11, 12]], dtype=torch.double)
    ]
    result_torch = hadamard_product(tensors_torch)
    expected_result_torch = torch.tensor([[45, 120], [231, 384]], dtype=torch.double)
    assert torch.equal(result_torch, expected_result_torch), f"Expected {expected_result_torch}, but got {result_torch}"

    # Test with mixed NumPy and PyTorch tensors
    tensors_mixed = [
        np.array([[1, 2], [3, 4]]),
        torch.tensor([[5, 6], [7, 8]], dtype=torch.double),
        np.array([[9, 10], [11, 12]])
    ]
    result_mixed = hadamard_product(tensors_mixed)
    expected_result_mixed = np.array([[45, 120], [231, 384]])
    assert np.array_equal(result_mixed, expected_result_mixed), f"Expected {expected_result_mixed}, but got {result_mixed}"