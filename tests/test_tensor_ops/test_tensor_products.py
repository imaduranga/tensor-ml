import numpy as np
import pytest
from tensor_ml import TensorProducts as tp

torch = pytest.importorskip("torch")


class TestTensorProducts:

    def setup_method(self):
        self.factor_matrices_np = [
            np.array([[1, 2, 3], [4, 5, 6]]),
            np.array([[7, 8, 9], [10, 11, 12]]),
            np.array([[13, 14, 15], [16, 17, 18]])
        ]
        self.factor_matrices_torch = [torch.tensor(matrix, dtype=torch.double) for matrix in self.factor_matrices_np]

    def test_kronecker_product(self):
        # Test with NumPy
        result_np = tp.kronecker_product(self.factor_matrices_np)
        assert isinstance(result_np, np.ndarray)
        assert result_np.shape == (8, 27)
        expected_column_0_np = np.array([91, 364, 130, 520, 112, 448, 160, 640])
        assert np.allclose(result_np[:, 0],
                           expected_column_0_np), "NumPy: Column 0 values do not match the expected values"

        # Test with Torch
        result_torch = tp.kronecker_product(self.factor_matrices_torch)
        assert isinstance(result_torch, torch.Tensor)
        assert result_torch.shape == (8, 27)
        expected_column_0_torch = torch.tensor([91, 364, 130, 520, 112, 448, 160, 640], dtype=torch.double)
        assert torch.allclose(result_torch[:, 0],
                              expected_column_0_torch), "Torch: Column 0 values do not match the expected values"

        # Test with an empty list
        matrices_empty = []
        try:
            tp.kronecker_product(matrices_empty)
        except ValueError as e:
            assert str(e) == "The list of factor_matrices is empty."

    def test_khatri_rao_product(self):
        # Test with NumPy
        result_np = tp.khatri_rao_product(self.factor_matrices_np)
        assert isinstance(result_np, np.ndarray)
        assert result_np.shape == (8, 3)
        expected_column_0_np = np.array([91, 364, 130, 520, 112, 448, 160, 640])
        assert np.allclose(result_np[:, 0],
                           expected_column_0_np), "NumPy: Column 0 values do not match the expected values"

        # Test with Torch
        result_torch = tp.khatri_rao_product(self.factor_matrices_torch)
        assert isinstance(result_torch, torch.Tensor)
        assert result_torch.shape == (8, 3)
        expected_column_0_torch = torch.tensor([91, 364, 130, 520, 112, 448, 160, 640], dtype=torch.double)
        assert torch.allclose(result_torch[:, 0],
                              expected_column_0_torch), "Torch: Column 0 values do not match the expected values"

        # Test with an empty list
        matrices_empty = []
        try:
            tp.khatri_rao_product(matrices_empty)
        except ValueError as e:
            assert str(e) == "The list of matrices is empty."


    def test_tensor_product(self):
        # Test with NumPy arrays
        result_np = tp.tensor_product(self.factor_matrices_np)
        expected_shape_np = (2, 3, 2, 3, 2, 3)
        assert result_np.shape == expected_shape_np, f"Expected shape {expected_shape_np}, but got {result_np.shape}"
        assert isinstance(result_np, np.ndarray), f"Expected result type np.ndarray, but got {type(result_np)}"

        # Test with PyTorch tensors
        result_torch = tp.tensor_product(self.factor_matrices_torch)
        expected_shape_torch = (2, 3, 2, 3, 2, 3)
        assert result_torch.shape == expected_shape_torch, f"Expected shape {expected_shape_torch}, but got {result_torch.shape}"
        assert isinstance(result_torch,
                          torch.Tensor), f"Expected result type torch.Tensor, but got {type(result_torch)}"

        # Test with mixed NumPy and PyTorch tensors
        matrices_mixed = [
            self.factor_matrices_np[0],
            self.factor_matrices_torch[1],
            self.factor_matrices_np[2]
        ]
        result_mixed = tp.tensor_product(matrices_mixed)
        expected_shape_mixed = (2, 3, 2, 3, 2, 3)
        assert result_mixed.shape == expected_shape_mixed, f"Expected shape {expected_shape_mixed}, but got {result_mixed.shape}"
        assert isinstance(result_mixed, np.ndarray), f"Expected result type np.ndarray, but got {type(result_mixed)}"

        # Test with a single matrix
        matrix_single = self.factor_matrices_np[0]
        result_single = tp.tensor_product([matrix_single])
        expected_shape_single = (2, 3)
        assert result_single.shape == expected_shape_single, f"Expected shape {expected_shape_single}, but got {result_single.shape}"
        assert np.array_equal(result_single, matrix_single), "The result should be the same as the input matrix"

        # Test with an empty list
        matrices_empty = []
        try:
            tp.tensor_product(matrices_empty)
        except ValueError as e:
            assert str(e) == "The list of matrices is empty."


def test_hadamard_product():
    # Test with NumPy arrays
    tensors_np = [
        np.array([[1, 2], [3, 4]]),
        np.array([[5, 6], [7, 8]]),
        np.array([[9, 10], [11, 12]])
    ]
    result_np = tp.hadamard_product(tensors_np)
    expected_result_np = np.array([[45, 120], [231, 384]])
    assert np.array_equal(result_np, expected_result_np), f"Expected {expected_result_np}, but got {result_np}"

    # Test with PyTorch tensors
    tensors_torch = [
        torch.tensor([[1, 2], [3, 4]], dtype=torch.double),
        torch.tensor([[5, 6], [7, 8]], dtype=torch.double),
        torch.tensor([[9, 10], [11, 12]], dtype=torch.double)
    ]
    result_torch = tp.hadamard_product(tensors_torch)
    expected_result_torch = torch.tensor([[45, 120], [231, 384]], dtype=torch.double)
    assert torch.equal(result_torch, expected_result_torch), f"Expected {expected_result_torch}, but got {result_torch}"

    # Test with mixed NumPy and PyTorch tensors
    tensors_mixed = [
        np.array([[1, 2], [3, 4]]),
        torch.tensor([[5, 6], [7, 8]], dtype=torch.double),
        np.array([[9, 10], [11, 12]])
    ]
    result_mixed = tp.hadamard_product(tensors_mixed)
    expected_result_mixed = np.array([[45, 120], [231, 384]])
    assert np.array_equal(result_mixed, expected_result_mixed), f"Expected {expected_result_mixed}, but got {result_mixed}"

    # Test with an empty list
    matrices_empty = []
    try:
        tp.hadamard_product(matrices_empty)
    except ValueError as e:
        assert str(e) == "The list of tensors is empty."