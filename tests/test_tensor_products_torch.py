import torch
from tensor_ml.tensorops._tensor_products_torch import _kronecker_product_torch, _khatri_rao_product_torch, \
    _tensor_product_torch


class TestTensorProductsTorch:

    def setup_method(self):
        self.factor_matrices = [
            torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.double),
            torch.tensor([[7, 8, 9], [10, 11, 12]], dtype=torch.double),
            torch.tensor([[13, 14, 15], [16, 17, 18]], dtype=torch.double)
        ]
        self.column_indices = [0, 1, 2]
        self.X = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                               [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
                               [[19, 20, 21], [22, 23, 24], [25, 26, 27]]], dtype=torch.double)
        self.tensor_shape = [3, 3, 3]
        self.active_columns = [0, 1, 2]
        self.active_indices = [0, 1, 2]

    def test_kronecker_product_torch(self):
        result = _kronecker_product_torch(self.factor_matrices)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (8, 27)
        expected_column_0 = torch.tensor([91, 364, 130, 520, 112, 448, 160, 640], dtype=torch.double)
        assert torch.allclose(result[:, 0], expected_column_0), "Column 0 values do not match the expected values"

    def test_kronecker_product_torch_single_matrix(self):
        result = _kronecker_product_torch([self.factor_matrices[0]])
        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, 3)
        assert torch.allclose(result, self.factor_matrices[0]), "Values do not match the expected values"

    def test_kronecker_product_torch_empty_list(self):
        matrices = []
        try:
            _kronecker_product_torch(matrices)
        except ValueError as e:
            assert str(e) == "The list of matrices is empty."

    def test_khatri_rao_product_torch(self):
        result = _khatri_rao_product_torch(self.factor_matrices)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (8, 3)
        expected_column_0 = torch.tensor([91, 364, 130, 520, 112, 448, 160, 640], dtype=torch.double)
        assert torch.allclose(result[:, 0], expected_column_0), "Column 0 values do not match the expected values"

    def test_khatri_rao_product_torch_different_columns(self):
        matrices = [torch.rand(2, 3, dtype=torch.double), torch.rand(2, 4, dtype=torch.double)]
        try:
            _khatri_rao_product_torch(matrices)
        except AssertionError as e:
            assert str(e) == "All matrices must have the same number of columns."


    def test_tensor_product_torch_2d_arrays(self):
        matrices = [
            torch.tensor([[1, 2], [3, 4]], dtype=torch.double),
            torch.tensor([[5, 6], [7, 8]], dtype=torch.double),
            torch.tensor([[9, 10], [11, 12]], dtype=torch.double)
        ]
        result = _tensor_product_torch(matrices)
        expected_shape = (2, 2, 2, 2, 2, 2)
        assert result.shape == expected_shape, f"Expected shape {expected_shape}, but got {result.shape}"

        # Calculate the expected values for a specific column
        expected_column = torch.tensor([45, 55], dtype=torch.double)
        assert torch.allclose(result[:, 0, 0, 0, 0, 0], expected_column), "Column values do not match the expected values"

    def test_tensor_product_torch_1d_arrays(self):
        matrices = [
            torch.tensor([1, 2], dtype=torch.double),
            torch.tensor([3, 4], dtype=torch.double),
            torch.tensor([5, 6], dtype=torch.double)
        ]
        result = _tensor_product_torch(matrices)
        expected_shape = (2, 2, 2)
        assert result.shape == expected_shape, f"Expected shape {expected_shape}, but got {result.shape}"

    def test_tensor_product_torch_single_matrix(self):
        matrix = torch.tensor([[1, 2], [3, 4]], dtype=torch.double)
        result = _tensor_product_torch([matrix])
        expected_shape = (2, 2)
        assert result.shape == expected_shape, f"Expected shape {expected_shape}, but got {result.shape}"
        assert torch.equal(result, matrix), "The result should be the same as the input matrix"

    def test_tensor_product_torch_empty_arrays(self):
        matrices = [
            torch.tensor([], dtype=torch.double),
            torch.tensor([], dtype=torch.double),
            torch.tensor([], dtype=torch.double)
        ]
        try:
            _tensor_product_torch(matrices)
        except ValueError as e:
            assert str(e) == "The list of matrices is empty."