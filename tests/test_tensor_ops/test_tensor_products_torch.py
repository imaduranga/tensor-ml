import numpy as np
import pytest
from tensor_ml.tensor_ops.tensor_products_torch import TorchTensorProducts

torch = pytest.importorskip("torch")


class TestTensorProductsTorch:

    def setup_method(self):
        self.tp = TorchTensorProducts()
        self.factor_matrices = [
            torch.tensor([[1, 2, 3], [4, 5, 6]]),
            torch.tensor([[7, 8, 9], [10, 11, 12]]),
            torch.tensor([[13, 14, 15], [16, 17, 18]])
        ]
        self.column_indices = [0, 1, 2]
        self.X = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                           [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
                           [[19, 20, 21], [22, 23, 24], [25, 26, 27]]])
        self.Y = torch.tensor([[[1, 2], [4, 5]],
                           [[10, 11], [13, 14]]])
        self.tensor_shape = [3, 3, 3]
        self.active_columns = [0, 1, 2]
        self.active_indices = [0, 1, 2]

    def test_kronecker_product_torch(self):
        result = self.tp.kronecker_product(self.factor_matrices)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (8, 27)
        expected_column_0 = torch.tensor([91, 364, 130, 520, 112, 448, 160, 640])
        assert torch.allclose(result[:, 0], expected_column_0), "Column 0 values do not match the expected values"

    def test_kronecker_product_torch_single_matrix(self):
        result = self.tp.kronecker_product([self.factor_matrices[0]])
        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, 3)
        assert torch.allclose(result, self.factor_matrices[0]), "Values do not match the expected values"

    def test_kronecker_product_torch_empty_list(self):
        matrices = []
        try:
            self.tp.kronecker_product(matrices)
        except ValueError as e:
            assert str(e) == "The list of matrices is empty."

    def test_khatri_rao_product_torch(self):
        result = self.tp.khatri_rao_product(self.factor_matrices)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (8, 3)
        expected_column_0 = torch.tensor([91, 364, 130, 520, 112, 448, 160, 640])
        assert torch.allclose(result[:, 0], expected_column_0), "Column 0 values do not match the expected values"

    def test_khatri_rao_product_torch_different_columns(self):
        matrices = [torch.rand(2, 3, dtype=torch.double), torch.rand(2, 4, dtype=torch.double)]
        try:
            self.tp.khatri_rao_product(matrices)
        except ValueError as e:
            assert "same number of columns" in str(e)


    def test_tensor_product_torch_2d_arrays(self):
        matrices = [
            torch.tensor([[1, 2], [3, 4]], dtype=torch.double),
            torch.tensor([[5, 6], [7, 8]], dtype=torch.double),
            torch.tensor([[9, 10], [11, 12]], dtype=torch.double)
        ]
        result = self.tp.tensor_product(matrices)
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
        result = self.tp.tensor_product(matrices)
        expected_shape = (2, 2, 2)
        assert result.shape == expected_shape, f"Expected shape {expected_shape}, but got {result.shape}"

    def test_tensor_product_torch_single_matrix(self):
        matrix = torch.tensor([[1, 2], [3, 4]], dtype=torch.double)
        result = self.tp.tensor_product([matrix])
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
            self.tp.tensor_product(matrices)
        except ValueError as e:
            assert str(e) == "The list of matrices is empty."

    def test_full_multilinear_product_no_transpose(self):
        result = self.tp.full_multilinear_product(self.X, self.factor_matrices, use_transpose=False)
        expected_result = np.dot(self.tp.kronecker_product(self.factor_matrices), self.tp.vectorize(self.X))
        assert np.allclose(self.tp.vectorize(result), expected_result), f"Expected {expected_result}, but got {result}"

    def test_full_multilinear_product_with_transpose(self):
        result = self.tp.full_multilinear_product(self.Y, self.factor_matrices, use_transpose=True)
        expected_result = np.dot(self.tp.kronecker_product(self.factor_matrices).T, self.tp.vectorize(self.Y))
        assert np.allclose(self.tp.vectorize(result), expected_result), f"Expected {expected_result}, but got {result}"

    def test_kronecker_matrix_vector_product(self):
        x = torch.tensor([3.0, 5.0, 7.0], dtype=torch.double)
        tensor_shape = [3, 3, 3]
        active_columns = [0, 1, 2]

        # Without sub-tensor optimization (active_indices=None)
        result = self.tp.kronecker_matrix_vector_product(
            self.factor_matrices, x, tensor_shape, active_columns,
            active_indices=None, use_transpose=False)

        # Verify against full kronecker product: y = B[:, active_cols] @ x
        B = self.tp.kronecker_product(self.factor_matrices).to(dtype=torch.double)
        expected = B[:, active_columns] @ x
        assert np.allclose(result.numpy(), expected.numpy()), f"Without sub-tensor: Expected {expected}, but got {result}"

        # With sub-tensor optimization
        all_factor_indices = [set() for _ in range(len(self.factor_matrices))]
        for col in active_columns:
            inds = np.unravel_index(col, tensor_shape, order='F')
            for n in range(len(self.factor_matrices)):
                all_factor_indices[n].add(inds[n])
        active_indices = [sorted(s) for s in all_factor_indices]

        result_sub = self.tp.kronecker_matrix_vector_product(
            self.factor_matrices, x, tensor_shape, active_columns,
            active_indices=active_indices, use_transpose=False)
        assert np.allclose(result_sub.numpy(), expected.numpy()), f"With sub-tensor: Expected {expected}, but got {result_sub}"