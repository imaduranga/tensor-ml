import numpy as np
from tensor_ml.tensor_ops.tensor_products_numpy import NumpyTensorProducts


class TestTensorProductsNumpy:

    def setup_method(self):
        self.tp = NumpyTensorProducts()
        self.factor_matrices = [
            np.array([[1, 2, 3], [4, 5, 6]]),
            np.array([[7, 8, 9], [10, 11, 12]]),
            np.array([[13, 14, 15], [16, 17, 18]])
        ]
        self.column_indices = [0, 1, 2]
        self.X = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                           [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
                           [[19, 20, 21], [22, 23, 24], [25, 26, 27]]])
        self.Y = np.array([[[1, 2], [4, 5]],
                           [[10, 11], [13, 14]]])
        self.tensor_shape = [3, 3, 3]
        self.active_columns = [0, 1, 2]
        self.active_indices = [0, 1, 2]

    def test_kronecker_product_numpy(self):
        result = self.tp.kronecker_product(self.factor_matrices)
        assert isinstance(result, np.ndarray)
        assert result.shape == (8, 27)
        expected_column_0 = np.array([91, 364, 130, 520, 112, 448, 160, 640])
        assert np.allclose(result[:, 0], expected_column_0), "Column 0 values do not match the expected values"

    def test_kronecker_product_numpy_single_matrix(self):
        result = self.tp.kronecker_product([self.factor_matrices[0]])
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 3)
        assert np.allclose(result, self.factor_matrices[0]), "Values do not match the expected values"

    def test_kronecker_product_numpy_empty_list(self):
        matrices = []
        try:
            self.tp.kronecker_product(matrices)
        except ValueError as e:
            assert str(e) == "The list of matrices is empty."

    def test_khatri_rao_product_numpy(self):
        result = self.tp.khatri_rao_product(self.factor_matrices)
        assert isinstance(result, np.ndarray)
        assert result.shape == (8, 3)
        expected_column_0 = np.array([91, 364, 130, 520, 112, 448, 160, 640])
        assert np.allclose(result[:, 0], expected_column_0), "Column 0 values do not match the expected values"

    def test_khatri_rao_product_numpy_different_columns(self):
        matrices = [np.random.rand(2, 3), np.random.rand(2, 4)]
        try:
            self.tp.khatri_rao_product(matrices)
        except ValueError as e:
            assert "same number of columns" in str(e)

    def test_tensor_product_numpy_2d_arrays(self):
        matrices = [
            np.array([[1, 2], [3, 4]]),
            np.array([[5, 6], [7, 8]]),
            np.array([[9, 10], [11, 12]])
        ]
        result = self.tp.tensor_product(matrices)
        expected_shape = (2, 2, 2, 2, 2, 2)
        assert result.shape == expected_shape, f"Expected shape {expected_shape}, but got {result.shape}"

        # Calculate the expected values for a specific column
        expected_column = np.array([45, 55])
        assert np.allclose(result[:, 0, 0, 0, 0, 0], expected_column), "Column values do not match the expected values"

    def test_tensor_product_numpy_1d_arrays(self):
        matrices = [
            np.array([1, 2]),
            np.array([3, 4]),
            np.array([5, 6])
        ]
        result = self.tp.tensor_product(matrices)
        expected_shape = (2, 2, 2)
        assert result.shape == expected_shape, f"Expected shape {expected_shape}, but got {result.shape}"

    def test_tensor_product_numpy_single_matrix(self):
        matrix = np.array([[1, 2], [3, 4]])
        result = self.tp.tensor_product([matrix])
        expected_shape = (2, 2)
        assert result.shape == expected_shape, f"Expected shape {expected_shape}, but got {result.shape}"
        assert np.array_equal(result, matrix), "The result should be the same as the input matrix"

    def test_tensor_product_numpy_empty_arrays(self):
        matrices = [
            np.array([]),
            np.array([]),
            np.array([])
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

    def test_kronecker_matrix_vector_product_numpy(self):
        x = np.array([3.0, 5.0, 7.0])
        tensor_shape = [3, 3, 3]
        active_columns = [0, 1, 2]

        # Without sub-tensor optimization (active_indices=None)
        result = self.tp.kronecker_matrix_vector_product(
            self.factor_matrices, x, tensor_shape, active_columns,
            active_indices=None, use_transpose=False)

        # Verify against full kronecker product: y = B[:, active_cols] @ x
        B = self.tp.kronecker_product(self.factor_matrices)
        expected = B[:, active_columns] @ x
        assert np.allclose(result, expected), f"Without sub-tensor: Expected {expected}, but got {result}"

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
        assert np.allclose(result_sub, expected), f"With sub-tensor: Expected {expected}, but got {result_sub}"