import numpy as np
import tensor_ml.tensorops._tensor_products_numpy as npt

class TestTensorProductsNumpy:

    def setup_method(self):
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
        result = npt._kronecker_product(self.factor_matrices)
        assert isinstance(result, np.ndarray)
        assert result.shape == (8, 27)
        expected_column_0 = np.array([91, 364, 130, 520, 112, 448, 160, 640])
        assert np.allclose(result[:, 0], expected_column_0), "Column 0 values do not match the expected values"

    def test_kronecker_product_numpy_single_matrix(self):
        result = npt._kronecker_product([self.factor_matrices[0]])
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 3)
        assert np.allclose(result, self.factor_matrices[0]), "Values do not match the expected values"

    def test_kronecker_product_numpy_empty_list(self):
        matrices = []
        try:
            npt._kronecker_product(matrices)
        except ValueError as e:
            assert str(e) == "The list of matrices is empty."

    def test_khatri_rao_product_numpy(self):
        result = npt._khatri_rao_product(self.factor_matrices)
        assert isinstance(result, np.ndarray)
        assert result.shape == (8, 3)
        expected_column_0 = np.array([91, 364, 130, 520, 112, 448, 160, 640])
        assert np.allclose(result[:, 0], expected_column_0), "Column 0 values do not match the expected values"

    def test_khatri_rao_product_numpy_different_columns(self):
        matrices = [np.random.rand(2, 3), np.random.rand(2, 4)]
        try:
            npt._khatri_rao_product(matrices)
        except AssertionError as e:
            assert str(e) == "All matrices must have the same number of columns."

    def test_tensor_product_numpy_2d_arrays(self):
        matrices = [
            np.array([[1, 2], [3, 4]]),
            np.array([[5, 6], [7, 8]]),
            np.array([[9, 10], [11, 12]])
        ]
        result = npt._tensor_product(matrices)
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
        result = npt._tensor_product(matrices)
        expected_shape = (2, 2, 2)
        assert result.shape == expected_shape, f"Expected shape {expected_shape}, but got {result.shape}"

    def test_tensor_product_numpy_single_matrix(self):
        matrix = np.array([[1, 2], [3, 4]])
        result = npt._tensor_product([matrix])
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
            npt._tensor_product(matrices)
        except ValueError as e:
            assert str(e) == "The list of matrices is empty."

    def test_full_multilinear_product_no_transpose(self):
        result = npt._full_multilinear_product(self.X, self.factor_matrices, use_transpose=False)
        expected_result = np.dot(npt._kronecker_product(self.factor_matrices), npt._vectorize(self.X))
        assert np.allclose(npt._vectorize(result), expected_result), f"Expected {expected_result}, but got {result}"

    def test_full_multilinear_product_with_transpose(self):
        result = npt._full_multilinear_product(self.Y, self.factor_matrices, use_transpose=True)
        expected_result = np.dot(npt._kronecker_product(self.factor_matrices).T, npt._vectorize(self.Y))
        assert np.allclose(npt._vectorize(result), expected_result), f"Expected {expected_result}, but got {result}"

    def test_kronecker_matrix_vector_product_numpy(self):
        x = np.array([3, 5, 7])
        tensor_shape = [3, 3, 3]
        active_columns = [0, 1, 2]
        active_indices = [0, 1, 2]
        result = npt._kronecker_matrix_vector_product(self.factor_matrices, x, tensor_shape, active_columns, active_indices,
                                                      use_transpose=False)
        expected_result = np.array([45, 55, 99, 121, 153, 187, 207, 253])
        assert np.allclose(result, expected_result), f"Expected {expected_result}, but got {result}"