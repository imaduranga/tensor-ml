import numpy as np
from tensor_ml.tensorops._tensor_products_numpy import _kronecker_product_numpy, _khatri_rao_product_numpy, \
    _tensor_product_numpy


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
        self.tensor_shape = [3, 3, 3]
        self.active_columns = [0, 1, 2]
        self.active_indices = [0, 1, 2]

    def test_kronecker_product_numpy(self):
        result = _kronecker_product_numpy(self.factor_matrices)
        assert isinstance(result, np.ndarray)
        assert result.shape == (8, 27)
        expected_column_0 = np.array([91, 364, 130, 520, 112, 448, 160, 640])
        assert np.allclose(result[:, 0], expected_column_0), "Column 0 values do not match the expected values"

    def test_kronecker_product_numpy_single_matrix(self):
        result = _kronecker_product_numpy([self.factor_matrices[0]])
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 3)
        assert np.allclose(result, self.factor_matrices[0]), "Values do not match the expected values"

    def test_kronecker_product_numpy_empty_list(self):
        matrices = []
        try:
            _kronecker_product_numpy(matrices)
        except ValueError as e:
            assert str(e) == "The list of matrices is empty."

    def test_khatri_rao_product_numpy(self):
        result = _khatri_rao_product_numpy(self.factor_matrices)
        assert isinstance(result, np.ndarray)
        assert result.shape == (8, 3)
        expected_column_0 = np.array([91, 364, 130, 520, 112, 448, 160, 640])
        assert np.allclose(result[:, 0], expected_column_0), "Column 0 values do not match the expected values"

    def test_khatri_rao_product_numpy_different_columns(self):
        matrices = [np.random.rand(2, 3), np.random.rand(2, 4)]
        try:
            _khatri_rao_product_numpy(matrices)
        except AssertionError as e:
            assert str(e) == "All matrices must have the same number of columns."

    def test_tensor_product_numpy_2d_arrays(self):
        matrices = [
            np.array([[1, 2], [3, 4]]),
            np.array([[5, 6], [7, 8]]),
            np.array([[9, 10], [11, 12]])
        ]
        result = _tensor_product_numpy(matrices)
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
        result = _tensor_product_numpy(matrices)
        expected_shape = (2, 2, 2)
        assert result.shape == expected_shape, f"Expected shape {expected_shape}, but got {result.shape}"

    def test_tensor_product_numpy_single_matrix(self):
        matrix = np.array([[1, 2], [3, 4]])
        result = _tensor_product_numpy([matrix])
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
            _tensor_product_numpy(matrices)
        except ValueError as e:
            assert str(e) == "The list of matrices is empty."