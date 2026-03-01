import numpy as np
import pytest
from tensor_ml.tensor_models.multilinear.multilinear_model import MultilinearModel
from tensor_ml.enums import BackendType

class DummyModel(MultilinearModel):
    def __init__(self, backend=None, device=None):
        super().__init__(backend=backend, device=device)
        self._fit_called = False
        self._predict_value = None
    def fit(self, X, y=None, **kwargs):
        self._fit_called = True
        return self
    def predict(self, X, **kwargs):
        # For testing, just return a constant or echo input
        if self._predict_value is not None:
            return self._predict_value
        return X

class TestMultilinearModel:
    def test_get_backend_numpy(self):
        model = DummyModel(backend='numpy')
        arr = np.zeros((2,2))
        assert model.get_backend(arr) == BackendType.NUMPY

    def test_normalize_input_numpy(self):
        model = DummyModel(backend='numpy')
        arr = np.array([[1,2],[3,4]])
        assert np.all(model.normalize_input(arr) == arr)

    def test_score_perfect(self):
        model = DummyModel(backend='numpy')
        y_true = np.array([1,2,3])
        model._predict_value = y_true.copy()
        assert model.score(None, y_true) == 1.0

    def test_score_zero(self):
        model = DummyModel(backend='numpy')
        y_true = np.array([1,2,3])
        model._predict_value = np.array([0,0,0])
        assert model.score(None, y_true) < 1.0

    def test_mse(self):
        model = DummyModel(backend='numpy')
        y_true = np.array([1,2,3])
        model._predict_value = np.array([1,2,4])
        assert np.isclose(model.mse(None, y_true), 1/3)

    def test_mae(self):
        model = DummyModel(backend='numpy')
        y_true = np.array([1,2,3])
        model._predict_value = np.array([1,2,4])
        assert np.isclose(model.mae(None, y_true), 1/3)

    @pytest.mark.skipif('torch' not in globals(), reason='torch not available')
    def test_backend_torch(self):
        import torch
        model = DummyModel(backend='torch')
        arr = torch.zeros(2,2)
        assert model.get_backend(arr) == BackendType.TORCH
        arr2 = model.normalize_input(arr)
        assert isinstance(arr2, torch.Tensor)

    def test_backend_pandas_input_to_numpy(self):
        """Pandas DataFrames passed to a NUMPY backend are converted to ndarray."""
        try:
            import pandas as pd
        except ImportError:
            pytest.skip('pandas not available')
        model = DummyModel(backend='numpy')
        df = pd.DataFrame([[1, 2], [3, 4]])
        arr = model.normalize_input(df)
        assert isinstance(arr, np.ndarray)

