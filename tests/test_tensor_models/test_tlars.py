import numpy as np
import pytest
from tensor_ml.tensor_models.multilinear.tlars import TLARS

class TestTLARS:
    def test_tlars_numpy(self):
        # Simple synthetic data
        Y = np.random.randn(10, 10, 10)
        factor_matrices = [np.random.randn(10, 5) for _ in range(3)]
        model = TLARS(backend='numpy', iterations=50, tolerance=1e-2)
        model.fit(factor_matrices, Y)
        assert model.coef_tensor_ is not None
        assert model.active_columns_ is not None
        assert model.coef_ is not None
        # Predict and check shape
        Y_pred = model.predict(factor_matrices)
        assert Y_pred.shape == Y.shape
        # Score should be between -inf and 1
        score = model.score(factor_matrices, Y)
        assert isinstance(score, float)
        assert score <= 1.0

    @pytest.mark.skipif('torch' not in globals(), reason='torch not available')
    def test_tlars_torch(self):
        import torch
        Y = torch.randn(8, 8, 8)
        factor_matrices = [torch.randn(8, 4) for _ in range(3)]
        model = TLARS(backend='torch', device='cpu', iterations=30, tolerance=1e-2)
        model.fit(factor_matrices, Y)
        assert model.coef_tensor_ is not None
        assert model.active_columns_ is not None
        assert model.coef_ is not None
        Y_pred = model.predict(factor_matrices)
        assert Y_pred.shape == Y.shape
        score = model.score(factor_matrices, Y)
        assert isinstance(score, float)
        assert score <= 1.0

