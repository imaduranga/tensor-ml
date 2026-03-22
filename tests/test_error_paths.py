"""Tests for error paths and input validation across tensor-ml."""

import numpy as np
import pytest

from tensor_ml.exceptions import (
    BackendError,
    NotFittedError,
    ShapeMismatchError,
    TensorMLError,
    ValidationError,
)


# ── Exception hierarchy ────────────────────────────────────────────

class TestExceptionHierarchy:
    """All custom exceptions inherit from TensorMLError."""

    def test_not_fitted_is_tensor_ml_error(self):
        assert issubclass(NotFittedError, TensorMLError)

    def test_backend_error_is_tensor_ml_error(self):
        assert issubclass(BackendError, TensorMLError)

    def test_shape_mismatch_is_value_error(self):
        assert issubclass(ShapeMismatchError, (TensorMLError, ValueError))

    def test_validation_error_is_value_error(self):
        assert issubclass(ValidationError, (TensorMLError, ValueError))

    def test_catch_all_via_base(self):
        with pytest.raises(TensorMLError):
            raise ValidationError("test")


# ── TLARSConfig validation ────────────────────────────────────────

class TestTLARSConfigValidation:

    def test_negative_tolerance(self):
        from tensor_ml.tensor_models import TLARSConfig
        with pytest.raises(Exception):  # Pydantic ValidationError
            TLARSConfig(tolerance=-1)

    def test_zero_tolerance(self):
        from tensor_ml.tensor_models import TLARSConfig
        with pytest.raises(Exception):
            TLARSConfig(tolerance=0)

    def test_invalid_mask_type(self):
        from tensor_ml.tensor_models import TLARSConfig
        with pytest.raises(Exception):
            TLARSConfig(mask_type="INVALID")

    def test_negative_iterations(self):
        from tensor_ml.tensor_models import TLARSConfig
        with pytest.raises(Exception):
            TLARSConfig(iterations=-5)

    def test_valid_config(self):
        from tensor_ml.tensor_models import TLARSConfig
        cfg = TLARSConfig(tolerance=0.1, mask_type="KR", l0_mode=True)
        assert cfg.tolerance == 0.1
        assert cfg.mask_type == "KR"
        assert cfg.l0_mode is True


# ── TLARS error paths ─────────────────────────────────────────────

class TestTLARSErrors:

    def test_predict_before_fit(self):
        from tensor_ml import TLARS
        model = TLARS()
        with pytest.raises(NotFittedError):
            model.predict([np.eye(3)])

    def test_fit_empty_factor_matrices(self):
        from tensor_ml import TLARS
        model = TLARS()
        with pytest.raises(ValidationError):
            model.fit(factor_matrices=[], Y=np.ones((3, 3)))

    def test_fit_none_Y(self):
        from tensor_ml import TLARS
        model = TLARS()
        with pytest.raises(ValidationError):
            model.fit(factor_matrices=[np.eye(3)], Y=None)

    def test_fit_shape_mismatch(self):
        from tensor_ml import TLARS
        model = TLARS()
        # 3D target but only 2 factor matrices → mismatch
        Y = np.ones((3, 3, 3))
        with pytest.raises(ShapeMismatchError):
            model.fit(factor_matrices=[np.eye(3), np.eye(3)], Y=Y)

    def test_repr_default(self):
        from tensor_ml import TLARS
        model = TLARS()
        assert repr(model) == "TLARS()"

    def test_repr_custom(self):
        from tensor_ml import TLARS
        model = TLARS(l0_mode=True, tolerance=0.5)
        r = repr(model)
        assert "l0_mode=True" in r
        assert "tolerance=0.5" in r

    def test_get_set_params(self):
        from tensor_ml import TLARS
        model = TLARS(tolerance=0.1)
        params = model.get_params()
        assert params["tolerance"] == 0.1

        model.set_params(tolerance=0.5, l0_mode=True)
        params = model.get_params()
        assert params["tolerance"] == 0.5
        assert params["l0_mode"] is True

    def test_cpu_cuda_chaining(self):
        from tensor_ml import TLARS
        model = TLARS()
        result = model.cpu()
        assert result is model  # returns self


# ── Backend errors ─────────────────────────────────────────────────

class TestBackendErrors:

    def test_infer_backend_bad_type(self):
        from tensor_ml.utils import infer_backend
        with pytest.raises(BackendError, match="Cannot infer backend"):
            infer_backend("not_an_array")

    def test_infer_backend_explicit(self):
        from tensor_ml.utils import infer_backend
        from tensor_ml.enums import BackendType
        result = infer_backend("anything", backend=BackendType.NUMPY)
        assert result == BackendType.NUMPY


# ── Tensor products validation ─────────────────────────────────────

class TestTensorProductsValidation:

    def test_kronecker_empty_list(self):
        from tensor_ml import TensorProducts
        with pytest.raises(ValueError):
            TensorProducts.kronecker_product([])

    def test_khatri_rao_empty_list(self):
        from tensor_ml import TensorProducts
        with pytest.raises(ValueError):
            TensorProducts.khatri_rao_product([])

    def test_khatri_rao_column_mismatch(self):
        from tensor_ml import NumpyTensorProducts
        tp = NumpyTensorProducts()
        A = np.ones((3, 2))
        B = np.ones((3, 5))
        with pytest.raises(ValueError, match="same number of columns"):
            tp.khatri_rao_product([A, B])

    def test_hadamard_empty_list(self):
        from tensor_ml import TensorProducts
        with pytest.raises(ValueError):
            TensorProducts.hadamard_product([])

    def test_full_multilinear_product_none_X(self):
        from tensor_ml import TensorProducts
        with pytest.raises(ValueError):
            TensorProducts.full_multilinear_product(None, [np.eye(3)])

    def test_tensorize_none_x(self):
        from tensor_ml import TensorProducts
        with pytest.raises(ValueError):
            TensorProducts.tensorize(None, (3,), np.array([0]))
