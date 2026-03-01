"""tensor_ml.tensorops – public API for tensor operations and tensor products."""

# ── Element-wise tensor operations (TensorOps) ──────────────────
from tensor_ml.tensorops.tensor_ops import (
    TensorOps,
    NumpyOps,
    TensorOpsFactory,
)

# ── Tensor products (TensorProducts) ────────────────────────────
from tensor_ml.tensorops.tensor_products import (
    TensorProductsBase,
    TensorProducts,
    TensorProductsFactory,
)

from tensor_ml.tensorops.tensor_products_numpy import NumpyTensorProducts

__all__ = [
    # TensorOps
    "TensorOps", "NumpyOps", "TensorOpsFactory",
    # TensorProducts
    "TensorProductsBase", "TensorProducts", "TensorProductsFactory",
    "NumpyTensorProducts",
]

try:
    from tensor_ml.tensorops.tensor_ops import TorchOps
    from tensor_ml.tensorops.tensor_products_torch import TorchTensorProducts
    __all__ += ["TorchOps", "TorchTensorProducts"]
except ImportError:
    pass
