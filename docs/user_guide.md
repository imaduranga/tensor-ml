# tensor-ml User Guide

A comprehensive guide to the tensor-ml library — concepts, architecture, usage patterns, and extension points.

---

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Core Concepts](#core-concepts)
4. [Backend System](#backend-system)
5. [Tensor Operations](#tensor-operations)
6. [Tensor Products](#tensor-products)
7. [T-LARS: Sparse Tensor Recovery](#t-lars-sparse-tensor-recovery)
8. [Device Management](#device-management)
9. [Error Handling](#error-handling)
10. [Logging & Debugging](#logging--debugging)
11. [Extending tensor-ml](#extending-tensor-ml)

---

## Overview

**tensor-ml** is a Python library for tensor-based machine learning. It provides:

- **Element-wise tensor operations** (norm, normalize, argmax, …) via a backend-agnostic API.
- **Tensor products** (Kronecker, Khatri-Rao, Hadamard, full multilinear) with automatic backend dispatch.
- **T-LARS** — a Tensor Least Angle Regression and Selection algorithm for sparse tensor recovery from compressed measurements.

The library auto-detects whether inputs are NumPy arrays or PyTorch tensors and dispatches to the correct backend transparently.

---

## Installation

```bash
uv sync --all-extras       # recommended: creates .venv and installs all deps
# or
pip install -e .           # development install from source
```

**Dependencies:**

| Package | Required | Purpose |
|---------|----------|---------|
| numpy | Yes | Core array operations |
| pydantic | Yes | Configuration validation |
| torch | Optional | GPU-accelerated backend |
| tqdm | Optional | Progress bars (`pip install tensor-ml[progress]`) |
| matplotlib | Optional | Visualisation in example notebooks |

---

## Core Concepts

### Kronecker-Structured Dictionaries

Many tensor problems involve dictionaries that factor as Kronecker products:

$$\mathbf{D} = \mathbf{D}_N \otimes \mathbf{D}_{N-1} \otimes \cdots \otimes \mathbf{D}_1$$

This structure avoids explicitly forming the (potentially huge) full dictionary. tensor-ml exploits this decomposition throughout.

### Sparse Tensor Recovery

Given compressed measurements $\mathbf{y} = \boldsymbol{\Phi} \mathbf{D} \mathbf{c}$, the goal is to recover a sparse coefficient vector $\mathbf{c}$. T-LARS does this by iteratively selecting and deselecting dictionary atoms along the LARS/LASSO regularisation path.

### Backend Agnosticism

Every operation is defined by an abstract base class. Concrete implementations for NumPy and PyTorch are registered in a factory. User code calls a static facade that infers the backend from input types — no code changes needed when switching between CPU and GPU.

---

## Backend System

### How It Works

```
User code  →  Static facade (TensorProducts / TensorOps)
                 ↓
           infer_backend(data) → BackendType
                 ↓
           Factory.get(backend) → concrete instance
                 ↓
           NumpyTensorProducts / TorchTensorProducts
```

1. **Inference:** `infer_backend()` inspects the input's type (`np.ndarray` → NUMPY, `torch.Tensor` → TORCH).
2. **Factory lookup:** `TensorOpsFactory.get()` or `TensorProductsFactory.get()` returns a cached instance.
3. **Dispatch:** The facade method calls the matching backend implementation.

### Explicit Backend Selection

```python
from tensor_ml import TensorOpsFactory, BackendType

ops = TensorOpsFactory.get(BackendType.NUMPY)
ops.norm(x)
```

### Backend Registration

To add a custom backend:

```python
from tensor_ml import TensorOps, TensorOpsFactory, BackendType

class MyOps(TensorOps):
    ...  # implement all abstract methods

# You'd register under a new BackendType or override an existing one:
TensorOpsFactory.register(BackendType.NUMPY, MyOps)
```

---

## Tensor Operations

Element-wise operations accessed through `TensorOps` subclasses or the `TensorOpsFactory`.

```python
from tensor_ml import NumpyOps
import numpy as np

ops = NumpyOps()
x = np.array([3.0, 4.0])

ops.norm(x)         # 5.0
ops.abs(x)          # array([3., 4.])
ops.zeros((2, 3))   # 2×3 zero matrix
ops.eye(3)          # 3×3 identity
```

All operations are documented in the [API Reference](api_reference.md#tensorops-abc).

---

## Tensor Products

### Static Facade (Recommended)

```python
from tensor_ml import TensorProducts
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.eye(2)

# Kronecker product: B ⊗ A (applied in reverse order)
TensorProducts.kronecker_product([A, B])

# Khatri-Rao product (column-wise Kronecker)
TensorProducts.khatri_rao_product([A, B])

# Full multilinear product: X ×₁ A ×₂ B
X = np.random.randn(2, 2)
TensorProducts.full_multilinear_product(X, [A, B])
```

### Index Helpers

Convert between linear (Fortran-order) indices and per-mode subscripts:

```python
# Linear index → subscript indices
TensorProducts.get_tensor_indices(5, (3, 4))  # (2, 1)

# Subscript → linear index
TensorProducts.get_vector_index((2, 1), (3, 4))  # 5

# Kronecker column → per-factor column indices
TensorProducts.get_kronecker_factor_column_indices(5, (3, 4))
```

---

## T-LARS: Sparse Tensor Recovery

### Quick Example

```python
import numpy as np
from tensor_ml import TLARS

# Build problem
D1 = np.random.randn(8, 16)      # mode-1 dictionary
D2 = np.random.randn(8, 16)      # mode-2 dictionary
D_full = np.kron(D2, D1)          # full Kronecker dictionary

c_true = np.zeros(256)
c_true[[10, 42, 100]] = [1.5, -0.8, 2.3]

Phi = np.random.randn(30, 64)     # sensing matrix
y = Phi @ D_full @ c_true          # compressed measurements

# Fit
model = TLARS(l0_mode=True, debug_mode=True)
model.fit(
    y=y,
    sensing_matrix=Phi,
    dictionary_matrices=[D1, D2],
)

# Evaluate
print("R²:", model.score())
print("Iterations:", model.n_iter_)
print("Active atoms:", len(model.active_columns_))
```

### Configuration

All parameters are validated through `TLARSConfig` (Pydantic):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tolerance` | 0.075 | Stop when ‖residual‖ falls below this threshold. |
| `l0_mode` | False | **True** = greedy L0 (never removes atoms). **False** = L1/LASSO (can remove). |
| `mask_type` | 'KP' | `'KP'` = Kronecker Product structure. `'KR'` = Khatri-Rao structure. |
| `debug_mode` | False | Emit per-iteration log messages at DEBUG level. |
| `active_coefficients` | 1,000,000 | Cap on active dictionary atoms. |
| `iterations` | 1,000,000 | Maximum LARS iterations. |
| `precision_factor` | 5 | Multiplier for machine epsilon rounding. |

Invalid values are rejected immediately:

```python
from tensor_ml.tensor_models import TLARSConfig

TLARSConfig(tolerance=-1)    # ValidationError: tolerance must be > 0
TLARSConfig(mask_type='XX')  # ValidationError: must be 'KP' or 'KR'
```

### scikit-learn-style Interface

```python
model = TLARS(tolerance=0.05)

# Inspect
model.get_params()
# {'tolerance': 0.05, 'l0_mode': False, 'mask_type': 'KP', ...}

# Update
model.set_params(l0_mode=True, debug_mode=True)
```

### Fitted Attributes

After calling `fit()`, the following attributes are available:

- `model.coef_tensor_` — sparse coefficient tensor.
- `model.active_columns_` — indices of selected dictionary atoms.
- `model.coef_` — coefficient values for active columns only.
- `model.norm_r_` — residual norm at each iteration.
- `model.n_iter_` — total iterations executed.
- `model.tensor_norm_` — norms of tensor dictionary columns.

### L0 vs L1 Mode

| | L0 (`l0_mode=True`) | L1 (`l0_mode=False`) |
|---|---|---|
| **Column removal** | Never | Yes (LASSO sign changes) |
| **Path** | Greedy forward | Full LARS/LASSO path |
| **Speed** | Faster (no sign checks) | Slower but more accurate |
| **Use case** | Known high sparsity | Unknown sparsity, regularisation needed |

---

## Device Management

### Device Switching

```python
model = TLARS(backend='numpy')
model.to(backend='torch', device='cuda')  # switch to GPU
model.cpu()   # shorthand for .to(device='cpu')
model.cuda()  # shorthand for .to(backend='torch', device='cuda')
```

### PyTorch Device Placement

When using the `torch` backend, tensors are automatically placed on the configured device. The `TorchOps` and `TorchTensorProducts` classes handle device propagation internally.

```python
import torch
from tensor_ml import TensorProducts

A = torch.randn(3, 3, device='cuda')
B = torch.randn(3, 3, device='cuda')
result = TensorProducts.kronecker_product([A, B])  # stays on CUDA
```

---

## Error Handling

tensor-ml provides a custom exception hierarchy rooted at `TensorMLError`:

```python
from tensor_ml import TensorMLError, NotFittedError, BackendError

try:
    model.predict([D1, D2])
except NotFittedError:
    print("Call fit() first")
except TensorMLError:
    print("Caught any tensor-ml error")
```

| Exception | When |
|-----------|------|
| `NotFittedError` | Calling `predict()` / `score()` before `fit()`. |
| `BackendError` | Backend inference fails or PyTorch is not installed. |
| `ShapeMismatchError` | Factor matrices don't match tensor dimensions. |
| `ValidationError` | Invalid parameter values or input types. |

All inherit from both `TensorMLError` and (where applicable) `ValueError`, so existing `except ValueError` clauses still work.

---

## Logging & Debugging

tensor-ml uses Python's built-in `logging` module. To see per-iteration T-LARS logs:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

model = TLARS(debug_mode=True)
model.fit(y=y, sensing_matrix=Phi, dictionary_matrices=[D1, D2])
```

Each iteration logs: active set changes, residual norms, step sizes, and convergence status.

For long runs, enable the progress bar:

```python
model = TLARS(show_progress=True)   # requires tqdm
model.fit(factor_matrices=[D1, D2], Y=Y)
```

To silence logs:

```python
logging.getLogger('tensor_ml').setLevel(logging.WARNING)
```

---

## Extending tensor-ml

### Adding a New Backend

1. Subclass `TensorOps` and `TensorProductsBase`.
2. Implement all abstract methods.
3. Register via the factories:

```python
TensorOpsFactory.register(MyBackendType, MyOps)
TensorProductsFactory.register(MyBackendType, MyTensorProducts)
```

### Adding a New Model

1. Subclass `MultilinearModel` (or `BaseTensorModel` for non-multilinear models).
2. Implement `fit()` and `predict()`.
3. Use `self.ops` and `self.tp` for backend-agnostic computation.

```python
from tensor_ml.tensor_models.multilinear import MultilinearModel

class MyModel(MultilinearModel):
    def fit(self, X, y=None, **kwargs):
        self._resolve_backend(X)
        # use self.ops and self.tp for computation
        return self

    def predict(self, X=None, **kwargs):
        ...
```

---

## Further Reading

- [Quickstart Tutorial](quickstart.ipynb) — hands-on notebook
- [T-LARS Image Reconstruction](examples/tlars_image_reconstruction.ipynb) — visual demo with DCT dictionaries
- [API Reference](api_reference.md) — full class/method documentation
