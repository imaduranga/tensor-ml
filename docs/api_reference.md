# tensor-ml API Reference

Complete class and method reference for every public symbol exported by `tensor_ml`.

```python
import tensor_ml
tensor_ml.__version__   # e.g. '0.2.0'
```

---

## `tensor_ml.exceptions`

All library-specific exceptions inherit from `TensorMLError`, allowing a single catch clause.

| Exception | Bases | Raised when |
|-----------|-------|-------------|
| `TensorMLError` | `Exception` | Base class for all tensor-ml errors. |
| `NotFittedError` | `TensorMLError` | A model method is called before `fit()`. |
| `BackendError` | `TensorMLError` | Backend detection fails or torch is missing. |
| `ShapeMismatchError` | `TensorMLError`, `ValueError` | Array shapes are incompatible. |
| `ValidationError` | `TensorMLError`, `ValueError` | A parameter or input fails validation. |

---

## `tensor_ml.enums`

### `BackendType`

```python
class BackendType(Enum):
    NUMPY = "numpy"
    TORCH = "torch"
```

Enumeration of supported compute backends.

---

## `tensor_ml.utils`

### `infer_backend(data, backend=None)`

Infer the compute backend from input data.

| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | array-like | `np.ndarray`, `torch.Tensor`, or a list of them. |
| `backend` | `BackendType`, optional | If provided, returned directly without inference. |

**Returns:** `BackendType`

---

## `tensor_ml.tensor_ops`

### Element-wise Operations

#### `TensorOps` (ABC)

Abstract base class defining the contract for element-wise tensor operations. All methods below are abstract; concrete backends implement them.

| Method | Signature | Description |
|--------|-----------|-------------|
| `norm` | `(x) → float` | L2 norm of *x*. |
| `normalize` | `(D) → array` | Column-normalise matrix *D*. |
| `zeros` | `(shape) → array` | Zero-filled array. |
| `ones` | `(shape) → array` | Ones-filled array. |
| `abs` | `(x) → array` | Element-wise absolute value. |
| `sign` | `(x) → array` | Element-wise sign. |
| `argmax` | `(x) → int` | Index of the maximum element. |
| `argmin` | `(x) → int` | Index of the minimum element. |
| `concatenate` | `(arrs) → array` | Concatenate along the first axis. |
| `inf` | `→ float` | Positive infinity constant (property). |
| `asarray` | `(x) → array` | Convert to native array type. |
| `flatten` | `(x) → array` | Flatten in Fortran (column-major) order. |
| `to_device` | `(x) → array` | Move to configured device (no-op for NumPy). |
| `nonzero` | `(x) → array` | Indices of non-zero elements. |
| `mean` | `(x) → float` | Arithmetic mean. |
| `sum` | `(x) → scalar` | Sum of all elements. |
| `gramian` | `(D) → array` | Gramian D^T D. |
| `copy` | `(x) → array` | Deep copy / clone. |
| `to_scalar` | `(x) → float` | Extract Python float from scalar. |
| `has_nan` | `(x) → bool` | True if *x* contains NaN. |
| `pinv` | `(x) → array` | Moore–Penrose pseudo-inverse. |
| `to_numpy` | `(x) → np.ndarray` | Convert to NumPy ndarray. |
| `find_index` | `(arr, val) → int` | First index of *val* in *arr*. |
| `numel` | `(x) → int` | Number of elements along axis 0. |
| `eye` | `(n) → array` | n × n identity matrix. |
| `allclose` | `(a, b) → bool` | Element-wise approximate equality. |
| `max` | `(x) → scalar` | Maximum element. |

#### `NumpyOps(TensorOps)`

NumPy implementation. Instantiated with no arguments.

#### `TorchOps(TensorOps)`

PyTorch implementation.

```python
TorchOps(device='cuda')
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `device` | `str \| torch.device \| None` | `'cuda'` | Target device. Defaults to CUDA if available, else CPU. |

#### `TensorOpsFactory`

Registry-based factory.

| Method | Description |
|--------|-------------|
| `register(backend_type, ops_class)` | Register a `TensorOps` subclass. |
| `get(backend, device=None) → TensorOps` | Return an instance for the given backend. |

---

### Tensor Products

#### `TensorProductsBase` (ABC)

Abstract base class for all tensor-product operations. Fully documented parameter signatures are in the source docstrings.

| Method | Description |
|--------|-------------|
| `kronecker_product(matrices)` | Kronecker product A^(N) ⊗ … ⊗ A^(1). |
| `khatri_rao_product(matrices)` | Column-wise Kronecker (Khatri-Rao) product. |
| `tensor_product(matrices)` | Tensor (outer) product. |
| `hadamard_product(tensors)` | Element-wise (Hadamard) product. |
| `get_kronecker_matrix_column(factor_matrices, column_indices)` | Single column of the implicit Kronecker matrix. |
| `full_multilinear_product(X, factor_matrices, use_transpose=False)` | X ×₁ A₁ ×₂ A₂ … ×_N A_N. |
| `kronecker_matrix_vector_product(factor_matrices, x, tensor_shape, active_columns, ...)` | Kronecker-structured matrix–vector product. |
| `tensorize(x, tensor_shape, active_elements)` | Scatter coefficients into a tensor. |
| `vectorize(X)` | Flatten tensor in Fortran order. |
| `get_gramian(gramians, active_columns, tensor_shape)` | Gramian sub-matrix for active columns. |
| `get_direction_vector(GInv, zI, gramians, active_columns, ...)` | Schur-complement update + direction vector. |
| `tround(tensor, precision_order=0)` | Round to significant digits. |

#### `NumpyTensorProducts(TensorProductsBase)`

NumPy implementation. Instantiated with no arguments.

#### `TorchTensorProducts(TensorProductsBase)`

PyTorch implementation.

```python
TorchTensorProducts(device='cuda')
```

#### `TensorProducts` (static facade)

Auto-detects backend from input arrays and delegates to the correct implementation. All methods mirror `TensorProductsBase` with identical signatures.

Additional backend-independent helpers:

| Method | Description |
|--------|-------------|
| `get_vector_index(tensor_indices, tensor_shape) → int` | Linear (Fortran-order) index from subscript indices. |
| `get_tensor_indices(vector_index, tensor_shape) → tuple` | Subscript indices from a linear index. |
| `get_kronecker_factor_column_indices(column_index, tensor_shape) → list` | Per-factor column indices for a Kronecker column. |

#### `TensorProductsFactory`

| Method | Description |
|--------|-------------|
| `register(backend_type, products_class)` | Register a `TensorProductsBase` subclass. |
| `get(backend, device=None) → TensorProductsBase` | Return an instance for the given backend. |

---

## `tensor_ml.tensor_models`

### `BaseTensorModel` (ABC)

Abstract base class for all tensor-based models.

| Method | Description |
|--------|-------------|
| `get_params(deep=True) → dict` | Get estimator parameters. |
| `set_params(**params) → self` | Set estimator parameters. |
| `fit(X, y=None, **kwargs) → self` | Fit the model (abstract). |
| `predict(X, **kwargs) → array` | Generate predictions (abstract). |
| `score(X, y=None, **kwargs) → float` | Default score metric. |

---

### `MultilinearModel(BaseTensorModel)`

Base class for multilinear tensor models with backend-agnostic ops.

```python
MultilinearModel(backend=None, device=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backend` | `str \| BackendType \| None` | `None` | Compute backend. Inferred from data if `None`. |
| `device` | `str \| torch.device \| None` | `None` | Device hint (PyTorch only). |

| Method | Description |
|--------|-------------|
| `get_backend(X=None) → BackendType` | Return the backend, inferring from *X* if needed. |
| `normalize_input(X) → array` | Convert input to the backend's native type. |
| `score(X, y=None) → float` | R² coefficient of determination. |
| `mse(X, y) → float` | Mean squared error. |
| `mae(X, y) → float` | Mean absolute error. |

---

### `TLARSConfig`

Pydantic model for validated T-LARS configuration.

```python
TLARSConfig(
    tolerance=0.075,
    l0_mode=False,
    mask_type='KP',
    debug_mode=False,
    active_coefficients=1_000_000,
    iterations=1_000_000,
    precision_factor=5,
    backend=None,
    device=None,
)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `tolerance` | `float` | `0.075` | Residual norm stopping threshold (must be > 0). |
| `l0_mode` | `bool` | `False` | Greedy L0 selection (no column removal). |
| `mask_type` | `str` | `'KP'` | `'KP'` (Kronecker Product) or `'KR'` (Khatri-Rao). |
| `debug_mode` | `bool` | `False` | Emit per-iteration DEBUG log messages. |
| `active_coefficients` | `int` | `1_000_000` | Maximum active (non-zero) coefficients. |
| `iterations` | `int` | `1_000_000` | Maximum LARS iterations. |
| `precision_factor` | `int` | `5` | Machine-epsilon multiplier. |
| `show_progress` | `bool` | `False` | Display a `tqdm` progress bar during fitting. |
| `backend` | `str \| BackendType \| None` | `None` | Compute backend (inferred if `None`). |
| `device` | `str \| None` | `None` | Device hint (e.g. `'cpu'`, `'cuda'`). |

---

### `TLARS(MultilinearModel)`

Tensor Least Angle Regression and Selection. Solves sparse tensor recovery via the LARS/LASSO path over a Kronecker-structured dictionary.

```python
model = TLARS(tolerance=0.075, l0_mode=True, debug_mode=True)
```

All keyword arguments are forwarded to `TLARSConfig`.

#### Methods

**`fit(y, sensing_matrix, dictionary_matrices, mode_dimensions=None, dictionary_dimensions=None, num_measurements=None)`**

Fit the T-LARS model.

| Parameter | Type | Description |
|-----------|------|-------------|
| `y` | array-like | Measurement vector of shape `(m,)` or `(m, 1)`. |
| `sensing_matrix` | array-like | Sensing / measurement matrix of shape `(m, n)`. |
| `dictionary_matrices` | list[array-like] | Per-mode dictionary factor matrices. |
| `mode_dimensions` | list[int], optional | Signal dimensions per mode (inferred from dictionaries if omitted). |
| `dictionary_dimensions` | list[int], optional | Atom counts per mode (inferred from dictionaries if omitted). |
| `num_measurements` | int, optional | Number of rows in the sensing matrix (inferred if omitted). |

Returns `self`.

**`predict(X=None)`**

Reconstruct the measurement vector from the fitted coefficients.

Returns the predicted measurement vector.

**`score(X=None, y=None)`**

R² score on the training data.

**`get_params(deep=True) → dict`**

Return current configuration as a dictionary.

**`set_params(**params) → self`**

Update configuration parameters. Validates via `TLARSConfig`.

**`to(backend=None, device=None) → self`**

Switch the compute backend and/or device.

**`cpu() → self`**

Shorthand for `to(device='cpu')`.

**`cuda() → self`**

Shorthand for `to(backend='torch', device='cuda')`.

**`__repr__() → str`**

Return a readable string showing only non-default parameters, e.g. `TLARS(l0_mode=True, tolerance=0.5)`.

#### Fitted Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `config` | `TLARSConfig` | Validated parameter configuration. |
| `coef_tensor_` | array-like | Sparse coefficient tensor. |
| `active_columns_` | array-like | Indices of selected dictionary columns. |
| `coef_` | array-like | Non-zero coefficient vector. |
| `norm_r_` | `list[float]` | Residual norm history. |
| `n_iter_` | `int` | Number of iterations executed. |
| `tensor_norm_` | `float` | Norm of the input tensor (used for de-normalisation). |

---

## Inheritance Hierarchy

```
BaseTensorModel (ABC)
  └── MultilinearModel
        └── TLARS
```

```
TensorOps (ABC)
  ├── NumpyOps
  └── TorchOps

TensorProductsBase (ABC)
  ├── NumpyTensorProducts
  └── TorchTensorProducts
```
