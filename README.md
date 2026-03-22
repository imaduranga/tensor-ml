# tensor-ml

[![PyPI version](https://img.shields.io/pypi/v/tensor-ml.svg)](https://pypi.org/project/tensor-ml/)
[![Python](https://img.shields.io/pypi/pyversions/tensor-ml.svg)](https://pypi.org/project/tensor-ml/)
[![Tests](https://github.com/IW276/tensor-ml/actions/workflows/tests.yml/badge.svg)](https://github.com/IW276/tensor-ml/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A Python library for tensor analysis, multilinear algebra, tensor regression, and multidimensional sparse signal representations.

## Features

- **Backend-agnostic** — unified API for NumPy and PyTorch (auto-detects input type)
- **Tensor products** — Kronecker, Khatri-Rao, Hadamard, full multilinear product
- **T-LARS** — Tensor Least Angle Regression & Selection for sparse tensor recovery
- **scikit-learn-style interface** — `fit` / `predict` / `score` / `get_params` / `set_params`
- **Device management** — `.to('cuda')`, `.cpu()`, `.cuda()` for PyTorch backend
- **Validated configuration** — Pydantic-based parameter validation with clear error messages

## Installation

```bash
pip install tensor-ml
```

With optional PyTorch support:

```bash
pip install "tensor-ml[torch]"
```

Development install from source (using [uv](https://docs.astral.sh/uv/)):

```bash
git clone https://github.com/IW276/tensor-ml.git
cd tensor-ml
uv sync --all-extras       # creates .venv and installs all deps
uv run pytest              # run the test suite
```

Or with pip:

```bash
pip install -e ".[torch]"
```

## Quick Start

### Tensor Products

```python
import numpy as np
from tensor_ml import TensorProducts

A = np.array([[1, 2], [3, 4]])
B = np.eye(2)

# Kronecker product
K = TensorProducts.kronecker_product([A, B])

# Full multilinear product: X ×₁ A ×₂ B
X = np.random.randn(2, 2)
Y = TensorProducts.full_multilinear_product(X, [A, B])
```

### T-LARS: Sparse Tensor Recovery

```python
import numpy as np
from tensor_ml import TLARS

# Per-mode dictionaries
D1 = np.random.randn(8, 16)
D2 = np.random.randn(8, 16)

# Target tensor
Y = np.random.randn(8, 8)

# Fit
model = TLARS(tolerance=0.01, l0_mode=True)
model.fit(factor_matrices=[D1, D2], Y=Y)

# Predict & score
Y_hat = model.predict([D1, D2])
r2 = model.score([D1, D2], Y)
print(f"R² = {r2:.4f}, iterations = {model.n_iter_}")
```

### PyTorch Backend

```python
import torch
from tensor_ml import TensorProducts

A = torch.randn(3, 3)
B = torch.randn(3, 3)
K = TensorProducts.kronecker_product([A, B])  # auto-uses TorchTensorProducts
```

## Documentation

| Resource | Description |
|----------|-------------|
| [Quickstart Tutorial](docs/quickstart.ipynb) | Interactive notebook walkthrough |
| [T-LARS Image Reconstruction](docs/examples/tlars_image_reconstruction.ipynb) | Visual demo with DCT dictionaries |
| [API Reference](docs/api_reference.md) | Complete class and method reference |
| [User Guide](docs/user_guide.md) | Concepts, architecture, and extension guide |

## Architecture

```
tensor_ml/
├── enums.py              # BackendType enum
├── exceptions.py         # Custom exception hierarchy
├── utils.py              # Backend inference
├── tensor_ops/
│   ├── tensor_ops.py     # TensorOps ABC + NumpyOps + TorchOps + Factory
│   ├── tensor_products_base.py   # TensorProductsBase ABC
│   ├── tensor_products_numpy.py  # NumPy backend
│   ├── tensor_products_torch.py  # PyTorch backend
│   └── tensor_products.py        # Static facade + Factory
└── tensor_models/
    ├── base.py            # BaseTensorModel ABC
    └── multilinear/
        ├── multilinear_model.py  # MultilinearModel base
        └── tlars.py              # TLARS algorithm + TLARSConfig
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Run tests (`pytest`)
4. Submit a pull request

## Citation

If you use tensor-ml in your research, please cite:

```bibtex
@software{tensor_ml,
  title={tensor-ml: Tensor Machine Learning Library},
  author={Ishan Wickramasingha},
  url={https://github.com/IW276/tensor-ml},
  license={MIT}
}
```

## License

MIT — see [LICENSE](LICENSE) for details.
