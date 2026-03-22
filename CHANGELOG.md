# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Migrated from Poetry to **uv** for dependency management and build system (hatchling)

### Added
- Custom exception hierarchy (`TensorMLError`, `NotFittedError`, `BackendError`, `ShapeMismatchError`, `ValidationError`).
- `TLARS.cpu()` and `TLARS.cuda()` convenience methods.
- `TLARS.__repr__()` for readable string representation.
- Optional `tqdm` progress bar for TLARS iterations (install with `pip install tensor-ml[progress]`).
- `py.typed` marker for PEP 561 compliance.
- `__version__` attribute on the package.
- Comprehensive README with installation, quick start, and documentation links.
- GitHub Actions CI workflow for automated testing on Python 3.10–3.13.
- API reference documentation (`docs/api_reference.md`).
- User guide documentation (`docs/user_guide.md`).
- Quickstart tutorial notebook (`docs/quickstart.ipynb`).
- T-LARS image reconstruction example notebook (`docs/examples/tlars_image_reconstruction.ipynb`).
- Error-path tests for invalid inputs and unfitted model access.

### Changed
- All domain-specific errors now raise custom exceptions instead of built-in `ValueError`/`ImportError`.
- Pydantic v2 API throughout (`TLARSConfig` uses `@field_validator`, `model_dump()`, etc.).
- NumPy-style docstrings across all public modules.
- `pydantic ≥ 2.0` is now a required dependency.

### Fixed
- `TorchTensorProducts` now preserves device placement via `_ensure_tensor()`.
- `MultilinearModel.device` default changed from `'cuda'` to `None` (safe on CPU-only systems).
- TLARS `fit()` no longer references `self.debug_mode` etc. directly; uses validated config values.

## [0.1.0] — 2024-01-01

### Added
- Initial release with NumPy and PyTorch backends.
- T-LARS sparse tensor recovery algorithm.
- Kronecker, Khatri-Rao, Hadamard, and full multilinear product operations.
- Backend auto-detection from input array types.
