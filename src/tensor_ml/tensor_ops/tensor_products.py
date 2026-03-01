"""
Backend-agnostic tensor products API.

Provides:
- ``TensorProducts``        – static facade with automatic backend detection
- ``TensorProductsFactory`` – registry-based factory

The abstract base class ``TensorProductsBase`` lives in
``tensor_products_base``.
"""

from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np

from tensor_ml.enums import BackendType
from tensor_ml.utils import infer_backend
from tensor_ml.tensor_ops.tensor_products_base import TensorProductsBase


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class TensorProductsFactory:
    """Registry-based factory for tensor-products backends."""

    _registry: dict[BackendType, type[TensorProductsBase]] = {}

    @classmethod
    def register(
        cls,
        backend_type: BackendType,
        products_class: type[TensorProductsBase],
    ) -> None:
        """Register a ``TensorProductsBase`` subclass for *backend_type*."""
        cls._registry[backend_type] = products_class

    @classmethod
    def get(
        cls,
        backend: Union[str, BackendType],
        device: Any = None,
    ) -> TensorProductsBase:
        """Return a ``TensorProductsBase`` instance for *backend*.

        Parameters
        ----------
        backend : str | BackendType
            The backend identifier.
        device : optional
            Device hint (only used by the PyTorch backend).

        Returns
        -------
        instance : TensorProductsBase
        """
        if isinstance(backend, str):
            backend = BackendType(backend.lower())
        products_class = cls._registry.get(backend)
        if products_class is None:
            if backend == BackendType.TORCH:
                raise ImportError(
                    "PyTorch is required for the TORCH backend but is not installed. "
                    "Install it with: pip install torch"
                )
            raise ValueError(f"No TensorProducts registered for backend: {backend}")
        if backend == BackendType.TORCH:
            return products_class(device=device)
        return products_class()


# ---------------------------------------------------------------------------
# Built-in registrations (lazy for torch)
# ---------------------------------------------------------------------------

def _register_builtins() -> None:
    from tensor_ml.tensor_ops.tensor_products_numpy import NumpyTensorProducts
    TensorProductsFactory.register(BackendType.NUMPY, NumpyTensorProducts)
    try:
        from tensor_ml.tensor_ops.tensor_products_torch import TorchTensorProducts
        TensorProductsFactory.register(BackendType.TORCH, TorchTensorProducts)
    except ImportError:
        pass


_register_builtins()


# ---------------------------------------------------------------------------
# Singleton cache (one instance per (backend, device) pair)
# ---------------------------------------------------------------------------

_INSTANCE_CACHE: dict[tuple[BackendType, Any], TensorProductsBase] = {}


def _get_instance(backend: BackendType, device: Any = None) -> TensorProductsBase:
    """Return a cached ``TensorProductsBase`` instance for the given backend."""
    key = (backend, device)
    if key not in _INSTANCE_CACHE:
        _INSTANCE_CACHE[key] = TensorProductsFactory.get(backend, device)
    return _INSTANCE_CACHE[key]


def _resolve(data: Any, backend: Optional[BackendType] = None) -> TensorProductsBase:
    """Infer backend from *data* and return the matching cached instance."""
    bt = infer_backend(data, backend)
    return _get_instance(bt)


# ---------------------------------------------------------------------------
# Static facade (public API)
# ---------------------------------------------------------------------------

class TensorProducts:
    """Static facade for tensor-product operations with automatic backend detection.

    Usage::

        from tensor_ml import TensorProducts as tp
        result = tp.kronecker_product(matrices)

    Every method infers the backend (NumPy / PyTorch) from the input data
    and delegates to the appropriate :class:`TensorProductsBase` subclass.
    """

    # ── Products ───────────────────────────────────────────────────

    @staticmethod
    def kronecker_product(factor_matrices: list[Any]) -> Any:
        """Compute the Kronecker product of *factor_matrices* (N → 1).

        Parameters
        ----------
        factor_matrices : list[array-like]
            Factor matrices, applied in reverse order.

        Returns
        -------
        result : array-like
        """
        if not factor_matrices:
            raise ValueError("The list of factor_matrices is empty.")
        return _resolve(factor_matrices).kronecker_product(factor_matrices)

    @staticmethod
    def khatri_rao_product(matrices: list[Any]) -> Any:
        """Compute the Khatri-Rao (column-wise Kronecker) product.

        Parameters
        ----------
        matrices : list[array-like]
            Matrices with the same number of columns.

        Returns
        -------
        result : array-like
        """
        if not matrices:
            raise ValueError("The list of matrices is empty.")
        return _resolve(matrices).khatri_rao_product(matrices)

    @staticmethod
    def tensor_product(matrices: list[Any]) -> Any:
        """Compute the tensor (outer) product of *matrices*.

        Parameters
        ----------
        matrices : list[array-like]
            Input matrices.

        Returns
        -------
        result : array-like
        """
        if not matrices:
            raise ValueError("The list of matrices is empty.")
        return _resolve(matrices).tensor_product(matrices)

    @staticmethod
    def hadamard_product(tensors: list[Any]) -> Any:
        """Compute the element-wise (Hadamard) product of *tensors*.

        Parameters
        ----------
        tensors : list[array-like]
            Tensors of identical shape.

        Returns
        -------
        result : array-like
        """
        if not tensors:
            raise ValueError("The list of tensors is empty.")
        return _resolve(tensors).hadamard_product(tensors)

    # ── Kronecker helpers ──────────────────────────────────────────

    @staticmethod
    def get_kronecker_matrix_column(
        factor_matrices: list[Any],
        column_indices: list[int] | tuple[int, ...],
    ) -> Any:
        """Return a single column of the implicit Kronecker matrix.

        Parameters
        ----------
        factor_matrices : list[array-like]
            Factor matrices.
        column_indices : list[int] | tuple[int, ...]
            Per-factor column indices.

        Returns
        -------
        column : array-like
        """
        if not factor_matrices:
            raise ValueError("The list of factor_matrices is empty.")
        if not column_indices:
            raise ValueError("The list of column_indices is empty.")
        return _resolve(factor_matrices).get_kronecker_matrix_column(factor_matrices, column_indices)

    @staticmethod
    def full_multilinear_product(
        X: Any,
        factor_matrices: list[Any],
        use_transpose: bool = False,
    ) -> Any:
        """Full multilinear product of tensor *X* with *factor_matrices*.

        Parameters
        ----------
        X : array-like
            Input tensor.
        factor_matrices : list[array-like]
            One matrix per tensor mode.
        use_transpose : bool, default=False
            Multiply by transposed factors.

        Returns
        -------
        Y : array-like
        """
        if X is None:
            raise ValueError("The input tensor X is empty or None.")
        return _resolve(X).full_multilinear_product(X, factor_matrices, use_transpose)

    @staticmethod
    def kronecker_matrix_vector_product(
        factor_matrices: list[Any],
        x: Any,
        tensor_shape: tuple[int, ...] | list[int],
        active_columns: Any,
        active_indices: Optional[list[list[int]]] = None,
        use_transpose: bool = False,
    ) -> Any:
        """Kronecker-structured matrix–vector product  y = A x.

        Parameters
        ----------
        factor_matrices : list[array-like]
            Factor matrices.
        x : array-like
            Coefficient vector.
        tensor_shape : tuple[int, ...] | list[int]
            Core tensor shape.
        active_columns : array-like
            Active column indices.
        active_indices : list[list[int]], optional
            Per-mode active indices for sub-tensor optimisation.
        use_transpose : bool, default=False
            Multiply by transposed factors.

        Returns
        -------
        y : array-like
        """
        return _resolve(x).kronecker_matrix_vector_product(
            factor_matrices, x, tensor_shape, active_columns, active_indices, use_transpose
        )

    # ── Vectorize / tensorize ──────────────────────────────────────

    @staticmethod
    def tensorize(
        x: Any,
        tensor_shape: tuple[int, ...] | list[int],
        active_elements: Any,
    ) -> Any:
        """Scatter a coefficient vector into a tensor.

        Parameters
        ----------
        x : array-like
            Coefficient vector.
        tensor_shape : tuple[int, ...] | list[int]
            Output tensor shape.
        active_elements : array-like
            Linear indices.

        Returns
        -------
        X : array-like
        """
        if x is None:
            raise ValueError("The input vector x is None.")
        return _resolve(x).tensorize(x, tensor_shape, active_elements)

    @staticmethod
    def vectorize(X: Any) -> Any:
        """Vectorise a tensor in Fortran (column-major) order.

        Parameters
        ----------
        X : array-like
            Input tensor.

        Returns
        -------
        x : array-like
        """
        return _resolve(X).vectorize(X)

    # ── Gramian / direction vector ─────────────────────────────────

    @staticmethod
    def get_gramian(
        gramians: list[Any],
        active_columns: Any,
        tensor_shape: tuple[int, ...] | list[int],
    ) -> Any:
        """Gramian sub-matrix for the active columns.

        Parameters
        ----------
        gramians : list[array-like]
            Per-mode Gramian matrices.
        active_columns : array-like
            Active column indices.
        tensor_shape : tuple[int, ...] | list[int]
            Core tensor shape.

        Returns
        -------
        GI : array-like
        """
        if gramians is None:
            raise ValueError("The input list gramians is None.")
        return _resolve(gramians).get_gramian(gramians, active_columns, tensor_shape)

    @staticmethod
    def get_direction_vector(
        GInv: Any,
        zI: Any,
        gramians: list[Any],
        active_columns: Any,
        add_column_flag: bool,
        changed_dict_column_index: int,
        changed_active_column_index: int,
        tensor_shape: tuple[int, ...] | list[int],
        precision_order: int = 10,
    ) -> tuple[Any, Any]:
        """Schur-complement Gramian update + direction vector.

        Parameters
        ----------
        GInv : array-like
            Current inverse Gramian.
        zI : array-like
            Sign vector.
        gramians : list[array-like]
            Per-mode Gramian matrices.
        active_columns : array-like
            Active column indices.
        add_column_flag : bool
            ``True`` to add, ``False`` to remove.
        changed_dict_column_index : int
            Global column index being changed.
        changed_active_column_index : int
            Position within the active set.
        tensor_shape : tuple[int, ...] | list[int]
            Core tensor shape.
        precision_order : int, default=10
            Rounding precision.

        Returns
        -------
        dI : array-like
            Direction vector.
        GInv : array-like
            Updated inverse Gramian.
        """
        return _resolve(zI).get_direction_vector(
            GInv, zI, gramians, active_columns,
            add_column_flag, changed_dict_column_index, changed_active_column_index,
            tensor_shape, precision_order
        )

    # ── Rounding ───────────────────────────────────────────────────

    @staticmethod
    def tround(tensor: Any, precision_order: int = 0) -> Any:
        """Round *tensor* elements to *precision_order* significant digits.

        Parameters
        ----------
        tensor : array-like
            Input tensor (or scalar).
        precision_order : int, default=0
            Number of significant decimal digits.

        Returns
        -------
        rounded : array-like
        """
        if isinstance(tensor, (int, float)):
            tensor = np.array(tensor)
        return _resolve(tensor).tround(tensor, precision_order)

    # ── Pure-math helpers (backend-independent) ────────────────────

    @staticmethod
    def get_vector_index(
        tensor_indices: tuple[int, ...],
        tensor_shape: tuple[int, ...] | list[int],
    ) -> int:
        """Linear (Fortran-order) index from subscript indices.

        Parameters
        ----------
        tensor_indices : tuple[int, ...]
            Subscript indices.
        tensor_shape : tuple[int, ...] | list[int]
            Tensor shape.

        Returns
        -------
        index : int
        """
        return TensorProductsBase.get_vector_index(tensor_indices, tensor_shape)

    @staticmethod
    def get_kronecker_factor_column_indices(
        column_index: int,
        tensor_shape: tuple[int, ...] | list[int],
    ) -> tuple[int, ...]:
        """Decompose a linear index into per-factor column indices.

        Parameters
        ----------
        column_index : int
            Linear index.
        tensor_shape : tuple[int, ...] | list[int]
            Core tensor shape.

        Returns
        -------
        indices : tuple[int, ...]
        """
        return TensorProductsBase.get_kronecker_factor_column_indices(column_index, tensor_shape)
