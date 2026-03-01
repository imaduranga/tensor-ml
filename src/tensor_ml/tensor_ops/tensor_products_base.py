"""Abstract base class (contract) for tensor-product backends.

Concrete subclasses live in ``tensor_products_numpy`` and
``tensor_products_torch``.  The public facade and factory remain in
``tensor_products``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class TensorProductsBase(ABC):
    """Abstract base class for backend-specific tensor product operations.

    Concrete subclasses (e.g. ``NumpyTensorProducts``, ``TorchTensorProducts``)
    must implement every ``@abstractmethod`` listed here.
    """

    # ── Products ───────────────────────────────────────────────────

    @abstractmethod
    def kronecker_product(self, matrices: list[Any]) -> Any:
        """Compute the Kronecker product  A^(N) ⊗ … ⊗ A^(1).

        Parameters
        ----------
        matrices : list[array-like]
            Factor matrices, applied in reverse order.

        Returns
        -------
        result : array-like
            The Kronecker product matrix.
        """
        ...

    @abstractmethod
    def khatri_rao_product(self, matrices: list[Any]) -> Any:
        """Compute the column-wise Kronecker (Khatri-Rao) product.

        Parameters
        ----------
        matrices : list[array-like]
            Matrices with the same number of columns.

        Returns
        -------
        result : array-like
            The Khatri-Rao product matrix.
        """
        ...

    @abstractmethod
    def tensor_product(self, matrices: list[Any]) -> Any:
        """Compute the tensor (outer) product of a list of matrices.

        Parameters
        ----------
        matrices : list[array-like]
            Input matrices.

        Returns
        -------
        result : array-like
            The tensor product.
        """
        ...

    @abstractmethod
    def hadamard_product(self, tensors: list[Any]) -> Any:
        """Compute the element-wise (Hadamard) product of tensors.

        Parameters
        ----------
        tensors : list[array-like]
            Tensors of identical shape.

        Returns
        -------
        result : array-like
            The element-wise product.
        """
        ...

    # ── Kronecker helpers ──────────────────────────────────────────

    @abstractmethod
    def get_kronecker_matrix_column(
        self,
        factor_matrices: list[Any],
        column_indices: list[int] | tuple[int, ...],
    ) -> Any:
        """Return a single column of the implicit Kronecker matrix.

        Parameters
        ----------
        factor_matrices : list[array-like]
            Factor matrices whose Kronecker product defines the full matrix.
        column_indices : list[int] | tuple[int, ...]
            Per-factor column indices selecting the desired column.

        Returns
        -------
        column : array-like
            The selected column vector.
        """
        ...

    @abstractmethod
    def full_multilinear_product(
        self,
        X: Any,
        factor_matrices: list[Any],
        use_transpose: bool = False,
    ) -> Any:
        """Full multilinear product  X ×₁ A₁ ×₂ A₂ … ×_N A_N.

        Parameters
        ----------
        X : array-like
            Input tensor of order *N*.
        factor_matrices : list[array-like]
            One matrix per tensor mode.
        use_transpose : bool, default=False
            If ``True``, multiply by A_nᵀ instead of A_n.

        Returns
        -------
        Y : array-like
            Result tensor.
        """
        ...

    @abstractmethod
    def kronecker_matrix_vector_product(
        self,
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
            Factor matrices defining the Kronecker structure.
        x : array-like
            Coefficient vector (length = number of active columns).
        tensor_shape : tuple[int, ...] | list[int]
            Shape of the core tensor.
        active_columns : array-like
            Indices of the active columns.
        active_indices : list[list[int]], optional
            Per-mode active index lists for sub-tensor optimisation.
        use_transpose : bool, default=False
            If ``True``, multiply by transposed factors.

        Returns
        -------
        y : array-like
            Result vector.
        """
        ...

    # ── Vectorize / tensorize ──────────────────────────────────────

    @abstractmethod
    def tensorize(
        self,
        x: Any,
        tensor_shape: tuple[int, ...] | list[int],
        active_elements: Any,
    ) -> Any:
        """Scatter a coefficient vector into a tensor (Fortran order).

        Parameters
        ----------
        x : array-like
            Coefficient vector.
        tensor_shape : tuple[int, ...] | list[int]
            Desired output tensor shape.
        active_elements : array-like
            Linear indices where *x* values are placed.

        Returns
        -------
        X : array-like
            Tensor of shape *tensor_shape*.
        """
        ...

    @abstractmethod
    def vectorize(self, X: Any) -> Any:
        """Vectorise a tensor in Fortran (column-major) order.

        Parameters
        ----------
        X : array-like
            Input tensor.

        Returns
        -------
        x : array-like
            Flattened vector.
        """
        ...

    # ── Gramian / direction vector ─────────────────────────────────

    @abstractmethod
    def get_gramian(
        self,
        gramians: list[Any],
        active_columns: Any,
        tensor_shape: tuple[int, ...] | list[int],
    ) -> Any:
        """Assemble the Gramian sub-matrix for the active columns.

        Parameters
        ----------
        gramians : list[array-like]
            Per-mode Gramian matrices (D_nᵀ D_n).
        active_columns : array-like
            Linear indices of the active Kronecker columns.
        tensor_shape : tuple[int, ...] | list[int]
            Shape of the core tensor.

        Returns
        -------
        GI : array-like
            Gramian sub-matrix of shape ``(n_active, n_active)``.
        """
        ...

    @abstractmethod
    def get_direction_vector(
        self,
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
        """Schur-complement update of G⁻¹ and compute  dI = G⁻¹ zI.

        Parameters
        ----------
        GInv : array-like
            Current inverse Gramian.
        zI : array-like
            Sign vector of correlations for the active set.
        gramians : list[array-like]
            Per-mode Gramian matrices.
        active_columns : array-like
            Linear indices of the active Kronecker columns.
        add_column_flag : bool
            ``True`` if a column is being added, ``False`` if removed.
        changed_dict_column_index : int
            Global index of the column being added/removed.
        changed_active_column_index : int
            Position within the active set.
        tensor_shape : tuple[int, ...] | list[int]
            Shape of the core tensor.
        precision_order : int, default=10
            Rounding precision (significant digits).

        Returns
        -------
        dI : array-like
            Direction vector.
        GInv : array-like
            Updated inverse Gramian.
        """
        ...

    # ── Rounding ───────────────────────────────────────────────────

    @abstractmethod
    def tround(self, tensor: Any, precision_order: int = 0) -> Any:
        """Round tensor elements to *precision_order* significant digits.

        Parameters
        ----------
        tensor : array-like
            Input tensor.
        precision_order : int, default=0
            Number of significant decimal digits to keep.

        Returns
        -------
        rounded : array-like
            Rounded tensor of the same shape.
        """
        ...

    # ── Pure-math helpers (backend-independent) ────────────────────

    @staticmethod
    def get_vector_index(
        tensor_indices: tuple[int, ...],
        tensor_shape: tuple[int, ...] | list[int],
    ) -> int:
        """Convert subscript indices to a linear (Fortran-order) index.

        Parameters
        ----------
        tensor_indices : tuple[int, ...]
            Subscript indices, one per mode.
        tensor_shape : tuple[int, ...] | list[int]
            Shape of the tensor.

        Returns
        -------
        index : int
            Linear index.
        """
        return int(np.ravel_multi_index(tensor_indices, tensor_shape, order='F'))

    @staticmethod
    def get_kronecker_factor_column_indices(
        column_index: int,
        tensor_shape: tuple[int, ...] | list[int],
    ) -> tuple[int, ...]:
        """Decompose a linear (Fortran-order) index into per-factor indices.

        Parameters
        ----------
        column_index : int
            Linear index into the Kronecker product.
        tensor_shape : tuple[int, ...] | list[int]
            Shape of the core tensor.

        Returns
        -------
        indices : tuple[int, ...]
            Per-factor column indices.
        """
        return np.unravel_index(column_index, tensor_shape, order='F')
