"""Backend enumeration types for tensor-ml."""

from enum import Enum

__all__ = ["BackendType"]


class BackendType(Enum):
    """Supported compute backends."""

    NUMPY = "numpy"
    TORCH = "torch"

