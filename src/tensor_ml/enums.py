from enum import Enum


class BackendType(Enum):
    """Supported compute backends."""

    NUMPY = "numpy"
    TORCH = "torch"
    PANDAS = "pandas"

