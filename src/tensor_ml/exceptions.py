"""Custom exception hierarchy for tensor-ml.

All public exceptions inherit from :class:`TensorMLError` so that
callers can catch every library-specific error with a single clause.
"""

__all__ = [
    "TensorMLError",
    "NotFittedError",
    "BackendError",
    "ShapeMismatchError",
    "ValidationError",
]


class TensorMLError(Exception):
    """Base class for all tensor-ml exceptions."""


class NotFittedError(TensorMLError):
    """Raised when a model method is called before :meth:`fit`."""


class BackendError(TensorMLError):
    """Raised for backend-related failures (missing torch, unsupported type)."""


class ShapeMismatchError(TensorMLError, ValueError):
    """Raised when array shapes are incompatible."""


class ValidationError(TensorMLError, ValueError):
    """Raised when a parameter or input fails validation."""
