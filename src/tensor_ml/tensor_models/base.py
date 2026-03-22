"""Abstract base class for tensor-based models."""

from abc import ABC, abstractmethod
from typing import Optional, Any, Dict

__all__ = ["BaseTensorModel"]


class BaseTensorModel(ABC):
    """Abstract base class for tensor-based machine learning models.

    Supports NumPy and PyTorch backends.
    Subclasses must implement :meth:`fit` and :meth:`predict`.
    """

    def __init__(self) -> None:
        pass

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, return parameters for contained sub-objects.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        raise NotImplementedError("get_params must be implemented by subclass.")

    def set_params(self, **params: Any) -> 'BaseTensorModel':
        """Set the parameters of this estimator.

        Parameters
        ----------
        **params
            Estimator parameters.

        Returns
        -------
        self : BaseTensorModel
        """
        raise NotImplementedError("set_params must be implemented by subclass.")

    @abstractmethod
    def fit(self, X: Any, y: Optional[Any] = None, **kwargs: Any) -> 'BaseTensorModel':
        """Fit the model to data.

        Parameters
        ----------
        X : array-like
            Input data (``np.ndarray`` or ``torch.Tensor``).
        y : array-like, optional
            Target data.

        Returns
        -------
        self : BaseTensorModel
            The fitted model instance.
        """
        ...

    @abstractmethod
    def predict(self, X: Any, **kwargs: Any) -> Any:
        """Generate predictions from the fitted model.

        Parameters
        ----------
        X : array-like
            Input data.

        Returns
        -------
        predictions : array-like
            Model predictions.
        """
        ...

    def score(self, X: Any, y: Optional[Any] = None, **kwargs: Any) -> float:
        """Return a default score for predictions (e.g., R², accuracy).

        Parameters
        ----------
        X : array-like
            Input data.
        y : array-like, optional
            Ground-truth target data.

        Returns
        -------
        score : float
        """
        raise NotImplementedError("score method must be implemented by subclass.")
