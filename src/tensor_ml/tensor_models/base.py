from typing import Optional, Any, Union
import numpy as np

class BaseTensorModel:
    """Generic base class for tensor-based machine learning models supporting NumPy, PyTorch, and pandas DataFrame."""

    def __init__(self):
        pass

    def fit(self, X: Any, y: Optional[Any] = None, **kwargs: Any) -> 'BaseTensorModel':
        """
        Fit the model to data.

        :param X: np.ndarray, torch.Tensor, or pandas.DataFrame
        :param y: np.ndarray, torch.Tensor, or pandas.DataFrame, optional
        :return: self
        """
        raise NotImplementedError("fit method must be implemented by subclass.")

    def predict(self, X: Any, **kwargs: Any) -> Any:
        """
        Predict using the model.

        :param X: np.ndarray, torch.Tensor, or pandas.DataFrame
        :return: predictions as np.ndarray, torch.Tensor, or pandas.DataFrame
        """
        raise NotImplementedError("predict method must be implemented by subclass.")

    def score(self, X: Any, y: Optional[Any] = None, **kwargs: Any) -> float:
        """
        Return a score for predictions (e.g., accuracy, R^2, etc.).

        :param X: np.ndarray, torch.Tensor, or pandas.DataFrame
        :param y: np.ndarray, torch.Tensor, or pandas.DataFrame, optional
        :return: score as float
        """
        raise NotImplementedError("score method must be implemented by subclass.")
