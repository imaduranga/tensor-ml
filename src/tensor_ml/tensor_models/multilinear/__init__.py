"""Multi-linear tensor models."""

from tensor_ml.tensor_models.multilinear.multilinear_model import MultilinearModel
from tensor_ml.tensor_models.multilinear.tlars import TLARS, TLARSConfig

__all__ = ["MultilinearModel", "TLARS", "TLARSConfig"]
