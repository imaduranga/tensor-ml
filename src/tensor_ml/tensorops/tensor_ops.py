from tensor_ml.enums import BackendType
import numpy as np

class TensorOps:
    def norm(self, x):
        return np.linalg.norm(x)
    def normalize(self, D):
        return D / np.linalg.norm(D, axis=0, keepdims=True)
    def zeros(self, shape):
        return np.zeros(shape)
    def ones(self, shape):
        return np.ones(shape)
    def abs(self, x):
        return np.abs(x)
    def sign(self, x):
        return np.sign(x)
    def argmax(self, x):
        return np.argmax(x)
    def argmin(self, x):
        return np.argmin(x)
    def concatenate(self, arrs):
        return np.concatenate(arrs)
    @property
    def inf(self):
        return np.inf
    def asarray(self, x):
        return np.asarray(x)
    def flatten(self, x):
        return x.flatten(order='F')
    def to_device(self, x, device=None):
        return x
    def nonzero(self, x):
        return np.nonzero(x)[0]
    def mean(self, x):
        return np.mean(x)
    def sum(self, x):
        return np.sum(x)
    def gramian(self, D):
        return D.T @ D

class NumpyOps(TensorOps):
    pass

class TorchOps(TensorOps):
    def __init__(self, device='cuda'):
        import torch
        import torch.nn.functional as F
        self.torch = torch
        self.F = F
        # Simplified device selection logic with robust torch.device check
        if device is None or (isinstance(device, str) and device == 'cuda'):
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(device, torch.device):
            self.device = device
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            raise TypeError(f"device must be None, a string, or torch.device, got {type(device)}")
    def norm(self, x):
        return self.torch.norm(x)
    def normalize(self, D):
        return self.F.normalize(D, dim=0)
    def zeros(self, shape):
        return self.torch.zeros(shape, device=self.device)
    def ones(self, shape):
        return self.torch.ones(shape, device=self.device)
    def abs(self, x):
        return self.torch.abs(x)
    def sign(self, x):
        return self.torch.sign(x)
    def argmax(self, x):
        return self.torch.argmax(x)
    def argmin(self, x):
        return self.torch.argmin(x)
    def concatenate(self, arrs):
        return self.torch.cat(arrs)
    @property
    def inf(self):
        return float('inf')
    def asarray(self, x):
        return self.torch.as_tensor(x, device=self.device)
    def flatten(self, x):
        if x.ndim <= 1:
            return x.contiguous().flatten()
        return x.permute(*reversed(range(x.ndim))).contiguous().flatten()
    def to_device(self, x, device=None):
        return x.to(self.device)
    def nonzero(self, x):
        return self.torch.nonzero(x, as_tuple=True)[0]
    def mean(self, x):
        return self.torch.mean(x)
    def sum(self, x):
        return self.torch.sum(x)
    def gramian(self, D):
        return D.t() @ D

class TensorOpsFactory:
    @staticmethod
    def get(backend, device=None):
        if backend == BackendType.TORCH or (isinstance(backend, str) and backend.lower() == 'torch'):
            return TorchOps(device)
        elif backend == BackendType.NUMPY or (isinstance(backend, str) and backend.lower() == 'numpy'):
            return NumpyOps()
        else:
            raise ValueError(f"Unsupported backend: {backend}")
