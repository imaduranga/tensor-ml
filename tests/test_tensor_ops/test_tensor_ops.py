import numpy as np
import pytest
from tensor_ml.tensor_ops.tensor_ops import TensorOps, NumpyOps

class TestTensorOps:
    def test_norm(self):
        ops = NumpyOps()
        x = np.array([3, 4])
        assert np.isclose(ops.norm(x), 5.0)

    def test_normalize(self):
        ops = NumpyOps()
        D = np.array([[3, 0], [4, 5]])
        Dn = ops.normalize(D)
        assert np.allclose(np.linalg.norm(Dn, axis=0), 1)

    def test_zeros_ones(self):
        ops = NumpyOps()
        assert np.all(ops.zeros((2, 2)) == 0)
        assert np.all(ops.ones((2, 2)) == 1)

    def test_abs_sign(self):
        ops = NumpyOps()
        x = np.array([-1, 0, 2])
        assert np.all(ops.abs(x) == np.abs(x))
        assert np.all(ops.sign(x) == np.sign(x))

    def test_argmax_argmin(self):
        ops = NumpyOps()
        x = np.array([1, 3, 2])
        assert ops.argmax(x) == 1
        assert ops.argmin(x) == 0

    def test_concatenate(self):
        ops = NumpyOps()
        arrs = [np.ones((2,)), np.zeros((2,))]
        out = ops.concatenate(arrs)
        assert np.all(out == np.array([1, 1, 0, 0]))

    def test_inf_property(self):
        ops = NumpyOps()
        assert ops.inf == np.inf

    def test_asarray_flatten(self):
        ops = NumpyOps()
        x = [1, 2, 3]
        arr = ops.asarray(x)
        assert isinstance(arr, np.ndarray)
        assert np.all(ops.flatten(arr) == np.array([1, 2, 3]))

    def test_to_device(self):
        ops = NumpyOps()
        x = np.array([1, 2])
        assert np.all(ops.to_device(x) == x)

    def test_nonzero(self):
        ops = NumpyOps()
        x = np.array([0, 2, 0, 3])
        nz = ops.nonzero(x)
        assert np.all(nz == np.array([1, 3]))

    def test_numpyops_inheritance(self):
        ops = NumpyOps()
        assert isinstance(ops, TensorOps)
        assert np.isclose(ops.norm(np.array([6, 8])), 10.0)

# TorchOps tests (if torch is available)
torch = pytest.importorskip("torch")
from tensor_ml.tensor_ops.tensor_ops import TorchOps

class TestTorchOps:
    def test_norm(self):
        ops = TorchOps()
        x = torch.tensor([3.0, 4.0])
        assert abs(ops.norm(x) - 5.0) < 1e-6

    def test_normalize(self):
        ops = TorchOps()
        D = torch.tensor([[3.0, 0.0], [4.0, 5.0]])
        Dn = ops.normalize(D)
        assert torch.allclose(torch.norm(Dn, dim=0), torch.ones(2))

    def test_zeros_ones(self):
        ops = TorchOps()
        assert torch.all(ops.zeros((2, 2)) == 0)
        assert torch.all(ops.ones((2, 2)) == 1)

    def test_abs_sign(self):
        ops = TorchOps()
        x = torch.tensor([-1.0, 0.0, 2.0])
        assert torch.all(ops.abs(x) == torch.abs(x))
        assert torch.all(ops.sign(x) == torch.sign(x))

    def test_argmax_argmin(self):
        ops = TorchOps()
        x = torch.tensor([1.0, 3.0, 2.0])
        assert ops.argmax(x) == 1
        assert ops.argmin(x) == 0

    def test_concatenate(self):
        ops = TorchOps()
        arrs = [torch.ones(2), torch.zeros(2)]
        out = ops.concatenate(arrs)
        assert torch.all(out == torch.tensor([1.0, 1.0, 0.0, 0.0]))

    def test_inf_property(self):
        ops = TorchOps()
        assert ops.inf == float('inf')

    def test_asarray_flatten(self):
        ops = TorchOps()
        x = [1.0, 2.0, 3.0]
        arr = ops.asarray(x)
        assert isinstance(arr, torch.Tensor)
        assert torch.all(ops.flatten(arr) == torch.tensor([1.0, 2.0, 3.0]))

    def test_to_device(self):
        ops = TorchOps()
        x = torch.tensor([1.0, 2.0])
        assert torch.all(ops.to_device(x) == x)

    def test_nonzero(self):
        ops = TorchOps()
        x = torch.tensor([0.0, 2.0, 0.0, 3.0])
        nz = ops.nonzero(x)
        assert torch.all(nz == torch.tensor([1, 3]))

    def test_torchops_inheritance(self):
        ops = TorchOps()
        assert isinstance(ops, TensorOps)
        assert abs(ops.norm(torch.tensor([6.0, 8.0])) - 10.0) < 1e-6
