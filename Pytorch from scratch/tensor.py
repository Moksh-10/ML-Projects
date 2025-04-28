from typing import Union, List, Callable, Optional
from dataclasses import dataclass
import numpy as np

Scalar = Union[int, float]

Data = Union[Scalar, list, np.ndarray]

@dataclass(frozen=True)
class Leaf:
    value: "Tensor"
    grad_fn: Callable[[np.ndarray], np.ndarray]

class Tensor:
    def __init__(self, data: Data, requires_grad: bool = False,
                 dependencies: Optional[List[Leaf]] = None, dtype=np.float32):
        self._data = Tensor.build_ndarray(data, dtype)
        self.requires_grad = requires_grad
        self.dependencies = dependencies or []
        self.dtype = dtype
        self.grad: np.ndarray = np.zeros_like(self._data) if requires_grad else None

    @property
    def data(self) -> np.ndarray:
        return self._data

    @data.setter
    def data(self, data: Data):
        self._data = Tensor.build_ndarray(data)
        if self.requires_grad:
            self.zero_grad()

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def shape(self) -> int:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @staticmethod
    def build_ndarray(data: Data, dtype=np.float32) -> np.ndarray:
        if isinstance(data, Tensor):
            return np.array(data.data, dtype=dtype)
        if isinstance(data, np.ndarray):
            return data.astype(dtype=dtype)
        return np.array(data, dtype=dtype)

    def __repr__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad}, shape={self.shape})"

    def zero_grad(self):
        if self.grad is None:
            self.grad = np.zeros_like(self._data)
        else:
            self.grad.fill(0.0)
#
# t = Tensor([1, 2, 3], requires_grad=True)
# t.data = [5, 5, 5]
# print(t, t.grad, t.ndim, t.shape)


