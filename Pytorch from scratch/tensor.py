from typing import Union, List, Callable
from dataclasses import dataclass
import numpy as np

Scalar = Union[int, float]

Data = Union[Scalar, list, np.ndarray]

@dataclass(frozen=True)
class Leaf:
    value: "Tensor"
    grad_fn: Callable[[np.ndarray], np.ndarray]


class Tensor:
    def __init__(self, data: Data, requires_grad: bool = False, dependencies: Optional[List[Leaf]] = None, dtype=np.float32):
        self._data = data
        self.requires_grad = requires_grad
        self.dependencies = dependencies or []
        self.dtype = dtype