from typing import Union
import numpy as np

Scalar = Union[int, float]

Data = Union[Scalar, list, np.ndarray]

class Tensor:
    def __init__(self, data: Data):
        self._data = data