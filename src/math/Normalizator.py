from typing import Callable

import numpy as np

from src.math.Approximator import Approximator


class Normalizator(Approximator):
    def __init__(self, func: Callable):
        super().__init__(func)

    def normalize(self, data: dict[str, np.ndarray]):
        pass
