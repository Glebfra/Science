from typing import Callable

import numpy as np
from scipy.optimize import curve_fit


class Approximator(object):
    func: Callable
    popt: tuple | None

    def __init__(self, func: Callable):
        self.func = func
        self.popt = None

    def approximate(self, *args, **kwargs) -> tuple:
        self.popt, _ = curve_fit(self.func, *args, **kwargs)
        return self.popt

    def calculate(self, x) -> np.ndarray | list:
        return self.func(x, *self.popt)
