import numpy as np
import requests
from scipy.optimize import curve_fit
from typing import Callable


class SaturationLine:
    URL = 'http://api.localhost/database/saturation/?format=json'
    mapping: Callable[[tuple[np.ndarray], float], np.ndarray] = None

    def __init__(self):
        data = requests.get(self.URL).json()
        self.temperature = np.array([dat['temperature'] for dat in data])
        self.pressure = np.array([dat['pressure'] for dat in data])
        self.density = np.array([dat['density'] for dat in data])

    @property
    def data(self) -> list[np.ndarray]:
        return [self.temperature, self.pressure, self.density]

    @property
    def linear_data(self) -> list[np.ndarray]:
        return [1 / self.temperature, np.log(self.pressure), np.log(self.density)]

    @property
    def line_popt(self) -> list:
        popt, pcov = curve_fit(self.mapping, (1 / self.temperature, np.log(self.pressure)), np.log(self.density))
        return popt

    def linearize(self) -> list:
        return [1 / self.temperature, np.log(self.pressure),
                np.log(self.mapping((1 / self.temperature, np.log(self.pressure)), *self.line_popt()))]
