import numpy as np

from .Interface.ConnectionInterface import ConnectionInterface


class PhaseDiagram(ConnectionInterface):
    URL = 'http://api.localhost/database/phase_diagram/'
    URL_STORAGE = 'http://api.localhost/database/storage/'
    ELEMENT_HG_ID = 4
    A: float = None
    B: float = None
    C: float = None
    D: float = None

    critical_temperature = 1750
    critical_pressure = 172e6

    def __init__(self):
        self._data = self.unsecure_get(url=f'{self.URL}').json()
        self.temperature = np.array([dat['temperature'] for dat in self._data])
        self.pressure = np.array([dat['pressure'] for dat in self._data])
        self.density = np.array([dat['density'] for dat in self._data])

    @property
    def data(self) -> tuple:
        return self.temperature, self.pressure, self.density

    def vander_vals_coeff(self) -> tuple:
        b = 8.31 * self.critical_temperature / (8 * self.critical_pressure)
        a = 27 * b ** 2 * self.critical_pressure
        return a, b

    def plate(self, temperatures=(453, 2273), pressures=(1e3, 2e7)) -> tuple:
        x, y = np.meshgrid(np.linspace(*np.log(temperatures), 100), np.linspace(*np.log(pressures), 100))
        z = (-self.A * x - self.B * y - self.D) / self.C
        return x, y, z

    def calculate_plate_coeff(self):
        first_point = self.unsecure_get(f'{self.URL}?temperature={180 + 273.15}&pressure={1000}').json()[0]
        second_point = self.unsecure_get(f'{self.URL}?temperature={2000 + 273.15}&pressure={1000}').json()[0]
        third_point = self.unsecure_get(f'{self.URL}?temperature={2000 + 273.15}&pressure={20000000}').json()[0]

        points = np.array(
            [
                [np.log(first_point['temperature']), np.log(first_point['pressure']), np.log(first_point['density'])],
                [np.log(second_point['temperature']), np.log(second_point['pressure']), np.log(second_point['density'])],
                [np.log(third_point['temperature']), np.log(third_point['pressure']), np.log(third_point['density'])]
            ]
        )
        p0, p1, p2 = points
        x0, y0, z0 = p0
        x1, y1, z1 = p1
        x2, y2, z2 = p2

        ux, uy, uz = u = [x1 - x0, y1 - y0, z1 - z0]
        vx, vy, vz = v = [x2 - x0, y2 - y0, z2 - z0]

        u_cross_v = [uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx]
        point = np.array(p0)
        normal = np.array(u_cross_v)

        d = -point.dot(normal)
        self.A, self.B, self.C = normal
        self.D = d

    def save_plate_coeff(self):
        self.secure_post(
            url=self.URL_STORAGE,
            data={
                "query": "plate_parameters",
                "element": self.ELEMENT_HG_ID,
                "values": {
                    "A": self.A,
                    "B": self.B,
                    "C": self.C,
                    "D": self.D,
                },
            }
        )

    def load_plate_coeff(self):
        parameters = self.secure_get(
            url=f'{self.URL_STORAGE}?query=plate_parameters'
        ).json()[0]['values']
        self.A = parameters['A']
        self.B = parameters['B']
        self.C = parameters['C']
        self.D = parameters['D']
