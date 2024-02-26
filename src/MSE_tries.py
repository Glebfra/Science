import numpy as np
from scipy.optimize import curve_fit

from PhaseDiagram.Interface.ConnectionInterface import ConnectionInterface
from maths.Error import r_squared
from plotter.Plotter import Plotter

connection = ConnectionInterface()
data = connection.unsecure_get(
    f'http://77.246.98.155/database/phase_diagram/?format=json'
).json()
pressure, density = np.array([dat['pressure'] for dat in data]), np.array([dat['density'] for dat in data])
temperature = np.array([dat['temperature'] for dat in data])

R = 8.31
M = 200.59e-3


def phase_MSE():
    def mapping(x, b2, b3):
        return x + b2 * (x ** 2) + b3 * (x ** 3)

    M = 200.59e-3
    R = 8.31

    connection = ConnectionInterface()

    temperatures = []
    for i in range(300, 720, 20):
        temperatures.append(i + 273.15)
    for i in range(800, 1100, 100):
        temperatures.append(i + 273.15)
    for i in range(1200, 2200, 200):
        temperatures.append(i + 273.15)

    coefficients = []
    for temperature in temperatures:
        data = connection.secure_get(
            f'http://api.localhost/database/phase_diagram/?temperature={temperature}'
        ).json()
        pressure, density = np.array([dat['pressure'] for dat in data]), np.array([dat['density'] for dat in data])
        temp = np.array([dat['temperature'] for dat in data])

        popt, pcov = curve_fit(mapping, density, pressure * M / (R * temp))

        coefficients.append(popt)

    coefficients = np.array(coefficients).T

    plotter = Plotter()
    plotter.plot(temperatures, coefficients[0])
    plotter.xlabel(r'$T$ $(K)$')
    plotter.ylabel(r'$B_i$')
    plotter.grid(True)
    plotter.show()


def ideal_gas():
    new_pressure = R / M * (density * temperature)

    plotter = Plotter(dimension='3d')
    plotter.scatter(temperature, pressure, density, color='b')
    plotter.scatter(temperature, new_pressure, density, color='r')
    plotter.xlabel(r'$T$ $(K)$')
    plotter.ylabel(r'$p$ $(Pa)$')
    plotter.zlabel(r'$\rho$ $(kg/m^3)$')
    plotter.show()


def my_mse():
    coefficients = 0
    matrix_A = np.zeros((coefficients, coefficients))
    matrix_B = np.zeros(coefficients)
    for i in range(coefficients):
        for j in range(coefficients):
            matrix_A[i, j] = np.sum(density ** (i + j + 2))
        matrix_B[i] = -np.sum(pressure * M / (density * R * temperature) + density ** (i + 1))
    popts = np.linalg.solve(matrix_A, matrix_B)
    print(f'popts: {popts}')

    new_pressure = 1
    for i, popt in enumerate(popts):
        new_pressure += popt * density ** (i + 1)
    new_pressure *= density * R * temperature / M

    R_squared = r_squared(pressure, new_pressure)
    print(f'R^2: {R_squared}')

    plotter = Plotter(dimension='3d')
    plotter.scatter(temperature, new_pressure, density, color='r')
    plotter.scatter(temperature, pressure, density, color='b')
    plotter.xlabel(r'$T$ $(K)$')
    plotter.ylabel(r'$p$ $(Pa)$')
    plotter.zlabel(r'$\rho$ $(kg/m^3)$')
    plotter.show()


def new_mse():
    def mapping(x, b2):
        return 1 + b2 * x

    popt, pcov = curve_fit(mapping, pressure * M / (R * density * temperature), density)
    new_pressure = mapping(density, *popt) * density * temperature * R / M

    plotter = Plotter(dimension='3d')
    plotter.scatter(temperature, pressure, density, color='b')
    plotter.scatter(temperature, new_pressure, density, color='r')
    plotter.show()


def try_to_linearize():
    def mapping(x, a):
        return a * x

    temperatures = []
    for i in range(300, 720, 20):
        temperatures.append(i + 273.15)
    for i in range(800, 1100, 100):
        temperatures.append(i + 273.15)
    for i in range(1200, 2200, 200):
        temperatures.append(i + 273.15)

    coefficients = []
    temperature_coefficients = {}
    for temperature_i in temperatures:
        data = connection.unsecure_get(
            f'http://api.localhost/database/phase_diagram/?temperature={temperature_i}'
        ).json()
        pressure_i, density_i = np.array([dat['pressure'] for dat in data]), np.array([dat['density'] for dat in data])
        temp = np.array([dat['temperature'] for dat in data])

        popt, pcov = curve_fit(mapping, density_i, pressure_i * M / (R * temp))
        temperature_coefficients[str(temperature_i)] = popt
        coefficients.append(popt)

    coefficients = np.array(coefficients).T

    new_pressure = []
    new_density = []
    new_temperature = []
    for i in range(len(temperature)):
        if str(temperature[i]) in temperature_coefficients.keys():
            new_pressure.append(
                temperature[i] * R / M * (density[i] * temperature_coefficients[str(temperature[i])][0])
                )
            new_density.append(density[i])
            new_temperature.append(temperature[i])
    new_pressure = np.array(new_pressure)
    new_density = np.array(new_density)
    new_temperature = np.array(new_temperature)

    plotter = Plotter()
    plotter.plot(temperatures, coefficients[0])
    # plotter.plot(temperatures, coefficients[1])
    plotter.xlabel(r'$T$ $(K)$')
    plotter.ylabel(r'$B_1$')
    plotter.grid(True)
    plotter.show()

    plotter = Plotter(dimension='3d')
    plotter.scatter(temperature, pressure, density, color='b')
    plotter.scatter(new_temperature, new_pressure, new_density, color='r')
    plotter.xlabel(r'$T$ $(K)$')
    plotter.ylabel(r'$p$ $(Pa)$')
    plotter.zlabel(r'$\rho$ $(kg/m^3)$')
    plotter.show()


if __name__ == '__main__':
    try_to_linearize()
