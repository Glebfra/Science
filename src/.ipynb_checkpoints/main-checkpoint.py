import numpy as np
import tensorflow as tf
from scipy.optimize import curve_fit

from PhaseDiagram.Interface.ConnectionInterface import ConnectionInterface
from PhaseDiagram.PhaseDiagram import PhaseDiagram
from SaturationLine.SaturationLine import SaturationLine
from maths.Error import r_squared
from plotter.Plotter import Plotter


def get_phase_model():
    inputs = tf.keras.layers.Input(shape=(2,))
    x = tf.keras.layers.Dense(10, activation='gelu')(inputs)
    x = tf.keras.layers.Dense(10, activation='gelu')(x)
    outputs = tf.keras.layers.Dense(1, activation='linear')(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='Adam', loss="mse", metrics=['accuracy'])

    return model


def network():
    triple_temperature, critical_temperature = 234.3156, 1750
    triple_pressure, critical_pressure = 1.65e-4, 172e6

    saturation_line = SaturationLine()
    temperature, pressure, density = saturation_line.kalium_data

    linearized_temperature = (temperature - triple_temperature) / (critical_temperature - triple_temperature)
    linearized_pressure = (pressure - triple_pressure) / (critical_pressure - triple_pressure)
    linearized_density = (density - min(density)) / (max(density) - min(density))

    x_train = np.array([linearized_temperature, linearized_pressure]).T
    y_train = np.array([linearized_density]).T

    inputs = tf.keras.layers.Input(shape=(2,))
    x = tf.keras.layers.Dense(30)(inputs)
    x = tf.keras.layers.Dense(30)(x)
    outputs = tf.keras.layers.Dense(1)(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="Adam", loss="mse", metrics=["mae"])
    model.fit(x_train, y_train, epochs=50)
    model.summary()

    x_test = np.array([linearized_temperature, linearized_pressure]).T
    y_test = model.predict(x_test).T

    plotter = Plotter(dimension='3d')
    plotter.plot(linearized_temperature, linearized_pressure, linearized_density)
    plotter.scatter(linearized_temperature, linearized_pressure, y_test)
    plotter.show()


def network_phase():
    phase_diagram = PhaseDiagram()
    temperature, pressure, density = phase_diagram.kalium_data

    linearized_temperature = (temperature - min(temperature)) / (max(temperature) - min(temperature))
    linearized_pressure = (pressure - min(pressure)) / (max(pressure) - min(pressure))
    linearized_density = (density - min(density)) / (max(density) - min(density))

    x_train = np.array([linearized_temperature, linearized_pressure]).T
    y_train = np.array([linearized_density]).T

    model = get_phase_model()
    model.fit(x_train, y_train, epochs=500)

    x_test = np.array([linearized_temperature, linearized_pressure]).T
    y_test = model.predict(x_test).T

    plotter = Plotter(dimension='3d')
    plotter.scatter(linearized_temperature, linearized_pressure, linearized_density, color='b')
    plotter.scatter(linearized_temperature, linearized_pressure, y_test, color='r')
    plotter.show()

    linear_x = np.linspace(0, 1, 100)
    linear_y = linear_x

    plotter = Plotter(dimension='2d')
    plotter.scatter(linearized_density, y_test, alpha=0.5)
    plotter.plot(linear_x, linear_y, 'b-')
    plotter.xlabel(r'$Experimental$ $data$')
    plotter.ylabel(r'$Predicted$ $data$')
    plotter.grid(True)
    plotter.show()


def phase_MSE():
    def mapping(x, b1, b2, b3):
        return b1 * x + b2 * x ** 2 + b3 * x ** 3

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

        popt, pcov = curve_fit(mapping, density, pressure * M / (R * temperature))

        coefficients.append(popt)

    coefficients = np.array(coefficients).T

    plotter = Plotter()
    plotter.plot(temperatures, coefficients[0])
    plotter.plot(temperatures, coefficients[1])
    plotter.plot(temperatures, coefficients[2])
    plotter.grid(True)
    plotter.show()


def saturation():
    saturation_line = SaturationLine()
    saturation_line.mapping = lambda xy, a, b, c, d: a + b * xy[0] + c * xy[1] + d * xy[0] * xy[1]

    plotter = Plotter(dimension='3d')
    plotter.scatter(*saturation_line.kalium_data, color='b')

    new_rho = np.exp(
        saturation_line.mapping(
            (1 / saturation_line.lithium_temperature, np.log(saturation_line.lithium_pressure)), *saturation_line.line_popt
        )
    )
    print(f'popt: {saturation_line.line_popt}')
    print(f'R: {r_squared(saturation_line.kalium_data[2], new_rho)}')
    plotter.plot(*saturation_line.kalium_data[0:2], new_rho, 'c-')
    plotter.xlabel(r'$T$ $(K)$')
    plotter.ylabel(r'$p$ $(Pa)$')
    plotter.zlabel(r'$\rho$ $(kg/m^3)$')

    plotter.show()


def phase():
    phase_diagram = PhaseDiagram()
    phase_diagram.load_plate_coeff()
    plate = phase_diagram.plate()

    saturation = SaturationLine()

    analytical_temperature, analytical_pressure = np.meshgrid(
        np.linspace(235, 2500, 100), np.linspace(1000, 20000000, 100)
    )
    analytical_density = analytical_pressure / analytical_temperature * np.exp(-phase_diagram.D / phase_diagram.A)

    print(f'A: {phase_diagram.A}, B: {phase_diagram.B}, C: {phase_diagram.C}, D: {phase_diagram.D}')

    plotter = Plotter(dimension='3d')
    plotter.title(r'$Linearized$ $phase$ $data$')
    phase_data = list(map(np.array, phase_diagram.kalium_data))
    saturation_data = list(map(np.array, saturation.kalium_data))
    plotter.plot_wireframe(*plate)
    plotter.scatter(np.log(phase_data[0]), np.log(phase_data[1]), np.log(phase_data[2]), color='b')
    plotter.scatter(np.log(saturation_data[0]), np.log(saturation_data[1]), np.log(saturation_data[2]), color='r')
    plotter.xlabel(r'$ln(T)$')
    plotter.ylabel(r'$ln(p)$')
    plotter.zlabel(r'$ln(\rho)$')
    plotter.show()

    plotter = Plotter(dimension='3d')
    plotter.title(r'$Phase$ $data$')
    plotter.scatter(*phase_diagram.kalium_data, color='r')
    plotter.plot_wireframe(analytical_temperature, analytical_pressure, analytical_density)
    plotter.xlabel(r'$T$ $(K)$')
    plotter.ylabel(r'$p$ $(Pa)$')
    plotter.zlabel(r'$\rho$ $(kg/m^3)$')
    # plotter.plot(*saturation.data, color='m')
    plotter.show()


def only_phase():
    phase_diagram = PhaseDiagram()
    saturation_line = SaturationLine()

    plotter = Plotter(dimension='3d')
    plotter.scatter(*phase_diagram.kalium_data)
    plotter.plot(*saturation_line.kalium_data)
    plotter.xlabel(r'$T$ $(K)$')
    plotter.ylabel(r'$p$ $(Pa)$')
    plotter.zlabel(r'$\rho$ $(kg/m^3)$')
    plotter.show()


if __name__ == '__main__':
    network_phase()
