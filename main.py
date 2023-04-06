from pathlib import Path

import dotenv
import numpy as np

from src.data.FileService import FileService
from src.plotter.Plotter import Plotter

from NeuralNetwork.NeuralNetwork import NeuralNetwork

# Enabling environ variables
dotenv.load_dotenv('.env')
PROJECT_DIR = Path(__file__).resolve().parent
dotenv.set_key(f'{PROJECT_DIR}/.env', 'PROJECT_DIR', str(PROJECT_DIR))

# Loading file service
file_service = FileService()


def approx_T(x):
    a = 144.04338419277755
    b = 215.6913065948331
    c = -0.10619280166072138
    return a*(x+c)**(1/3)+b


def approx_V(x):
    a = 0.2439937981419252
    b = 0.01526354824634014
    c = 0.014487170791688337
    return a / (x + b) + c


# Loading data
datas = file_service.load_file(filename='ртуть.json', type='json')

temperature, pressure, density = [], [], []
train_inputs, train_outputs = [], []
for data in datas:
    T_m = approx_T(data['p'])
    V_m = approx_V(data['p'])
    temperature_norm = T_m / (data['T'] + 273)
    pressure_norm = 1 / data['p']
    density_norm = data['V']*V_m

    temperature.append(temperature_norm)
    pressure.append(pressure_norm)
    density.append(density_norm)

    train_inputs.append([temperature_norm, pressure_norm])
    train_outputs.append([density_norm])

temperature, pressure, density = np.array(temperature), np.array(pressure), np.array(density)

# Creating neural network
# network = NeuralNetwork.create_default_network(
#     number_of_input_neurons=2,
#     number_of_hidden_layers=2,
#     number_of_hidden_neurons=3,
#     number_of_output_neurons=1
# )
network = NeuralNetwork()
network.add_input_layer(2)
network.add_hidden_layers(3, 2)
network.add_output_layer(1)
network.output_layer.set_activation_function(lambda x: x, lambda x: 1)
for hidden_layer in network.hidden_layers:
    hidden_layer.set_activation_function(lambda x: np.tanh(x), lambda x: np.cosh(x)**(-2))

epochs = 1000
errors = network.train(
    train_inputs=train_inputs,
    train_outputs=train_outputs,
    epochs=epochs,
    learning_rate=0.005
)
epochs = np.linspace(1, epochs, epochs)

temperature_x, pressure_x = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
density_x = np.zeros((100, 100))
for index_1, value_1 in enumerate(temperature_x):
    for index_2, value_2 in enumerate(temperature_x[index_1]):
        density_x[index_1][index_2] = network.feed_forward([temperature_x[index_1][index_2], pressure_x[index_1][index_2]])

# density_x = np.zeros((len(density), 1))
# for index, value in enumerate(temperature):
#     density_x[index] = network.feed_forward([temperature[index], pressure[index]])

# Plotting
plotter = Plotter()
plotter.surface(temperature_x, pressure_x, density_x, xlabel=r'$1/T$ norm', ylabel=r'$1/P$ norm', zlabel=r'$1/\rho$ norm')
plotter.scatter3(temperature, pressure, density, xlabel=r'$1/T$ norm', ylabel=r'$1/P$ norm', zlabel=r'$1/\rho$ norm')
# plotter.scatter3(temperature, pressure, density_x)

plotter1 = Plotter()
plotter1.plot(epochs, errors, xlabel='epochs', ylabel='MSE')

plotter.show()
plotter1.show()
