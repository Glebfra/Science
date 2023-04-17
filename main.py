from pathlib import Path

import dotenv
import numpy as np

from NeuralNetwork.NeuralNetwork import NeuralNetwork
from src.data.FileService import FileService
from src.math.Approximator import Approximator
from src.plotter.Plotter import Plotter

# Enabling environ variables
dotenv.load_dotenv('.env')
PROJECT_DIR = Path(__file__).resolve().parent
dotenv.set_key(f'{PROJECT_DIR}/.env', 'PROJECT_DIR', str(PROJECT_DIR))

# Loading file service
file_service = FileService()

# Loading data
datas = file_service.load_file(filename='ртуть.json', type='json')
temperature, pressure, density = np.zeros(len(datas)), np.zeros(len(datas)), np.zeros(len(datas))
for index, data in enumerate(datas):
    temperature[index] = data['T'] + 273
    pressure[index] = data['p']
    density[index] = 1/data['V']


# # Creating neural network
# network = NeuralNetwork()
# network.add_input_layer(2)
# network.add_hidden_layers(3, 2)
# network.add_output_layer(1)
# network.output_layer.set_activation_function(lambda x: x, lambda x: 1)
# for hidden_layer in network.hidden_layers:
#     hidden_layer.set_activation_function(lambda x: np.tanh(x), lambda x: np.cosh(x) ** (-2))
#
# epochs = 1000
# errors = network.train(
#     train_inputs=train_inputs,
#     train_outputs=train_outputs,
#     epochs=epochs,
#     learning_rate=0.005
# )
# epochs = np.linspace(1, epochs, epochs)
#
# temperature_x, pressure_x = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
# density_x = np.zeros((100, 100))
# for index_1, value_1 in enumerate(temperature_x):
#     for index_2, value_2 in enumerate(temperature_x[index_1]):
#         density_x[index_1][index_2] = network.feed_forward(
#             [temperature_x[index_1][index_2], pressure_x[index_1][index_2]]
#         )
#

# Plotting
plotter = Plotter()
plotter.scatter3(1/temperature, np.log(pressure), np.log(density), xlabel=r'$T (K)$', ylabel=r'$P (Bar)$', zlabel=r'$\rho (kg/m^3)$')
plotter.show()
