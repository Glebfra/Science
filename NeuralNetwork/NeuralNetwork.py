import numpy as np
from matplotlib import pyplot as plt

from Layers import *


class NeuralNetwork(object):
    input_layer: InputLayer | None
    hidden_layers: list[HiddenLayer] | None
    output_layer: OutputLayer | None

    def __init__(self):
        self.input_layer = None
        self.hidden_layers = None
        self.output_layer = None

    def add_input_layer(self, number_of_neurons: int):
        self.input_layer = InputLayer(number_of_neurons)

    def add_hidden_layer(self, number_of_neurons: int):
        if self.hidden_layers is None:
            self.hidden_layers = [HiddenLayer(self.input_layer, number_of_neurons)]
        self.hidden_layers.append(HiddenLayer(self.hidden_layers[len(self.hidden_layers)-1], number_of_neurons))

    def add_hidden_layers(self, number_of_neurons: int, number_of_layers):
        for _ in range(number_of_layers):
            self.add_hidden_layer(number_of_neurons)

    def add_output_layer(self, number_of_neurons: int):
        self.output_layer = OutputLayer(self.hidden_layers[len(self.hidden_layers)-1], number_of_neurons)

    def feed_forward(self, inputs) -> np.ndarray:
        output = self.input_layer.feed_forward(inputs)
        for hidden_layer in self.hidden_layers:
            output = hidden_layer.feed_forward(output)
        output = self.output_layer.feed_forward(output)
        return output

    def back_propagation(self, inputs: list | np.ndarray, truth_out: list | np.ndarray, learning_rate) -> float:
        if isinstance(inputs, list):
            inputs = np.array(inputs)
        inputs = inputs.reshape((len(inputs), 1))
        if isinstance(truth_out, list):
            truth_out = np.array(truth_out)
        truth_out = truth_out.reshape((len(truth_out), 1))

        out: np.ndarray = self.feed_forward(inputs)
        error: np.ndarray = self._error_derivative(out, truth_out)
        layer_error: np.ndarray = error * self.output_layer.get_derivative_error()
        out_error = self.output_layer.back_propagation(layer_error, learning_rate)
        for hidden_layer in reversed(self.hidden_layers):
            out_error = hidden_layer.back_propagation(out_error, learning_rate)

        return self._error(out, truth_out).mean()

    def train(self, train_inputs, train_outputs, epochs: int, learning_rate):
        error = []
        for epoch in range(epochs):
            input_error = []
            for index, train_input in enumerate(train_inputs):
                input_error.append(self.back_propagation(train_input, train_outputs[index], learning_rate))
            error.append(np.array(input_error).mean())
            if epoch % 100 == 0:
                # self.back_propagation(np.random.random((self.input_layer.number_of_neurons, 1)), np.random.random((self.output_layer.number_of_neurons, 1)), learning_rate)
                print(f'epoch: {epoch}')
        return error

    @staticmethod
    def _error(out, truth_out) -> np.ndarray:
        return (out - truth_out) ** 2

    @staticmethod
    def _error_derivative(out, truth_out) -> np.ndarray:
        return 2 * (out - truth_out)

    @classmethod
    def create_default_network(
            cls,
            number_of_input_neurons: int,
            number_of_hidden_neurons: int,
            number_of_hidden_layers: int,
            number_of_output_neurons: int
    ):
        class_object = cls()
        class_object.add_input_layer(number_of_input_neurons)
        class_object.add_hidden_layers(number_of_hidden_neurons, number_of_hidden_layers)
        class_object.add_output_layer(number_of_output_neurons)

        return class_object


if __name__ == '__main__':
    network = NeuralNetwork.create_default_network(
        number_of_input_neurons=3,
        number_of_hidden_neurons=2,
        number_of_hidden_layers=1,
        number_of_output_neurons=2
    )
    train_inputs = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 1]]
    train_outputs = [[0], [1], [0], [1], [0], [1], [1]]
    epochs = 5000

    errors = network.train(train_inputs, train_outputs, epochs, learning_rate=0.02)
    epochs = np.linspace(1, epochs, epochs)

    print(network.feed_forward([1, 1, 0]))
    print(network.feed_forward([0, 1, 0]))
    print(network.feed_forward([1, 1, 1]))

    plt.plot(epochs, errors)
    plt.show()
