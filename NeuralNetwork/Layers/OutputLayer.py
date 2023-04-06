import numpy as np

from .AbstractLayer import AbstractLayer


class OutputLayer(AbstractLayer):
    def __init__(self, previous_layer, number_of_neurons: int):
        super().__init__(number_of_neurons)
        self.set_previous_layer(previous_layer)

        self.weights = np.random.random((self.number_of_neurons, previous_layer.number_of_neurons))
        self.biases = np.random.random((self.number_of_neurons, 1))

    def feed_forward(self, inputs: list[float] | np.ndarray[float]) -> np.ndarray:
        return super().feed_forward(inputs)

    def back_propagation(self, error, learning_rate) -> np.ndarray:
        error = error.reshape((len(error), 1))
        out_error = self.weights.T @ error
        self.weights -= learning_rate * (error @ self.inputs.reshape((1, len(self.inputs))))
        self.biases -= learning_rate * error
        return out_error
