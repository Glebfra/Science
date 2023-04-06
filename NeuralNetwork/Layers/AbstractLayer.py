from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

NoneArrays = list[float] | np.ndarray[float] | None
ArrayFunction = Callable[[np.ndarray[float]], np.ndarray]


class AbstractLayer(ABC):
    weights: NoneArrays = None
    biases: NoneArrays = None

    inputs: NoneArrays = None
    outputs: NoneArrays = None

    def __init__(self, number_of_neurons: int):
        self.number_of_neurons: int = number_of_neurons

        self.activation_function: ArrayFunction = lambda x: \
            1 / (1 + np.exp(-x))
        self.activation_function_derivative: ArrayFunction = lambda x: \
            self.activation_function(x) * (1 - self.activation_function(x))

        self.previous_layer: AbstractLayer | None = None

    @abstractmethod
    def feed_forward(self, inputs: list[float] | np.ndarray[float]) -> np.ndarray:
        if isinstance(inputs, list):
            inputs = np.array(inputs)
        self.inputs = inputs.reshape((len(inputs), 1))
        self.outputs = self.activation_function(self.weights @ self.inputs + self.biases)
        return self.outputs

    def get_derivative_error(self) -> np.ndarray:
        return self.activation_function_derivative(self.weights @ self.inputs + self.biases)

    def set_activation_function(self, function: ArrayFunction, function_derivative: ArrayFunction) -> None:
        self.activation_function = function
        self.activation_function_derivative = function_derivative

    def set_previous_layer(self, previous_layer):
        self.previous_layer = previous_layer

    def __str__(self) -> str:
        return f'Weights: {self.weights}, biases: {self.biases}'
