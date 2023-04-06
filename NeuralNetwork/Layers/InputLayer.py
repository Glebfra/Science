import numpy as np

from .AbstractLayer import AbstractLayer


class InputLayer(AbstractLayer):
    def __init__(self, number_of_neurons):
        super().__init__(number_of_neurons)

    def feed_forward(self, inputs: list[float] | np.ndarray[float]) -> np.ndarray:
        if isinstance(inputs, list):
            inputs = np.array(inputs)
        self.inputs = inputs.reshape((len(inputs), 1))
        self.outputs = self.inputs
        return self.outputs


if __name__ == '__main__':
    from AbstractLayer import AbstractLayer
