import numpy as np
import tensorflow as tf


class Network:
    def __init__(self):
        inputs = tf.keras.layers.Input(shape=(3,))
        outputs = tf.keras.layers.Dense(1)(inputs)
        self.model = tf.keras.models.Model(
            inputs=inputs,
            outputs=outputs
        )
        self.model.compile(optimizer="Adam", loss="mse", metrics=["mae"])

    def train(self, *args, **kwargs):
        self.model.fit(*args, **kwargs)


if __name__ == '__main__':
    x = np.random.random((500, 3)).astype('float32')
    y = np.random.random((500, 1)).astype('float32')
    print(x)
    # network = Network()
    # network.train(x, y, epochs=5)
