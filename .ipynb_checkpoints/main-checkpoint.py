import tensorflow as tf

from src.PhaseDiagram.PhaseDiagram import PhaseDiagram


class BinomialOutputLayer(tf.keras.layers.Dense):
    def __init__(self, x, **kwargs):
        super().__init__(**kwargs)
        self.x = x

    def call(self, inputs, *args, **kwargs):
        return self.activation()


class PhaseNetwork:
    def __init__(self, activation='gelu'):
        self.phase = PhaseDiagram()
        self.model = self._model(activation)

    @property
    def linearized_data(self):
        temp = []
        for dat in self.phase.data:
            temp.append((dat - min(dat)) / (max(dat) - min(dat)))
        return temp

    @staticmethod
    def _model(activation='gelu'):
        inputs = tf.keras.layers.Input(shape=(2,))
        x = tf.keras.layers.Dense(10, activation=activation)(inputs)
        x = tf.keras.layers.Dense(10, activation=activation)(x)
        outputs = tf.keras.layers.Dense(1, activation='linear')(x)
        model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='Adam', loss="mse", metrics=['accuracy'])
        return model


if __name__ == '__main__':
    phase_network = PhaseNetwork()
    print(phase_network.linearized_data)
