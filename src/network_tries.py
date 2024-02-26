import keras
import numpy as np
import tensorflow as tf
from keras import Model
from keras.src.layers import Conv2D

from PhaseDiagram.Interface.ConnectionInterface import ConnectionInterface

connection = ConnectionInterface()
data = connection.secure_get(
    f'http://api.localhost/database/phase_diagram/'
).json()
pressure, density = np.array([dat['pressure'] for dat in data]), np.array([dat['density'] for dat in data])
temperature = np.array([dat['temperature'] for dat in data])

R = 8.31
M = 200.59e-3

inputs = tf.keras.layers.Input(shape=(837,))
x = tf.keras.layers.Dense(837)(inputs)
x = tf.keras.layers.Dense(837)(x)
outputs = tf.keras.layers.Dense(5)(x)

model = keras.Model(inputs=inputs, outputs=outputs)
model.summary()

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.RMSprop(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

model.fit(epochs=10)


class CustomLoss(tf.keras.losses.Loss):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def call(self, y_true, y_pred):
        pass
