from multiprocessing import Process

import requests
import numpy as np
import tensorflow as tf

from src.maths.Error import r_squared

functions = ['sigmoid', 'tanh', 'relu', 'gelu', 'linear']


data = requests.get('http://77.246.98.155/database/phase_diagram/?format=json').json()
temperature = np.array([dat['temperature'] for dat in data if dat['temperature']])
pressure = np.array([dat['pressure'] for dat in data if dat['temperature']])
density = np.array([dat['density'] for dat in data if dat['temperature']])

MIN_TEMPERATURE = min(temperature)
MAX_TEMPERATURE = max(temperature)
MIN_PRESSURE = min(pressure)
MAX_PRESSURE = max(pressure)
MIN_DENSITY = min(density)
MAX_DENSITY = max(density)

linearized_temperature = (temperature - MIN_TEMPERATURE) / (MAX_TEMPERATURE - MIN_TEMPERATURE)
linearized_pressure = (pressure - MIN_PRESSURE) / (MAX_PRESSURE - MIN_PRESSURE)
linearized_density = (density - MIN_DENSITY) / (MAX_DENSITY - MIN_DENSITY)

M = 200.59e-3
R = 8.31


def train(*args):
    func1, func2, x_train, y_train = args

    inputs = tf.keras.layers.Input(shape=(2,))
    x = tf.keras.layers.Dense(20, activation=func1)(inputs)
    x = tf.keras.layers.Dense(20, activation=func2)(x)
    outputs = tf.keras.layers.Dense(1, activation='linear')(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs, name=f'{func1}/{func2}')
    model.compile(optimizer='Adam', loss="mse", metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=200, use_multiprocessing=True, verbose=0)

    y_test = model.predict(x_train).T
    y_test_reshape = np.reshape(y_test, (837,))
    rs = r_squared(y_test_reshape, linearized_density)
    response = {"rs": rs, "func1": func1, "func2": func2}
    print(response)


if __name__ == '__main__':
    x_train = np.array([linearized_temperature, linearized_pressure]).T
    y_train = np.array([linearized_density]).T

    for func1 in functions:
        for func2 in functions:
            process = Process(target=train, args=(func1, func2, x_train, y_train))
            process.start()
