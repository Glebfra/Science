from typing_extensions import override

import matplotlib.pyplot as plt
import numpy as np


class Plotter:
    def __init__(self, figure=1, dimension='2d'):
        self.fig = plt.figure(figure)
        if dimension == '2d':
            self.ax = self.fig.add_subplot()
        elif dimension == '3d':
            self.ax = self.fig.add_subplot(projection='3d')

    @override
    def plot(self, *args, **kwargs):
        self.ax.plot(*args, **kwargs)

    @override
    def scatter(self, *args, **kwargs):
        self.ax.scatter(*args, **kwargs)

    @override
    def plot_surface(self, *args, **kwargs):
        self.ax.plot_surface(*args, **kwargs)

    @override
    def plot_wireframe(self, *args, **kwargs):
        self.ax.plot_wireframe(*args, **kwargs)

    @override
    def quiver(self, *args, **kwargs):
        self.ax.quiver(*args, **kwargs)

    @override
    def xlabel(self, *args, **kwargs):
        self.ax.set_xlabel(*args, **kwargs)

    @override
    def ylabel(self, *args, **kwargs):
        self.ax.set_ylabel(*args, **kwargs)

    @override
    def zlabel(self, *args, **kwargs):
        self.ax.set_zlabel(*args, **kwargs)

    @override
    def grid(self, *args, **kwargs):
        self.ax.grid(*args, **kwargs)

    @override
    def title(self, *args, **kwargs):
        self.ax.set_title(*args, **kwargs)

    @override
    def legend(self, *args, **kwargs):
        self.ax.legend(*args, **kwargs)

    def save(self, *args, **kwargs):
        self.fig.savefig(*args, **kwargs)

    @staticmethod
    @override
    def show():
        plt.show()


if __name__ == '__main__':
    plotter = Plotter(dimension='3d')

    [x, y] = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
    z = x ** 2 + y ** 2
    plotter.plot_wireframe(x, y, z)
    plotter.show()
