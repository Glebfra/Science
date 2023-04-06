import matplotlib.pyplot as plt


class Plotter(object):
    def __init__(self, **kwargs):
        self.fig = plt.figure()
        self.ax = None

        if 'use_latex' in kwargs and kwargs['use_latex']:
            plt.rcParams.update({"text.usetex": kwargs['use_latex']})

        if 'font_family' in kwargs:
            plt.rcParams.update({"font.family": kwargs['font_family']})
        else:
            pass
            # plt.rcParams.update({"font.family": "Helvetica"})

    def plot(self, *args, **kwargs):
        if 'xlabel' in kwargs:
            plt.xlabel(kwargs['xlabel'])
            del kwargs['xlabel']
        if 'ylabel' in kwargs:
            plt.ylabel(kwargs['ylabel'])
            del kwargs['ylabel']
        plt.plot(*args, **kwargs)

    def plot3(self, *args, **kwargs):
        if self.ax is None:
            self.ax = self.fig.add_subplot(111, projection='3d', **kwargs)
        self.ax.plot(*args)

    def scatter3(self, *args, **kwargs):
        if self.ax is None:
            self.ax = self.fig.add_subplot(111, projection='3d', **kwargs)
        self.ax.scatter(*args)

    def surface(self, *args, **kwargs):
        if self.ax is None:
            self.ax = self.fig.add_subplot(111, projection='3d', **kwargs)
        self.ax.plot_wireframe(*args)

    @staticmethod
    def show():
        plt.show()
