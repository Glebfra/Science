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
            plt.rcParams.update({"font.family": "Helvetica"})

    def plot(self, *args, **kwargs):
        pass

    def plot3(self, *args, **kwargs):
        self.ax = self.fig.add_subplot(111, projection='3d', **kwargs)
        self.ax.plot(*args)

    def scatter3(self, *args, **kwargs):
        self.ax = self.fig.add_subplot(111, projection='3d', **kwargs)
        self.ax.scatter(*args)

    @staticmethod
    def show():
        plt.show()
