from .SaturationLine import SaturationLine


class Genetics:
    def __init__(self):
        self.functions = [
            'a*x+b*y+c',
            'a*x+b*y+c*x*y+d',
            'a*x+b*y+c*x*y+d*x**2+e',
            'a*x+b*y+c*x*y+d*x**2+e*y**2+f',
        ]
        self.saturation_line = SaturationLine()
        self.saturation_line.load_data()

    def get_max(self):
        for function in self.functions:
            def func(*args, **kwargs):
                return exec(function)
            self.saturation_line.mapping = func
            data = self.saturation_line.line_data
            print(self.saturation_line.r_squared(self.saturation_line.rho, data[2]))


if __name__ == '__main__':
    genetics = Genetics()
    genetics.get_max()
