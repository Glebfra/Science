import numpy as np


class Error(object):
    def __init__(self):
        pass

    def R(self, y, y_tr):
        return 1 - sum((y-y_tr)**2) / sum((y-y.mean())**2)
