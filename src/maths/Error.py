import numpy as np


def r_squared(y, y_tr):
    y = y / np.max(np.abs(y))
    y_tr = y_tr / np.max(np.abs(y_tr))
    return 1 - np.sum((y - y_tr) ** 2) / np.sum((y - y.mean()) ** 2)
