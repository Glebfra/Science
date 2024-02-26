import numpy as np


def r_squared(y, y_tr):
    y = y / max(np.abs(y))
    y_tr = y_tr / max(np.abs(y_tr))
    return 1 - sum((y - y_tr) ** 2) / sum((y - y.mean()) ** 2)
