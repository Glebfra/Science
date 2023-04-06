from pathlib import Path

import dotenv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from numpy.polynomial import Polynomial
from scipy.optimize import curve_fit

dotenv.load_dotenv('.env')
PROJECT_DIR = Path(__file__).resolve().parent
dotenv.set_key(f'{PROJECT_DIR}/.env', 'PROJECT_DIR', str(PROJECT_DIR))

workbook = pd.read_excel(open('./data/ртуть.xlsx', 'rb'), sheet_name='Линия насыщения')
data_len = len(workbook['Unnamed: 1'])
T = np.array(list(map(float, workbook['Unnamed: 1'][2:data_len])))
p = np.array(list(map(float, workbook['Unnamed: 2'][2:data_len])))
V = np.array(list(map(float, workbook['Unnamed: 3'][2:data_len])))


def mapping(x, a, b, c):
    return a/(x+b)+c

#
popt, _ = curve_fit(mapping, p, V)
a, b, c = popt
print(f'a: {a}, b: {b}, c: {c}')
p_x = np.linspace(0, 8, 1000)
V_x = mapping(p_x, a, b, c)
# plt.plot(p, V, 'b.')
plt.plot(p, V, 'b.', p_x, V_x, 'b-')
plt.show()
