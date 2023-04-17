import os

import dotenv
import matplotlib.pyplot as plt
import numpy as np

from src.data.FileService import FileService
from src.math.Error import Error
from src.plotter.Plotter import Plotter

dotenv.load_dotenv('.env')
PROJECT_DIR: str = os.getenv('PROJECT_DIR')

error = Error()
file_service = FileService()
datas = file_service.load_file('ртуть_насыщение_2.json', 'json')
train_datas = file_service.load_file('ртуть.json', 'json')

temperature, pressure, rho = np.zeros(len(datas)), np.zeros(len(datas)), np.zeros(len(datas))
for index, data in enumerate(datas):
    temperature[index] = data['T']
    pressure[index] = data['p'] * 10
    rho[index] = 1 / data['V']

train_temperature, train_pressure, train_density = np.zeros(len(train_datas)), np.zeros(len(train_datas)), np.zeros(len(train_datas))
for index, data in enumerate(train_datas):
    train_temperature[index] = data['T'] + 273
    train_pressure[index] = data['p']
    train_density[index] = 1/data['V']

x1, y1, z1 = np.log(pressure[0]), 1 / temperature[0], np.log(rho[0])
x2, y2, z2 = np.log(1720), 1 / 1750, np.log(rho[len(pressure) - 1])

y = 1/np.linspace(234, 1750, 1000)
a1 = 1/(y2 - y1)*(x2-x1)
b1 = -y2*a1 + x2
a2 = 1/(y2 - y1)*(z2-z1)
b2 = - y2*a2 + z2
x = a1*y + b1
z = a2*y + b2

print(f'x = {round(a1, 5)}*y+{round(b1, 5)}')
print(f'z = {round(a2, 5)}*y+{round(b2, 5)}')

plt.figure()
plt.plot(1/y, np.exp(x))
plt.plot([1750, 1750], [0, 1800], 'k--')
plt.scatter(1750, 1720, c='r')
plt.scatter(234, 0.165, c='b')
plt.scatter(train_temperature, train_pressure)
plt.text(800, 800, '$Liquid$')
plt.text(1500, 400, '$Vapor$')
plt.text(1800, 1750, '$Supercritical$ $fluid$')
plt.xlabel('$T (K)$')
plt.ylabel('$P (bar)$')
plt.show()

# plotter = Plotter()
# plotter.scatter3(pressure, temperature, rho, xlabel=r'$P (bar)$', ylabel=r'$T (K)$', zlabel=r'$\rho (kg/m^3)$')
# plotter.scatter3(pressure1, temperature1, rho1, xlabel=r'$P (Bar)$', ylabel=r'$T (K)$', zlabel=r'$\rho (rg/m^3)$')
# plotter.plot3(np.exp(x), 1/y, np.exp(z))
# plotter.show()
