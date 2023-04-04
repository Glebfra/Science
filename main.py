from pathlib import Path

import dotenv
import numpy as np

from src.data.FileService import FileService
from src.plotter.Plotter import Plotter

dotenv.load_dotenv('.env')
PROJECT_DIR = Path(__file__).resolve().parent
dotenv.set_key(f'{PROJECT_DIR}/.env', 'PROJECT_DIR', str(PROJECT_DIR))

file_service = FileService()

datas = file_service.load_file(filename='ртуть.json', type='json')

temperature, pressure, density = [], [], []
for data in datas:
    temperature.append((data['T'] + 273))
    pressure.append(data['p'])
    density.append(1 / data['V'])
temperature, pressure, density = np.array(temperature), np.array(pressure), np.array(density)

plotter = Plotter(use_latex=True, font_family='Poppins')
plotter.scatter3(temperature, pressure, density, xlabel=r'$T$ (K)', ylabel=r'$P$ (bar)', zlabel=r'$\rho$ (kg/$m^3$)')
plotter.show()
