import os
import numpy as np


distance_matrix = np.load('distance_matrix.npy')

eta_charge = 0.90

kwh_per_mi = 141.6122004 / 1000 * 1.609344 * eta_charge
energy_matrix = distance_matrix * kwh_per_mi
np.save('energies_2023_Dacia_Spring_Essential.npy', energy_matrix)

