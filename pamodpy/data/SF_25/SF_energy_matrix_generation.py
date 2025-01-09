import os
import numpy as np
import matplotlib.pyplot as plt


distance_matrix = np.load('distance_matrix.npy')

eta_charge = 0.90

kwh_per_mi = 135 / 1000 * 1.609344 * eta_charge
print(eta_charge / kwh_per_mi)
energy_matrix = distance_matrix * kwh_per_mi
# np.save('energies_2024_Dacia_Spring_Electric_65_Expression.npy', energy_matrix)

# ioniqhyb_gal_per_mi = 1 / 52.3
# ioniqhyb_energy_matrix = distance_matrix * ioniqhyb_gal_per_mi
# np.save('energies_2021_Hyundai_IONIQ_Hybrid_SE.npy', ioniqhyb_energy_matrix)

od_matrix = np.load('od_matrix.npy')
od_matrix[25:, 25:, :] = 0
duration_matrix = np.load('duration_matrix.npy') / (60 * 60)
duration_matrix2 = duration_matrix.copy()
distance_matrix2 = distance_matrix.copy()

distance_matrix2[25, :, :] += 13
distance_matrix2[:, 25, :] += 13
duration_matrix2[25, :, :] += 20 / 60
duration_matrix2[:, 25, :] += 20 / 60

distance_matrix2[26, :, :] += 10
distance_matrix2[:, 26, :] += 10
duration_matrix2[26, :, :] += 25 / 60
duration_matrix2[:, 26, :] += 25 / 60

distance_matrix2[27, :, :] += 23
distance_matrix2[:, 27, :] += 23
duration_matrix2[27, :, :] += 15 / 60
duration_matrix2[:, 27, :] += 15 / 60

energy_matrix2 = distance_matrix2 * kwh_per_mi + duration_matrix2 * 0
print("20th percentile = {} kWh".format(np.percentile(np.repeat(energy_matrix2[energy_matrix2 > 0].flatten(), np.round(od_matrix[energy_matrix2 > 0].flatten()).astype(int)), 20)))
