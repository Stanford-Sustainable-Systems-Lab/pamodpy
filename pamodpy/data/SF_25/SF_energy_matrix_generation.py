import os
import numpy as np
import matplotlib.pyplot as plt


distance_matrix = np.load('distance_matrix.npy')

eta_charge = 0.90

kwh_per_mi = 148 / 1000 * 1.609344 * eta_charge
energy_matrix = distance_matrix * kwh_per_mi
np.save('energies_2023_Tesla_Model_3_Long_Range_AWD.npy', energy_matrix)

# plt.figure()
# plt.hist(energy_matrix[energy_matrix > 0].flatten(), bins=20)
# plt.show()

print("20th percentile = {} kWh".format(np.percentile(energy_matrix[energy_matrix > 0].flatten(), 20)))


# ioniqhyb_gal_per_mi = 1 / 62.8
# ioniqhyb_energy_matrix = distance_matrix * ioniqhyb_gal_per_mi
# np.save('energies_2021_Hyundai_IONIQ_Hybrid_SE.npy', ioniqhyb_energy_matrix)