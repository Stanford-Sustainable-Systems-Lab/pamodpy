import os
import numpy as np
import matplotlib.pyplot as plt

# PHYSICAL CONSTANTS
rho_a = 1.225  # [kg/m3]
g = 9.80665  # [m/s2]
DAYS_PER_YEAR = 365.2422
HOURS_PER_YEAR = 8765.8127
HOURS_PER_MONTH = HOURS_PER_YEAR / 12

# FILE PATHS
PATH_TO_DATA = os.path.join(os.path.dirname(__file__), '..', 'data')
PATH_TO_CAR_PARAMS = os.path.join(PATH_TO_DATA, 'Vehicle_parameters.csv')

# DATA
def generate_p_elec(T, create_plot=False):
    # https://www.pge.com/tariffs/assets/pdf/tariffbook/ELEC_SCHEDS_BEV.pdf
    p_elec = 0.18603 * np.ones(T)                                          # Off-peak
    p_elec[int(np.round(16/24 * T)): int(np.round(21/24 * T))] = 0.39926  # Peak [$/kWh]
    p_elec[int(np.round(9/24 * T)):int(np.round(14/24 * T))] = 0.16276    # Super-off-peak
    if create_plot:
        plt.figure(dpi=200, figsize=(12, 8))
        plt.step(np.arange(0, 24, T/24), p_elec, where='post')
        plt.xlabel('Time of Day [hr]', fontsize=24)
        plt.ylabel('Electricity Price [$ / kWh]', fontsize=24)
        # plt.title('PG&E Time of Use Commercial EV Rate')
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.grid()
        plt.ylim(bottom=0)
        plt.show()
    return p_elec

def generate_carbon_intensity_grid(T, create_plot=False):
    # https://ww2.arb.ca.gov/sites/default/files/classic/fuels/lcfs/fuelpathways/comments/tier2/2023_elec_update.pdf
    carbon_intensity_grid_hourly = np.load(os.path.join(PATH_TO_DATA, 'CA_carbon_intensity_hourly_yearavg.npy')) / 1e6 * 3.6  # [ton CO2 / kWh], 0 = Q1, 3 = Q4
    carbon_intensity_grid = np.ones(T)
    for hour, value in enumerate(carbon_intensity_grid_hourly):
        carbon_intensity_grid[int(np.round(hour/24 * T)):int(np.round((hour+1)/24 * T))] = value
    if create_plot:
        plt.figure(dpi=200, figsize=(12, 8))
        plt.step(np.arange(0, 24, T/24), carbon_intensity_grid, where='post')
        plt.xlabel('Time of Day [hr]', fontsize=24)
        plt.ylabel('Carbon Intensity California Grid [ton CO2 / kWh]', fontsize=24)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.grid()
        plt.ylim(bottom=0)
        plt.show()
    return carbon_intensity_grid

p_demand_month = np.round(95.56 / 50, decimals=5)                                               # [$/kW]
INTEREST_RATE = 0.07
EVSE_LIFESPAN = 10                                                      # [yrs]

TONS_CO2_PER_GAL_GAS = 0.008887
KWH_PER_GAL_GAS = 33.7
