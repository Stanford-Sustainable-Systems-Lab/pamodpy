import os
import pandas as pd

from .utils.constants import PATH_TO_CAR_PARAMS

class Vehicle():
    def __init__(self, name):
        df = pd.read_csv(PATH_TO_CAR_PARAMS)
        if name not in df['name'].values:
            print('Warning: vehicle with name "{}" not found.'.format(name) +
                  '\nHere is a list of available names from "Vehicle_parameters.csv":\n{}'.format(df['name']) +
                  '\nDefaulting to "2022 Dacia Spring Comfort"')
            name = "2022 Dacia Spring Comfort"
        idx = df.index[df['name'] == name]
        self.name = name
        self.make = df.loc[idx]['make'].values[0]
        self.model = df.loc[idx]['model'].values[0]
        self.trim = df.loc[idx]['trim'].values[0]
        self.year = df.loc[idx]['year'].values[0]
        self.m = df.loc[idx]['m'].values[0]                     # [kg]
        self.powertrain = df.loc[idx]['powertrain'].values[0]   # {'electric', 'hybrid', 'conventional'}
        self.Af = df.loc[idx]['Af'].values[0]                   # [m2]
        self.Cd = df.loc[idx]['Cd'].values[0]                   # [1]
        self.Cr = df.loc[idx]['Cr'].values[0]                   # [1]
        self.batt_cap = df.loc[idx]['batt_cap'].values[0]       # [kWh]
        self.max_charge_rate_AC = float(df.loc[idx]['max_charge_rate_AC'].values[0])  # [kW]
        self.max_charge_rate_DC = float(df.loc[idx]['max_charge_rate_DC'].values[0])  # [kW]
        self.mi_per_kWh = df.loc[idx]['mi_per_kWh'].values[0]   # [mi/kWh]
        self.mi_per_gal = float(df.loc[idx]['mi_per_gal'].values[0]) # [mi/gal]
        self.price = df.loc[idx]['price'].values[0]             # [$]
        self.energies_filename = df.loc[idx]['energies_filename'].values[0]

        self.eta_regen = 0.6
        self.eta_discharge = 0.95
        self.eta_charge = 0.90
        self.aux_power = 4                                   # [kW]
