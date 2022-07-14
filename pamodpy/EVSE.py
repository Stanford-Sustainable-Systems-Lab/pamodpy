import pandas as pd
import os

class EVSE():
    def __init__(self, name, num_units=0, station=None):
        df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'EVSE_parameters.csv'))
        if name not in df['name'].values:
            print('Warning: station with name "{}" not found.'.format(name) +
                  '\nHere is a list of available names from "EVSE_parameters.csv":\n{}'.format(df['name']) +
                  '\nDefaulting to "DC 22.5kW"')
            name = "DC 22.5kW"
        idx = df.index[df['name'] == name]
        self.name = name
        self.num_units = num_units
        self.station = station
        self.manufacturer = df.loc[idx]['manufacturer'].values[0]
        self.model = df.loc[idx]['model'].values[0]
        self.year = df.loc[idx]['year'].values[0]
        self.rate_AC = df.loc[idx]['rate_AC'].values[0]                 # [kW]
        self.rate_DC = df.loc[idx]['rate_DC'].values[0]                 # [kW]
        self.rate = max(self.rate_AC, self.rate_DC)
        self.evse_unit_cost = df.loc[idx]['evse_unit_cost'].values[0]   # [$]
        self.p_infra_marginal = self.evse_unit_cost + (84 + 240)/2 + (200 + 250)/2
        self.p_infra_capital = 0.1 * (df.loc[idx]['capital_10unit_cost'].values[0] + (1500+3500)/2 + (325+1000)/2)
        if self.station is not None:
            self.evse_id = str(self.station.name) + '_' + str(self.name)
        else:
            self.evse_id = None