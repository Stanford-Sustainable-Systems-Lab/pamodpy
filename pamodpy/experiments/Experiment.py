"""
MIT License

Copyright (c) 2022 Justin Luke

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""
import os
import itertools
import pickle
from copy import deepcopy
from abc import ABC, abstractmethod

import numpy as np

from ..Vehicle import Vehicle
from ..utils.constants import *
from ..utils.load_data import *
from ..EVSE import EVSE


class Experiment(ABC):
    """
    Abstract base class for an experiment configuration.
    """
    def __init__(self, config):
        self.config = config
        self.name = config['name']
        self.region = config['region']

        # Key Parameters
        self.locations = None
        self.locations_excl_passthrough = None
        self.deltaT = config['deltaT']  # time step duration [hr]
        self.T = int(np.round(config['num_hours'] / self.deltaT))  # total number of time steps
        self.startT = int(np.round(config['start_hour'] / self.deltaT))  # start time in time steps
        self.endT = self.startT + self.T  # end time in time steps
        self.deltaC = config['deltaC']  # energy step [kWh]
        self.batt_cap_range = config['batt_cap_range']  # 1.0 - 0.2
        self.charge_throttle = False
        self.Vehicles = [Vehicle(vehicle_name) for vehicle_name in config['Vehicles']]  # 2022 Dacia Spring Comfort, 2022 Nissan Leaf S, 2022 Chevrolet Bolt EV 1LT,
        # 2021 Hyundai IONIQ Electric SE, 2021 Hyundai IONIQ Hybrid SE # list of Vehicle models used in fleet
        self.energy_ODs = None  # List of numpy array of OD matrix with trip energies in [kWh] for each vehicle in self.Vehicles
        self.fleet_sizes = config['fleet_sizes']  # max(np.sum(self.od_matrix, axis=(0, 1)))         # List of number of vehicles of each Vehicle model in fleet, corresponding to self.Vehicles for each vehicle in self.Vehicles

        # Data
        self.data_path = None
        self.shp_file_path = None
        self.streetlight_df = None
        self.results_path = None                      # directory where results are saved
        self.time_matrix = None  # (L, L, 24) Numpy array of OD matrix with trip durations in [s]
        self.dist_matrix = None  # (L, L, 24) Numpy array of OD matrix with trip distances in [mi]
        self.od_matrix = None        # (L, L, 24) Numpy array of OD matrix with travel volume [# vehicles]
        self.top_idx = None                      # Numpy array of indices (not TAZ) of non-zero roads in matched_od_matrix_top
        self.EVSEs = [EVSE(evse_name) for evse_name in config['EVSEs']]  # 7.7, 20, 50.0, 150.0
        self.charge_rate = np.sort(
            np.unique(np.array([evse.rate for evse in self.EVSEs])))  # list of available charging rates [kW]

        # Optimization Settings
        self.save_opt = True
        self.load_opt = False
        self.boundary = True  # constraint requiring fleet distribution and SOCs across all location at t=-1 be same as t=0
        self.optimize_fleet_size = config['optimize_fleet_size']                                             # Whether to optimize for fleet size or constrain it
        self.optimize_infra = config['optimize_infra']                                                 # Whether to optimize charging infrastructure placement
        self.optimize_infra_mip = False                                             # True: integer variable for capex or step costs; False: approximate costs as linear
        self.congestion_constr_road = False                                         # Whether to have congestion threshold limits on road paths
        self.congestion_constr_charge = True                                       # Whether to have congestion threshold limits at charging stations
        self.drop_trips = config['drop_trips']
        self.use_baseline_charge_stations = config['use_baseline_charge_stations']

        # Costs and prices
        self.revenue_matrix = None  # (L, L, 24) Numpy array of OD matrix with trip revenue in [$]
        self.p_elec_demand = p_demand_month
        self.p_travel = config['p_travel']  # 0.0770 [$ / mi] 0.30 * 1.60934 https://newsroom.aaa.com/wp-content/uploads/2021/08/2021-YDC-Brochure-Live.pdf # travel cost (maintenance)
        self.p_ownership_excl_deprec = config['p_ownership_excl_deprec']  # 1381 + 155  # [$ / yr] https://newsroom.aaa.com/wp-content/uploads/2021/08/2021-YDC-Brochure-Live.pdf # insurance, fees. Finance charges ($692) also excluded
        self.p_travel_ICE = config['p_travel_ICE']  #0.0878 # [$ / mi] https://newsroom.aaa.com/wp-content/uploads/2021/08/2021-YDC-Brochure-Live.pdf # travel cost (maintenance)
        self.p_gas = config['p_gas']  #4.127 # [$ / gal] https://www.eia.gov/dnav/pet/pet_pri_gnd_dcus_y05sf_a.htm
        self.p_carbon = config['p_carbon']  #0 # [$ / ton CO2]

        # Generated variables
        self.p_elec = generate_p_elec(int(np.round(24 / self.deltaT)))
        self.logger = None

    @abstractmethod
    def build(self):
        """
        Generate (or regenerate) instance variables that are a function of other instance variables.
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def run(self):
        """
        Run the experiment.
        :return: No return.
        """
        raise NotImplementedError

    @abstractmethod
    def save(self):
        """
        Save the experiment run's resulting output.
        :return: No return.
        """
        raise NotImplementedError

    @abstractmethod
    def plot(self):
        raise NotImplementedError


class SF_190(Experiment):
    def __init__(self, config):
        super().__init__(config)
        self.region = "SF_190"
        self.data_path = os.path.join(os.path.dirname(__file__), '..', 'data', self.region)
        self.shp_file_path = os.path.join(self.data_path,
                     'Justin-Luke---Academic_SF-TAZ-with-added-boundary-pass-through_ZoneSet',
                     'zone_set_SF_TAZ_with_added_boundary_pass_through.shp')
        if config['results_path'] is not None and os.path.exists(os.path.normpath(config['results_path'])):
            self.results_path = os.path.join(os.path.normpath(config['results_path']), self.region, self.name)
        else:
            self.results_path = os.path.join(os.path.dirname(__file__), '..', '..', 'results', self.region, self.name)
        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path, exist_ok=True)
        print("Results will be saved in {}".format(self.results_path))
        self.streetlight_df = load_streetlight(os.path.join(self.data_path,
                                                            "92009_SF_all_weekday_each_hour_added_pass_od_trip_all.csv"))
        self.locations = np.sort(self.streetlight_df['Origin Zone Name'].unique()).tolist()
        self.locations_excl_passthrough = np.array(self.locations)[
            (np.array(self.locations) != 191) & (np.array(self.locations) != 192) & (np.array(self.locations) != 193)].tolist()
        self.energy_ODs = [np.load(
            os.path.join(self.data_path,
                         self.Vehicles[vehicle_idx].energies_filename)) for vehicle_idx in range(len(self.Vehicles))]  # Numpy array of OD matrix with trip energies in [kWh]
        self.time_matrix = np.nan_to_num(np.load(os.path.join(self.data_path,
                                                              'duration_matrix (1).npy')))  # (193, 193, 24) Numpy array of OD matrix with trip durations in [s]
        self.dist_matrix = np.nan_to_num(np.load(os.path.join(self.data_path,
                                                              'distance_matrix (1).npy')))  # (193, 193, 24) Numpy array of OD matrix with trip distances in [mi]
        self.od_matrix = np.load(os.path.join(self.data_path,
                                              'matched_od_matrix_top.npy'))  # (193, 193, 24) Numpy array of OD matrix with travel volume [# vehicles]
        for x in itertools.product([self.locations.index(191), self.locations.index(192), self.locations.index(193)], [self.locations.index(191), self.locations.index(192), self.locations.index(193)]):
            self.od_matrix[x[0], x[1], :] = 0
        self.top_idx = np.load(os.path.join(self.data_path,
                                            'top_idx.npy'))  # (2, 1663) Numpy array of indices (not TAZ) of non-zero roads in matched_od_matrix_top
        self.revenue_matrix = self.dist_matrix * 0.91 + self.time_matrix / 60 * 0.39 + 2.20 + 2.70  # (193, 193, 24) Numpy array of OD matrix with trip revenue in [$]

class SF_5(Experiment):
    def __init__(self, config):
        super().__init__(config)
        self.region = "SF_5"
        self.data_path = os.path.join(os.path.dirname(__file__), '..', 'data', self.region)
        self.shp_file_path = os.path.join(self.data_path,
                     'Justin-Luke---Academic_SF_spectral_clusters_with_passthroughs_ZoneSet',
                     'zone_set_SF_spectral_clusters_with_passthroughs.shp')
        if config['results_path'] is not None and os.path.exists(os.path.normpath(config['results_path'])):
            self.results_path = os.path.join(os.path.normpath(config['results_path']), self.region, self.name)
        else:
            self.results_path = os.path.join(os.path.dirname(__file__), '..', '..', 'results', self.region, self.name)
        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path, exist_ok=True)
        print("Results will be saved in {}".format(self.results_path))
        self.streetlight_df = load_streetlight(os.path.join(self.data_path,
                                                            "143230_SF_5zones_weekday_hourly_passthroughs_od_trip_all.csv"))
        self.locations = np.arange(1, 9).tolist()
        self.locations_excl_passthrough = np.array(self.locations)[
            (np.array(self.locations) != 6) & (np.array(self.locations) != 7) & (np.array(self.locations) != 8)].tolist()
        self.energy_ODs = [np.load(
            os.path.join(self.data_path,
                         self.Vehicles[vehicle_idx].energies_filename)) for vehicle_idx in range(len(self.Vehicles))] # Numpy array of OD matrix with trip energies in [kWh]
        self.time_matrix = np.nan_to_num(np.load(os.path.join(self.data_path,
                                                              'duration_matrix.npy')))  # (8, 8, 24) Numpy array of OD matrix with trip durations in [s]
        self.dist_matrix = np.nan_to_num(np.load(os.path.join(self.data_path,
                                                              'distance_matrix.npy')))  # (8, 8, 24) Numpy array of OD matrix with trip distances in [mi]
        self.od_matrix = np.load(os.path.join(self.data_path,
                                              'od_matrix.npy'))  # (8, 8, 24) Numpy array of OD matrix with travel volume [# vehicles]
        for x in itertools.product([self.locations.index(6), self.locations.index(7), self.locations.index(8)], [self.locations.index(6), self.locations.index(7), self.locations.index(8)]):
            self.od_matrix[x[0], x[1], :] = 0
        self.revenue_matrix = self.dist_matrix * 0.91 + self.time_matrix / 60 * 0.39 + 2.20 + 2.70  # (8, 8, 24) Numpy array of OD matrix with trip revenue in [$]

class SF_25(Experiment):
    def __init__(self, config):
        super().__init__(config)
        self.region = "SF_25"
        self.data_path = os.path.join(os.path.dirname(__file__), '..', 'data', self.region)
        self.shp_file_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'SF_190',
                     'Justin-Luke---Academic_SF-TAZ-with-added-boundary-pass-through_ZoneSet',
                     'zone_set_SF_TAZ_with_added_boundary_pass_through.shp')
        if config['results_path'] is not None and os.path.exists(os.path.normpath(config['results_path'])):
            self.results_path = os.path.join(os.path.normpath(config['results_path']), self.region, self.name)
        else:
            self.results_path = os.path.join(os.path.dirname(__file__), '..', '..', 'results', self.region, self.name)
        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path, exist_ok=True)
        print("Results will be saved in {}".format(self.results_path))
        self.streetlight_df = None
        self.locations = np.arange(1, 29).tolist()
        self.locations_excl_passthrough = np.array(self.locations)[
            (np.array(self.locations) != 26) & (np.array(self.locations) != 27) & (np.array(self.locations) != 28)].tolist()
        self.energy_ODs = [np.load(
            os.path.join(self.data_path,
                         self.Vehicles[vehicle_idx].energies_filename)) for vehicle_idx in range(len(self.Vehicles))]  # Numpy array of OD matrix with trip energies in [kWh]
        self.time_matrix = np.nan_to_num(np.load(os.path.join(self.data_path,
                                                              'duration_matrix.npy')))  # (28, 28, 24) Numpy array of OD matrix with trip durations in [s]
        self.dist_matrix = np.nan_to_num(np.load(os.path.join(self.data_path,
                                                              'distance_matrix.npy')))  # (28, 28, 24) Numpy array of OD matrix with trip distances in [mi]
        self.od_matrix = np.load(os.path.join(self.data_path,
                                              'od_matrix.npy'))  # (28, 28, 24) Numpy array of OD matrix with travel volume [# vehicles]
        for x in itertools.product([self.locations.index(26), self.locations.index(27), self.locations.index(28)], [self.locations.index(26), self.locations.index(27), self.locations.index(28)]):
            self.od_matrix[x[0], x[1], :] = 0
        self.revenue_matrix = self.dist_matrix * 0.91 + self.time_matrix / 60 * 0.39 + 2.20 + 2.70  # (28, 28, 24) Numpy array of OD matrix with trip revenue in [$]

        if self.use_baseline_charge_stations:
            with open(os.path.join(self.data_path, 'SF_charging_stations_to_25_cluster.p'), 'rb') as f:
                self.charge_stations = pickle.load(f)
            desired_total_installed_capacity = 358257.08870379557#374898.8370658811 #629912.7705068741
            current_total_installed_capacity = 0.0
            for l in self.locations_excl_passthrough:
                for station in self.charge_stations[l]:
                    for evse in station.EVSEs:
                        current_total_installed_capacity += evse.rate * evse.num_units
            print("Current Installed Capacity = {} kW".format(current_total_installed_capacity))
            scaling_factor = desired_total_installed_capacity / current_total_installed_capacity
            print("Scaling Factor = {}".format(scaling_factor))
            for l in self.charge_stations:
                for station in self.charge_stations[l]:
                    for evse in station.EVSEs:
                        evse.num_units *= scaling_factor

    def convert_25_to_190(self, data):
        self.cluster_to_taz = {
            1: [56, 57],
            2: [52, 62, 65, 66],
            3: [43, 48, 49, 50, 51, 70, 71, 72],
            4: [6, 7, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 26, 39, 40, 41, 44, 45, 46, 47, 73, 74, 75],
            5: [1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17, 22, 23, 24, 25, 26, 37, 38, 42],
            6: [54, 55, 58, 59, 60],
            7: [53, 61, 63, 64, 67, 90],
            8: [68, 69, 83, 84, 85, 86, 87, 88, 89],
            9: [8, 9, 10, 11, 20, 76, 77, 78, 79, 80, 81, 82, 104, 105, 106, 107],
            10: [18, 19, 21, 108, 109, 110],
            11: [178, 179, 180, 181, 184],
            12: [172, 173, 174, 175, 176, 177, 185],
            13: [91, 92, 93, 94, 95, 96, 129, 171],
            14: [97, 98, 99, 100, 101, 102, 103, 116, 117, 118, 119, 122, 123, 128],
            15: [111, 112, 113, 114, 115, 120, 121, 142],
            16: [182, 183, 186, 187],
            17: [169, 170, 188],
            18: [130, 131, 132, 133, 134],
            19: [124, 125, 126, 127, 135, 136, 137, 138, 152],
            20: [139, 140, 141, 143, 144, 145, 146, 147, 150],
            21: [190],
            22: [168, 189],
            23: [161, 162, 163, 164, 165, 166, 167],
            24: [155, 156, 157, 158, 159, 160],
            25: [148, 149, 151, 153, 154]
        }

        output = np.zeros(190 + 3)
        for idx, cluster in enumerate(range(1, 26)):
            output[self.cluster_to_taz[cluster]] = data[idx]
        output[190:] = data[25:]
        return output
