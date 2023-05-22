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
import itertools
import time
import pickle
import logging
import sys

import networkx as nx

from .Experiment import Experiment, SF_5, SF_25, SF_190, NYC_manh
from ..utils.load_experiment import config
from ..utils.constants import *
from ..plotting.plot_results import plot_PAMoDFleet_results, plot_animiation
from ..algorithms.PAMoD_optimization_gurobi import PAMoD_optimization_gurobi
from ..algorithms.PAMoD_optimization_pyomo import PAMoD_optimization_pyomo
from ..Station import Station
from ..EVSE import EVSE

class MetaPAMoDFleet(type):
    def __new__(cls, name, bases, dct):
        if config.current_experiment_region == "SF_5":
            bases = (SF_5,)
        elif config.current_experiment_region == "SF_25":
            bases = (SF_25,)
        elif config.current_experiment_region == "SF_190":
            bases = (SF_190,)
        elif config.current_experiment_region == "NYC_manh":
            bases = (NYC_manh,)
        else:
            raise ValueError('No Experiment of class name "{}" found in Experiment.py.'.format(config.current_experiment_region))
        return type(name, bases, dct)

class PAMoDFleet(metaclass=MetaPAMoDFleet):
    def __init__(self, experiment_config):
        # Experiment Parameters
        super().__init__(experiment_config)

        self.results_path = os.path.join(self.results_path, 'PAMoDFleet')
        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path, exist_ok=True)
        self.UMax_charge = np.zeros((len(self.locations_excl_passthrough), len(self.EVSEs))) # (L, P) max num vehicles charging for charge station at location L and rating P
        # https://rmi.org/wp-content/uploads/2020/01/RMI-EV-Charging-Infrastructure-Costs.pdf
        self.p_infra_capital = np.zeros((len(self.locations_excl_passthrough), len(self.EVSEs))) # (L, P) capital cost for charge station at location L and rating P
        self.p_infra_marginal = np.zeros((len(self.locations_excl_passthrough), len(self.EVSEs)))  # (L, P) capital cost for charge station at location L and rating P

        if not self.use_baseline_charge_stations:
            self.charge_stations = {}  # dictionary of charge station configuration, with keys being locations and values being a list with one Station
            for lep_idx, l in enumerate(self.locations_excl_passthrough):
                station = Station(l, l)
                self.charge_stations[l] = [station]
                for evse_idx, evse in enumerate(self.EVSEs):
                    UMax_charge_l_eidx = 0.4 * sum(self.fleet_sizes) # TODO have actual charge station spots limit
                    station.EVSEs.append(EVSE(evse.name, UMax_charge_l_eidx, station))
                    self.UMax_charge[lep_idx, evse_idx] = UMax_charge_l_eidx
                    self.p_infra_capital[lep_idx, evse_idx] = evse.p_infra_capital * (INTEREST_RATE / (1 - (1 + INTEREST_RATE) ** -EVSE_LIFESPAN)) * (
                    self.T * self.deltaT / HOURS_PER_YEAR)
                    self.p_infra_marginal[lep_idx, evse_idx] = evse.p_infra_marginal * (INTEREST_RATE / (1 - (1 + INTEREST_RATE) ** -EVSE_LIFESPAN)) * (
                    self.T * self.deltaT / HOURS_PER_YEAR)
                    if not self.optimize_infra_mip:
                        self.p_infra_marginal[lep_idx, evse_idx] += self.p_infra_capital[lep_idx, evse_idx]
                    else:
                        pass  # TODO should have higher capital cost
        else:
            for lep_idx, l in enumerate(self.locations_excl_passthrough):
                    for station in self.charge_stations[l]:
                        for evse in station.EVSEs:
                            evse_idx = [e.name for e in self.EVSEs].index(evse.name)
                            self.UMax_charge[lep_idx, evse_idx] = evse.num_units
                            self.p_infra_capital[lep_idx, evse_idx] = evse.p_infra_capital * (INTEREST_RATE / (1 - (1 + INTEREST_RATE) ** -EVSE_LIFESPAN)) * (
                            self.T * self.deltaT / HOURS_PER_YEAR)
                            self.p_infra_marginal[lep_idx, evse_idx] = evse.p_infra_marginal * (INTEREST_RATE / (1 - (1 + INTEREST_RATE) ** -EVSE_LIFESPAN)) * (
                            self.T * self.deltaT / HOURS_PER_YEAR)
        self.p_infra_capital = self.p_infra_capital.round(2)
        self.p_infra_marginal = self.p_infra_marginal.round(2)

        # # Generated Variables (from above settings)
        # self.deltaC = None                  # energy step [kWh]
        self.L = None                       # number of physical locations (TAZs)
        self.PAMoDVehicles = []    # List of PAMoDVehicle for each Vehicle in self.Vehicles
        self.build()

    class PAMoDVehicle():
        def __init__(self, Fleet, Vehicle, energy_OD, fleet_size):
            self.Fleet = Fleet                  # outer class, PAMoDFleet()
            self.Vehicle = Vehicle                      # corresponding Vehicle
            self.energy_OD = energy_OD          # corresponding energy_OD numpy array
            self.fleet_size = fleet_size        # corresponding fleet_size

            self.G = None                       # NetworkX graph of the extended P-AMoD graph (locations, SOC level, time)
            self.C = None                       # number of charge levels
            self.N = None                       # number of nodes
            self.E = None                       # number of edges, both road and charge edges, in the extended graph G
            self.road_arcs = None               # set of (O, D) tuples that have a road arc connecting them in the graph
            self.A = None                       # incidence matrix of extended graph G with +1 flow in, -1 flow out
            self.A_outflows = None              # incidence matrix of extended graph G with only -1 flow out
            self.A_inflows = None               # incidence matrix of extended graph G with only 1 flow in
            self.Dist = None                    # (E, ) trip distances each edge [mi]
            self.Demand = None                  # (E, ) travel demand each edge [# vehicles]
            self.Dur = None                     # (E, ) trip or charge durations each edge [hr]
            self.UMax_road = None               # (E, ) max vehicle flow for each edge
            self.E_road_idx = None              # list of edge indices corresponding to road edges
            self.E_charge_idx = None            # list of edge indices corresponding to charge edges
            self.energy_conv = None             # (E, ) conversion factor for each edge from num vehicles to kWh (or gal gasoline if non-electric)
            self.power_conv = None              # (E, ) conversion factor for each charge edge from num vehicles to kW

            self.G_edges_O_arr = None           # (E, ) np.array of edge origins in G.edges()
            self.G_edges_D_arr = None           # (E, ) np.array of edge destinations in G.edges()
            self.G_edges_t_arr = None           # (E, ) np.array of origin times in G.edges()
            self.G_edges_road_mask = None       # (E, ) boolean np.array indicating road edge in G.edges()
            self.G_edges_charge_mask = None     # (E, ) boolean np.array indicating charge edge in G.edges()
            self.G_edges_evse_id_arr = None     # (E, ) np.array of evse_ids for charge edges in G.edges()
            self.G_edges_road_idle_mask = None  # (E, ) boolean np.array indicating idle road edge in G.edges()
            self.G_nodes_arr = None             # (N, ) np.array of nodes from G.nodes()

            self.num_dropped_trips = 0
            self.num_incl_trips = 0

            # Results
            self.X = None
            self.U = None
            self.U_trip_charge_idle = None
            self.U_rebal = None
            self.costs_list = None
            self.power_matrix = None

        def set_incidence_matrices(self):
            self.A = (nx.incidence_matrix(self.G, oriented=True))
            self.A_outflows = (-1 * (self.A == -1).astype(np.float64))
            self.A_inflows = ((self.A == 1).astype(np.float64))

        def remove_incidence_matrices(self):
            self.A = None
            self.A_outflows = None
            self.A_inflows = None

        def add_road_edges(self, O, D, t, uMax_road, Car):
            O_idx = self.Fleet.locations.index(O)
            D_idx = self.Fleet.locations.index(D)
            if np.all(self.Fleet.od_matrix[O_idx, D_idx] == 0):  # TODO: use location index
                return False

            dur = self.Fleet.time_matrix[O_idx, D_idx, int(np.floor((t * self.Fleet.deltaT) % (24 / self.Fleet.deltaT)))] / (60 * 60)
            dist = self.Fleet.dist_matrix[O_idx, D_idx, int(np.floor((t * self.Fleet.deltaT) % (24 / self.Fleet.deltaT)))]
            energy = 0  # TODO: for (O,D) with pass_through, it should include energy needed to get to the boundary

            last_area_zone = len(self.Fleet.locations_excl_passthrough)
            golden_gate = last_area_zone + 1
            bay_bridge = last_area_zone + 2
            south = last_area_zone + 3

            if O <= last_area_zone and D <= last_area_zone:
                energy += self.energy_OD[O_idx, D_idx, int(np.floor((t * self.Fleet.deltaT) % (24 / self.Fleet.deltaT)))] + Car.compute_power * dur  # TODO: fix this relic (first two rows empty, arbitrarily)

            # if O == golden_gate or D == golden_gate:
            #     dur += 40 / 60
            #     dist += 20
            #     if Car.powertrain == 'electric':
            #         energy += 20 / (Car.mi_per_kWh / Car.eta_charge) + Car.compute_power * dur
            #     else:
            #         energy += 20 / Car.mi_per_gal + Car.compute_power * dur / KWH_PER_GAL_GAS
            # if O == bay_bridge or D == bay_bridge:
            #     dur += 20 / 60
            #     dist += 15
            #     if Car.powertrain == 'electric':
            #         energy += 15 / (Car.mi_per_kWh / Car.eta_charge) + Car.compute_power * dur
            #     else:
            #         energy += 15 / Car.mi_per_gal + Car.compute_power * dur / KWH_PER_GAL_GAS
            # if O == south or D == south:
            #     dur += 45 / 60
            #     dist += 30
            #     if Car.powertrain == 'electric':
            #         energy += 30 / (Car.mi_per_kWh / Car.eta_charge) + Car.compute_power * dur
            #     else:
            #         energy += 30 / Car.mi_per_gal + Car.compute_power * dur / KWH_PER_GAL_GAS

            dur_deltaTs = self.Fleet.round_time(dur, min_val=1)
            if Car.powertrain == 'electric':
                energy_deltaCs = self.Fleet.round_energy(energy, min_val=1)
            else:
                energy_deltaCs = 0

            if t + dur_deltaTs > self.Fleet.endT - 1:  # not enough time to make the trip
                return False

            hour = int(np.floor((t * self.Fleet.deltaT) % (24 / self.Fleet.deltaT)))
            demand = self.Fleet.od_matrix[O_idx, D_idx, hour] * self.Fleet.deltaT

            if energy_deltaCs > self.C:
                self.Fleet.logger.error(
                    "Not enough battery capacity for O={}, D={}, t={}, energy={}. Trips dropped={}".format(O, D, t,
                                                                                                           energy,
                                                                                                           demand))
                self.num_dropped_trips += demand
                return False
            elif energy_deltaCs > self.C / 2 and (D == golden_gate or D == bay_bridge or D == south):  # TODO
                self.Fleet.logger.error(
                    "Not enough battery capacity for O={}, D={}, t={}, energy={}. Trips dropped={}".format(O, D, t,
                                                                                                           energy,
                                                                                                           demand))
                self.num_dropped_trips += demand
                return False

            for c in reversed(range(self.C)):
                if c - energy_deltaCs < 0:  # not enough energy; trip infeasible, do not create an edge
                    break
                self.G.add_edge((O, c, t), (D, min(c - energy_deltaCs, self.C - 1), t + dur_deltaTs), dur=dur,
                                energy=energy,
                                dist=dist, uMax_road=uMax_road, demand=demand)

            self.num_incl_trips += demand
            return True

        def add_idle_road_edges(self, l, uMax_road):
            for t in range(self.Fleet.startT, self.Fleet.endT - 1):
                for c in range(self.C):
                    self.G.add_edge((l, c, t), (l, c, t + 1), dur=self.Fleet.deltaT, energy=0, dist=0, uMax_road=uMax_road,
                                    idle=True)

        def add_charge_edges(self, l, rate, Car, throttle=True, evse_id=None):
            dur_deltaTs = 1
            dur = self.Fleet.deltaT * dur_deltaTs
            energy_deltaCs = self.Fleet.round_energy(rate * dur * Car.eta_charge)
            energy = self.Fleet.deltaC * energy_deltaCs
            if energy_deltaCs < 1:
                self.Fleet.logger.error(
                    "With C={}, rate={} is too low to charge one level during dur_deltaTs={} and deltaTs={}".format(
                        self.C, rate, dur_deltaTs, self.Fleet.deltaT))
                return
            for t in range(self.Fleet.startT, self.Fleet.endT - dur_deltaTs):
                for c in range(self.C - 1):
                    if c + energy_deltaCs <= self.C - 1:
                        if throttle:
                            self.G.add_edge((l, c, t), (l, c + energy_deltaCs, t + dur_deltaTs), dur=dur,
                                            energy=energy,
                                            power=energy / dur,
                                            energy_grid=energy / Car.eta_charge,
                                            power_grid=energy / Car.eta_charge / dur)
                        else:
                            self.G.add_edge((l, c, t), (l, c + energy_deltaCs, t + dur_deltaTs), dur=dur,
                                            energy=energy,
                                            power=energy / dur,
                                            energy_grid=energy / Car.eta_charge,
                                            power_grid=energy / Car.eta_charge / dur,
                                            rating=rate,
                                            evse_id=evse_id)
                    else:
                        if not throttle:
                            energy_net_topoff = self.Fleet.deltaC * (self.C - 1 - c)
                            dur_topoff = dur * ((self.C - 1 - c) / energy_deltaCs)
                            # self.G.add_edge()  # TODO
                if throttle:
                    for energy_deltaCs_throttled in reversed(range(1, energy_deltaCs)):
                        energy_throttled = self.Fleet.deltaC * energy_deltaCs_throttled
                        for c in range(self.C - 1):
                            if c + energy_deltaCs_throttled <= self.C - 1:
                                self.G.add_edge((l, c, t), (l, c + energy_deltaCs_throttled, t + dur_deltaTs), dur=dur,
                                                energy=energy_throttled,
                                                power=energy_throttled / dur,
                                                energy_grid=energy_throttled / Car.eta_charge,
                                                power_grid=energy_throttled / Car.eta_charge / dur)

        def filter_node_idx(self, l=None, c=None, t=None):
            """
            Returns the indices of the PAMoDVehicle.G graph's node list of nodes that have coordinates matching
            the location, charge_level, and/or time parameters.

            Parameters
            ----------
            l : str, optional
                Location name.
            c : int, optional
                Charge level.
            t : int or ndarray, optional
                Time step or array of time steps.

            Returns
            -------
            output : integer ndarray
                Indices in the graph's node list that have coordinates matching the parameters.
            """
            mask_l = np.ones(self.N).astype(bool)
            mask_c = np.ones(self.N).astype(bool)
            mask_t = np.ones(self.N).astype(bool)

            if l is not None:
                mask_l = self.G_nodes_arr[:, 0] == l
            if c is not None:
                mask_c = self.G_nodes_arr[:, 1] == c
            if t is not None and type(t) == int:
                mask_t = self.G_nodes_arr[:, 2] == t
            elif t is not None and type(t) == np.ndarray:
                mask_t = np.in1d(self.G_nodes_arr[:, 2], t)

            return np.argwhere((mask_l) & (mask_c) & (mask_t)).flatten()

        def filter_edge_idx(self, edge_type, O=None, D=None, evse_id=None, idle=None, t=None, power_grid=None):
            if edge_type == 'road':
                mask_edge_type = self.G_edges_road_mask
            elif edge_type == 'charge':
                mask_edge_type = self.G_edges_charge_mask
            else:
                raise ValueError("Invalid edge_type '{}'.  Must be 'road' or 'charge'.".format(edge_type))
            mask_O = np.ones(self.E).astype(bool)
            mask_D = np.ones(self.E).astype(bool)
            mask_evse_id = np.ones(self.E).astype(bool)
            mask_idle = np.ones(self.E).astype(bool)
            mask_t = np.ones(self.E).astype(bool)
            mask_power = np.ones(self.E).astype(bool)
            if O is not None:
                mask_O = (self.G_edges_O_arr == O)
            if D is not None:
                mask_D = (self.G_edges_D_arr == D)
            if evse_id is not None:
                mask_evse_id = (self.G_edges_evse_id_arr == evse_id)
            if idle is not None:
                if idle:
                    mask_idle = self.G_edges_road_idle_mask
                else:
                    mask_idle = ~self.G_edges_road_idle_mask
            if t is not None:
                mask_t = (self.G_edges_t_arr == t)
            if power_grid is not None:
                mask_power = (self.power_conv > power_grid[0]) & (self.power_conv <= power_grid[1])
            return np.argwhere((mask_O) & (mask_D) & (mask_edge_type) & (mask_evse_id) & (mask_idle) & (mask_t) & (
                mask_power)).flatten()

    def build(self):
        self.set_logger()
        self.logger.info('Building experiment {}...'.format(self.name))
        self.p_elec = generate_p_elec(int(np.round(24 / self.deltaT)))
        self.carbon_intensity_grid = generate_carbon_intensity_grid(int(np.round(24 / self.deltaT)))
        self.L = len(self.locations)

        for vehicle_idx in range(len(self.Vehicles)):
            self.PAMoDVehicles.append(self.PAMoDVehicle(self, self.Vehicles[vehicle_idx], self.energy_ODs[vehicle_idx], self.fleet_sizes[vehicle_idx]))

        for PAMoDVehicle in self.PAMoDVehicles:
            if PAMoDVehicle.Vehicle.powertrain == 'electric':
                PAMoDVehicle.C = int(np.round(PAMoDVehicle.Vehicle.batt_cap * self.batt_cap_range / self.deltaC)) + 1
            else:
                PAMoDVehicle.C = 1
            PAMoDVehicle.G = nx.MultiDiGraph()
            PAMoDVehicle.G.add_nodes_from(itertools.product(self.locations, np.arange(PAMoDVehicle.C), np.arange(self.startT, self.endT)))
            PAMoDVehicle.N = PAMoDVehicle.G.number_of_nodes()

        self.road_arcs = set()

        for PAMoDVehicle in self.PAMoDVehicles:
            tic = time.time()
            self.logger.info('-Adding road edges for {}'.format(PAMoDVehicle.Vehicle.name))
            for idx, x in enumerate(itertools.product(self.locations, self.locations, np.arange(self.startT, self.endT))):
                status = PAMoDVehicle.add_road_edges(x[0], x[1], x[2], 0.1 * sum(self.fleet_sizes), PAMoDVehicle.Vehicle)  # TODO have actual road congestion
                if status:
                    self.road_arcs.add((x[0], x[1]))
            print(PAMoDVehicle.num_dropped_trips, PAMoDVehicle.num_incl_trips)
            for l in self.locations:  # idling # TODO: should idling be allowed at passthrough locations?
                PAMoDVehicle.add_idle_road_edges(l, 0.4 * sum(self.fleet_sizes))  # TODO have actual idle congestion

            if PAMoDVehicle.Vehicle.powertrain == 'electric':
                self.logger.info('-Adding charge edges')
                for l, station_list in self.charge_stations.items():
                    if station_list == []:
                        continue
                    station = station_list[0]  # TODO: assuming here that each location has at most one station
                    ratings = [evse.rate for evse in station.EVSEs]
                    if self.charge_throttle:
                        rate = min(max(ratings), max(PAMoDVehicle.Vehicle.max_charge_rate_AC, PAMoDVehicle.Vehicle.max_charge_rate_DC))  # TODO: this will not work; max_charge_rate_AC limit will not be enforced
                        PAMoDVehicle.add_charge_edges(l, rate, PAMoDVehicle.Vehicle, throttle=True)  # TODO: loop Vehicles
                    else:
                        for evse in station.EVSEs:
                            rate, evse_id = evse.rate, evse.evse_id
                            if rate <= 19.2:  # AC TODO: sloppy fix for identifying AC or DC station
                                if PAMoDVehicle.Vehicle.max_charge_rate_AC == None or PAMoDVehicle.Vehicle.max_charge_rate_AC == 0:
                                    continue
                                elif rate > PAMoDVehicle.Vehicle.max_charge_rate_AC:
                                    rate = PAMoDVehicle.Vehicle.max_charge_rate_AC
                            else:  # DC
                                if PAMoDVehicle.Vehicle.max_charge_rate_DC == None or PAMoDVehicle.Vehicle.max_charge_rate_DC == 0:
                                    continue
                                elif rate > PAMoDVehicle.Vehicle.max_charge_rate_DC:
                                    rate = PAMoDVehicle.Vehicle.max_charge_rate_DC
                            PAMoDVehicle.add_charge_edges(l, rate, PAMoDVehicle.Vehicle, throttle=False, evse_id=evse_id)  # TODO: loop Vehicles
                assert(PAMoDVehicle.N == PAMoDVehicle.G.number_of_nodes())  # Make sure no new nodes were created when adding edges

            PAMoDVehicle.E = len(PAMoDVehicle.G.edges)
            PAMoDVehicle.set_incidence_matrices()
            PAMoDVehicle.Dist = np.array([dist for O, D, dist in PAMoDVehicle.G.edges(data="dist", default=0)])
            PAMoDVehicle.Demand = np.array([demand for O, D, demand in PAMoDVehicle.G.edges(data="demand", default=0)])
            PAMoDVehicle.Dur = np.array([dur for O, D, dur in PAMoDVehicle.G.edges(data="dur", default=0)])
            PAMoDVehicle.UMax_road = np.array([uMax_road for O, D, uMax_road in PAMoDVehicle.G.edges(data="uMax_road", default=0)])
            PAMoDVehicle.E_road_idx = [idx for idx, (O, D, uMax_road) in enumerate(PAMoDVehicle.G.edges(data="uMax_road")) if uMax_road]
            PAMoDVehicle.E_charge_idx = [idx for idx, (O, D, energy_grid) in
                                                enumerate(PAMoDVehicle.G.edges(data="energy_grid")) if
                                                energy_grid]

            if PAMoDVehicle.Vehicle.powertrain == 'electric':
                PAMoDVehicle.energy_conv = np.array([energy for O, D, energy in PAMoDVehicle.G.edges(data="energy_grid", default=0)]).round(1)
            else:
                PAMoDVehicle.energy_conv = np.array(
                    [energy for O, D, energy in PAMoDVehicle.G.edges(data="energy", default=0)]).round(2)
            PAMoDVehicle.power_conv = np.array(
                [power for O, D, power in PAMoDVehicle.G.edges(data="power_grid", default=0)]).round(1)

            assert set(list(range(PAMoDVehicle.E))) == set(PAMoDVehicle.E_road_idx).union(
                set(PAMoDVehicle.E_charge_idx))  # check that all graph's edges are accounted for in E_road_idx, E_charge_idx

            PAMoDVehicle.G_edges_O_arr = np.array(list(PAMoDVehicle.G.edges()))[:, 0, 0]
            PAMoDVehicle.G_edges_D_arr = np.array(list(PAMoDVehicle.G.edges()))[:, 1, 0]
            PAMoDVehicle.G_edges_t_arr = np.array(list(PAMoDVehicle.G.edges()))[:, 0, 2]
            PAMoDVehicle.G_edges_road_mask = (np.array(list(PAMoDVehicle.G.edges(data='uMax_road', default=0)))[:, 2] != 0)
            PAMoDVehicle.G_edges_charge_mask = (np.array(list(PAMoDVehicle.G.edges(data='energy_grid', default=0)))[:, 2] != 0)
            PAMoDVehicle.G_edges_evse_id_arr = np.array(list(PAMoDVehicle.G.edges(data='evse_id', default=None)))[:, 2]
            PAMoDVehicle.G_edges_road_idle_mask = np.array(list(PAMoDVehicle.G.edges(data='idle', default=False)))[:, 2].astype(bool)
            PAMoDVehicle.G_nodes_arr = np.array(PAMoDVehicle.G.nodes())

            if PAMoDVehicle.Vehicle.powertrain == 'electric':
                self.logger.info("-Created extended graph with {} nodes and {} edges ({} charge, {} road). Total time elapsed={:.2f}".format(PAMoDVehicle.N, PAMoDVehicle.E,
                                                                                                    len(PAMoDVehicle.E_charge_idx),
                                                                                                    len(PAMoDVehicle.E_road_idx),
                                                                                                    time.time() - tic))
            else:
                self.logger.info(
                    "-Created extended graph with {} nodes and {} edges. Total time elapsed={:.2f}".format(
                        PAMoDVehicle.N, PAMoDVehicle.E, time.time() - tic))

    def run(self):
        if self.config['algorithm'] == 'PAMoD_optimization_gurobi':
            [X, U, U_trip_charge_idle, U_rebal, elec_energy, elec_demand, dist, revenue, fleet_cost, elec_carbon, infra, gas, gas_carbon] = PAMoD_optimization_gurobi(self)
        elif self.config['algorithm'] == 'PAMoD_optimization_pyomo':
            [X, U, U_trip_charge_idle, U_rebal, elec_energy, elec_demand, dist, revenue, fleet_cost, elec_carbon, infra, gas,
             gas_carbon] = PAMoD_optimization_pyomo(self)
        else:
            raise ValueError('"{}" is not a valid algorithm for the experiment_type {}'.format(self.config['algorithm'], self.config['experiment_type']))
        self.X_list = X
        self.U_list = U
        self.U_trip_charge_idle_list = U_trip_charge_idle
        self.U_rebal_list = U_rebal
        self.costs_list = np.array([elec_energy, elec_demand, dist, revenue, fleet_cost, elec_carbon, infra, gas, gas_carbon])
        self.power_matrix_list = [np.zeros((len(self.locations_excl_passthrough), self.T - 1)) for _ in range(len(self.Vehicles))]
        for vehicle_idx, PAMoDVehicle in enumerate(self.PAMoDVehicles):
            for l_idx, l in enumerate(self.locations_excl_passthrough):
                for t_idx, t in enumerate(range(self.startT, self.endT - 1)):
                    E_charge_idx_l_t = PAMoDVehicle.filter_edge_idx('charge', l, l, t=t)
                    self.power_matrix_list[vehicle_idx][l_idx, t_idx] = np.sum(
                        np.multiply(self.U_list[vehicle_idx][E_charge_idx_l_t], PAMoDVehicle.power_conv[E_charge_idx_l_t]))
        self.logger.info('total={}, elec_energy={}, elec_demand={}, dist={}, revenue={}, fleet_cost={}, elec_carbon={}, infra={}, gas={}, gas_carbon={}'.format(np.sum(self.costs_list),
                                                                                     elec_energy, elec_demand, dist, revenue, fleet_cost, elec_carbon, infra, gas, gas_carbon))
    def save(self):
        np.save(os.path.join(self.results_path, 'power_matrix_list.npy'), self.power_matrix_list)
        np.save(os.path.join(self.results_path, 'X_list.npy'), self.X_list)
        np.save(os.path.join(self.results_path, 'U_list.npy'), self.U_list)
        np.save(os.path.join(self.results_path, 'U_trip_charge_idle_list.npy'), self.U_trip_charge_idle_list)
        np.save(os.path.join(self.results_path, 'U_rebal_list.npy'), self.U_rebal_list)
        if self.optimize_infra:
            np.save(os.path.join(self.results_path, 'UMax_charge.npy'), self.UMax_charge)
        np.save(os.path.join(self.results_path, 'cost_list.npy'),self.costs_list)
        self.logger = None
        pickle.dump(self, open(os.path.join(self.results_path, '{}.p'.format(self.name)), 'wb'))

    def plot(self, plot_graphs=True, plot_anim=None):
        if plot_anim is not None:
            for node_type in plot_anim:
                plot_animiation(self, node_type)
        if plot_graphs:
            plot_PAMoDFleet_results(self)

    def round_time(self, dur, min_val=None):
        if min_val is not None:
            return max(int(np.round(dur / self.deltaT)), min_val)
        else:
            return int(np.round(dur / self.deltaT))

    def round_energy(self, energy, min_val=None):
        if min_val is not None:
            return max(int(np.round(energy / self.deltaC)), min_val)
        else:
            return int(np.round(energy / self.deltaC))

    def set_logger(self):
        self.logger = logging.getLogger(self.name)
        logging.Logger.manager.loggerDict[self.name] = self.logger
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(logging.StreamHandler(sys.stdout))
        self.logger.addHandler(logging.FileHandler(os.path.join(self.results_path, 'print_log.txt')))

    def get_lep_idx_evse_idx(self, l, evse_id):
        lep_idx = self.locations_excl_passthrough.index(l)
        evse_idx = [evse.name for evse in self.EVSEs].index(evse_id.split('_')[1])
        return lep_idx, evse_idx
