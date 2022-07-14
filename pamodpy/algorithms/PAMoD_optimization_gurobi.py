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
import time
import gc
import logging
from itertools import repeat, count

import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import pathos.multiprocessing as pmp
from scipy import sparse

from ..utils.constants import *

U_const, UMax_const, PMax_const = 1000, 1, 10000

def PAMoD_optimization_gurobi(experiment):
    def obj_elec_energy_carbon_and_constr_UMax_charge(U_list, UMax_charge, build=True):
        if experiment.charge_throttle:
            l_t_eid_gen = ((l, t) for l in experiment.locations_excl_passthrough
                   for t in range(experiment.startT, experiment.endT))
        else:
            l_t_eid_gen = ((l, t, evse.evse_id) for l, station_list in experiment.charge_stations.items()
                            for station in station_list
                            for evse in station.EVSEs
                            for t in range(experiment.startT, experiment.endT))

        with pmp.ThreadingPool() as p:
            outputs = p.map(obj_elec_energy_carbon_and_constr_UMax_charge_worker,
                            repeat(U_list), repeat(UMax_charge), repeat(build), l_t_eid_gen, repeat(experiment), count())
        return outputs

    def obj_revenue_and_constr_UMax_road(U_list, trip_flow, build=True):
        o_d_t_gen = ((x[0], x[1], t) for x in experiment.road_arcs
                     for t in range(experiment.startT, experiment.endT))
        idx_gen = range(len(experiment.road_arcs) * experiment.T)
        with pmp.ThreadingPool() as p:
            outputs = p.map(obj_revenue_and_constr_UMax_road_worker,
                            repeat(U_list), repeat(trip_flow), repeat(build), o_d_t_gen, idx_gen, repeat(experiment), count())
        return outputs

    def obj_elec_demand(U_list, PMax, build=True):
        elec_demand = 0
        for l_idx, l in enumerate(experiment.charge_stations.keys()):
            elec_demand += PMax[l_idx] * experiment.p_elec_demand * (
                        experiment.T * experiment.deltaT / HOURS_PER_MONTH) * (PMax_const / U_const)

        if build:
            lidx_l_t = ((l_idx, l, t) for l_idx, l in enumerate(experiment.charge_stations.keys())
                        for t in range(experiment.startT, experiment.endT))
            with pmp.ThreadingPool() as p:
                outputs = p.map(obj_elec_demand_worker,
                                repeat(U_list), repeat(PMax), lidx_l_t, repeat(experiment), count())
            return elec_demand, outputs
        else:
            return elec_demand

    def constr_infra(U_list, UMax_charge):
        l_lepidx_ridx_t_gen = ((l, lep_idx, rating_idx, t) for lep_idx, l in enumerate(experiment.locations_excl_passthrough)
                       for rating_idx, rating in enumerate(experiment.charge_rate)
                       for t in range(experiment.startT, experiment.endT))

        with pmp.ThreadingPool() as p:
            outputs = p.map(constr_infra_worker,
                            repeat(U_list), repeat(UMax_charge), l_lepidx_ridx_t_gen, repeat(experiment), count())
        return outputs

    def post_opt_U_rebal(U_value, PAMoDVehicle):
        o_d_t_gen = ((x[0], x[1], t) for x in experiment.road_arcs
                     for t in range(experiment.startT, experiment.endT))
        with pmp.ThreadingPool() as p:
            outputs = p.map(post_opt_U_rebal_worker, repeat(U_value), repeat(PAMoDVehicle), o_d_t_gen, repeat(experiment))
        return outputs

    def post_opt_X(U_value, PAMoDVehicle):
        t_gen = (t for t in range(experiment.startT + 1, experiment.endT - 1))
        with pmp.ThreadingPool() as p:
            outputs = p.map(post_opt_X_worker, repeat(U_value), repeat(PAMoDVehicle), t_gen)
        return outputs


    try:
        experiment.logger.setLevel(logging.WARNING)
        with open(os.path.join(experiment.results_path, 'gurobi_log.log'), 'w') as f:
            pass
        gp.setParam("LogFile", os.path.join(experiment.results_path, 'gurobi_log.log'))

        if experiment.load_opt:
            m = gp.read(os.path.join(experiment.results_path, 'gurobi_model.mps'))
            experiment.logger.setLevel(logging.WARNING)
            m.read(os.path.join(experiment.results_path, 'gurobi_model.prm'))
            experiment.logger.setLevel(logging.INFO)
            if experiment.optimize_fleet_size:
                fleet_size_const = 10000
            else:
                fleet_sizes = experiment.fleet_sizes
                fleet_size_const = 1
        else:
            # Create a new model
            m = gp.Model("PAMoD_optimization_gurobi")
            experiment.logger.setLevel(logging.INFO)

            # Create variables
            U_list = []
            for vehicle_idx, PAMoDVehicle in enumerate(experiment.PAMoDVehicles):
                U_list.append(m.addMVar(shape=PAMoDVehicle.E, lb=0.0, name="U_{}".format(vehicle_idx)))
            if experiment.optimize_infra:
                UMax_charge = m.addMVar(shape=(len(experiment.locations_excl_passthrough), len(experiment.EVSEs)), lb=0.0, name="UMax_charge")
                if experiment.optimize_infra_mip:
                    B = m.addMVar(shape=(len(experiment.locations_excl_passthrough), len(experiment.EVSEs)), vtype='B', name="B")
            else:
                UMax_charge = experiment.UMax_charge
            PMax = m.addMVar(shape=(len(experiment.locations_excl_passthrough), 1), lb=0.0, name="PMax")
            if experiment.drop_trips:
                trip_flow = m.addMVar(shape=len(experiment.road_arcs) * experiment.T, name="trip_flow")
            else:
                trip_flow = np.zeros(len(experiment.road_arcs) * experiment.T)
            fleet_sizes = []
            if experiment.optimize_fleet_size:
                fleet_size_const = 10000
                for vehicle_idx, PAMoDVehicle in enumerate(experiment.PAMoDVehicles):
                    fleet_sizes.append(m.addMVar(shape=1, lb=0.0, name="fleet_size_{}".format(vehicle_idx)))
            else:
                    fleet_sizes = experiment.fleet_sizes
                    fleet_size_const = 1

            m.update()
            experiment.logger.info("-Creating optimization problem")
            tic_start = time.time()

            tic = time.time()
            for vehicle_idx, PAMoDVehicle in enumerate(experiment.PAMoDVehicles):
                nodes_start = PAMoDVehicle.filter_node_idx(None, None, experiment.startT)
                m.addConstr((np.ones(len(nodes_start)) * -PAMoDVehicle.A_outflows[nodes_start] @ U_list[vehicle_idx]) == fleet_sizes[vehicle_idx] * (fleet_size_const / U_const), name="fleet_size_{}".format(vehicle_idx))
                experiment.logger.info("--Created fleet size constraints (elapsed={:.2f})".format(time.time() - tic))

            if experiment.charge_throttle:
                if experiment.optimize_infra or experiment.congestion_constr_charge:
                    tic = time.time()
                    outputs = obj_elec_energy_carbon_and_constr_UMax_charge(U_list, UMax_charge)
                    elec_energy = 0
                    elec_carbon = 0
                    for output in sorted(outputs, key=lambda item: item[-1]):
                        elec_energy += output[0]
                        elec_carbon += output[1]
                    del outputs
                    experiment.logger.info(
                        "--Created elec_energy obj term (elapsed={:.2f})".format(time.time() - tic))

            else:
                if experiment.optimize_infra or experiment.congestion_constr_charge:
                    tic = time.time()
                    outputs = obj_elec_energy_carbon_and_constr_UMax_charge(U_list, UMax_charge)
                    elec_energy = 0
                    elec_carbon = 0
                    for output in sorted(outputs, key=lambda item: item[-1]):
                        elec_energy += output[0]
                        elec_carbon += output[1]
                        if output[2] is not None:
                            m.addConstr(output[2])
                    del outputs
                    experiment.logger.info("--Created elec_energy obj term and UMax_charge constraint (elapsed={:.2f})".format(time.time() - tic))

            infra = 0
            if experiment.optimize_infra:
                tic = time.time()
                if not experiment.optimize_fleet_size:
                    M = sum(experiment.fleet_sizes)
                else:
                    M = np.amax(np.sum(experiment.od_matrix, axis=(0, 1)))
                if experiment.optimize_infra_mip:
                    for lep_idx, l in enumerate(experiment.locations_excl_passthrough):
                        for evse_idx, evse in enumerate(experiment.EVSEs):
                            infra += (B[lep_idx, evse_idx] * experiment.p_infra_capital[lep_idx, evse_idx] +
                                      experiment.p_infra_marginal[
                                          lep_idx, evse_idx] * UMax_charge[lep_idx, evse_idx:evse_idx+1] * UMax_const) * (
                                             1 / U_const)
                            m.addConstr(
                                UMax_charge[lep_idx, evse_idx:evse_idx+1] <= M * B[lep_idx, evse_idx] * (1 / UMax_const))
                else:
                    for lep_idx, l in enumerate(experiment.locations_excl_passthrough):
                        for evse_idx, evse in enumerate(experiment.EVSEs):
                            infra += (experiment.p_infra_marginal[lep_idx, evse_idx] * UMax_charge[
                                lep_idx, evse_idx:evse_idx+1] * UMax_const) * (
                                             1 / U_const)
                experiment.logger.info(
                    "--Created infra obj term (elapsed={:.2f})".format(
                        time.time() - tic))

            if experiment.charge_throttle:
                tic = time.time()
                outputs = constr_infra(U_list, UMax_charge)
                for output in sorted(outputs, key=lambda item: item[-1]):
                    m.addConstr(output[0])
                del outputs
                experiment.logger.info(
                    "--Created infra constraint (elapsed={:.2f})".format(
                        time.time() - tic))

            tic = time.time()
            outputs = obj_revenue_and_constr_UMax_road(U_list, trip_flow)
            revenue = 0
            for output in sorted(outputs, key=lambda item: item[-1]):
                revenue -= output[0]
                if output[1] is not None:
                    m.addConstr(output[1])
                if output[2] is not None:
                    m.addConstr(output[2])
                if experiment.congestion_constr_road:
                    m.addConstr(output[3])
            del outputs
            experiment.logger.info("--Created revenue obj term and UMax_road constraint (elapsed={:.2f})".format(
                time.time() - tic))

            tic = time.time()
            for vehicle_idx, PAMoDVehicle in enumerate(experiment.PAMoDVehicles):
                nodes_t = PAMoDVehicle.filter_node_idx(None, None, np.array(range(experiment.startT + 1, experiment.endT - 1)))
                m.addConstr(PAMoDVehicle.A[nodes_t] @ U_list[vehicle_idx] == 0)

            if experiment.boundary:
                for vehicle_idx, PAMoDVehicle in enumerate(experiment.PAMoDVehicles):
                    nodes_start = PAMoDVehicle.filter_node_idx(None, None, experiment.startT)
                    nodes_end = PAMoDVehicle.filter_node_idx(None, None, experiment.endT - 1)
                    m.addConstr(-(PAMoDVehicle.A_outflows[nodes_start] @ U_list[vehicle_idx]) == (PAMoDVehicle.A_inflows[nodes_end] @ U_list[vehicle_idx]), name="boundary_{}".format(vehicle_idx))
            experiment.logger.info("--Created fleet dynamics (elapsed={:.2f})".format(time.time() - tic))

            tic = time.time()
            [elec_demand, outputs] = obj_elec_demand(U_list, PMax)
            for output in sorted(outputs, key=lambda item: item[-1]):
                if output[0] is not None:
                    m.addConstr(output[0])
            del outputs
            dist = sum([U_list[vehicle_idx] @ PAMoDVehicle.Dist * experiment.p_travel for vehicle_idx, PAMoDVehicle in enumerate(experiment.PAMoDVehicles)])
            experiment.logger.info("--Created elec_demand and dist obj terms (elapsed={:.2f})".format(time.time() - tic))

            gas = 0
            gas_carbon = 0
            for vehicle_idx, PAMoDVehicle in enumerate(experiment.PAMoDVehicles):
                if PAMoDVehicle.Vehicle.powertrain != 'electric':
                    gas += U_list[vehicle_idx] @ PAMoDVehicle.energy_conv * experiment.p_gas
                    gas_carbon += U_list[vehicle_idx] @ PAMoDVehicle.energy_conv * tons_CO2_per_gal_gas * experiment.p_carbon
            experiment.logger.info("--Done creating optimization problem. Total time elapsed={:.2f}".format(time.time() - tic_start))

            # Set objective
            fleet_cost = 0
            for vehicle_idx in range(len(experiment.Vehicles)):
                fleet_cost += fleet_sizes[vehicle_idx] * (experiment.Vehicles[vehicle_idx].price * 0.2 + experiment.p_ownership_excl_deprec) * (experiment.T * experiment.deltaT / HOURS_PER_YEAR) * (
                        fleet_size_const / U_const)

            obj = elec_energy + elec_demand + elec_carbon + dist + revenue + fleet_cost + infra + gas + gas_carbon
            m.setObjective(obj, GRB.MINIMIZE)

            # Clean-up

            for PAMoDVehicle in experiment.PAMoDVehicles:
                PAMoDVehicle.remove_incidence_matrices()
            gc.collect()

            # Optimization Settings
            experiment.logger.setLevel(logging.WARNING)
            if experiment.optimize_infra_mip:
                # m.Params.Method = 1            # dual simplex only; default does simplex and barrier
                # m.Params.MarkowitzTol = 0.0625
                # m.Params.MIPFocus = 3           # 0 is default (balanced); 1 focuses on feasible soln quickly; 2 on proving optimality; 3 on improving best upper bound
                m.Params.MIPGap = 2*1e-4        # default is 1e-4
                # m.Params.MIPGapAbs = 5000/U_const       # default is 1e-10
            else:
                m.Params.Method = 2  # Barrier only; default -1 does simplex and barrier
                # m.Params.BarConvTol = 1e-10     # default is 1e-8; make tighter to spend less time in crossover
                m.Params.Crossover = 0
                # m.Params.Presolve = 2             # default is -1 (auto); 2 is aggressive
                # m.Params.BarOrder = 1             # default is -1 (auto); 0 is Approximate Minimum Degree, 1 is Nested Dissection (usual?)

            experiment.logger.setLevel(logging.INFO)

            # Save
            if experiment.save_opt:
                m.write(os.path.join(experiment.results_path, 'gurobi_model.mps'))
                m.write(os.path.join(experiment.results_path, 'gurobi_model.prm'))

        # Optimize model
        experiment.logger.info("-Starting optimization:")
        experiment.logger.setLevel(logging.WARNING)
        tic = time.time()
        m.optimize()
        experiment.logger.setLevel(logging.INFO)
        experiment.logger.info("-Finished optimization: (elapsed={:.2f})".format(time.time() - tic))
        experiment.logger.info('Obj: %g' % m.objVal)
        for PAMoDVehicle in experiment.PAMoDVehicles:
            PAMoDVehicle.set_incidence_matrices()

        if experiment.load_opt:
            m.write(os.path.join(experiment.results_path, 'gurobi_model.sol'))
            df = pd.read_csv(os.path.join(experiment.results_path, 'gurobi_model.sol'),
                             names=['var', 'val'], skiprows=2, delimiter=' ')
            U_value_list = [np.array(df['val'][df['var'].str.contains("U_{}".format(vehicle_idx))].to_list()) for vehicle_idx in range(len(experiment.Vehicles))]
            PMax_value = np.array(df['val'][df['var'].str.contains("PMax")].to_list())
            trip_flow_value = np.array(df['val'][df['var'].str.contains("trip_flow")].to_list())
            if experiment.optimize_fleet_size:
                experiment.fleet_sizes = [df['val'][df['var'].str.contains("fleet_size_{}".format(vehicle_idx))].to_list()[0] * fleet_size_const for vehicle_idx in range(len(experiment.Vehicles))]
        else:
            U_value_list = [U.X for U in U_list]
            PMax_value = PMax.X
            if experiment.drop_trips:
                trip_flow_value = trip_flow.X
            else:
                trip_flow_value = trip_flow
            if experiment.optimize_fleet_size:
                experiment.fleet_sizes = [float(fleet_size.X * fleet_size_const) for fleet_size in fleet_sizes]
            if experiment.optimize_infra:
                experiment.UMax_charge = UMax_charge.X * UMax_const
                print(UMax_charge.X * UMax_const)
                if experiment.optimize_infra_mip:
                    print(B.X)

        fleet_cost = 0
        for vehicle_idx in range(len(experiment.Vehicles)):
            fleet_cost += experiment.fleet_sizes[vehicle_idx] * (
                    experiment.Vehicles[vehicle_idx].price * 0.2 + experiment.p_ownership_excl_deprec) * (
                                      experiment.T * experiment.deltaT / HOURS_PER_YEAR) * (
                                  1 / U_const)

        experiment.logger.info("fleet_sizes = {}".format(experiment.fleet_sizes))
        experiment.logger.info("U value range: [{}, {}]".format(max([np.max(U_value) for U_value in U_value_list]), min([np.min(U_value) for U_value in U_value_list])))
        experiment.logger.info("PMax value range: [{}, {}]".format(np.max(PMax_value), np.min(PMax_value)))

        X_list = []
        for U_value, PAMoDVehicle in zip(U_value_list, experiment.PAMoDVehicles):
            X = np.zeros(PAMoDVehicle.N)
            outputs = post_opt_X(U_value, PAMoDVehicle)
            for output in outputs:
                X[output[0]] = output[1]
            del outputs
            nodes_start = PAMoDVehicle.filter_node_idx(None, None, experiment.startT)
            nodes_end = PAMoDVehicle.filter_node_idx(None, None, experiment.endT - 1)
            X[nodes_start] = -PAMoDVehicle.A_outflows[nodes_start] * U_value * U_const
            X[nodes_end] = PAMoDVehicle.A_inflows[nodes_end] * U_value * U_const
            X_list.append(X)

        U_rebal_list = []
        U_trip_charge_idle_list = []
        U_rebal_dist_costs = []

        for U_value, PAMoDVehicle in zip(U_value_list, experiment.PAMoDVehicles):
            U_rebal = np.zeros(PAMoDVehicle.E)
            outputs = post_opt_U_rebal(U_value, PAMoDVehicle)
            for output in outputs:
                if output is not None:
                    U_rebal[output[0]] = output[1]
            U_rebal_list.append(U_rebal)
            U_rebal_dist_costs.append((U_rebal @ PAMoDVehicle.Dist) * experiment.p_travel  * U_const)
            del outputs
            U_trip_charge_idle = U_value - U_rebal
            U_trip_charge_idle_list.append(U_trip_charge_idle)
        experiment.logger.info("U_rebal distance costs = {}".format(sum(U_rebal_dist_costs)))

        infra_value = 0
        if experiment.optimize_infra:
            if experiment.optimize_infra_mip:
                for lep_idx, l in enumerate(experiment.locations_excl_passthrough):
                    for evse_idx, evse in enumerate(experiment.EVSEs):
                        infra_value += (B.X[lep_idx, evse_idx] * experiment.p_infra_capital[lep_idx, evse_idx] +
                                  experiment.p_infra_marginal[
                                      lep_idx, evse_idx] * UMax_charge.X[lep_idx, evse_idx] * UMax_const) * (
                                         1 / U_const)
            else:
                for lep_idx, l in enumerate(experiment.locations_excl_passthrough):
                    for evse_idx, evse in enumerate(experiment.EVSEs):
                        infra_value += (experiment.p_infra_marginal[lep_idx, evse_idx] * UMax_charge.X[lep_idx, evse_idx] * UMax_const) * (
                                         1 / U_const)
        elif experiment.use_baseline_charge_stations:
            for lep_idx, l in enumerate(experiment.locations_excl_passthrough):
                for evse_idx, evse in enumerate(experiment.EVSEs):
                    infra_value += (experiment.p_infra_marginal[lep_idx, evse_idx] * experiment.UMax_charge[
                        lep_idx, evse_idx]) * (
                                           1 / U_const)

        infra_value *= U_const
        experiment.logger.info("infra_value = {}".format(infra_value))

        if experiment.drop_trips:
            revenue_final = revenue.getValue() * U_const
        else:
            revenue_final = revenue
        if any(Vehicle.powertrain != 'electric' for Vehicle in experiment.Vehicles):
            gas_final = gas.getValue()[0] * U_const
            gas_carbon_final = gas_carbon.getValue()[0] * U_const
        else:
            gas_final = 0
            gas_carbon_final = 0
        if any(Vehicle.powertrain == 'electric' for Vehicle in experiment.Vehicles):
            elec_energy_final = elec_energy.getValue()[0] * U_const
            elec_demand_final = elec_demand.getValue()[0] * U_const
            elec_carbon_final = elec_carbon.getValue()[0] * U_const
        else:
            elec_energy_final = 0
            elec_demand_final = 0
            elec_carbon_final = 0


        return [X_list, np.array(U_value_list) * U_const, np.array(U_trip_charge_idle_list) * U_const, np.array(U_rebal_list) * U_const,
                elec_energy_final,
                elec_demand_final,
                dist.getValue()[0] * U_const,
                revenue_final,
                fleet_cost * U_const,
                elec_carbon_final,
                gas_final,
                gas_carbon_final
                ]  # TODO this won't work for load_opt == True


    except gp.GurobiError as e:
        experiment.logger.info('Error code ' + str(e.errno) + ": " + str(e))

    except AttributeError:
        experiment.logger.info('Encountered an attribute error')


def obj_elec_energy_carbon_and_constr_UMax_charge_worker(U_list, UMax_charge, build, l_t_eid, experiment, count):
    UMax_charge_constr_lhs = []
    elec_energy_list = []
    elec_carbon_list = []
    invalid = 0
    n_elec_vehicles = 0
    lep_idx, evse_idx = None, None

    for vehicle_idx, PAMoDVehicle in enumerate(experiment.PAMoDVehicles):
        if PAMoDVehicle.Vehicle.powertrain == 'electric':
            n_elec_vehicles += 1
            if experiment.charge_throttle:
                l, t = l_t_eid
                E_charge_idx_l_eid_t = PAMoDVehicle.filter_edge_idx('charge', l, l, t=t)
                if len(E_charge_idx_l_eid_t) == 0:
                    invalid += 1
                else:
                    elec_energy_list.append(
                        (U_list[vehicle_idx][E_charge_idx_l_eid_t] @ PAMoDVehicle.energy_conv[E_charge_idx_l_eid_t]) *
                        experiment.p_elec[t])
            else:
                l, t, evse_id = l_t_eid
                E_charge_idx_l_eid_t = PAMoDVehicle.filter_edge_idx('charge', l, l, evse_id=evse_id, t=t)
                if len(E_charge_idx_l_eid_t) == 0:
                    invalid += 1
                else:
                    elec_energy_list.append(
                        (U_list[vehicle_idx][E_charge_idx_l_eid_t] @ PAMoDVehicle.energy_conv[E_charge_idx_l_eid_t]) *
                        experiment.p_elec[t])
                    elec_carbon_list.append(
                        (U_list[vehicle_idx][E_charge_idx_l_eid_t] @ PAMoDVehicle.energy_conv[
                            E_charge_idx_l_eid_t]) *
                        experiment.carbon_intensity_grid[t] * experiment.p_carbon
                    )
                    lep_idx, evse_idx = experiment.get_lep_idx_evse_idx(l, evse_id)
                    UMax_charge_constr_lhs.append((U_list[vehicle_idx][E_charge_idx_l_eid_t].sum()))

    if invalid < max(n_elec_vehicles, 1):
        elec_energy_term = sum(elec_energy_list)
        elec_carbon_term = sum(elec_carbon_list)
        if experiment.charge_throttle or n_elec_vehicles == 0:
            UMax_charge_constr = None
        else:
            UMax_charge_constr = (sum(UMax_charge_constr_lhs) <= (UMax_charge[lep_idx, evse_idx:evse_idx+1]) * (
                            UMax_const / U_const))
        if build:
            return elec_energy_term, elec_carbon_term, UMax_charge_constr, count
        else:
            return elec_energy_term, elec_carbon_term
    else:
        if build:
            return 0, 0, None, count
        else:
            return 0, 0

def constr_infra_worker(U_list, UMax_charge, l_lidx_ridx_t, experiment, count):
    l, lep_idx, rating_idx, t = l_lidx_ridx_t
    if rating_idx == 0:
        power_lb = 0
    else:
        power_lb = experiment.charge_rate[rating_idx - 1]
    infra_constr_lhs = []
    for vehicle_idx, PAMoDVehicle in enumerate(experiment.PAMoDVehicles):
        if PAMoDVehicle.Vehicle.powertrain == 'electric':
            E_charge_idx_l_t_rates = PAMoDVehicle.filter_edge_idx('charge', l, l, t=t,
                                                                    power_grid=(power_lb+0.5*experiment.deltaC/experiment.deltaT,
                                                                           experiment.charge_rate[-1]+0.5*experiment.deltaC/experiment.deltaT))
            infra_constr_lhs.append(U_list[vehicle_idx][E_charge_idx_l_t_rates].sum())
    infra_constr = (sum(infra_constr_lhs) <= (UMax_charge[lep_idx, rating_idx:].sum()) * (UMax_const / U_const))
    return infra_constr, count

def obj_revenue_and_constr_UMax_road_worker(U_list, trip_flow, build, o_d_t, idx, experiment, count):
    O, D, t = o_d_t
    hour = int(np.floor((t * experiment.deltaT) % (24 / experiment.deltaT)))
    if experiment.drop_trips:
        revenue_term = trip_flow[idx] * experiment.revenue_matrix[O - 1, D - 1, hour]
    else:
        revenue_term = 0
    if build:
        if experiment.drop_trips:
            trip_flow_constr1 = (trip_flow[idx] <= experiment.od_matrix[O - 1, D - 1, hour] * experiment.deltaT / U_const)
        else:
            trip_flow_constr1 = None

        trip_flow_constr2_lhs = []
        UMax_road_constr_lhs = []
        UMax_road_constr = None
        invalid = 0

        for vehicle_idx, PAMoDVehicle in enumerate(experiment.PAMoDVehicles):
            E_road_idx_r_nonidle_t = PAMoDVehicle.filter_edge_idx('road', O, D, idle=False, t=t)
            if len(E_road_idx_r_nonidle_t) != 0:
                if experiment.drop_trips:
                    trip_flow_constr2_lhs.append((U_list[vehicle_idx][E_road_idx_r_nonidle_t]).sum())
                else:
                    trip_flow_constr2_lhs.append((U_list[vehicle_idx][E_road_idx_r_nonidle_t]).sum())
                if experiment.congestion_constr_road:
                    UMax_road_constr_lhs.append((U_list[vehicle_idx][E_road_idx_r_nonidle_t]).sum())
            else:
                invalid += 1

        if invalid < len(experiment.PAMoDVehicles):
            if experiment.drop_trips:
                trip_flow_constr2 = (sum(trip_flow_constr2_lhs) >= trip_flow[idx])
            else:
                trip_flow_constr2 = (sum(trip_flow_constr2_lhs) >= experiment.od_matrix[
                        O - 1, D - 1, hour] * experiment.deltaT / U_const)
            if experiment.congestion_constr_road:
                UMax_road_constr = (sum(UMax_road_constr_lhs) <= (0.1 * sum(experiment.fleet_sizes)) / U_const) # TODO have actual road congestion
            return revenue_term, trip_flow_constr1, trip_flow_constr2, UMax_road_constr, count
        else:
            return 0, None, None, None, count
    else:
        return revenue_term


def obj_elec_demand_worker(U_list, PMax, lidx_l_t, experiment, count):
    l_idx, l, t = lidx_l_t

    obj_elec_demand_lhs = []
    invalid = 0
    n_elec_vehicles = 0

    for vehicle_idx, PAMoDVehicle in enumerate(experiment.PAMoDVehicles):
        if PAMoDVehicle.Vehicle.powertrain == 'electric':
            n_elec_vehicles += 1
        else:
            continue
        E_charge_idx_l_t = PAMoDVehicle.filter_edge_idx('charge', l, l, t=t)

        if len(E_charge_idx_l_t) != 0:
            obj_elec_demand_lhs.append(U_list[vehicle_idx][E_charge_idx_l_t] @ PAMoDVehicle.power_conv[E_charge_idx_l_t])
        else:
            invalid += 1

    if invalid < max(n_elec_vehicles, 1):
        return sum(obj_elec_demand_lhs) <= PMax[l_idx] * (PMax_const / U_const), count
    else:
        return None, count

def post_opt_U_rebal_worker(U_value, PAMoDVehicle, o_d_t, experiment):
    O, D, t = o_d_t
    hour = int(np.floor((t * experiment.deltaT) % (24 / experiment.deltaT)))
    E_road_idx_r_nonidle_t = PAMoDVehicle.filter_edge_idx('road', O, D, idle=False, t=t)
    if len(E_road_idx_r_nonidle_t) == 0:
        return None
    U_non_idle_r_t_sum = np.sum(U_value[E_road_idx_r_nonidle_t])
    demand_r_t = experiment.od_matrix[O - 1, D - 1, hour] * experiment.deltaT
    if U_non_idle_r_t_sum >= demand_r_t / U_const and U_non_idle_r_t_sum > 0:
        return E_road_idx_r_nonidle_t, U_value[E_road_idx_r_nonidle_t] * (
                U_non_idle_r_t_sum - demand_r_t / U_const) / U_non_idle_r_t_sum

def post_opt_X_worker(U_value, PAMoDVehicle, t):
    nodes_t = PAMoDVehicle.filter_node_idx(None, None, t)
    return nodes_t, PAMoDVehicle.A_inflows[nodes_t] * U_value * U_const
