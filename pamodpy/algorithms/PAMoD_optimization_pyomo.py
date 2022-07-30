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
import pyomo.environ as pe
from pyomo.contrib import appsi
from pyomo.core.util import sum_product, quicksum
import pathos.multiprocessing as pmp
from scipy import sparse

from ..utils.constants import *

U_const, UMax_const, PMax_const = 1000, 1, 10000

def PAMoD_optimization_pyomo(experiment):
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

    def constr_conservation(A_nodes_t, U):
        A_nodes_t_index_gen = ((A_nodes_t[i, :], np.nonzero(A_nodes_t[i, :])[0].tolist()) for i in range(A_nodes_t.shape[0]))
        with pmp.ThreadingPool() as p:
            outputs = p.map(constr_conservation_worker, A_nodes_t_index_gen, repeat(U), count())
        return outputs

    def constr_boundary(A_out_nodes_start, A_in_nodes_end, U):
        A_out_nodes_start_index_gen = ((A_out_nodes_start[i, :], np.nonzero(A_out_nodes_start[i, :])[0].tolist()) for i in range(A_out_nodes_start.shape[0]))
        A_in_nodes_end_index_gen = ((A_in_nodes_end[i, :], np.nonzero(A_in_nodes_end[i, :])[0].tolist()) for i in range(A_in_nodes_end.shape[0]))
        with pmp.ThreadingPool() as p:
            outputs = p.map(constr_boundary_worker, A_out_nodes_start_index_gen, A_in_nodes_end_index_gen, repeat(U), count())
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

        if experiment.load_opt:
            m = None
            raise NotImplementedError()
        else:
            # Create a new Pyomo model
            m = pe.ConcreteModel()
            experiment.logger.setLevel(logging.INFO)

            # Create Pyomo Sets and Variables
            E_Set_list = []
            m.Lep = pe.RangeSet(0, len(experiment.locations_excl_passthrough) - 1)
            m.RT = pe.RangeSet(0, len(experiment.road_arcs) * experiment.T - 1)
            m.n_EVSEs = pe.RangeSet(0, len(experiment.EVSEs) - 1)

            U_list = []
            for vehicle_idx, PAMoDVehicle in enumerate(experiment.PAMoDVehicles):
                m.add_component('E_{}'.format(vehicle_idx), pe.RangeSet(0, PAMoDVehicle.E - 1))
                E_Set_list.append(getattr(m, 'E_{}'.format(vehicle_idx)))
                m.add_component('U_{}'.format(vehicle_idx), pe.Var(E_Set_list[-1], domain=pe.NonNegativeReals))
                U_list.append(getattr(m, 'U_{}'.format(vehicle_idx)))
            if experiment.optimize_infra:
                m.UMax_charge = pe.Var(m.Lep, m.n_EVSEs, domain=pe.NonNegativeReals)
                if experiment.optimize_infra_mip:
                    m.B = pe.Var(m.Lep, domain=pe.Binary)
            else:
                m.UMax_charge = pe.Param(m.Lep, m.n_EVSEs, initialize=experiment.UMax_charge)
            m.PMax = pe.Var(m.Lep, domain=pe.NonNegativeReals)
            if experiment.drop_trips:
                m.trip_flow = pe.Var(m.RT, domain=pe.NonNegativeReals)
            else:
                m.trip_flow = pe.Param(m.RT, initialize=0, domain=pe.NonNegativeReals)
            fleet_sizes = []
            if experiment.optimize_fleet_size:
                fleet_size_const = 10000
                for vehicle_idx, PAMoDVehicle in enumerate(experiment.PAMoDVehicles):
                    m.add_component('fleet_size_{}'.format(vehicle_idx), pe.Var(domain=pe.NonNegativeReals))
                    fleet_sizes.append(getattr(m, 'fleet_size_{}'.format(vehicle_idx)))
            else:
                fleet_sizes = experiment.fleet_sizes
                fleet_size_const = 1

            experiment.logger.info("-Creating optimization problem")
            tic_start = time.time()

            tic = time.time()
            m.constr_fleet_sizes = pe.ConstraintList()
            for vehicle_idx, PAMoDVehicle in enumerate(experiment.PAMoDVehicles):
                nodes_start = PAMoDVehicle.filter_node_idx(None, None, experiment.startT)
                m.constr_fleet_sizes.add(expr=sum_product(np.ones(len(nodes_start)) * -PAMoDVehicle.A_outflows[nodes_start], U_list[vehicle_idx]) == fleet_sizes[vehicle_idx] * (fleet_size_const / U_const))
                experiment.logger.info("--Created fleet size constraints (elapsed={:.2f})".format(time.time() - tic))

            if experiment.charge_throttle:
                if experiment.optimize_infra or experiment.congestion_constr_charge:
                    tic = time.time()
                    outputs = obj_elec_energy_carbon_and_constr_UMax_charge(U_list, m.UMax_charge)
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
                    outputs = obj_elec_energy_carbon_and_constr_UMax_charge(U_list, m.UMax_charge)
                    elec_energy = 0
                    elec_carbon = 0
                    m.constr_UMax_charge = pe.ConstraintList()
                    for output in sorted(outputs, key=lambda item: item[-1]):
                        elec_energy += output[0]
                        elec_carbon += output[1]
                        if output[2] is not None:
                            m.constr_UMax_charge.add(expr=output[2])
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
                    m.constr_infra_mip = pe.ConstraintList()
                    for lep_idx, l in enumerate(experiment.locations_excl_passthrough):
                        for evse_idx, evse in enumerate(experiment.EVSEs):
                            infra += (m.B[lep_idx, evse_idx] * experiment.p_infra_capital[lep_idx, evse_idx] +
                                      experiment.p_infra_marginal[
                                          lep_idx, evse_idx] * m.UMax_charge[lep_idx, evse_idx] * UMax_const) * (
                                             1 / U_const)
                            m.constr_infra_mip.add(expr=
                                m.UMax_charge[lep_idx, evse_idx] <= M * m.B[lep_idx, evse_idx] * (1 / UMax_const))
                else:
                    for lep_idx, l in enumerate(experiment.locations_excl_passthrough):
                        for evse_idx, evse in enumerate(experiment.EVSEs):
                            infra += (experiment.p_infra_marginal[lep_idx, evse_idx] * m.UMax_charge[
                                lep_idx, evse_idx] * UMax_const) * (
                                             1 / U_const)
                experiment.logger.info(
                    "--Created infra obj term (elapsed={:.2f})".format(
                        time.time() - tic))

            if experiment.charge_throttle:
                tic = time.time()
                m.constr_infra_throttle = pe.ConstraintList()
                outputs = constr_infra(U_list, m.UMax_charge)
                for output in sorted(outputs, key=lambda item: item[-1]):
                    m.constr_infra_throttle.add(expr=output[0])
                del outputs
                experiment.logger.info(
                    "--Created infra constraint (elapsed={:.2f})".format(
                        time.time() - tic))

            tic = time.time()
            outputs = obj_revenue_and_constr_UMax_road(U_list, m.trip_flow)
            revenue = 0
            m.trip_flow_constr1 = pe.ConstraintList()
            m.trip_flow_constr2 = pe.ConstraintList()
            m.UMax_road_constr = pe.ConstraintList()
            for output in sorted(outputs, key=lambda item: item[-1]):
                revenue -= output[0]
                if output[1] is not None:
                    m.trip_flow_constr1.add(expr=output[1])
                if output[2] is not None:
                    m.trip_flow_constr2.add(expr=output[2])
                if experiment.congestion_constr_road:
                    m.UMax_road_constr.add(expr=output[3])
            del outputs
            experiment.logger.info("--Created revenue obj term and UMax_road constraint (elapsed={:.2f})".format(
                time.time() - tic))

            tic = time.time()
            m.constr_flow_conservation = pe.ConstraintList()
            for vehicle_idx, PAMoDVehicle in enumerate(experiment.PAMoDVehicles):
                nodes_t = PAMoDVehicle.filter_node_idx(None, None, np.array(range(experiment.startT + 1, experiment.endT - 1)))
                outputs = constr_conservation((PAMoDVehicle.A[nodes_t]).toarray(), U_list[vehicle_idx])
                for output in sorted(outputs, key=lambda item: item[-1]):
                    m.constr_flow_conservation.add(expr=output[0])
                del outputs

            m.constr_boundary = pe.ConstraintList()
            if experiment.boundary:
                for vehicle_idx, PAMoDVehicle in enumerate(experiment.PAMoDVehicles):
                    nodes_start = PAMoDVehicle.filter_node_idx(None, None, experiment.startT)
                    nodes_end = PAMoDVehicle.filter_node_idx(None, None, experiment.endT - 1)
                    assert np.all(PAMoDVehicle.G_nodes_arr[nodes_start][:, :2] == PAMoDVehicle.G_nodes_arr[nodes_end][:, :2])
                    outputs = constr_boundary((PAMoDVehicle.A_outflows[nodes_start]).toarray(), (PAMoDVehicle.A_inflows[nodes_end]).toarray(), U_list[vehicle_idx])
                    for output in sorted(outputs, key=lambda item: item[-1]):
                        m.constr_boundary.add(expr=output[0])
                    del outputs
            experiment.logger.info("--Created fleet dynamics (elapsed={:.2f})".format(time.time() - tic))

            tic = time.time()
            [elec_demand, outputs] = obj_elec_demand(U_list, m.PMax)
            m.constr_PMax = pe.ConstraintList()
            for output in sorted(outputs, key=lambda item: item[-1]):
                if output[0] is not None:
                    m.constr_PMax.add(expr=output[0])
            del outputs
            dist = quicksum([sum_product(PAMoDVehicle.Dist * experiment.p_travel, U_list[vehicle_idx]) for vehicle_idx, PAMoDVehicle in enumerate(experiment.PAMoDVehicles)])
            experiment.logger.info("--Created elec_demand and dist obj terms (elapsed={:.2f})".format(time.time() - tic))

            gas = 0
            gas_carbon = 0
            for vehicle_idx, PAMoDVehicle in enumerate(experiment.PAMoDVehicles):
                if PAMoDVehicle.Vehicle.powertrain != 'electric':
                    gas += sum_product(U_list[vehicle_idx], PAMoDVehicle.energy_conv * experiment.p_gas)
                    gas_carbon += sum_product(U_list[vehicle_idx], PAMoDVehicle.energy_conv * tons_CO2_per_gal_gas * experiment.p_carbon)
            experiment.logger.info("--Done creating optimization problem. Total time elapsed={:.2f}".format(time.time() - tic_start))

            # Set objective
            fleet_cost = 0
            for vehicle_idx in range(len(experiment.Vehicles)):
                fleet_cost += fleet_sizes[vehicle_idx] * (experiment.Vehicles[vehicle_idx].price * 0.2 + experiment.p_ownership_excl_deprec) * (experiment.T * experiment.deltaT / HOURS_PER_YEAR) * (
                        fleet_size_const / U_const)

            obj = elec_energy + elec_demand + elec_carbon + dist + revenue + fleet_cost + infra + gas + gas_carbon
            m.obj = pe.Objective(expr=obj, sense=pe.minimize)

            # Clean-up
            for PAMoDVehicle in experiment.PAMoDVehicles:
                PAMoDVehicle.remove_incidence_matrices()
            gc.collect()

            # Save
            if experiment.save_opt:
                pass

        # Optimize model
        experiment.logger.info("-Starting optimization:")
        experiment.logger.setLevel(logging.WARNING)
        tic = time.time()
        opt = appsi.solvers.Gurobi()
        if not experiment.optimize_infra_mip:
            opt.gurobi_options['Method'] = 2
            opt.gurobi_options['Crossover'] = 0
        res = opt.solve(m)
        experiment.logger.setLevel(logging.INFO)
        experiment.logger.info("-Finished optimization: (elapsed={:.2f})".format(time.time() - tic))
        experiment.logger.info('Obj: %g' % m.obj())
        for PAMoDVehicle in experiment.PAMoDVehicles:
            PAMoDVehicle.set_incidence_matrices()

        if experiment.load_opt:
            raise NotImplementedError
        else:
            U_value_list = [np.array(list(U.extract_values().values())) for U in U_list]
            PMax_value = np.array(list(m.PMax.extract_values().values()))
            trip_flow_value = np.array(list(m.trip_flow.extract_values().values()))
            if experiment.optimize_fleet_size:
                experiment.fleet_sizes = [float(list(fleet_size.extract_values().values())[0] * fleet_size_const) for fleet_size in fleet_sizes]
            if experiment.optimize_infra:
                experiment.UMax_charge = np.array(list(m.UMax_charge.extract_values().values())).reshape((len(experiment.locations_excl_passthrough), len(experiment.EVSEs))) * UMax_const
                print(experiment.UMax_charge)
                if experiment.optimize_infra_mip:
                    print(np.array(list(m.B.extract_values().values())).reshape((len(experiment.locations_excl_passthrough), len(experiment.EVSEs))))

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
            U_rebal_dist_costs.append((U_rebal @ PAMoDVehicle.Dist) * experiment.p_travel * U_const)
            del outputs
            U_trip_charge_idle = U_value - U_rebal
            U_trip_charge_idle_list.append(U_trip_charge_idle)
        experiment.logger.info("U_rebal distance costs = {}".format(sum(U_rebal_dist_costs)))

        infra_value = 0
        if experiment.optimize_infra:
            UMax_charge_value = np.array(list(m.UMax_charge.extract_values().values())).reshape(
                (len(experiment.locations_excl_passthrough), len(experiment.EVSEs)))
            if experiment.optimize_infra_mip:
                B_value = np.array(list(m.B.extract_values().values())).reshape(
                    (len(experiment.locations_excl_passthrough), len(experiment.EVSEs)))
                infra_value = np.sum(B_value * experiment.p_infra_capital + experiment.p_infra_marginal * UMax_charge_value * UMax_const) * (1 / U_const)
            else:
                infra_value = np.sum(experiment.p_infra_marginal * UMax_charge_value * UMax_const) * (1 / U_const)

        elif experiment.use_baseline_charge_stations:
            infra_value = np.sum(experiment.p_infra_marginal * experiment.UMax_charge) * (1 / U_const)

        infra_value *= U_const
        experiment.logger.info("infra_value = {}".format(infra_value))

        if experiment.drop_trips:
            revenue_final = revenue() * U_const
        else:
            revenue_final = revenue
        if any(Vehicle.powertrain != 'electric' for Vehicle in experiment.Vehicles):
            gas_final = gas() * U_const
            gas_carbon_final = gas_carbon() * U_const
        else:
            gas_final = 0
            gas_carbon_final = 0
        if any(Vehicle.powertrain == 'electric' for Vehicle in experiment.Vehicles):
            elec_energy_final = elec_energy() * U_const
            elec_demand_final = elec_demand() * U_const
            elec_carbon_final = elec_carbon * U_const
        else:
            elec_energy_final = 0
            elec_demand_final = 0
            elec_carbon_final = 0


        return [X_list, np.array(U_value_list) * U_const, np.array(U_trip_charge_idle_list) * U_const, np.array(U_rebal_list) * U_const,
                elec_energy_final,
                elec_demand_final,
                dist() * U_const,
                revenue_final,
                fleet_cost * U_const,
                elec_carbon_final,
                gas_final,
                gas_carbon_final
                ]  # TODO this won't work for load_opt == True

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
                        (sum_product(U_list[vehicle_idx], PAMoDVehicle.energy_conv, index=E_charge_idx_l_eid_t)) *
                        experiment.p_elec[t])
            else:
                l, t, evse_id = l_t_eid
                E_charge_idx_l_eid_t = PAMoDVehicle.filter_edge_idx('charge', l, l, evse_id=evse_id, t=t)
                if len(E_charge_idx_l_eid_t) == 0:
                    invalid += 1
                else:
                    elec_energy_list.append(
                        (sum_product(U_list[vehicle_idx], PAMoDVehicle.energy_conv, index=E_charge_idx_l_eid_t)) *
                        experiment.p_elec[t])
                    elec_carbon_list.append(
                        (sum_product(U_list[vehicle_idx], PAMoDVehicle.energy_conv, index=E_charge_idx_l_eid_t)) *
                        experiment.carbon_intensity_grid[t] * experiment.p_carbon
                    )
                    lep_idx, evse_idx = experiment.get_lep_idx_evse_idx(l, evse_id)
                    UMax_charge_constr_lhs.append((sum_product(U_list[vehicle_idx], index=E_charge_idx_l_eid_t)))

    if invalid < max(n_elec_vehicles, 1):
        elec_energy_term = quicksum(elec_energy_list)
        elec_carbon_term = quicksum(elec_carbon_list)
        if experiment.charge_throttle or n_elec_vehicles == 0:
            UMax_charge_constr = None
        else:
            UMax_charge_constr = (quicksum(UMax_charge_constr_lhs) <= (UMax_charge[lep_idx, evse_idx]) * (
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
            infra_constr_lhs.append(sum_product(U_list[vehicle_idx], index=E_charge_idx_l_t_rates))
    infra_constr = (quicksum(infra_constr_lhs) <= (quicksum(UMax_charge[lep_idx, rating_idx:])) * (UMax_const / U_const))
    return infra_constr, count

def obj_revenue_and_constr_UMax_road_worker(U_list, trip_flow, build, o_d_t, idx, experiment, count):
    O, D, t = o_d_t
    O_idx = experiment.locations.index(O)
    D_idx = experiment.locations.index(D)
    hour = int(np.floor((t * experiment.deltaT) % (24 / experiment.deltaT)))
    if experiment.drop_trips:
        revenue_term = trip_flow[idx] * experiment.revenue_matrix[O_idx, D_idx, hour]
    else:
        revenue_term = 0
    if build:
        if experiment.drop_trips:
            trip_flow_constr1 = (trip_flow[idx] <= experiment.od_matrix[O_idx, D_idx, hour] * experiment.deltaT / U_const)
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
                    trip_flow_constr2_lhs.append(sum_product(U_list[vehicle_idx], index=E_road_idx_r_nonidle_t))
                else:
                    trip_flow_constr2_lhs.append(sum_product(U_list[vehicle_idx], index=E_road_idx_r_nonidle_t))
                if experiment.congestion_constr_road:
                    UMax_road_constr_lhs.append(sum_product(U_list[vehicle_idx], index=E_road_idx_r_nonidle_t))
            else:
                invalid += 1

        if invalid < len(experiment.PAMoDVehicles):
            if experiment.drop_trips:
                trip_flow_constr2 = (quicksum(trip_flow_constr2_lhs) >= trip_flow[idx])
            else:
                trip_flow_constr2 = (quicksum(trip_flow_constr2_lhs) >= experiment.od_matrix[
                        O_idx, D_idx, hour] * experiment.deltaT / U_const)
            if experiment.congestion_constr_road:
                UMax_road_constr = (quicksum(UMax_road_constr_lhs) <= (quicksum(experiment.fleet_sizes)) / U_const) # TODO have actual road congestion
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
            obj_elec_demand_lhs.append(sum_product(U_list[vehicle_idx], PAMoDVehicle.power_conv, index=E_charge_idx_l_t))
        else:
            invalid += 1

    if invalid < max(n_elec_vehicles, 1):
        return quicksum(obj_elec_demand_lhs) <= PMax[l_idx] * (PMax_const / U_const), count
    else:
        return None, count

def constr_conservation_worker(A_node_t_index, U, count):
    A_node_t, index = A_node_t_index
    return sum_product(A_node_t, U, index=index) == 0, count

def constr_boundary_worker(A_out_node_start_index, A_in_node_end_index, U, count):
    A_out_node_start, start_index = A_out_node_start_index
    A_in_node_end, end_index = A_in_node_end_index
    return sum_product(A_in_node_end, U, index=end_index) + sum_product(A_out_node_start, U, index=start_index) == 0, count

def post_opt_U_rebal_worker(U_value, PAMoDVehicle, o_d_t, experiment):
    O, D, t = o_d_t
    O_idx = experiment.locations.index(O)
    D_idx = experiment.locations.index(D)
    hour = int(np.floor((t * experiment.deltaT) % (24 / experiment.deltaT)))
    E_road_idx_r_nonidle_t = PAMoDVehicle.filter_edge_idx('road', O, D, idle=False, t=t)
    if len(E_road_idx_r_nonidle_t) == 0:
        return None
    U_non_idle_r_t_sum = np.sum(U_value[E_road_idx_r_nonidle_t])
    demand_r_t = experiment.od_matrix[O_idx, D_idx, hour] * experiment.deltaT
    if U_non_idle_r_t_sum >= demand_r_t / U_const and U_non_idle_r_t_sum > 0:
        return E_road_idx_r_nonidle_t, U_value[E_road_idx_r_nonidle_t] * (
                U_non_idle_r_t_sum - demand_r_t / U_const) / U_non_idle_r_t_sum

def post_opt_X_worker(U_value, PAMoDVehicle, t):
    nodes_t = PAMoDVehicle.filter_node_idx(None, None, t)
    return nodes_t, PAMoDVehicle.A_inflows[nodes_t] * U_value * U_const
