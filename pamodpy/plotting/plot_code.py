import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FuncFormatter
import time
import matplotlib.animation
import matplotlib.colors as mpl_colors
# from matplotlib import rc
import numpy as np
import pandas as pd
import os
import itertools
import geopandas as gpd
from geopandas.plotting import plot_polygon_collection
import pathos.multiprocessing as pmp
from itertools import repeat

def num_veh_each_l(time_vec, startT, endT, experiment, vehicle_idx):
    fig = plt.figure(dpi=200, figsize=(12,8))
    plt.grid()
    x = np.tile(time_vec[:-1], len(experiment.locations))
    y = np.zeros_like(x)
    location_matrix = np.zeros((len(experiment.locations), experiment.T))
    for l_idx, l in enumerate(experiment.locations):
        num_veh_l = []
        for t_idx, t in enumerate(range(startT, endT)):
            nodes_l_t = experiment.PAMoDVehicles[vehicle_idx].filter_node_idx(l, None, t)
            num_veh_l.append(np.sum(experiment.X_list[vehicle_idx][nodes_l_t]))
        plt.plot(time_vec, num_veh_l, label='{}'.format(l))
        location_matrix[l_idx, :] = num_veh_l
    plt.legend()
    # plt.hist2d(x, y, bins=40, cmap=plt.cm.BuGn_r)
    # plt.title("Number of Vehicles at each Location")
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.ylabel("Number of Vehicles", fontsize=24)
    plt.xlabel("Time [hr]", fontsize=24)
    # plt.show()
    fig.savefig(os.path.join(experiment.results_path, experiment.Vehicles[vehicle_idx].name, 'num_veh_each_l.png'), dpi=fig.dpi)

def num_veh_each_c(time_vec, startT, endT, experiment, vehicle_idx):
    if experiment.Vehicles[vehicle_idx].powertrain != 'electric':
        return
    fig = plt.figure(dpi=200, figsize=(12,8))
    plt.grid()
    C = experiment.PAMoDVehicles[vehicle_idx].C
    for c in range(C):
        num_veh_c = []
        for t_idx, t in enumerate(range(startT, endT)):
            nodes_c_t = experiment.PAMoDVehicles[vehicle_idx].filter_node_idx(None, c, t)
            num_veh_c.append(np.sum(experiment.X_list[vehicle_idx][nodes_c_t]) * (c / (C - 1)))
        if c != C - 1:
            plt.plot(time_vec, num_veh_c, label=c)
        else:
            plt.plot(time_vec, num_veh_c, 'r*', label=c)
    # plt.title("Number of Vehicles at each SOC level")
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.ylabel("Number of Vehicles")
    plt.xlabel("Time [hr]")
    plt.legend()
    # plt.show()
    fig.savefig(os.path.join(experiment.results_path, experiment.Vehicles[vehicle_idx].name, 'num_veh_each_c.png'), dpi=fig.dpi)

def avg_soc(time_vec, startT, endT, experiment, vehicle_idx):
    if experiment.Vehicles[vehicle_idx].powertrain != 'electric':
        return
    fig = plt.figure(dpi=200, figsize=(12,8))
    plt.grid()
    C = experiment.PAMoDVehicles[vehicle_idx].C
    weighted_SOC = np.zeros((C, len(time_vec)))
    for c in range(C):
        for t_idx, t in enumerate(range(startT, endT)):
            nodes_c_t = experiment.PAMoDVehicles[vehicle_idx].filter_node_idx(None, c, t)
            weighted_SOC[c, t_idx] = np.sum(experiment.X_list[vehicle_idx][nodes_c_t]) * (c / (C - 1))
    avg_SOC = np.sum(weighted_SOC, axis=0) / experiment.fleet_sizes[vehicle_idx]
    plt.plot(time_vec, avg_SOC)
    # plt.title("Average SOC, entire fleet")
    y_formatter = ScalarFormatter(useOffset=False)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(y_formatter)
    plt.ylim(0, 1)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.ylabel("SOC [1]", fontsize=24)
    plt.xlabel("Time [hr]", fontsize=24)
    # plt.show()
    fig.savefig(os.path.join(experiment.results_path, experiment.Vehicles[vehicle_idx].name, 'avg_soc.png'), dpi=fig.dpi)

def charging_load_each_l(time_vec, startT, endT, experiment, vehicle_idx):
    if experiment.Vehicles[vehicle_idx].powertrain != 'electric':
        return
    fig = plt.figure(dpi=200, figsize=(12,8))
    plt.grid()
    x = np.tile(time_vec[:-1], len(experiment.locations_excl_passthrough))
    y = np.zeros_like(x)
    for l_idx, l in enumerate(experiment.locations_excl_passthrough):
        charge_l_arr = []
        for t in range(startT, endT - 1):
            E_charge_idx_l_t = experiment.PAMoDVehicles[vehicle_idx].filter_edge_idx('charge', l, l, t=t)
            charge_l_arr.append(np.sum(np.multiply(experiment.U_list[vehicle_idx][E_charge_idx_l_t], experiment.PAMoDVehicles[vehicle_idx].power_conv[E_charge_idx_l_t])))
        # if max(charge_l_arr) > 5000:
        #     plt.plot(time_vec[:-1], np.array(charge_l_arr), '*-', label=l)
        # else:
        #     plt.plot(time_vec[:-1], np.array(charge_l_arr))
        # plt.plot(time_vec[:-1], np.array(charge_l_arr))
        y[l_idx*len(time_vec[:-1]):(l_idx + 1)*len(time_vec[:-1])] = charge_l_arr
    plt.hist2d(x, y)
    # plt.legend()
    # plt.title("Charging at each location")
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.ylabel("Power [kW]", fontsize=24)
    plt.xlabel("Time [hr]", fontsize=24)
    # plt.show()
    fig.savefig(os.path.join(experiment.results_path, experiment.Vehicles[vehicle_idx].name, 'charging_load_each_l.png'), dpi=fig.dpi)

def charging_load_total(time_vec, startT, endT, experiment, vehicle_idx, top_lim=None):
    if experiment.Vehicles[vehicle_idx].powertrain != 'electric':
        return
    fig = plt.figure(dpi=200, figsize=(12, 8))
    plt.grid()
    charge_arr = []
    for t in range(startT, endT):
        E_charge_idx_t = experiment.PAMoDVehicles[vehicle_idx].filter_edge_idx('charge', t=t)
        charge_arr.append(
            np.sum(np.multiply(experiment.U_list[vehicle_idx][E_charge_idx_t], experiment.PAMoDVehicles[vehicle_idx].power_conv[E_charge_idx_t])))
    print(max(charge_arr))
    plt.step(time_vec, np.array(charge_arr) / 1000, where='post')
    # plt.title("Charging summed across all locations")
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.ylabel("Power [MW]", fontsize=24)
    plt.xlabel("Time [hr]", fontsize=24)
    if top_lim is not None:
        plt.ylim(top=top_lim)
    ax = plt.gca()
    ax.ticklabel_format(useOffset=False)
    # plt.show()
    fig.savefig(os.path.join(experiment.results_path, experiment.Vehicles[vehicle_idx].name, 'charging_load_total.png'), dpi=fig.dpi)

def vehicle_status(time_vec, startT, endT, experiment, vehicle_idx, top_lim=None):
    def charging_arr(t_idx, t, E_charge_idx_t):
        idx_gen = (idx for idx in E_charge_idx_t)

        with pmp.ThreadingPool() as p:
            outputs = p.map(charging_arr_worker,
                            idx_gen, repeat(experiment), repeat(vehicle_idx))
        return outputs

    def idle_arr():
        t_idx_t_gen = ((t_idx, t) for t_idx, t in enumerate(range(startT, endT)))

        with pmp.ThreadingPool() as p:
            outputs = p.map(idle_arr_worker,
                            t_idx_t_gen, repeat(experiment), repeat(vehicle_idx))
        return outputs

    def passenger_rebalancing_arrs():
        t_idx_t_x_gen = ((t_idx, t, x) for t_idx, t in enumerate(range(startT, endT))
                         for x in experiment.road_arcs)

        with pmp.ThreadingPool() as p:
            outputs = p.map(passenger_rebalancing_arrs_worker,
                            t_idx_t_x_gen, repeat(experiment), repeat(vehicle_idx))
        return outputs

    charging = np.zeros(experiment.T)
    idle = np.zeros(experiment.T)
    passenger = np.zeros(experiment.T)
    rebalance = np.zeros(experiment.T)

    if experiment.Vehicles[vehicle_idx].powertrain == 'electric':
        for t_idx, t in enumerate(range(startT, endT)):
            E_charge_idx_t = experiment.PAMoDVehicles[vehicle_idx].filter_edge_idx('charge', t=t)
            outputs = charging_arr(t_idx, t, E_charge_idx_t)
            for output in outputs:
                charging[t_idx:t_idx + output[0]] += output[1]
        del outputs

    outputs = idle_arr()
    for output in outputs:
        idle[output[0]] += output[1]
    del outputs

    outputs = passenger_rebalancing_arrs()
    for output in outputs:
        if output is not None:
            passenger[output[0]:output[0] + output[1]] += output[2]
            rebalance[output[0]:output[0] + output[1]] += output[3]
    del outputs

    vehicle_status = pd.DataFrame({"Charging": np.maximum(charging, 0),
                                   "Passenger-carrying": np.maximum(passenger, 0),
                                   "Rebalancing": np.maximum(rebalance, 0),
                                   "Idle": np.maximum(idle, 0)}, index=time_vec)
    # ax = vehicle_status.plot.area(drawstyle='steps')
    # fig = ax.get_figure()
    fig, ax = plt.subplots(dpi=200)
    plt.stackplot(vehicle_status.index, vehicle_status.T, labels=vehicle_status.columns, step='post')
    fig.set_size_inches(12, 8)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylabel("Number of vehicles [-]", fontsize=20)
    plt.xlabel("Time of day [h]", fontsize=20)
    formatter = FuncFormatter(lambda h, x: time.strftime('%H:%M', time.gmtime(h * 3600)))
    ax.xaxis.set_ticks(np.arange(0, 24, 4))
    ax.xaxis.set_major_formatter(formatter)
    ax.ticklabel_format(axis='y', style='sci')
    # plt.title("Fleet Distribution by Vehicle Status")
    if top_lim is not None:
        plt.ylim(top=top_lim)
    plt.legend(fontsize=20)
    # plt.show()
    fig.savefig(os.path.join(experiment.results_path, experiment.Vehicles[vehicle_idx].name, 'vehicle_status.png'), dpi=200)

def charging_arr_worker(idx, experiment, vehicle_idx):
    dur_deltaTs = experiment.round_time(experiment.PAMoDVehicles[vehicle_idx].Dur[idx], min_val=1)
    return dur_deltaTs, experiment.U_list[vehicle_idx][idx]

def idle_arr_worker(t_idx_t, experiment, vehicle_idx):
    t_idx, t = t_idx_t
    E_roads_idle_t = experiment.PAMoDVehicles[vehicle_idx].filter_edge_idx('road', idle=True, t=t)
    return t_idx, np.sum(experiment.U_list[vehicle_idx][E_roads_idle_t])

def passenger_rebalancing_arrs_worker(t_idx_t_x, experiment, vehicle_idx):
    t_idx, t, x = t_idx_t_x
    O = x[0]
    D = x[1]
    E_road_idx_r_nonidle_t = experiment.PAMoDVehicles[vehicle_idx].filter_edge_idx('road', O, D, idle=False, t=t)
    if len(E_road_idx_r_nonidle_t) == 0:
        return None
    assert np.all(experiment.PAMoDVehicles[vehicle_idx].Dur[E_road_idx_r_nonidle_t] == experiment.PAMoDVehicles[vehicle_idx].Dur[E_road_idx_r_nonidle_t[0]])
    dur_deltaTs = experiment.round_time(experiment.PAMoDVehicles[vehicle_idx].Dur[E_road_idx_r_nonidle_t[0]], min_val=1)
    return t_idx, dur_deltaTs, np.sum(experiment.U_trip_charge_idle_list[vehicle_idx][E_road_idx_r_nonidle_t]), np.sum(experiment.U_rebal_list[vehicle_idx][E_road_idx_r_nonidle_t])

def travel_demand(time_vec, experiment):
    fig = plt.figure(dpi=200, figsize=(12, 8))
    travel_demand_arr = np.zeros(len(time_vec))
    num_deltaTs = int(np.round(1 / experiment.deltaT))
    for hour in range(24):
        travel_demand_arr[(hour)*num_deltaTs:(hour + 1)*num_deltaTs] = experiment.od_matrix[:, :, hour].sum() * experiment.deltaT
    plt.plot(time_vec, travel_demand_arr)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.ylabel("Total Travel Demand Flow [# / {}min]".format(experiment.deltaT * 60), fontsize=24)
    plt.xlabel("Time [hr]", fontsize=24)
    # plt.show()
    fig.savefig(os.path.join(experiment.results_path, 'travel_demand.png'), dpi=fig.dpi)

def location_timeseries(l, time_vec, startT, endT, experiment, vehicle_idx):
    fig, ax1 = plt.subplots()
    fig.dpi = 200
    fig.figsize = (12, 8)
    ax1.grid()
    charging = np.zeros(experiment.T)
    charging_150 = np.zeros(experiment.T)
    idle = np.zeros(experiment.T)
    passenger_arr = np.zeros(experiment.T)
    passenger_dep = np.zeros(experiment.T)
    rebalance_arr = np.zeros(experiment.T)
    rebalance_dep = np.zeros(experiment.T)

    for t_idx, t in enumerate(range(startT, endT)):
        if experiment.Vehicles[vehicle_idx].powertrain == 'electric':
            E_charge_idx_l_t = experiment.PAMoDVehicles[vehicle_idx].filter_edge_idx('charge', O=l, D=l, t=t)
            charging[t_idx] += np.sum(experiment.U_list[vehicle_idx][E_charge_idx_l_t])
            E_charge_idx_l_t_150 = experiment.PAMoDVehicles[vehicle_idx].filter_edge_idx('charge', O=l, D=l, t=t, power_grid=(50, 150))
            charging_150[t_idx] += np.sum(experiment.U_list[vehicle_idx][E_charge_idx_l_t_150])
        E_roads_idle_l_t = experiment.PAMoDVehicles[vehicle_idx].filter_edge_idx('road', O=l, D=l, idle=True, t=t)
        idle[t_idx] += np.sum(experiment.U_list[vehicle_idx][E_roads_idle_l_t])

        E_road_idx_arr_l_t = experiment.PAMoDVehicles[vehicle_idx].filter_edge_idx('road', D=l, idle=False, t=t)
        if len(E_road_idx_arr_l_t) != 0:
            passenger_arr[t_idx] += np.sum(experiment.U_trip_charge_idle_list[vehicle_idx][E_road_idx_arr_l_t])
            rebalance_arr[t_idx] += np.sum(experiment.U_rebal_list[vehicle_idx][E_road_idx_arr_l_t])
        E_road_idx_dep_l_t = experiment.PAMoDVehicles[vehicle_idx].filter_edge_idx('road', O=l, idle=False, t=t)
        if len(E_road_idx_dep_l_t) != 0:
            passenger_dep[t_idx] += np.sum(experiment.U_trip_charge_idle_list[vehicle_idx][E_road_idx_dep_l_t])
            rebalance_dep[t_idx] += np.sum(experiment.U_rebal_list[vehicle_idx][E_road_idx_dep_l_t])

    if experiment.Vehicles[vehicle_idx].powertrain == 'electric':
        C = experiment.PAMoDVehicles[vehicle_idx].C
        weighted_SOC = np.zeros((C, len(time_vec)))
        num_veh = np.zeros(len(time_vec))
        for c in range(C):
            for t_idx, t in enumerate(range(startT, endT)):
                nodes_l_c_t = experiment.PAMoDVehicles[vehicle_idx].filter_node_idx(l, c, t)
                weighted_SOC[c, t_idx] = np.sum(experiment.X_list[vehicle_idx][nodes_l_c_t]) * (c / (C - 1))
                num_veh[t_idx] = np.sum(experiment.X_list[vehicle_idx][nodes_l_c_t])
        avg_SOC = np.divide(np.sum(weighted_SOC, axis=0), num_veh)

    ax1.set_title("Vehicle Activity at Location {}".format(l))
    if experiment.Vehicles[vehicle_idx].powertrain == 'electric':
        ax1.plot(time_vec, charging, label="Charging")
        # ax1.plot(time_vec, charging_150, label="Charging (150kW)")
    ax1.plot(time_vec, idle, label="Idle")
    ax1.plot(time_vec, passenger_arr, label="Trip Arrival")
    ax1.plot(time_vec, passenger_dep, label="Trip Departure")
    ax1.plot(time_vec, rebalance_arr, label="Rebalance Arrival")
    ax1.plot(time_vec, rebalance_dep, label="Rebalance Departure")
    # ax1.tick_params(axis='x', labelsize=24)
    # ax1.tick_params(axis='y', labelsize=24)
    ax1.set_xlabel("Time [hr]")#fontsize=24)
    ax1.set_ylabel("Number of vehicles")#, fontsize=24)
    ax1.legend()

    # ax2 = ax1.twinx()
    # ax2.plot(time_vec, avg_SOC, color='grey')
    # ax2.set_ylabel('Average SOC [1]', color='grey')
    # ax2.tick_params(axis='y', labelcolor='grey')#, labelsize=24)

    # plt.show()
    fig.savefig(os.path.join(experiment.results_path, experiment.Vehicles[vehicle_idx].name, 'location_{}_timeseries.png'.format(l)), dpi=fig.dpi)

    # fig = plt.figure(dpi=200, figsize=(12, 8))
    # plt.grid()
    # plt.plot(time_vec, charging_150)
    # plt.show()
    # fig.savefig(os.path.join(experiment.results_path, 'location_{}_charging150.png'.format(l)), dpi=fig.dpi)

def infra(experiment, top_lim=None):
    fig = plt.figure()
    bottom = np.zeros(len(experiment.locations_excl_passthrough))
    for evse_idx, evse in enumerate(experiment.EVSEs):
        data = experiment.UMax_charge[:, evse_idx]
        if evse_idx == 0:
            plt.bar(experiment.locations_excl_passthrough, data, label=evse.name)
        else:
            plt.bar(experiment.locations_excl_passthrough, data, bottom=bottom, label=evse.name)
        bottom += data
    plt.legend()
    plt.ylabel("Number of plugs")
    if top_lim is not None:
        plt.ylim(top=top_lim)
    # plt.show()
    fig.savefig(os.path.join(experiment.results_path, 'infra.png'), dpi=200)

def infra_power(experiment, top_lim=None):
    fig = plt.figure()
    bottom = np.zeros(len(experiment.locations_excl_passthrough))
    for evse_idx, evse in enumerate(experiment.EVSEs):
        data = experiment.UMax_charge[:, evse_idx] * evse.rate
        if evse_idx == 0:
            plt.bar(experiment.locations_excl_passthrough, data, label=evse.name)
        else:
            plt.bar(experiment.locations_excl_passthrough, data, bottom=bottom, label=evse.name)
        bottom += data
    plt.legend()
    plt.ylabel("Power capacity installed [kW]")
    if top_lim is not None:
        plt.ylim(top=top_lim)
    # plt.show()
    fig.savefig(os.path.join(experiment.results_path, 'infra_power.png'), dpi=200)

    total_installed_capacity = 0.0
    for evse_idx, evse in enumerate(experiment.EVSEs):
        total_installed_capacity += np.sum(experiment.UMax_charge[:, evse_idx] * evse.rate)
    print("Total Installed Capacity = {} kW".format(total_installed_capacity))

def charging_power(experiment, vehicle_idx, top_lim=None):
    if experiment.Vehicles[vehicle_idx].powertrain != 'electric':
        return
    fig = plt.figure()
    first = True
    bottom = np.zeros(len(experiment.locations_excl_passthrough))
    for rate in np.unique([edge[2] for edge in experiment.PAMoDVehicles[vehicle_idx].G.edges(data='power_grid') if edge[2] is not None]):
        data_rate = []
        for l in experiment.locations_excl_passthrough:
            E_charge_idx_l_rate = experiment.PAMoDVehicles[vehicle_idx].filter_edge_idx('charge', l, l, power_grid=(rate - 1 / experiment.deltaT * experiment.deltaC, rate))
            if E_charge_idx_l_rate is not None:
                data_rate.append(np.sum(experiment.U_trip_charge_idle_list[vehicle_idx][E_charge_idx_l_rate]))
            else:
                data_rate.append(0)
        if np.sum(np.array(data_rate)) > 10:
            if first:
                plt.bar(experiment.locations_excl_passthrough, data_rate, label='{} kW'.format(rate))
                first = False
            else:
                plt.bar(experiment.locations_excl_passthrough, data_rate, bottom=bottom, label='{} kW'.format(rate))
                bottom += np.array(data_rate)
    plt.legend()
    plt.ylabel("Number of vehicles charging")
    if top_lim is not None:
        plt.ylim(top=top_lim)
    # plt.show()
    fig.savefig(os.path.join(experiment.results_path, experiment.Vehicles[vehicle_idx].name, 'charging_power.png'), dpi=200)

def charging_power_time(time_vec, experiment, vehicle_idx, top_lim=None):
    if experiment.Vehicles[vehicle_idx].powertrain != 'electric':
        return
    fig = plt.figure()
    for rate in np.unique([edge[2] for edge in experiment.PAMoDVehicles[vehicle_idx].G.edges(data='power_grid') if edge[2] is not None]):
        rate_arr = np.zeros(experiment.T)
        for t_idx, t in enumerate(range(experiment.startT, experiment.endT)):
            E_charge_idx_t_rate = experiment.PAMoDVehicles[vehicle_idx].filter_edge_idx('charge', t=t, power_grid=(rate - 1 / experiment.deltaT * experiment.deltaC, rate))
            if E_charge_idx_t_rate is not None:
                rate_arr[t_idx] = np.sum(experiment.U_trip_charge_idle_list[vehicle_idx][E_charge_idx_t_rate])
        if rate_arr.sum() > 10:
            plt.plot(time_vec, rate_arr, label='{} kW'.format(rate))
    plt.legend()
    plt.ylabel("Number of vehicles charging")
    if top_lim is not None:
        plt.ylim(top=top_lim)
    # plt.show()
    fig.savefig(os.path.join(experiment.results_path, experiment.Vehicles[vehicle_idx].name, 'charging_power_time.png'), dpi=200)

def heatmaps(startT, endT, experiment, power_matrix_list, vehicle_idx):
    if experiment.Vehicles[vehicle_idx].powertrain != 'electric':
        return
    location_matrix = np.zeros((len(experiment.locations), experiment.T))
    for l_idx, l in enumerate(experiment.locations):
        num_veh_l = []
        for t_idx, t in enumerate(range(startT, endT)):
            nodes_l_t = experiment.PAMoDVehicles[vehicle_idx].filter_node_idx(l, None, t)
            num_veh_l.append(np.sum(experiment.X_list[vehicle_idx][nodes_l_t]))
        location_matrix[l_idx, :] = num_veh_l
    # daily_total_power = np.concatenate((np.sum(power_matrix_list[vehicle_idx], axis=1), [0, 0, 0]))
    daily_total_power = np.sum(power_matrix_list[vehicle_idx], axis=1)  #for nyc_manh
    daily_total_location = np.sum(location_matrix[0:experiment.L, :], axis=1)
    hour_timesteps = int(np.round(1/experiment.deltaT))
    hourly_peaks = np.zeros((power_matrix_list[vehicle_idx].shape[0], experiment.T // hour_timesteps))
    for hour_idx, t_idx in enumerate(np.arange(0, experiment.T, hour_timesteps)):
        if t_idx + hour_timesteps <= experiment.T - 1:
            hourly_peaks[:, hour_idx] = np.max(power_matrix_list[vehicle_idx][:, t_idx:t_idx + hour_timesteps], axis=1)

    map_heatmap = gpd.read_file(experiment.shp_file_path)
    if experiment.region == "SF_190" or experiment.region == "SF_25" or experiment.region == "SF2_25":
        map_heatmap = map_heatmap.set_index('name')
    elif experiment.region == "SF_5":
        map_heatmap = map_heatmap.set_index('id')
    elif experiment.region == "NYC_manh":
        pass
    else:
        map_heatmap = None
    # map_heatmap.index = map_heatmap.index.astype(int)
    # map_heatmap = map_heatmap.sort_index()

    charging = daily_total_power / np.sum(daily_total_power)
    charging_vehicles = [np.sum(experiment.U_list[vehicle_idx][experiment.PAMoDVehicles[vehicle_idx].filter_edge_idx('charge', l, l)]) for l in experiment.locations]
    charging_vehicles = charging_vehicles / np.sum(charging_vehicles)
    num_veh = daily_total_location / np.sum(daily_total_location)
    demand_arr = experiment.od_matrix.sum(axis=(0, 2))
    demand_arr = demand_arr / np.sum(demand_arr)
    demand_dep = experiment.od_matrix.sum(axis=(1, 2))# + experiment.od_matrix.sum(axis=(0, 2))
    demand_dep = demand_dep / np.sum(demand_dep)
    infra_cap = np.zeros(len(experiment.locations_excl_passthrough))
    infra_cap_rates = np.zeros((len(experiment.locations_excl_passthrough), len(experiment.EVSEs)))
    for evse_idx, evse in enumerate(experiment.EVSEs):
        data = experiment.UMax_charge[:, evse_idx] * evse.rate
        infra_cap += data
        infra_cap_rates[:, evse_idx] += data
    infra_vmax = np.amax(infra_cap) / 1000
    # infra_cap = np.append(infra_cap, [0, 0, 0])
    # infra_cap_rates = np.concatenate((infra_cap_rates, np.zeros((3, len(experiment.EVSEs)))))

    if experiment.region == "SF_25" or experiment.region == "SF2_25":
        cluster_to_taz = {
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
            25: [148, 149, 151, 153, 154],
            26: [191],
            27: [192],
            28: [193]
        }
        for k, v in cluster_to_taz.items():
            map_heatmap.loc[v, 'cluster'] = k
        map_heatmap = map_heatmap.dissolve(by='cluster')
    map_heatmap['charging'] = charging
    map_heatmap['charging_vehicles'] = charging_vehicles
    map_heatmap['num_veh'] = num_veh
    map_heatmap['demand_dep'] = demand_dep
    map_heatmap['demand_arr'] = demand_arr
    if experiment.optimize_infra or experiment.use_baseline_charge_stations:
        map_heatmap['infra_cap'] = infra_cap / 1000
        for evse_idx, evse in enumerate(experiment.EVSEs):
            map_heatmap['infra_cap_{}'.format(evse.rate)] = infra_cap_rates[:, evse_idx] / 1000

    column_names = ['charging', 'charging_vehicles', 'num_veh', 'demand_dep', 'demand_arr']
    titles = ['Distribution of Daily Total Charging: Power',
              'Distribution of Daily Total Charging: Vehicles',
              'Distribution of Daily Total Location',
              'Distribution of Daily Total Travel Demand (Dep)',
              'Distribution of Daily Total Travel Demand (Arr)']
    filenames = [os.path.join(experiment.Vehicles[vehicle_idx].name, 'heatmap_daily_total_charging.png'),
                 os.path.join(experiment.Vehicles[vehicle_idx].name, 'heatmap_daily_total_charging_vehicles.png'),
                 os.path.join(experiment.Vehicles[vehicle_idx].name, 'heatmap_daily_total_location.png'),
                 os.path.join(experiment.Vehicles[vehicle_idx].name, 'heatmap_travel_demand.png'),
                 os.path.join(experiment.Vehicles[vehicle_idx].name, 'heatmap_travel_demand_arr.png')]

    if experiment.optimize_infra or experiment.use_baseline_charge_stations:
        column_names.extend(['infra_cap'] + ['infra_cap_{}'.format(evse.rate) for evse in experiment.EVSEs])
        titles.extend(['Charging infrastructure capacity [MW]']
                      + ['{} kW charging infrastructure capacity [MW]'.format(evse.rate) for evse in experiment.EVSEs])
        filenames.extend(['heatmap_infra_cap.png']
                         + ['heatmap_infra_cap_{}.png'.format(evse.rate) for evse in experiment.EVSEs])

    for column_name, title, filename in zip(column_names, titles, filenames):
        fig, ax = plt.subplots(figsize=(12, 8))
        ax1 = map_heatmap.plot(ax=ax, column=column_name, cmap=plt.cm.get_cmap('OrRd'), legend=True, edgecolor='black')#, vmin=0, vmax=infra_vmax)
        fig1 = ax1.figure
        plot_polygon_collection(ax, map_heatmap['geometry'], values=map_heatmap[column_name], cmap=plt.cm.get_cmap('OrRd'),
                                edgecolor='black')#, vmin=0, vmax=infra_vmax)
        map_heatmap.apply(lambda x: ax.annotate(int(x.name) + 1, xy=x.geometry.centroid.coords[0], ha='center', fontsize=15), axis=1)
        # plt.xlim((-122.525, -122.35))
        # plt.ylim((37.7, 37.850))
        plt.xticks([])
        plt.yticks([])
        plt.title(title, fontsize=20)
        cb_ax = fig1.axes[1]
        cb_ax.tick_params(labelsize=20)
        plt.tight_layout()
        # plt.show()
        fig.savefig(os.path.join(experiment.results_path, filename), dpi=200)
