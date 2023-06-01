import os

import matplotlib.pyplot as plt
import matplotlib.animation
# from matplotlib import rc
import numpy as np
import geopandas as gpd
from geopandas.plotting import plot_polygon_collection

from .plot_code import num_veh_each_l, num_veh_each_c, avg_soc, charging_load_each_l, charging_load_total
from .plot_code import vehicle_status, travel_demand, location_timeseries, infra, infra_power
from .plot_code import charging_power, charging_power_time, heatmaps

def plot_PAMoDFleet_results(experiment, startT=None, endT=None, power_matrix_list=None):
    if startT is None:
        startT = experiment.startT
    if endT is None:
        endT = experiment.endT
    if power_matrix_list is None:
        power_matrix_list = experiment.power_matrix_list

    time_vec = np.arange(startT * experiment.deltaT, endT * experiment.deltaT, experiment.deltaT)

    travel_demand(time_vec, experiment)

    for vehicle_idx, PAMoDVehicle in enumerate(experiment.PAMoDVehicles):
        # rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
        # rc('text', usetex=True)

        print("U_rebal distance costs = {}".format((experiment.U_rebal_list[vehicle_idx] @ PAMoDVehicle.Dist) * experiment.p_travel))
        if not os.path.exists(os.path.join(experiment.results_path, experiment.Vehicles[vehicle_idx].name)):
            os.mkdir(os.path.join(experiment.results_path, experiment.Vehicles[vehicle_idx].name))

        num_veh_each_l(time_vec, startT, endT, experiment, vehicle_idx)
        num_veh_each_c(time_vec, startT, endT, experiment, vehicle_idx)
        avg_soc(time_vec, startT, endT, experiment, vehicle_idx)
        charging_load_each_l(time_vec, startT, endT, experiment, vehicle_idx)
        charging_load_total(time_vec, startT, endT, experiment, vehicle_idx)

        for l in experiment.locations:
            location_timeseries(l, time_vec, startT, endT, experiment, vehicle_idx)

        charging_power(experiment, vehicle_idx)
        charging_power_time(time_vec, experiment, vehicle_idx)

        heatmaps(startT, endT, experiment, power_matrix_list, vehicle_idx)

        print("Generating vehicle status plot...")
        vehicle_status(time_vec, startT, endT, experiment, vehicle_idx)

    if experiment.optimize_infra or experiment.use_baseline_charge_stations:
        infra(experiment)
        infra_power(experiment)

def plot_animation(experiment, node_type):
    print("Creating {} animation for {}".format(node_type, experiment.name))

    for vehicle_idx, PAMoDVehicle in enumerate(experiment.PAMoDVehicles):
        fig, ax = plt.subplots(figsize=(12, 8))
        data = np.zeros((experiment.L, experiment.T))
        if node_type == 'numVeh':
            label1 = "\nNumber of Vehicles ({})".format(experiment.Vehicles[vehicle_idx].model)
            for t_idx, t in enumerate(range(experiment.startT, experiment.endT)):
                for i, l in enumerate(experiment.locations):
                    nodes_l_t = PAMoDVehicle.filter_node_idx(l, None, t)
                    data[i, t_idx] = np.sum(experiment.X_list[vehicle_idx][nodes_l_t])
        elif node_type == "chargingPower":
            if PAMoDVehicle.Vehicle.powertrain != 'electric':
                return
            label1 = "\nCharging Power [kW] ({})".format(experiment.Vehicles[vehicle_idx].model)
            for t_idx, t in enumerate(range(experiment.startT, experiment.endT)):
                for i, l in enumerate(experiment.locations):
                    E_charge_idx_l_t = PAMoDVehicle.filter_edge_idx('charge', O=l, D=l, t=t)
                    data[i, t_idx] = np.sum(np.multiply(experiment.U_list[vehicle_idx][E_charge_idx_l_t], experiment.PAMoDVehicles[vehicle_idx].power_conv[E_charge_idx_l_t]))
        elif node_type == "demandArr": # TODO for specific Vehicle
            label1 = "\nNumber of Vehicles"
            for t_idx, t in enumerate(range(experiment.startT, experiment.endT)):
                hour = int(np.floor((t * experiment.deltaT) % (24 / experiment.deltaT)))
                for i, l in enumerate(experiment.locations):
                    data[i, t_idx] = experiment.od_matrix[:, i, hour].sum()
        elif node_type == "demandDep": # TODO for specific Vehicle
            label1 = "\nNumber of Vehicles"
            for t_idx, t in enumerate(range(experiment.startT, experiment.endT)):
                hour = int(np.floor((t * experiment.deltaT) % (24 / experiment.deltaT)))
                for i, l in enumerate(experiment.locations):
                    data[i, t_idx] = experiment.od_matrix[i, :, hour].sum()
        elif node_type == "demandTot": # TODO for specific Vehicle
            label1 = "\nNumber of Vehicles"
            for t_idx, t in enumerate(range(experiment.startT, experiment.endT)):
                hour = int(np.floor((t * experiment.deltaT) % (24 / experiment.deltaT)))
                for i, l in enumerate(experiment.locations):
                    data[i, t_idx] = experiment.od_matrix[:, i, hour].sum() + experiment.od_matrix[i, :, hour].sum()
        elif node_type == "rebal":
            label1 = "\nNumber of Vehicles ({})".format(experiment.Vehicles[vehicle_idx].model)
            for t_idx, t in enumerate(range(experiment.startT, experiment.endT)):
                for i, l in enumerate(experiment.locations):
                    for j, l2 in enumerate(experiment.locations):
                        E_road_idx_r_nonidle_t = PAMoDVehicle.filter_edge_idx('road', l2, l, idle=False, t=t) # TODO parallelize
                        data[i, t_idx] += np.sum(experiment.U_rebal_list[vehicle_idx][E_road_idx_r_nonidle_t])
        elif node_type == "idle":
            label1 = "\nNumber of Vehicles ({})".format(experiment.Vehicles[vehicle_idx].model)
            for t_idx, t in enumerate(range(experiment.startT, experiment.endT)):
                for i, l in enumerate(experiment.locations):
                    E_roads_idle_l_t = PAMoDVehicle.filter_edge_idx('road', O=l, D=l, t=t, idle=True) # TODO parallelize
                    data[i, t_idx] = np.sum(experiment.U_list[vehicle_idx][E_roads_idle_l_t])
        else:
            print('"{}" is not a valid node_type option'.format(node_type))
            return

        node_min = np.min(data)
        node_max = np.max(data)

        map_heatmap = gpd.read_file(experiment.shp_file_path)
        if experiment.region == "SF_190" or experiment.region == "SF_25":
            map_heatmap = map_heatmap.set_index('name')
            map_heatmap.index = map_heatmap.index.astype(int)
            map_heatmap = map_heatmap.sort_index()
        elif experiment.region == "SF_5":
            map_heatmap = map_heatmap.set_index('id')
            map_heatmap.index = map_heatmap.index.astype(int)
            map_heatmap = map_heatmap.sort_index()

        if experiment.region == "SF_25":
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

        map_heatmap['values'] = 0
        ax1 = map_heatmap.plot(ax=ax, column='values', cmap=plt.cm.get_cmap('OrRd'), legend=True, edgecolor='black',
                          vmax=node_max, vmin=node_min)
        fig1 = ax1.figure
        plot_polygon_collection(ax, map_heatmap['geometry'], values=map_heatmap['values'], cmap=plt.cm.get_cmap('OrRd'),
                                edgecolor='black')
        cb_ax = fig1.axes[1]
        cb_ax.tick_params(labelsize=20)
        cb_ax.set_label(label1)

        if experiment.region in ["SF_190", "SF_25", "SF_5"]:
            plt.xlim((-122.525, -122.35))
            plt.ylim((37.7, 37.850))
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()

        def update(t, node_min, node_max):
            ax.clear()
            ax.set_axis_off()

            map_heatmap['values'] = data[:, t]

            plot_polygon_collection(ax, map_heatmap['geometry'], values=map_heatmap['values'], cmap=plt.cm.get_cmap('OrRd'),
                                    edgecolor='black', vmin=node_min, vmax=node_max)

            ax.set_title("t = {} (hour of day, sum = {})".format(t * experiment.deltaT, map_heatmap['values'].sum()))

            if experiment.region in ["SF_190", "SF_25", "SF_5"]:
                plt.xlim((-122.525, -122.35))
                plt.ylim((37.7, 37.850))
            plt.xticks([])
            plt.yticks([])
            plt.tight_layout()

        ani = matplotlib.animation.FuncAnimation(fig, update, frames=experiment.T,
                                                 fargs=(node_min, node_max), repeat=False)
        Writer = matplotlib.animation.writers['ffmpeg']
        writer = Writer(fps=int(np.round(1 / experiment.deltaT)))
        print("Saving animiation...")
        ani.save(os.path.join(experiment.results_path, experiment.Vehicles[vehicle_idx].name, '{}_{}.mp4'.format(experiment.name, node_type)), writer=writer, dpi=100)
        print("Done.")