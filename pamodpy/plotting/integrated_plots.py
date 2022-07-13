import os
import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib import rc
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import time
import geopandas as gpd
from geopandas.plotting import plot_polygon_collection

with open(os.path.join('results', 'SF_25',
            "CoordFleet_T96_C23_spring_rate7.7,16.8,30_0.6batt_infra_congestFalse_droptripsFalse",
            "CoordFleet_T96_C23_spring_rate7.7,16.8,30_0.6batt_infra_congestFalse_droptripsFalse.p"), 'rb') as f:
    exp_spring = pickle.load(f)

with open(os.path.join('results', 'SF_25', "CoordFleet_T96_C33_leafS_rate7.7,16.8,50,150_0.6batt_infra_congestFalse_droptripsFalse", "CoordFleet_T96_C33_leafS_rate7.7,16.8,50,150_0.6batt_infra_congestFalse_droptripsFalse.p"), 'rb') as f:
    exp_leafS = pickle.load(f)

with open(os.path.join('results', 'SF_25', "CoordFleet_T96_C33_leafS_baseline_0.6batt_congestroadFalse_droptripsFalse", "CoordFleet_T96_C33_leafS_baseline_0.6batt_congestroadFalse_droptripsFalse.p"), 'rb') as f:
    exp_leafS_baseline = pickle.load(f)

with open(os.path.join('results', 'SF_25', "CoordFleet_T96_C62_model3LRAWD_rate7.7,16.8,50,150_0.6batt_infra_congestFalse_droptripsFalse", "CoordFleet_T96_C62_model3LRAWD_rate7.7,16.8,50,150_0.6batt_infra_congestFalse_droptripsFalse.p"), 'rb') as f:
    exp_model3 = pickle.load(f)

T = 96
time_vec = np.arange(0, 24, 24/T)
# rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
# rc('text', usetex=True)

def total_charge():
    p_elec = 0.12586 * np.ones(T)                                          # Off-peak
    p_elec[int(np.round(16/24 * T)): int(np.round(21/24 * T))] = 0.33474   # Peak [$/kWh]
    p_elec[int(np.round(9/24 * T)):int(np.round(14/24 * T))] = 0.10320    # Super-off-peak

    charge_spring = np.zeros(T)
    charge_leafS = np.zeros(T)
    charge_leafS_baseline = np.zeros(T)
    charge_model3 = np.zeros(T)
    for t in range(T):
        E_charge_idx_t_spring = exp_spring.filter_edge_idx('charge', t=t)
        charge_spring[t] = np.multiply(exp_spring.U[E_charge_idx_t_spring], exp_spring.power_conv[E_charge_idx_t_spring]).sum()

        E_charge_idx_t_leafS = exp_leafS.filter_edge_idx('charge', t=t)
        charge_leafS[t] = np.multiply(exp_leafS.U[E_charge_idx_t_leafS], exp_leafS.power_conv[E_charge_idx_t_leafS]).sum()

        E_charge_idx_t_leafS_baseline = exp_leafS_baseline.filter_edge_idx('charge', t=t)
        charge_leafS_baseline[t] = np.multiply(exp_leafS_baseline.U[E_charge_idx_t_leafS_baseline], exp_leafS_baseline.power_conv[E_charge_idx_t_leafS_baseline]).sum()

        E_charge_idx_t_model3 = exp_model3.filter_edge_idx('charge', t=t)
        charge_model3[t] = np.multiply(exp_model3.U[E_charge_idx_t_model3], exp_model3.power_conv[E_charge_idx_t_model3]).sum()

    print("Max load for {} = {} MW".format('Spring', max(charge_spring) / 1000))
    print("Max load for {} = {} MW".format('Leaf S', max(charge_leafS) / 1000))
    print("Max load for {} = {} MW".format('Leaf S (baseline)', max(charge_leafS_baseline) / 1000))
    print("Max load for {} = {} MW".format('Model 3', max(charge_model3) / 1000))

    E_charge_idx_spring = exp_spring.filter_edge_idx('charge')
    E_charge_idx_leafS = exp_leafS.filter_edge_idx('charge')
    E_charge_idx_leafS_baseline = exp_leafS_baseline.filter_edge_idx('charge')
    E_charge_idx_model3 = exp_model3.filter_edge_idx('charge')
    print("Total energy for {} = {} MWh".format('Spring', np.sum(exp_spring.U[E_charge_idx_spring] @ exp_spring.energy_conv[E_charge_idx_spring]) / 1000))
    print("Total energy for {} = {} MWh".format('Leaf S', np.sum(exp_leafS.U[E_charge_idx_leafS] @ exp_leafS.energy_conv[E_charge_idx_leafS]) / 1000))
    print("Total energy for {} = {} MWh".format('Leaf S (baseline)', np.sum(exp_leafS_baseline.U[E_charge_idx_leafS_baseline] @ exp_leafS_baseline.energy_conv[E_charge_idx_leafS_baseline]) / 1000))
    print("Total energy for {} = {} MWh".format('Model 3', np.sum(exp_model3.U[E_charge_idx_model3] @ exp_model3.energy_conv[E_charge_idx_model3]) / 1000))

    fig, ax1 = plt.subplots(dpi=200, figsize=(12, 8))
    ax2 = ax1.twinx()
    ax1.grid()
    ax2_color = 'tab:red'
    ax2.step(time_vec, p_elec, linestyle='--', dashes=(5,12), color=ax2_color, where='post')
    ax1.step(time_vec, np.array(charge_spring) / 1000, color='green', where='post', label='Spring')
    ax1.step(time_vec, np.array(charge_leafS) / 1000, color='blue', where='post', label='Leaf S')
    ax1.step(time_vec, np.array(charge_model3) / 1000, color='orange', where='post', label='Model 3')

    ax1.tick_params(axis='x', labelsize=24)
    ax1.tick_params(axis='y', labelsize=24)
    ax2.tick_params(axis='y', labelsize=24, labelcolor=ax2_color)
    ax1.set_ylabel("Total charging load [MW]", fontsize=24)
    ax2.set_ylabel("Electricity price [$/kWh]", fontsize=24, color=ax2_color)
    ax2.set_ylim(bottom=0)
    ax1.set_xlabel("Time of day [h]", fontsize=24)
    formatter = FuncFormatter(lambda h, x: time.strftime('%H:%M', time.gmtime(h * 3600)))
    ax1.xaxis.set_ticks(np.arange(0, 24, 4))
    ax1.xaxis.set_major_formatter(formatter)
    # plt.ticklabel_format(useOffset=False)
    ax1.legend(fontsize=24)
    plt.show()
    fig.savefig(os.path.join('results', 'SF_25', 'total_charging_compare_evs.png'), dpi=fig.dpi)

    fig, ax1 = plt.subplots(dpi=200, figsize=(12, 8))
    ax2 = ax1.twinx()
    ax1.grid()
    ax2_color = 'tab:red'
    ax2.step(time_vec, p_elec, linestyle='--', dashes=(5, 12), color=ax2_color, where='post')
    ax1.step(time_vec, np.array(charge_leafS) / 1000, color='blue', where='post', label='Leaf S')
    ax1.step(time_vec, np.array(charge_leafS_baseline) / 1000, color='purple', where='post', label='Leaf S (baseline)')

    ax1.tick_params(axis='x', labelsize=24)
    ax1.tick_params(axis='y', labelsize=24)
    ax2.tick_params(axis='y', labelsize=24, labelcolor=ax2_color)
    ax1.set_ylabel("Total charging load [MW]", fontsize=24)
    ax2.set_ylabel("Electricity price [$/kWh]", fontsize=24, color=ax2_color)
    ax2.set_ylim(bottom=0)
    ax1.set_xlabel("Time of day [h]", fontsize=24)
    formatter = FuncFormatter(lambda h, x: time.strftime('%H:%M', time.gmtime(h * 3600)))
    ax1.xaxis.set_ticks(np.arange(0, 24, 4))
    ax1.xaxis.set_major_formatter(formatter)
    # plt.ticklabel_format(useOffset=False)
    ax1.legend(fontsize=24) # framealpha=0.9
    plt.show()
    fig.savefig(os.path.join('results', 'SF_25', 'total_charging_compare_baseline.png'), dpi=fig.dpi)

def infra_diff_heatmap():
    SF_map = gpd.read_file(exp_leafS.shp_file_path)
    SF_map = SF_map.set_index('name')
    SF_map.index = SF_map.index.astype(int)
    SF_map = SF_map.sort_index()

    infra_cap_leafS = np.zeros(len(exp_leafS.locations_excl_passthrough))
    for evse_idx, evse in enumerate(exp_leafS.EVSEs):
        data = exp_leafS.UMax_charge[:, evse_idx] * evse.rate
        infra_cap_leafS += data
    infra_cap_leafS = np.append(infra_cap_leafS, [0, 0, 0])

    infra_cap_leafS_baseline = np.zeros(len(exp_leafS_baseline.locations_excl_passthrough))
    for evse_idx, evse in enumerate(exp_leafS_baseline.EVSEs):
        data = exp_leafS_baseline.UMax_charge[:, evse_idx] * evse.rate
        infra_cap_leafS_baseline += data
    infra_cap_leafS_baseline = np.append(infra_cap_leafS_baseline, [0, 0, 0])

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
        SF_map.loc[v, 'cluster'] = k
    SF_map = SF_map.dissolve(by='cluster')
    SF_map['infra_cap'] = (infra_cap_leafS - infra_cap_leafS_baseline) / 1000

    fig, ax = plt.subplots(figsize=(12, 8))
    SF_map.plot(ax=ax, column='infra_cap', norm=TwoSlopeNorm(0, vmin=min(SF_map['infra_cap']),
                    vmax=max(SF_map['infra_cap'])), cmap=plt.cm.get_cmap('RdBu_r'), legend=True, edgecolor='black')
    plot_polygon_collection(ax, SF_map['geometry'], values=SF_map['infra_cap'], norm=TwoSlopeNorm(0, vmin=min(SF_map['infra_cap']),
                    vmax=max(SF_map['infra_cap'])), cmap=plt.cm.get_cmap('RdBu_r'),
                            edgecolor='black')
    SF_map.apply(lambda x: ax.annotate(s=int(x.name), xy=x.geometry.centroid.coords[0], ha='center', fontsize=15), axis=1)

    cb_ax = fig.axes[1]
    cb_ax.tick_params(labelsize=20)
    plt.title('Installed capacity difference (optimized - baseline) [MW]', fontsize=20)

    plt.xlim((-122.525, -122.35))
    plt.ylim((37.7, 37.850))
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.show()
    fig.savefig(os.path.join('results', 'SF_25', 'heatmap_infra_diff_cap.png'), dpi=200)

def infra_cap_rating():
    fig, ax = plt.subplots(figsize=(12, 10), dpi=200)
    bottom = np.zeros(4)
    x = ['Spring', 'Leaf S', 'Model 3', 'Baseline']

    cap7_7_rating = np.zeros(4)
    rating = 7.7
    cap7_7_rating[0] = exp_spring.UMax_charge[:, 0].sum() * rating
    cap7_7_rating[1] = exp_leafS.UMax_charge[:, 0].sum() * rating
    cap7_7_rating[2] = exp_model3.UMax_charge[:, 0].sum() * rating
    cap7_7_rating[3] = exp_leafS_baseline.UMax_charge[:, 0].sum() * rating
    cap7_7_rating /= 1000
    plt.bar(x, cap7_7_rating, label='{} kW'.format(rating))
    bottom += cap7_7_rating

    cap16_8_rating = np.zeros(4)
    rating = 16.8
    cap16_8_rating[0] = exp_spring.UMax_charge[:, 1].sum() * rating
    cap16_8_rating[1] = exp_leafS.UMax_charge[:, 1].sum() * rating
    cap16_8_rating[2] = exp_model3.UMax_charge[:, 1].sum() * rating
    cap16_8_rating /= 1000
    plt.bar(x, cap16_8_rating, bottom=bottom, label='{} kW'.format(rating))
    bottom += cap16_8_rating

    cap50_rating = np.zeros(4)
    rating = 50.0
    cap50_rating[1] = exp_leafS.UMax_charge[:, 2].sum() * rating
    cap50_rating[2] = exp_model3.UMax_charge[:, 2].sum() * rating
    cap50_rating[3] = exp_leafS_baseline.UMax_charge[:, 1].sum() * rating
    cap50_rating /= 1000
    plt.bar(x, cap50_rating, bottom=bottom, label='{} kW'.format(rating))
    bottom += cap50_rating

    cap150_rating = np.zeros(4)
    rating = 150.0
    cap150_rating[1] = exp_leafS.UMax_charge[:, 3].sum() * rating
    cap150_rating[2] = exp_model3.UMax_charge[:, 3].sum() * rating
    cap150_rating /= 1000
    plt.bar(x, cap150_rating, bottom=bottom, label='{} kW'.format(rating))

    plt.legend(fontsize=24)
    plt.ylabel("Installed charging station capacity [MW]", fontsize=24)
    ax.tick_params(axis='x', labelsize=24)
    ax.tick_params(axis='y', labelsize=24)
    plt.show()
    fig.savefig(os.path.join('results', 'SF_25', 'infra_cap_rating.png'), dpi=200)

    print(cap7_7_rating)
    print(cap16_8_rating)
    print(cap50_rating)
    print(cap150_rating)

def infra_cap_loc():
    fig, ax = plt.subplots(figsize=(12, 10), dpi=200)
    plt.grid()

    cap_spring = np.zeros(len(exp_spring.locations_excl_passthrough))
    cap_leafS = np.zeros(len(exp_leafS.locations_excl_passthrough))
    cap_model3 = np.zeros(len(exp_model3.locations_excl_passthrough))
    cap_leafS_baseline = np.zeros(len(exp_leafS_baseline.locations_excl_passthrough))

    for evse_idx, evse in enumerate(exp_spring.EVSEs):
        cap_spring += exp_spring.UMax_charge[:, evse_idx] * evse.rate
    for evse_idx, evse in enumerate(exp_leafS.EVSEs):
        cap_leafS += exp_leafS.UMax_charge[:, evse_idx] * evse.rate
    for evse_idx, evse in enumerate(exp_model3.EVSEs):
        cap_model3 += exp_model3.UMax_charge[:, evse_idx] * evse.rate
    for evse_idx, evse in enumerate(exp_leafS_baseline.EVSEs):
        cap_leafS_baseline += exp_leafS_baseline.UMax_charge[:, evse_idx] * evse.rate

    cap_spring.sort()
    cap_leafS.sort()
    cap_model3.sort()
    cap_leafS_baseline.sort()
    plt.plot(np.arange(1, 26), np.cumsum(np.flip(cap_spring / cap_spring.sum())), '.-', color='green', label='Spring')
    plt.plot(np.arange(1, 26), np.cumsum(np.flip(cap_leafS / cap_leafS.sum())), '.-', color='blue', label='Leaf S')
    plt.plot(np.arange(1, 26), np.cumsum(np.flip(cap_model3 / cap_model3.sum())), '.-', color='orange', label='Model 3')
    plt.plot(np.arange(1, 26), np.cumsum(np.flip(cap_leafS_baseline / cap_leafS_baseline.sum())), '.-', color='purple', label='Baseline')

    print(np.cumsum(np.flip(cap_spring / cap_spring.sum())))
    print(np.cumsum(np.flip(cap_leafS / cap_leafS.sum())))
    print(np.cumsum(np.flip(cap_model3 / cap_model3.sum())))
    print(np.cumsum(np.flip(cap_leafS_baseline / cap_leafS_baseline.sum())))

    plt.legend(fontsize=24)
    plt.xlabel("Locations, sorted by highest \nto lowest installed capacity", fontsize=24)
    plt.ylabel("Cumulative fraction of \ninstalled charging capacity [-]", fontsize=24)
    ax.tick_params(axis='x', labelsize=24)
    ax.tick_params(axis='y', labelsize=24)
    plt.show()
    fig.savefig(os.path.join('results', 'SF_25', 'infra_cap_location.png'), dpi=200)

# total_charge()
infra_cap_rating()
# infra_cap_loc()
# infra_diff_heatmap()
