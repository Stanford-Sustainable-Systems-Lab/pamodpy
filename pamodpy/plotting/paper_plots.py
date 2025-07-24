import ast
import os
import io
import pickle
from pickle import Unpickler
import zipfile

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
import seaborn as sns

from pamodpy.utils.generate_p_elec import generate_p_elec
from pamodpy.utils.constants import generate_carbon_intensity_grid

PATH_TO_RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'results')
PATH_TO_RESULTS_DATA = os.path.join(PATH_TO_RESULTS_DIR, 'SF_25')
PATH_TO_PAPER_PLOTS = os.path.join(PATH_TO_RESULTS_DIR, 'paper_plots')
if not os.path.exists(PATH_TO_PAPER_PLOTS):
    os.makedirs(PATH_TO_PAPER_PLOTS)

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

COLORS = {
    'total_d': sns.color_palette("muted")[4],
    'fleet_d': sns.color_palette("muted")[5],
    'dist_pass_d': sns.color_palette("muted")[6],
    'dist_rebal_d': sns.color_palette("muted")[0],
    'elec_energy_d': sns.color_palette("muted")[1],
    'elec_demand_d': sns.color_palette("muted")[2],
    'infra_d': sns.color_palette("muted")[3],
    'gas_d': sns.color_palette("muted")[7],
    'carbon_total_tco2': sns.color_palette("muted")[8],
    'cost_elec_per_kwh': sns.color_palette("muted")[9],
    'sum_d': sns.color_palette("dark")[0],
}

def boxplot_stackedbar(df: pd.DataFrame, exp_folder_name_to_labels: dict, filename_prefix: str = "") -> None:
    df_filtered = df[df['name'].isin([folder_name.split('.')[0] for folder_name in exp_folder_name_to_labels.keys()])].copy()
    cost_category_to_label = {
        # 'total_d': 'Total',
        'fleet_d': 'Vehicles',
        'dist_pass_d': 'Distance: Passenger',
        'dist_rebal_d': 'Distance: Rebalancing',
        'elec_energy_d': 'Electricity Energy Charges',
        'elec_demand_d': 'Electricity Demand Charges',
        'infra_d': 'Charging Infrastructure',
    }
    if df_filtered['gas_d'].sum() > 0:
        cost_category_to_label['gas_d'] = 'Gas'

    columns = []
    boxplot_colors = {}
    for cost_category in cost_category_to_label.keys():
        col_name = cost_category[:-1] + "cts_per_passenger_mile"
        df_filtered.loc[:, col_name] = df_filtered[cost_category] / df_filtered['dist_passenger_mi'] * 100
        columns.append(col_name)
        boxplot_colors[col_name] = COLORS[cost_category]
    df_boxplot = df_filtered[columns]
    sns.set_theme(style="whitegrid", font_scale=1.5)
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.boxplot(data=df_boxplot, ax=ax, palette=boxplot_colors, showfliers=False)
    sns.stripplot(data=df_boxplot, ax=ax, color='black', alpha=0.5, jitter=True, size=6)
    ax.set_xticklabels(cost_category_to_label.values(), rotation=45, ha='right', fontsize=14)
    ax.set_ylabel('Cost per Passenger Mile (cts/mi)', fontsize=16)
    ax.set_xlabel('Cost Category', fontsize=16)
    if SHOW_TITLE:
        ax.set_title('Cost Category Box Plots Across Different Vehicle Types', fontsize=18, pad=20)
    plt.tight_layout()
    for fig_format in FIG_FORMATS:
        plt.savefig(os.path.join(PATH_TO_PAPER_PLOTS, f"fig_{filename_prefix}_boxplot.{fig_format}"), dpi=300, bbox_inches='tight')

    cost_category_to_label = {
        # 'total_d': 'Total',
        'dist_pass_d': 'Distance: Passenger',
        'dist_rebal_d': 'Distance: Rebalancing',
        'elec_energy_d': 'Electricity Energy Charges',
        'elec_demand_d': 'Electricity Demand Charges',
        'infra_d': 'Charging Infrastructure',
        'fleet_d': 'Vehicles',
    }

    fig, ax = plt.subplots(figsize=(12, 8))
    bottom = np.zeros(len(df_filtered))
    for i, cost_category in enumerate(cost_category_to_label.keys()):
        if cost_category == 'total_d':
            continue
        col_name = cost_category[:-1] + "cts_per_passenger_mile"
        ax.bar(df_filtered['label'], df_filtered[col_name], label=cost_category_to_label[cost_category], bottom=bottom, color=COLORS[cost_category], edgecolor='black', linewidth=0.3)
        bottom += df_filtered[col_name]
    ax.set_ylabel('Cost per Passenger Mile (cts/mi)', fontsize=16)
    ax.set_xlabel('Vehicle Type', fontsize=16)
    if SHOW_TITLE:
        ax.set_title('Cost Breakdown for Each Vehicle Type', fontsize=18, pad=20)
    ax.legend(title="Cost Categories", fontsize=12, title_fontsize=14, loc='upper left', bbox_to_anchor=(1, 1))
    plt.xticks(rotation=45, ha='right', fontsize=14)
    plt.tight_layout()
    for fig_format in FIG_FORMATS:
        plt.savefig(os.path.join(PATH_TO_PAPER_PLOTS, f'fig_{filename_prefix}_stackedbar.{fig_format}'), dpi=300, bbox_inches='tight')

    fig, ax = plt.subplots(figsize=(12, 8))
    bottom = np.zeros(len(df_filtered))
    for i, cost_category in enumerate(cost_category_to_label.keys()):
        if cost_category in ['total_d', 'fleet_d', 'dist_pass_d']:
            continue
        col_name = cost_category[:-1] + "cts_per_passenger_mile"
        ax.bar(df_filtered['label'], df_filtered[col_name], label=cost_category_to_label[cost_category], bottom=bottom, color=COLORS[cost_category], edgecolor='black', linewidth=0.3)
        bottom += df_filtered[col_name]
    ax.set_ylabel('Variable Costs per Passenger Mile (cts/mi)', fontsize=16)
    ax.set_xlabel('Vehicle Type', fontsize=16)
    if SHOW_TITLE:
        ax.set_title('Variable Costs Breakdown for Each Vehicle Type', fontsize=18, pad=20)
    ax.legend(title="Cost Categories", fontsize=12, title_fontsize=14, loc='upper left', bbox_to_anchor=(1, 1))
    plt.xticks(rotation=45, ha='right', fontsize=14)
    plt.tight_layout()
    for fig_format in FIG_FORMATS:
        plt.savefig(os.path.join(PATH_TO_PAPER_PLOTS, f'fig_{filename_prefix}_var_costs_stackedbar.{fig_format}'), dpi=300, bbox_inches='tight')

def heatmap_infra_diff(exp_name_to_exp_obj: dict, exp_folder_name_and_label: tuple, filename_prefix: str = ""):
    exp1_name, exp1_label = exp_folder_name_and_label[0]
    exp2_name, exp2_label = exp_folder_name_and_label[1]

    exp1 = exp_name_to_exp_obj[exp1_name]
    exp2 = exp_name_to_exp_obj[exp2_name]

    SF_map = gpd.read_file(os.path.join(
        os.path.dirname(__file__), '..', 'data', 'SF_190',
        'Justin-Luke---Academic_SF-TAZ-with-added-boundary-pass-through_ZoneSet',
        'zone_set_SF_TAZ_with_added_boundary_pass_through.shp')
    )
    SF_map = SF_map.set_index('name')
    SF_map.index = SF_map.index.astype(int)
    SF_map = SF_map.sort_index()

    infra_cap1 = sum(exp1.UMax_charge[:, evse_idx] * evse.rate for evse_idx, evse in enumerate(exp1.EVSEs))
    infra_cap1 = np.append(infra_cap1, [0, 0, 0])

    infra_cap_2 = sum(exp2.UMax_charge[:, evse_idx] * evse.rate for evse_idx, evse in enumerate(exp2.EVSEs))
    infra_cap_2 = np.append(infra_cap_2, [0, 0, 0])

    for k, v in cluster_to_taz.items():
        SF_map.loc[v, 'cluster'] = k
    SF_map = SF_map.dissolve(by='cluster')
    SF_map['infra_cap'] = (infra_cap1 - infra_cap_2) / 1000

    sns.set_theme(style="whitegrid", font_scale=1.5)
    fig, ax = plt.subplots(figsize=(12, 8))
    vabsmax = max(abs(SF_map['infra_cap']))
    SF_map.plot(ax=ax, column='infra_cap', norm=TwoSlopeNorm(0, vmin=-vabsmax, vmax=vabsmax),
                cmap=plt.get_cmap('RdBu_r'), legend=True, edgecolor='black')
    SF_map.apply(lambda x: ax.annotate(f'{x.name:.0f}', xy=x.geometry.centroid.coords[0], ha='center', fontsize=12, color='black'), axis=1)
    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color=plt.get_cmap('RdBu_r')(plt.get_cmap('RdBu_r').N - 1), label=f'Greater capacity for {exp1_label}'),
            plt.Rectangle((0, 0), 1, 1, color=plt.get_cmap('RdBu_r')(0), label=f'Greater capacity for {exp2_label}')
        ],
        fontsize=14,
        loc='upper left',
    )

    cb_ax = fig.axes[1]
    cb_ax.tick_params(labelsize=14)
    cb_ax.set_ylabel('Installed charging infrastructure capacity difference (MW)', fontsize=16)
    if SHOW_TITLE:
        plt.title(f'Installed capacity difference ({exp1_label} minus {exp2_label})', fontsize=18, pad=20)

    plt.xlim((-122.525, -122.35))
    plt.ylim((37.7, 37.850))
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    for fig_format in FIG_FORMATS:
        plt.savefig(os.path.join(PATH_TO_PAPER_PLOTS, f'fig_{filename_prefix}_heatmap_infra_diff.{fig_format}'), dpi=300, bbox_inches='tight')

def cost_carbon_sensitivity(df: pd.DataFrame, exp_folder_name_to_attributes: dict, variables: list[str], filename_prefix: str = "", include_carbon: bool = False, include_avg_elec_cost: bool = False):
    n_var = len(variables)
    df_filtered = df[df['name'].isin([folder_name.split('.')[0] for folder_name in exp_folder_name_to_attributes.keys()])].copy()
    cost_category_to_label = {
        # 'total_d': 'Total',
        'dist_rebal_d': 'Distance: Rebalancing Only',
        'elec_energy_d': 'Electricity Energy Charges',
        'elec_demand_d': 'Electricity Demand Charges',
        'infra_d': 'Charging Infrastructure',
    }
    if df_filtered['gas_d'].sum() > 0:
        cost_category_to_label['gas_d'] = 'Gas'
        cost_category_to_label['fleet_d'] = 'Vehicles'
        cost_category_to_label['dist_pass_d'] = 'Distance: Passenger'

    df_filtered['sum_d'] = df_filtered[list(cost_category_to_label.keys())].sum(axis=1)
    cost_category_to_label['sum_d'] = 'Sum'

    base_case_name = None
    data = {variable : [] for variable in variables}

    for exp_folder_name, attributes in exp_folder_name_to_attributes.items():
        exp_name = exp_folder_name.split('.')[0]
        index = df_filtered[df_filtered['name'] == exp_name].index[0]
        for variable in variables:
            if variable.upper() in attributes['type']:
                data[variable].append((attributes[variable], index))

        if attributes.get('base_case'):
            if base_case_name is not None:
                raise ValueError(f"Multiple base cases found: {base_case_name} and {exp_name}. Please ensure only one base case is defined.")
            base_case_name = exp_name

    if base_case_name is None:
        raise ValueError("Base case not found. Please ensure that the base case is defined in the input data.")

    base_case_index = df_filtered[df_filtered['name'] == base_case_name].index[0]
    for variable in variables:
        data[variable].sort()

    values = {variable: [item[0] for item in data[variable]] for variable in variables}
    indices = {variable: [item[1] for item in data[variable]] for variable in variables}

    sns.set_theme(style="whitegrid", font_scale=1.5)

    labels_dict = {
        'energy_consump': 'Energy Consumption',
        'batt': 'Battery Capacity',
        'range': 'Range',
        'carbon_price': 'Carbon Price',
        'ev_price_ratio': 'EV : ICE Price Ratio',
        'compute_power': 'Autonomy Stack Power Consumption',
    }

    units_dict = {
        'energy_consump': 'Wh/mi',
        'batt': 'kWh',
        'range': 'mi',
        'carbon_price': '$/tCO2',
        'ev_price_ratio': '',
        'compute_power': 'W',
    }

    scale_multiplier = 1.5 if n_var > 1 else 1.0

    fig, ax = plt.subplots(1, n_var, figsize=(12 * n_var, 8 * scale_multiplier), sharey=True)
    if n_var == 1:
        ax = [ax]
    for v, variable in (enumerate(reversed(variables))):
        v = n_var - 1 - v  # Reverse the order of variables for plotting
        for i, (cost_category, cost_label) in enumerate(cost_category_to_label.items()):
            ax[v].plot(values[variable],
                    df_filtered.loc[indices[variable], cost_category] / df_filtered.loc[base_case_index, cost_category] * 100,
                    label=cost_label, marker='o', markersize=8, linestyle='-', color=COLORS[cost_category])
            print(variable)
            print(df_filtered.loc[indices[variable], cost_category] / df_filtered.loc[base_case_index, cost_category] * 100)

        ax[v].axhline(y=100, color='black', linestyle='--', linewidth=1.5)
        # Get the value in values[variable] that corresponds to the base case index
        base_case_value = None
        for i, value in enumerate(values[variable]):
            if indices[variable][i] == base_case_index:
                base_case_value = value
                break
        linear_line = (
            (2 - np.array(values[variable]) / base_case_value) * 100 if variable == 'batt' else
            (np.array(values[variable]) / base_case_value * 100)
        )
        ax[v].plot(
            values[variable],
            linear_line,
            label='Linear Change', color='gray', linestyle='--', linewidth=1.5
        )
        if units_dict[variable] == '':
            ax[v].set_xlabel(f'{labels_dict[variable]}', fontsize=16 * scale_multiplier)
        else:
            ax[v].set_xlabel(f'{labels_dict[variable]} ({units_dict[variable]})', fontsize=16 * scale_multiplier)

        if v == 0:
            ax[v].set_ylabel('Cost Relative to Base Case (%)', fontsize=16 * scale_multiplier)
            ax[v].legend(title="Cost Categories", fontsize=12 * scale_multiplier, title_fontsize=14 * scale_multiplier)

        if include_carbon:
            ax2 = ax[v].twinx()
            ax2.plot(
                values[variable],
                df_filtered.loc[indices[variable], 'carbon_total_tco2'] / df_filtered.loc[base_case_index, 'carbon_total_tco2'] * 100,
                label='Carbon Emissions', marker='o', markersize=8, linestyle='--', color=COLORS['carbon_total_tco2']
            )
            ax2.set_ylim(ax[n_var - 1].get_ylim())
            ax2.grid(False)
            if v < n_var - 1:
                ax2.set_yticklabels([])
                ax2.tick_params(axis='y', length=0)
            elif v == n_var - 1:
                ax2.set_ylabel('Carbon Emissions Relative to Base Case (%)', fontsize=16 * scale_multiplier,
                               color=COLORS['carbon_total_tco2'])
                ax2.tick_params(axis='y', labelcolor=COLORS['carbon_total_tco2'])

        if include_avg_elec_cost:
            ax3 = ax[v].twinx()
            ax3.plot(
                values[variable],
                df_filtered.loc[indices[variable], 'cost_elec_per_kwh'],
                label='Average Electricity Cost Paid ($/kWh)', marker='o', markersize=8, linestyle='--', color=COLORS['cost_elec_per_kwh']
            )
            ax3.grid(False)
            if v == n_var - 1:
                ax3.spines["right"].set_position(("axes", 1.15))
                ax3.spines["right"].set_visible(True)
                ax3.set_ylabel('Average Electricity Cost Paid ($/kWh)', fontsize=16 * scale_multiplier, color=COLORS['cost_elec_per_kwh'])
                ax3.tick_params(axis='y', labelcolor=COLORS['cost_elec_per_kwh'])


    if SHOW_TITLE:
        include_carbon_title = " and Carbon Emissions" if include_carbon else ""
        variables_title = ', '.join([labels_dict[v] for v in variables])
        fig.suptitle(f'Cost Category Sensitivity{include_carbon_title} to {variables_title}', fontsize=18 * scale_multiplier, pad=20)
    plt.tight_layout()
    include_carbon_filename = "_carbon" if include_carbon else ""
    variables_filename = '_'.join(variables)
    for fig_format in FIG_FORMATS:
        plt.savefig(os.path.join(PATH_TO_PAPER_PLOTS, f'fig_{filename_prefix}_cost{include_carbon_filename}_sensitivity_{variables_filename}.{fig_format}'), dpi=300,
                bbox_inches='tight')

def cost_carbon_and_elec_fleet_share_vs_carbon_price_or_vehicle_price(
        df: pd.DataFrame,
        exp_folder_name_to_attributes: dict,
        variable: str,
        filename_prefix: str = "",
        title_prefix: str = "",
        include_elec_fleet_share: bool = False
):
    df_filtered = df[df['name'].isin([folder_name.split('.')[0] for folder_name in exp_folder_name_to_attributes.keys()])].copy()

    if variable == 'carbon_price':
        x_axis_attr_name = 'carbon_price'
        x_axis_filename_string = 'carbon_price'
        x_axis_label = 'Carbon Price ($/tCO2)'
        title_label = 'Carbon Price'
    elif variable == 'vehicle_price':
        x_axis_attr_name = 'ev_price_ratio'
        x_axis_filename_string = 'vehicle_price'
        x_axis_label = 'EV : ICE Price Ratio'
        title_label = 'EV : ICE Price Ratio'
    else:
        raise ValueError("variable must be either 'carbon_price' or 'vehicle_price'.")

    x_axis_data = []

    for exp_folder_name, attributes in exp_folder_name_to_attributes.items():
        exp_name = exp_folder_name.split('.')[0]
        index = df_filtered[df_filtered['name'] == exp_name].index[0]
        x_axis_data.append((attributes[x_axis_attr_name], index))

    x_axis_data.sort()
    x_axis_values, x_axis_indices = zip(*x_axis_data)

    # df_filtered['variable_costs'] = (
    #         df_filtered['elec_energy_d'] + df_filtered['elec_demand_d'] +
    #         df_filtered['dist_rebal_d'] + df_filtered['infra_d'] + df_filtered['gas_d']
    # ) / df_filtered['dist_passenger_mi'] * 100
    df_filtered['total_cost_minus_carbon_cost'] = (
            df_filtered['total_d'] - df_filtered['elec_carbon_d'] - df_filtered['gas_carbon_d']
    ) / df_filtered['dist_passenger_mi'] * 100

    df_filtered['elec_fleet_share'] = df_filtered['fleet_sizes'].apply(lambda x: x[0] / sum(x) * 100)

    sns.set_theme(style="whitegrid", font_scale=1.5)
    fig, ax1 = plt.subplots(figsize=(12, 8))
    ax2 = ax1.twinx()
    ax3 = None
    if include_elec_fleet_share:
        ax3 = ax1.twinx()
        ax3.spines["right"].set_position(("axes", 1.15))
        ax3.spines["right"].set_visible(True)
    palette = sns.color_palette("muted", 3)
    ax1.plot(
        x_axis_values,
        df_filtered.loc[list(x_axis_indices), 'total_cost_minus_carbon_cost'],
        marker='o', markersize=8, linestyle='-', color=palette[0]
    )
    ax2.plot(
        x_axis_values,
        df_filtered.loc[list(x_axis_indices), 'carbon_total_tco2'] / df_filtered.loc[list(x_axis_indices), 'dist_passenger_mi'] * 1000 * 1000,
        marker='o', markersize=8, linestyle='--', color=COLORS['carbon_total_tco2']
    )
    if include_elec_fleet_share:
        ax3.plot(
            x_axis_values,
            df_filtered.loc[list(x_axis_indices), 'elec_fleet_share'],
            marker='o', markersize=8, linestyle='--', color=palette[2], label='Electric Fleet Share'
        )

    ax1.set_xlabel(x_axis_label, fontsize=16)
    ax1.set_ylabel('Total Cost (excluding carbon) per Passenger Mile (cts/mi)', fontsize=16, color=palette[0])
    ax1.tick_params(axis='y', labelcolor=palette[0])
    ax2.set_ylabel('Carbon Emissions per Passenger Mile (gCO2/mi)', fontsize=16, color=COLORS['carbon_total_tco2'])
    ax2.tick_params(axis='y', labelcolor=COLORS['carbon_total_tco2'])
    ax2.grid(False)
    if include_elec_fleet_share:
        ax3.set_ylabel('Electric Fleet Share (%)', fontsize=16, color=palette[2])
        ax3.tick_params(axis='y', labelcolor=palette[2])
        ax3.grid(False)
        ax3.set_ylim(top=100)
    title = f'Cost, Carbon Emissions, and Electric Fleet Share vs {title_label}' if include_elec_fleet_share else f'Cost and Carbon Emissions vs {title_label}'
    title = title_prefix + title
    if SHOW_TITLE:
        ax1.set_title(title, fontsize=18, pad=20)
    plt.tight_layout()

    for fig_format in FIG_FORMATS:
        filename = f'fig_{filename_prefix}_cost_carbon_and_elec_fleet_share_vs_{x_axis_filename_string}.{fig_format}' if include_elec_fleet_share else f'fig_{filename_prefix}_cost_and_carbon_vs_{x_axis_filename_string}.{fig_format}'
        plt.savefig(os.path.join(PATH_TO_PAPER_PLOTS, filename), dpi=300, bbox_inches='tight')

def charging_profile(exp_name_to_exp_obj: dict, exp_folder_name_to_labels: dict, legend_title: None, filename_prefix: str = ""):
    sns.set_theme(style="whitegrid", font_scale=1.5)
    fig, ax = plt.subplots(figsize=(12, 8))

    deltaT, time_vec = None, None
    for exp_name, label in exp_folder_name_to_labels.items():
        exp_obj = exp_name_to_exp_obj[exp_name]
        startT = exp_obj.startT
        endT = exp_obj.endT
        deltaT = exp_obj.deltaT
        time_vec = np.arange(startT * deltaT, endT * deltaT, deltaT)
        time_vec = pd.to_datetime(time_vec, unit='h', origin='unix').strftime('%H:%M')
        charge_arr = []
        num_vehicle_types = len(exp_obj.PAMoDVehicles)
        for i in range(num_vehicle_types):
            for t in range(startT, endT):
                E_charge_idx_t = exp_obj.PAMoDVehicles[i].filter_edge_idx('charge', t=t)
                charge_arr.append(
                    np.sum(np.multiply(exp_obj.U_list[i][E_charge_idx_t],
                                       exp_obj.PAMoDVehicles[i].power_conv[E_charge_idx_t])))
        ax.step(time_vec, np.array(charge_arr) / 1000, where='post', label=label)
    ax2 = ax.twinx()
    ax2.grid(False)
    carbon_intensity_grid = generate_carbon_intensity_grid(int(np.round(24 / deltaT))) * 1000 * 1000
    ax2.step(time_vec, carbon_intensity_grid, where='post', color=COLORS['carbon_total_tco2'], linestyle='--')
    ax2.set_ylabel('Carbon Intensity (gCO2/kWh)', fontsize=16, color=COLORS['carbon_total_tco2'])
    ax2.tick_params(axis='y', labelcolor=COLORS['carbon_total_tco2'])

    ax3 = ax.twinx()
    ax3.grid(False)
    ax3.spines["right"].set_position(("axes", 1.15))
    ax3.spines["right"].set_visible(True)
    p_elec_energy, _ = generate_p_elec('BEV-2-S', 1686812400, deltaT, 1, 0)
    ax3.step(time_vec, p_elec_energy, where='post', color=COLORS['cost_elec_per_kwh'], linestyle='--')
    ax3.set_ylabel('Electricity Energy Charge Rate ($/kWh)', fontsize=16, color=COLORS['cost_elec_per_kwh'])
    ax3.tick_params(axis='y', labelcolor=COLORS['cost_elec_per_kwh'])


    ax.set_xlabel('Time (hr)', fontsize=16)
    ax.set_ylabel('Charging Power (MW)', fontsize=16)
    if legend_title is not None:
        ax.legend(title=legend_title, fontsize=12, title_fontsize=14, loc='upper left')

    if SHOW_TITLE:
        ax.set_title('Fleet Charging Profiles', fontsize=18, pad=20)

    ax.set_xticks(time_vec[::int(1 / deltaT)])
    ax.set_xticklabels(time_vec[::int(1 / deltaT)], rotation=45, ha='right', fontsize=14)
    plt.tight_layout()
    for fig_format in FIG_FORMATS:
        plt.savefig(os.path.join(PATH_TO_PAPER_PLOTS, f'fig_{filename_prefix}_charging_profile.{fig_format}'), dpi=300, bbox_inches='tight')

def add_exp_to_df(df: pd.DataFrame, exp_folder_name: str, exp_label: str) -> pd.DataFrame:
    exp_name = exp_folder_name.split('.')[0]

    if exp_name in df['name'].values and exp_label in df.loc[df['name'] == exp_name, 'label'].values:
        return df

    if exp_folder_name.endswith('.zip'):
        zip_file = zipfile.ZipFile(os.path.join(PATH_TO_RESULTS_DATA, exp_folder_name), 'r')
        with zip_file.open('PAMoDFleet/print_log.txt', 'r') as f:
            df_row = get_data_from_log_file(io.TextIOWrapper(f, encoding='utf-8'), exp_name, exp_label)
    else:
        log_file = os.path.join(PATH_TO_RESULTS_DATA, exp_name, 'PAMoDFleet', 'print_log.txt')
        with open(log_file, 'r') as f:
            df_row = get_data_from_log_file(f, exp_name, exp_label)

    return pd.concat([df, df_row], ignore_index=True)

def get_data_from_log_file(log_file: io.TextIOWrapper, exp_name: str, label: str) -> pd.DataFrame:
    lines = log_file.readlines()
    data = {}
    data['name'] = exp_name
    data['label'] = label
    for line in lines:
        if line.startswith("fleet_sizes: "):
            fleet_sizes = ast.literal_eval(line.split("fleet_sizes: ")[1].strip())
            data['fleet_sizes'] = fleet_sizes
        elif line.startswith("costs: "):
            costs_dict = ast.literal_eval(line.split("costs: ")[1].strip())
            for key, value in costs_dict.items():
                data[key + "_d"] = value
            data['total_d'] = sum(costs_dict.values()) - costs_dict['dist_rebal']
            data['dist_pass_d'] = costs_dict['dist_total'] - costs_dict['dist_rebal']
        elif line.startswith("elec_energy: "):
            data['elec_energy_kwh'] = float(line.split("elec_energy: ")[1].strip())
        elif line.startswith("elec_demand: "):
            data['elec_demand_kw'] = float(line.split("elec_demand: ")[1].strip())
        elif line.startswith("dist_elec: "):
            data['dist_elec_mi'] = float(line.split("dist_elec: ")[1].strip())
        elif line.startswith("dist_gas: "):
            data['dist_gas_mi'] = float(line.split("dist_gas: ")[1].strip())
        elif line.startswith("dist_total: "):
            data['dist_total_mi'] = float(line.split("dist_total: ")[1].strip())
        elif line.startswith("dist_rebal: "):
            data['dist_rebal_mi'] = float(line.split("dist_rebal: ")[1].strip())
        elif line.startswith("dist_rebal_elec: "):
            data['dist_rebal_elec_mi'] = float(line.split("dist_rebal_elec: ")[1].strip())
        elif line.startswith("dist_rebal_gas: "):
            data['dist_rebal_gas_mi'] = float(line.split("dist_rebal_gas: ")[1].strip())
        elif line.startswith("dist_passenger: "):
            data['dist_passenger_mi'] = float(line.split("dist_passenger: ")[1].strip())
        elif line.startswith("dist_passenger_elec: "):
            data['dist_passenger_elec_mi'] = float(line.split("dist_passenger_elec: ")[1].strip())
        elif line.startswith("dist_passenger_gas: "):
            data['dist_passenger_gas_mi'] = float(line.split("dist_passenger_gas: ")[1].strip())
        elif line.startswith("carbon_total: "):
            data['carbon_total_tco2'] = float(line.split("carbon_total: ")[1].strip())
        elif line.startswith("carbon_elec: "):
            data['carbon_elec_tco2'] = float(line.split("carbon_elec: ")[1].strip())
        elif line.startswith("carbon_gas: "):
            data['carbon_gas_tco2'] = float(line.split("carbon_gas: ")[1].strip())
        elif line.startswith("infra_plugs: "):
            data['infra_plugs'] = float(line.split("infra_plugs: ")[1].strip())
        elif line.startswith("infra_capacity: "):
            data['infra_capacity_kw'] = float(line.split("infra_capacity: ")[1].strip())
    data['carbon_elec_tco2_per_kwh'] = data['carbon_elec_tco2'] / data['elec_energy_kwh']
    data['cost_per_mile'] = data['total_d'] / data['dist_total_mi']
    data['cost_per_passenger_mile'] = data['total_d'] / data['dist_passenger_mi']
    data['cost_elec_energy_per_kwh'] = data['elec_energy_d'] / data['elec_energy_kwh']
    data['cost_elec_per_kwh'] = (data['elec_energy_d'] + data['elec_demand_d']) / data['elec_energy_kwh']
    data['rebal_rate'] = data['dist_rebal_mi'] / data['dist_total_mi']
    data['rebal_ratio'] = data['dist_rebal_mi'] / data['dist_passenger_mi']
    return pd.DataFrame([data])

def load_exp(exp_folder_name: str, exp_name_to_exp_obj: dict):
    if exp_folder_name in exp_name_to_exp_obj:
        return
    if exp_folder_name.endswith('.zip'):
        zip_file = zipfile.ZipFile(os.path.join(PATH_TO_RESULTS_DATA, exp_folder_name), 'r')
        with zip_file.open(f'PAMoDFleet/{exp_folder_name[:-4]}.p', 'r') as f:
            try:
                exp = pickle.load(f)
            except ModuleNotFoundError:
                f.seek(0)
                exp = Scipy_1_10_Unpickler(f).load()
    else:
        exp_path = os.path.join(PATH_TO_RESULTS_DATA, exp_folder_name, 'PAMoDFleet', f'{exp_folder_name}.p')
        with open(exp_path, 'rb') as f:
            try:
                exp = pickle.load(f)
            except ModuleNotFoundError:
                f.seek(0)
                exp = Scipy_1_10_Unpickler(f).load()
    exp_name_to_exp_obj[exp_folder_name] = exp

# Class to handle unpickling scipy sparse matrices from older versions, scipy 1_10
class Scipy_1_10_Unpickler(Unpickler):
    def find_class(self, module, name):
        if module == 'scipy.sparse._arrays':
            module = 'scipy.sparse'
        return super().find_class(module, name)

SHOW_TITLE = False
FIG_FORMATS = ["png", "eps"]

if __name__ == "__main__":
    df = pd.DataFrame(columns=['name'])
    exp_name_to_exp_obj = {}

    sections_to_plot = [
        # "sec2",
        "sec3",
        # "sec4",
        # "sec5",
        # "sup1",
    ]

    if "sec2" in sections_to_plot:
        sec2_exp_folder_names_to_label = {
            'dacia_spring_electric' : 'Crossover City Car',
            'chevrolet_bolt_ev.zip': 'Subcompact Hatchback',
            'hyundai_ioniq_electric.zip': 'Compact Liftback',
            'tesla_model_3.zip': 'Mid-size Sedan',
            'hyundai_ioniq_5.zip': 'Compact Crossover SUV',
            'jaguar_ipace.zip': 'Crossover SUV',
        }
        for exp_folder_name, label in sec2_exp_folder_names_to_label.items():
            df = add_exp_to_df(df, exp_folder_name, label)
        boxplot_stackedbar(df, sec2_exp_folder_names_to_label, 'sec2')

        sec2_2_exp_folder_names_to_label = {
            'dacia_spring_electric': 'Crossover City Car',
            'jaguar_ipace.zip': 'Crossover SUV',
        }
        for exp_folder_name, _ in sec2_2_exp_folder_names_to_label.items():
            load_exp(exp_folder_name, exp_name_to_exp_obj)
        charging_profile(exp_name_to_exp_obj, sec2_2_exp_folder_names_to_label, 'Vehicle Type', 'sec2_2')
        heatmap_infra_diff(exp_name_to_exp_obj, tuple((key, value) for key, value in sec2_2_exp_folder_names_to_label.items()), 'sec2_2')

    if "sec3" in sections_to_plot:
        sec3_exp_folder_name_to_label = {
            'dacia_spring_electric': 'Joint',
            'dacia_spring_electric_use_baseline': 'Baseline',
        }
        for exp_folder_name, label in sec3_exp_folder_name_to_label.items():
            df = add_exp_to_df(df, exp_folder_name, label)
            load_exp(exp_folder_name, exp_name_to_exp_obj)

        boxplot_stackedbar(df, sec3_exp_folder_name_to_label, 'sec3')
        charging_profile(exp_name_to_exp_obj, sec3_exp_folder_name_to_label, 'Optimization', 'sec3')
        heatmap_infra_diff(exp_name_to_exp_obj, tuple((key, value) for key, value in sec3_exp_folder_name_to_label.items()), 'sec3')

    if "sec4" in sections_to_plot:
        sec4_1_exp_folder_names_to_attributes = {
            # 'dacia_175Whpkm' : {
            #     'label': '175 Wh/mi, 25 kWh',
            #     'type': ['ENERGY_CONSUMP'],
            #     'energy_consump': 175,
            #     'batt': 25,
            #     'range': 91.870,
            # },
            'dacia_170Whpkm': {
                'label': '170 Wh/mi, 25 kWh',
                'type': ['ENERGY_CONSUMP'],
                'energy_consump': 170,
                'batt': 25,
                'range': 94.382,
            },
            'dacia_165Whpkm' : {
                'label': '165 Wh/mi, 25 kWh',
                'type': ['ENERGY_CONSUMP'],
                'energy_consump': 165,
                'batt': 25,
                'range': 97.035,
            },
            'dacia_160Whpkm': {
                'label': '160 Wh/mi, 25 kWh',
                'type': ['ENERGY_CONSUMP'],
                'energy_consump': 160,
                'batt': 25,
                'range': 99.841,
            },
            'dacia_155Whpkm' : {
                'label': '155 Wh/mi, 25 kWh',
                'type': ['ENERGY_CONSUMP'],
                'energy_consump': 155,
                'batt': 25,
                'range': 102.815,
            },
            'dacia_150Whpkm': {
                'label': '150 Wh/mi, 25 kWh',
                'type': ['ENERGY_CONSUMP'],
                'energy_consump': 150,
                'batt': 25,
                'range': 105.971,
            },
            'dacia_145Whpkm' : {
                'label': '145 Wh/mi, 25 kWh',
                'type': ['ENERGY_CONSUMP', 'BATT'],
                'energy_consump': 145,
                'batt': 25,
                'range': 109.327,
                'base_case': True,
            },
            'dacia_140Whpkm' : {
                'label': '140 Wh/mi, 25 kWh',
                'type': ['ENERGY_CONSUMP'],
                'energy_consump': 140,
                'batt': 25,
                'range': 112.903,
            },
            'dacia_spring_electric' : {
                'label': '135 Wh/mi, 25 kWh',
                'type': ['ENERGY_CONSUMP'],
                'energy_consump': 135,
                'batt': 25,
                'range': 116.720,
            },
            'dacia_125Whpkm' : {
                'label': '125 Wh/mi, 25 kWh',
                'type': ['ENERGY_CONSUMP'],
                'energy_consump': 125,
                'batt': 25,
                'range': 125.186,
            },
            'dacia_115Whpkm' : {
                'label': '115 Wh/mi, 25 kWh',
                'type': ['ENERGY_CONSUMP'],
                'energy_consump': 115,
                'batt': 25,
                'range': 134.975,
            },
            'dacia_100Whpkm.zip': {
                'label': '100 Wh/mi, 25 kWh',
                'type': ['ENERGY_CONSUMP'],
                'energy_consump': 100,
                'batt': 25,
                'range': 152.911,
            },
            'dacia_21_583kWh': {
                'label': '145 Wh/mi, 21.583 kWh',
                'type': ['BATT'],
                'energy_consump': 145,
                'batt': 21.583,
                'range': 94.384,
            },
            'dacia_22_19kWh' : {
                'label': '145 Wh/mi, 22.190 kWh',
                'type': ['BATT'],
                'energy_consump': 145,
                'batt': 22.190,
                'range': 97.039,
            },
            'dacia_22_831kWh': {
                'label': '145 Wh/mi, 22.831 kWh',
                'type': ['BATT'],
                'energy_consump': 145,
                'batt': 22.831,
                'range': 99.842,
            },
            'dacia_23_511kWh' : {
                'label': '145 Wh/mi, 23.511 kWh',
                'type': ['BATT'],
                'energy_consump': 145,
                'batt': 23.511,
                'range': 102.815,
            },
            'dacia_24_233kWh' : {
                'label': '145 Wh/mi, 24.233 kWh',
                'type': ['BATT'],
                'energy_consump': 145,
                'batt': 24.233,
                'range': 105.973,
            },
            'dacia_25_818kWh' : {
                'label': '145 Wh/mi, 25.818 kWh',
                'type': ['BATT'],
                'energy_consump': 145,
                'batt': 25.818,
                'range': 112.904,
            },
            'dacia_26_691kWh.zip' : {
                'label': '145 Wh/mi, 26.691 kWh',
                'type': ['BATT'],
                'energy_consump': 145,
                'batt': 26.691,
                'range': 116.722,
            },
            'dacia_28_627kWh.zip' : {
                'label': '145 Wh/mi, 28.627 kWh',
                'type': ['BATT'],
                'energy_consump': 145,
                'batt': 28.627,
                'range': 125.188,
            },
            'dacia_30_865kWh.zip' : {
                'label': '145 Wh/mi, 30.865 kWh',
                'type': ['BATT'],
                'energy_consump': 145,
                'batt': 30.865,
                'range': 134.975,
            },
            'dacia_34_967kWh.zip' : {
                'label': '145 Wh/mi, 34.967 kWh',
                'type': ['BATT'],
                'energy_consump': 145,
                'batt': 34.967,
                'range': 152.913,
            },
        }
        for exp_folder_name, attributes in sec4_1_exp_folder_names_to_attributes.items():
            df = add_exp_to_df(df, exp_folder_name, attributes['label'])
        cost_carbon_sensitivity(df, sec4_1_exp_folder_names_to_attributes, ['batt'], 'sec4_1', )
        cost_carbon_sensitivity(df, sec4_1_exp_folder_names_to_attributes, ['energy_consump'], 'sec4_1', True, False)
        cost_carbon_sensitivity(df, sec4_1_exp_folder_names_to_attributes, ['batt', 'energy_consump'], 'sec4_1', False, False)

        sec4_2_exp_folder_names_to_attributes = {
            'dacia_0_1kW.zip': {
                'label': '100 W',
                'type': ['COMPUTE_POWER'],
                'compute_power': 100,
            },
            'dacia_spring_electric': {
                'label': '500 W',
                'type': ['COMPUTE_POWER'],
                'compute_power': 500,
                'base_case': True,
            },
            'dacia_1kW.zip': {
                'label': '1000 W',
                'type': ['COMPUTE_POWER'],
                'compute_power': 1000,
            },
            'dacia_1_5kW.zip': {
                'label': '1500 W',
                'type': ['COMPUTE_POWER'],
                'compute_power': 1500,
            },
            'dacia_2kW.zip': {
                'label': '2000 W',
                'type': ['COMPUTE_POWER'],
                'compute_power': 2000,
            },
        }
        for exp_folder_name, attributes in sec4_2_exp_folder_names_to_attributes.items():
            df = add_exp_to_df(df, exp_folder_name, attributes['label'])
        cost_carbon_sensitivity(df, sec4_2_exp_folder_names_to_attributes, ['compute_power'], 'sec4_2', True, True)

        sec4_3_exp_folder_names_to_label = {
            'dacia_spring_electric': '500 W',
            'dacia_1_5kW.zip': '1500 W',
        }
        for exp_folder_name, label in sec4_3_exp_folder_names_to_label.items():
            df = add_exp_to_df(df, exp_folder_name, label)
            load_exp(exp_folder_name, exp_name_to_exp_obj)
        charging_profile(exp_name_to_exp_obj, sec4_3_exp_folder_names_to_label, 'Autonomy Stack Power Consumption', 'sec4_3')
        heatmap_infra_diff(exp_name_to_exp_obj, tuple((key, value) for key, value in sec4_3_exp_folder_names_to_label.items()), 'sec4_3')

        sec4_4_exp_folder_names_to_label = {
            'dacia_100Whpkm.zip': '100 Wh/mi',
            'dacia_165Whpkm': '165 Wh/mi',
        }
        for exp_folder_name, label in sec4_4_exp_folder_names_to_label.items():
            load_exp(exp_folder_name, exp_name_to_exp_obj)
        charging_profile(exp_name_to_exp_obj, sec4_4_exp_folder_names_to_label, 'Energy Consumption', 'sec4_4')
        heatmap_infra_diff(exp_name_to_exp_obj, tuple((key, value) for key, value in sec4_4_exp_folder_names_to_label.items()), 'sec4_4')

    if "sec5" in sections_to_plot:
        sec5_1_exp_folder_names_to_attributes = {
            'ioniq_hybrid_electric_1_311.zip': {
                'label': 'Compact Liftback Mixed Fleet of Hybrid ICE and Electric Vehicles, 1.311',
                'ev_price_ratio': 1.311,
                'type': ['EV_PRICE_RATIO'],
                'base_case': True,
            },
            'ioniq_hybrid_electric_1_234.zip': {
                'label': 'Compact Liftback Mixed Fleet of Hybrid ICE and Electric Vehicles, 1.234',
                'ev_price_ratio': 1.234,
                'type': ['EV_PRICE_RATIO'],
            },
            'ioniq_hybrid_electric_1_156.zip': {
                'label': 'Compact Liftback Mixed Fleet of Hybrid ICE and Electric Vehicles, 1.156',
                'ev_price_ratio': 1.156,
                'type': ['EV_PRICE_RATIO'],
            },
            'ioniq_hybrid_electric_1_078.zip': {
                'label': 'Compact Liftback Mixed Fleet of Hybrid ICE and Electric Vehicles, 1.078',
                'ev_price_ratio': 1.078,
                'type': ['EV_PRICE_RATIO'],
            },
            'ioniq_hybrid_electric_1_000.zip': {
                'label': 'Compact Liftback Mixed Fleet of Hybrid ICE and Electric Vehicles, 1.000',
                'ev_price_ratio': 1.000,
                'type': ['EV_PRICE_RATIO'],
            },
        }
        for exp_folder_name, attributes in sec5_1_exp_folder_names_to_attributes.items():
            df = add_exp_to_df(df, exp_folder_name, attributes['label'])
        cost_carbon_and_elec_fleet_share_vs_carbon_price_or_vehicle_price(df, sec5_1_exp_folder_names_to_attributes,
                                                                          'vehicle_price', 'sec5_1',
                                                                          "Compact Liftback Mixed Fleet of Hybrid ICE and Electric Vehicles:\n",
                                                                          True)
        cost_carbon_sensitivity(df, sec5_1_exp_folder_names_to_attributes, ['ev_price_ratio'], 'sec5_1')

        sec5_2_exp_folder_names_to_attributes = {
            'ioniq_hybrid_electric_1_311.zip': {
                'label': 'Compact Liftback Mixed Fleet of Hybrid ICE and Electric Vehicles, $0/tCO2',
                'carbon_price': 0,
                'type': ['CARBON_PRICE'],
                'base_case': True,
            },
            'ioniq_hybrid_electric_1_311_44dptCO2.zip': {
                'label': 'Compact Liftback Mixed Fleet of Hybrid ICE and Electric Vehicles, $44/tCO2',
                'carbon_price': 44,
                'type': ['CARBON_PRICE'],
            },
            'ioniq_hybrid_electric_1_311_185dptCO2.zip': {
                'label': 'Compact Liftback Mixed Fleet of Hybrid ICE and Electric Vehicles, $185/tCO2',
                'carbon_price': 185,
                'type': ['CARBON_PRICE'],
            },
            'ioniq_hybrid_electric_1_311_413dptCO2.zip': {
                'label': 'Compact Liftback Mixed Fleet of Hybrid ICE and Electric Vehicles, $413/tCO2',
                'carbon_price': 413,
                'type': ['CARBON_PRICE'],
            },
            'ioniq_hybrid_electric_1_311_805dptCO2.zip': {
                'label': 'Compact Liftback Mixed Fleet of Hybrid ICE and Electric Vehicles, $805/tCO2',
                'carbon_price': 805,
                'type': ['CARBON_PRICE'],
            },
        }
        for exp_folder_name, attributes in sec5_2_exp_folder_names_to_attributes.items():
            df = add_exp_to_df(df, exp_folder_name, attributes['label'])
        cost_carbon_and_elec_fleet_share_vs_carbon_price_or_vehicle_price(df, sec5_2_exp_folder_names_to_attributes, 'carbon_price','sec5_2', "Compact Liftback Mixed Fleet of Hybrid ICE and Electric Vehicles:\n", True)
        cost_carbon_sensitivity(df, sec5_2_exp_folder_names_to_attributes, ['carbon_price'], 'sec5_2')

        sec5_3_exp_folder_names_to_attributes = {
            'dacia_spring_electric': {
                'label': 'Crossover City Car, $0/tCO2',
                'carbon_price': 0,
                'type': ['CARBON_PRICE'],
                'base_case': True,
            },
            'dacia_44dptCO2.zip': {
                'label': 'Crossover City Car, $44/tCO2',
                'carbon_price': 44,
                'type': ['CARBON_PRICE'],
            },
            'dacia_185dptCO2.zip': {
                'label': 'Crossover City Car, $185/tCO2',
                'carbon_price': 185,
                'type': ['CARBON_PRICE'],
            },
            'dacia_413dptCO2.zip': {
                'label': 'Crossover City Car, $413/tCO2',
                'carbon_price': 413,
                'type': ['CARBON_PRICE'],
            },
            'dacia_805dptCO2.zip': {
                'label': 'Crossover City Car, $805/tCO2',
                'carbon_price': 805,
                'type': ['CARBON_PRICE'],
            },
        }
        for exp_folder_name, attributes in sec5_3_exp_folder_names_to_attributes.items():
            df = add_exp_to_df(df, exp_folder_name, attributes['label'])
        cost_carbon_and_elec_fleet_share_vs_carbon_price_or_vehicle_price(df, sec5_3_exp_folder_names_to_attributes,
                                                                          'carbon_price', 'sec5_3',
                                                                          "Crossover City Car:\n")
        cost_carbon_sensitivity(df, sec5_3_exp_folder_names_to_attributes, ['carbon_price'], 'sec5_3')

        sec5_4_exp_folder_names_to_label = {
            'dacia_spring_electric': "$0/tCO2",
            'dacia_413dptCO2.zip': "$413/tCO2",
        }
        for exp_folder_name, label in sec5_4_exp_folder_names_to_label.items():
            load_exp(exp_folder_name, exp_name_to_exp_obj)
        charging_profile(exp_name_to_exp_obj, sec5_4_exp_folder_names_to_label, 'Carbon Price', 'sec5_4')
        heatmap_infra_diff(exp_name_to_exp_obj, tuple((key, value) for key, value in sec5_4_exp_folder_names_to_label.items()), 'sec5_4')

    if "sup1" in sections_to_plot:
        sup1_exp_folder_names_to_label = {
            'dacia_spring_electric_b20.zip' : 'Crossover City Car',
            'chevrolet_bolt_ev_b20.zip': 'Subcompact Hatchback',
            'hyundai_ioniq_electric_b20.zip': 'Compact Liftback',
            'tesla_model_3_b20.zip': 'Mid-size Sedan',
            'hyundai_ioniq_5_b20.zip': 'Compact Crossover SUV',
            'jaguar_ipace_b20.zip': 'Crossover SUV',
        }
        for exp_folder_name, label in sup1_exp_folder_names_to_label.items():
            df = add_exp_to_df(df, exp_folder_name, label)
        boxplot_stackedbar(df, sup1_exp_folder_names_to_label, 'sup1')