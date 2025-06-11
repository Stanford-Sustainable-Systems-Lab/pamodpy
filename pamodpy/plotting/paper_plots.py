import ast
import os
import io
import pickle
import zipfile

import geopandas as gpd
import matplotlib.pyplot as plt
from fontTools.ttLib.woff2 import bboxFormat
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
import seaborn as sns

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

def boxplot_stackedbar(df: pd.DataFrame, exp_folder_name_to_labels: dict, filename_prefix: str = "") -> None:
    df_filtered = df[df['name'].isin([folder_name.split('.')[0] for folder_name in exp_folder_name_to_labels.keys()])].copy()
    cost_category_to_label = {
        'total_d': 'Total',
        'fleet_d': 'Fleet',
        'dist_total_d': 'Distance',
        'elec_energy_d': 'Electricity Energy Charges',
        'infra_d': 'Charging Infrastructure',
        'dist_rebal_d': 'Distance: Rebalancing Only',
        'elec_demand_d': 'Electricity Demand Charges',
    }
    columns = []
    for cost_type in cost_category_to_label.keys():
        col_name = cost_type + "_per_passenger_mile"
        df_filtered.loc[:, col_name] = df_filtered[cost_type] / df_filtered['dist_passenger_mi']
        columns.append(col_name)
    df_boxplot = df_filtered[columns]
    sns.set_theme(style="whitegrid", font_scale=1.5)
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.boxplot(data=df_boxplot, ax=ax, palette="pastel", showfliers=False)
    sns.stripplot(data=df_boxplot, ax=ax, color='black', alpha=0.5, jitter=True, size=6)
    ax.set_xticklabels(cost_category_to_label.values(), rotation=45, ha='right', fontsize=14)
    ax.set_ylabel('Cost per passenger mile ($/mi)', fontsize=16)
    ax.set_xlabel('Cost Category', fontsize=16)
    ax.set_title('Cost Category Box Plots Across Different Vehicle Types', fontsize=18, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(PATH_TO_PAPER_PLOTS, f"{filename_prefix}_boxplot.png"), dpi=300, bbox_inches='tight')

    fig, ax = plt.subplots(figsize=(12, 8))
    bottom = np.zeros(len(df_filtered))
    for i, cost_type in enumerate(cost_category_to_label.keys()):
        if cost_type == 'total_d':
            continue
        col_name = cost_type + "_per_passenger_mile"
        ax.bar(df_filtered['label'], df_filtered[col_name], label=cost_category_to_label[cost_type], bottom=bottom, color=sns.color_palette("pastel")[i])
        bottom += df_filtered[col_name]
    ax.set_ylabel('Cost per passenger mile ($/mi)', fontsize=16)
    ax.set_xlabel('Vehicle Type', fontsize=16)
    ax.set_title('Cost Breakdown for Each Vehicle Type', fontsize=18, pad=20)
    ax.legend(title="Cost Categories", fontsize=12, title_fontsize=14, loc='upper left', bbox_to_anchor=(1, 1))
    plt.xticks(rotation=45, ha='right', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(PATH_TO_PAPER_PLOTS, f'{filename_prefix}_stackedbar.png'), dpi=300, bbox_inches='tight')

def heatmap_infra_diff(exp_name_to_exp_obj: dict, exp_folder_name_to_labels: tuple, filename_prefix: str = ""):
    exp1_name, exp1_label = exp_folder_name_to_labels[0]
    exp2_name, exp2_label = exp_folder_name_to_labels[1]

    exp1 = exp_name_to_exp_obj[exp1_name]
    exp2 = exp_name_to_exp_obj[exp2_name]

    SF_map = gpd.read_file(exp1.shp_file_path)
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
    SF_map.plot(ax=ax, column='infra_cap', norm=TwoSlopeNorm(0, vmin=min(SF_map['infra_cap']),
                    vmax=max(SF_map['infra_cap'])), cmap=plt.get_cmap('RdBu_r'), legend=True, edgecolor='black')
    SF_map.apply(lambda x: ax.annotate(f'{x.name:.0f}', xy=x.geometry.centroid.coords[0], ha='center', fontsize=12, color='black'), axis=1)

    cb_ax = fig.axes[1]
    cb_ax.tick_params(labelsize=14)
    cb_ax.set_ylabel('Installed capacity difference (MW)', fontsize=16)
    plt.title(f'Installed capacity difference ({exp1_label} - {exp2_label}) [MW]', fontsize=18, pad=20)

    plt.xlim((-122.525, -122.35))
    plt.ylim((37.7, 37.850))
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(PATH_TO_PAPER_PLOTS, f'{filename_prefix}_heatmap_infra_diff.png'), dpi=300, bbox_inches='tight')

def cost_sensitivity_to_vehicle_design(df: pd.DataFrame, exp_folder_name_to_attributes: dict, filename_prefix: str = ""):
    df_filtered = df[df['name'].isin([folder_name.split('.')[0] for folder_name in exp_folder_name_to_attributes.keys()])].copy()
    cost_category_to_label = {
        # 'total_d': 'Total',
        # 'fleet_d': 'Fleet',
        # 'dist_total_d': 'Distance',
        'elec_energy_d': 'Electricity Energy Charges',
        'infra_d': 'Charging Infrastructure',
        'dist_rebal_d': 'Distance: Rebalancing Only',
        'elec_demand_d': 'Electricity Demand Charges',
    }

    base_case_name = None
    energy_consump_data = []
    batt_data = []

    for exp_folder_name, attributes in exp_folder_name_to_attributes.items():
        exp_name = exp_folder_name.split('.')[0]
        index = df_filtered[df_filtered['name'] == exp_name].index[0]

        if 'ENERGY_CONSUMP' in attributes['type']:
            energy_consump_data.append((attributes['energy_consump'], index))
        if 'BATT' in attributes['type']:
            batt_data.append((attributes['batt'], index))
        if {'ENERGY_CONSUMP', 'BATT'}.issubset(attributes['type']):
            base_case_name = exp_name

    if base_case_name is None:
        raise ValueError("Base case not found. Please ensure that the base case is defined in the input data.")

    base_case_index = df_filtered[df_filtered['name'] == base_case_name].index[0]
    energy_consump_data.sort()
    batt_data.sort()

    energy_consump_list, energy_consump_indices = zip(*energy_consump_data)
    batt_list, batt_indices = zip(*batt_data)
    energy_consump_indices = list(energy_consump_indices)
    batt_indices = list(batt_indices)

    sns.set_theme(style="whitegrid", font_scale=1.5)

    fig, ax = plt.subplots(figsize=(12, 8))
    palette = sns.color_palette("muted", len(cost_category_to_label))

    for i, (cost_category, cost_label) in enumerate(cost_category_to_label.items()):
        ax.plot(energy_consump_list,
                df_filtered.loc[energy_consump_indices, cost_category] / df_filtered.loc[
                    base_case_index, cost_category] * 100,
                label=cost_label, marker='o', markersize=8, linestyle='-', color=palette[i])

    ax.axhline(y=100, color='gray', linestyle='--', linewidth=1.5)
    ax.set_ylim(60, 140)
    ax.set_xlabel('Energy Consumption (Wh/mi)', fontsize=16)
    ax.set_ylabel('Cost Relative to Base Case (%)', fontsize=16)
    ax.set_title('Cost Category Sensitivity to Energy Consumption', fontsize=18, pad=20)
    ax.legend(title="Cost Categories", fontsize=12, title_fontsize=14, loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(os.path.join(PATH_TO_PAPER_PLOTS, f'{filename_prefix}_cost_sensitivity_energy_consump.png'), dpi=300,
                bbox_inches='tight')

    fig, ax = plt.subplots(figsize=(12, 8))
    palette = sns.color_palette("muted", len(cost_category_to_label))

    for i, (cost_category, cost_label) in enumerate(cost_category_to_label.items()):
        ax.plot(batt_list,
                df_filtered.loc[batt_indices, cost_category] / df_filtered.loc[base_case_index, cost_category] * 100,
                label=cost_label, marker='o', markersize=8, linestyle='-', color=palette[i])

    ax.axhline(y=100, color='gray', linestyle='--', linewidth=1.5)
    ax.set_ylim(60, 140)
    ax.set_xlabel('Battery Size (kWh)', fontsize=16)
    ax.set_ylabel('Cost Relative to Base Case (%)', fontsize=16)
    ax.set_title('Cost Category Sensitivity to Battery Size', fontsize=18, pad=20)
    ax.legend(title="Cost Categories", fontsize=12, title_fontsize=14, loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(os.path.join(PATH_TO_PAPER_PLOTS, f'{filename_prefix}_cost_sensitivity_batt.png'), dpi=300,
                bbox_inches='tight')

def cost_carbon_and_elec_fleet_share_vs_carbon_price_or_vehicle_price(
        df: pd.DataFrame,
        exp_folder_name_to_attributes: dict,
        x_axis_variable: str,
        filename_prefix: str = "",
        title_prefix: str = "",
        elec_fleet_share: bool = False
):
    df_filtered = df[df['name'].isin([folder_name.split('.')[0] for folder_name in exp_folder_name_to_attributes.keys()])].copy()

    if x_axis_variable == 'carbon_price':
        x_axis_attr_name = 'dptCO2'
        x_axis_filename_string = 'carbon_price'
        x_axis_label = 'Carbon Price ($/tCO2)'
    elif x_axis_variable == 'vehicle_price':
        x_axis_attr_name = 'ev_price_ratio'
        x_axis_filename_string = 'vehicle_price'
        x_axis_label = 'Electric Vehicle : Hybrid ICE Price Ratio'
    else:
        raise ValueError("x_axis_variable must be either 'carbon_price' or 'vehicle_price'.")

    x_axis_data = []

    for exp_folder_name, attributes in exp_folder_name_to_attributes.items():
        exp_name = exp_folder_name.split('.')[0]
        index = df_filtered[df_filtered['name'] == exp_name].index[0]
        x_axis_data.append((attributes[x_axis_attr_name], index))

    x_axis_data.sort()
    x_axis_values, x_axis_indices = zip(*x_axis_data)

    df_filtered['elec_energy_elec_demand_dist_rebal_infra_d'] = df_filtered['elec_energy_d'] + df_filtered['elec_demand_d'] + df_filtered['dist_rebal_d'] + df_filtered['infra_d']

    if elec_fleet_share:
        df_filtered['elec_fleet_share'] = df_filtered['fleet_sizes'].apply(lambda x: x[0] / sum(x) * 100)

    sns.set_theme(style="whitegrid", font_scale=1.5)
    fig, ax1 = plt.subplots(figsize=(12, 8))
    ax2 = ax1.twinx()
    ax3 = None
    if elec_fleet_share:
        ax3 = ax1.twinx()
        ax3.spines["right"].set_position(("axes", 1.15))
        ax3.spines["right"].set_visible(True)
    palette = sns.color_palette("muted", 3)
    ax1.plot(
        x_axis_values,
        df_filtered.loc[list(x_axis_indices), 'elec_energy_elec_demand_dist_rebal_infra_d'],
        marker='o', markersize=8, linestyle='-', color=palette[0]
    )
    ax2.plot(
        x_axis_values,
        df_filtered.loc[list(x_axis_indices), 'carbon_total_tco2'],
        marker='o', markersize=8, linestyle='-', color=palette[1]
    )
    if elec_fleet_share:
        ax3.plot(
            x_axis_values,
            df_filtered.loc[list(x_axis_indices), 'elec_fleet_share'],
            marker='o', markersize=8, linestyle='--', color=palette[2], label='Electric Fleet Share'
        )

    ax1.set_xlabel(x_axis_label, fontsize=16)
    ax1.set_ylabel(
        'Cost: electricity energy charges + electricity demand charges\n+ distance (rebalancing only) + charging infrastructure ($)',
        fontsize=16, color=palette[0]
    )
    ax1.tick_params(axis='y', labelcolor=palette[0])
    ax2.set_ylabel('Carbon Emissions (tCO2)', fontsize=16, color=palette[1])
    ax2.tick_params(axis='y', labelcolor=palette[1])
    ax2.grid(False)
    if elec_fleet_share:
        ax3.set_ylabel('Electric Fleet Share (%)', fontsize=16, color=palette[2])
        ax3.tick_params(axis='y', labelcolor=palette[2])
        ax3.grid(False)
        ax3.set_ylim(top=100)
    title = 'Cost, Carbon Emissions, and Electric Fleet Share vs Carbon Price' if elec_fleet_share else 'Cost and Carbon Emissions vs Carbon Price'
    title = title_prefix + title
    ax1.set_title(title, fontsize=18, pad=20)
    plt.tight_layout()

    filename = f'{filename_prefix}_cost_carbon_and_elec_fleet_share_vs_{x_axis_filename_string}' if elec_fleet_share else f'{filename_prefix}_cost_and_carbon_vs_{x_axis_filename_string}'
    plt.savefig(os.path.join(PATH_TO_PAPER_PLOTS, filename), dpi=300, bbox_inches='tight')

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

def load_exp(exp_name: str, exp_name_to_exp_obj: dict):
    if exp_name in exp_name_to_exp_obj:
        return
    exp_path = os.path.join(PATH_TO_RESULTS_DATA, exp_name, 'PAMoDFleet', f'{exp_name}.p')
    with open(exp_path, 'rb') as f:
        exp = pickle.load(f)
    exp_name_to_exp_obj[exp_name] = exp

if __name__ == "__main__":
    df = pd.DataFrame(columns=['name'])
    exp_name_to_exp_obj = {}

    sections_to_plot = [
        # "sec2",
        # "sec3",
        # "sec4",
        "sec5",
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

    if "sec3" in sections_to_plot:
        sec3_exp_folder_names_to_label = (
            ('dacia_spring_electric', 'Joint'),
            ('dacia_spring_electric_use_baseline', 'Baseline'),
        )
        for exp_folder_name, label in sec3_exp_folder_names_to_label:
            df = add_exp_to_df(df, exp_folder_name, label)
            load_exp(exp_folder_name, exp_name_to_exp_obj)
        heatmap_infra_diff(exp_name_to_exp_obj, sec3_exp_folder_names_to_label, 'sec3')

    if "sec4" in sections_to_plot:
        sec4_exp_folder_names_to_attributes = {
            'dacia_175Whpkm' : {
                'label': '175 Wh/mi, 25 kWh',
                'type': ['ENERGY_CONSUMP'],
                'energy_consump': 175,
                'batt': 25,
                'range': 91.870,
            },
            'dacia_165Whpkm' : {
                'label': '165 Wh/mi, 25 kWh',
                'type': ['ENERGY_CONSUMP'],
                'energy_consump': 165,
                'batt': 25,
                'range': 97.035,
            },
            'dacia_155Whpkm' : {
                'label': '155 Wh/mi, 25 kWh',
                'type': ['ENERGY_CONSUMP'],
                'energy_consump': 155,
                'batt': 25,
                'range': 102.815,
            },
            'dacia_145Whpkm' : {
                'label': '145 Wh/mi, 25 kWh',
                'type': ['ENERGY_CONSUMP', 'BATT'],
                'energy_consump': 145,
                'batt': 25,
                'range': 109.327,
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
            'dacia_22_19kWh' : {
                'label': '145 Wh/mi, 22.190 kWh',
                'type': ['BATT'],
                'energy_consump': 145,
                'batt': 22.190,
                'range': 97.039,
            },
            'dacia_23_511kWh' : {
                'label': '145 Wh/mi, 23.511 kWh',
                'type': ['BATT'],
                'energy_consump': 145,
                'batt': 23.511,
                'range': 102.815,
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
        for exp_folder_name, attributes in sec4_exp_folder_names_to_attributes.items():
            df = add_exp_to_df(df, exp_folder_name, attributes['label'])
        cost_sensitivity_to_vehicle_design(df, sec4_exp_folder_names_to_attributes, 'sec4')

    if "sec5" in sections_to_plot:
        sec5_1_exp_folder_names_to_attributes = {
            'dacia_spring_electric': {
                'label': 'Crossover City Car, $0/tCO2',
                'dptCO2': 0,
            },
            'dacia_44dptCO2.zip': {
                'label': 'Crossover City Car, $44/tCO2',
                'dptCO2': 44,
            },
            'dacia_185dptCO2.zip': {
                'label': 'Crossover City Car, $185/tCO2',
                'dptCO2': 185,
            },
            'dacia_413dptCO2.zip': {
                'label': 'Crossover City Car, $413/tCO2',
                'dptCO2': 413,
            },
            'dacia_805dptCO2.zip': {
                'label': 'Crossover City Car, $805/tCO2',
                'dptCO2': 805,
            },
        }
        for exp_folder_name, attributes in sec5_1_exp_folder_names_to_attributes.items():
            df = add_exp_to_df(df, exp_folder_name, attributes['label'])
        cost_carbon_and_elec_fleet_share_vs_carbon_price_or_vehicle_price(df, sec5_1_exp_folder_names_to_attributes, 'carbon_price', 'sec5_1', "Crossover City Car:\n")

        sec5_2_exp_folder_names_to_attributes = {
            'ioniq_hybrid_electric_1_311.zip': {
                'label': 'Compact Liftback Mixed Fleet of Hybrid ICE and Electric Vehicles, $0/tCO2',
                'dptCO2': 0,
            },
            'ioniq_hybrid_electric_1_311_44dptCO2.zip': {
                'label': 'Compact Liftback Mixed Fleet of Hybrid ICE and Electric Vehicles, $44/tCO2',
                'dptCO2': 44,
            },
            'ioniq_hybrid_electric_1_311_185dptCO2.zip': {
                'label': 'Compact Liftback Mixed Fleet of Hybrid ICE and Electric Vehicles, $185/tCO2',
                'dptCO2': 185,
            },
            'ioniq_hybrid_electric_1_311_413dptCO2.zip': {
                'label': 'Compact Liftback Mixed Fleet of Hybrid ICE and Electric Vehicles, $413/tCO2',
                'dptCO2': 413,
            },
            'ioniq_hybrid_electric_1_311_805dptCO2.zip': {
                'label': 'Compact Liftback Mixed Fleet of Hybrid ICE and Electric Vehicles, $805/tCO2',
                'dptCO2': 805,
            },
        }
        for exp_folder_name, attributes in sec5_2_exp_folder_names_to_attributes.items():
            df = add_exp_to_df(df, exp_folder_name, attributes['label'])
        cost_carbon_and_elec_fleet_share_vs_carbon_price_or_vehicle_price(df, sec5_2_exp_folder_names_to_attributes, 'carbon_price','sec5_2', "Compact Liftback Mixed Fleet of Hybrid ICE and Electric Vehicles:\n", True)

        sec5_3_exp_folder_names_to_attributes = {
            'ioniq_hybrid_electric_1_311.zip': {
                'label': 'Compact Liftback Mixed Fleet of Hybrid ICE and Electric Vehicles, 1.311',
                'ev_price_ratio': 1.311,
            },
            'ioniq_hybrid_electric_1_234.zip': {
                'label': 'Compact Liftback Mixed Fleet of Hybrid ICE and Electric Vehicles, 1.234',
                'ev_price_ratio': 1.234,
            },
            'ioniq_hybrid_electric_1_156.zip': {
                'label': 'Compact Liftback Mixed Fleet of Hybrid ICE and Electric Vehicles, 1.156',
                'ev_price_ratio': 1.156,
            },
            'ioniq_hybrid_electric_1_078.zip': {
                'label': 'Compact Liftback Mixed Fleet of Hybrid ICE and Electric Vehicles, 1.078',
                'ev_price_ratio': 1.078,
            },
            'ioniq_hybrid_electric_1_000.zip': {
                'label': 'Compact Liftback Mixed Fleet of Hybrid ICE and Electric Vehicles, 1.000',
                'ev_price_ratio': 1.000,
            },
        }
        for exp_folder_name, attributes in sec5_3_exp_folder_names_to_attributes.items():
            df = add_exp_to_df(df, exp_folder_name, attributes['label'])
        cost_carbon_and_elec_fleet_share_vs_carbon_price_or_vehicle_price(df, sec5_3_exp_folder_names_to_attributes, 'vehicle_price', 'sec5_3', "Compact Liftback Mixed Fleet of Hybrid ICE and Electric Vehicles:\n", True)


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