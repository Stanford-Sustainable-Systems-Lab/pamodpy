import ast
import os
import io
import json
import zipfile
from typing import Union

import matplotlib.pyplot as plt
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

def boxplot_stackedbar(df: pd.DataFrame, experiment_folder_name_to_labels: dict, filename_prefix: str = "") -> None:
    df_filtered = df[df['name'].isin([folder_name.split('.')[0] for folder_name in experiment_folder_name_to_labels.keys()])].copy()
    cost_category_to_label = {
        'total_d': 'Total',
        'fleet_d': 'Fleet',
        'dist_total_d': 'Distance',
        'elec_energy_d': 'Elecricity Energy Charges',
        'infra_d': 'Charging Infrastructure',
        'dist_rebal_d': 'Distance - Rebalancing',
        'elec_demand_d': 'Electricity Demand Charges',
    }
    columns = []
    for cost_type in cost_category_to_label.keys():
        col_name = cost_type + "_per_passenger_mile"
        df_filtered.loc[:, col_name] = df_filtered[cost_type] / df_filtered['dist_passenger_mi']
        columns.append(col_name)
    df_boxplot = df_filtered[columns]
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.boxplot(data=df_boxplot, ax=ax, color='lightblue', showfliers=False)
    sns.stripplot(data=df_boxplot, ax=ax, color='black', alpha=0.5, jitter=True)
    ax.set_xticklabels(cost_category_to_label.values(), rotation=45)
    ax.set_ylabel('Cost per passenger mile ($/mi)')
    ax.set_xlabel('Cost Category')
    ax.set_title('Cost Breakdown Across Different Vehicle Types')
    plt.tight_layout()
    plt.savefig(os.path.join(PATH_TO_PAPER_PLOTS, f"{filename_prefix}_boxplot.png"), dpi=300)

    fig, ax = plt.subplots(figsize=(12, 8))
    bottom = np.zeros(len(df_filtered))
    for i, cost_type in enumerate(cost_category_to_label.keys()):
        if cost_type == 'total_d':
            continue
        col_name = cost_type + "_per_passenger_mile"
        ax.bar(df_filtered['label'], df_filtered[col_name], label=cost_category_to_label[cost_type], bottom=bottom)
        bottom += df_filtered[col_name]
    ax.set_ylabel('Cost per passenger mile ($/mi)')
    ax.set_xlabel('Vehicle Type')
    ax.set_title('Cost Breakdown Across Different Vehicle Types')
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(PATH_TO_PAPER_PLOTS, f'{filename_prefix}_stackedbar.png'), dpi=300)

def heatmap_infra_diff(df: pd.DataFrame, experiment_folder_name_to_labels: dict, filename_prefix: str = "") -> None:
    df_filtered = df[df['name'].isin([folder_name.split('.')[0] for folder_name in experiment_folder_name_to_labels.keys()])].copy()
    d

def add_experiment_to_df(df: pd.DataFrame, experiment_folder_name: str, experiment_label: str) -> pd.DataFrame:
    experiment_name = experiment_folder_name.split('.')[0]

    if experiment_name in df['name'].values:
        return df

    if experiment_folder_name.endswith('.zip'):
        zip_file = zipfile.ZipFile(os.path.join(PATH_TO_RESULTS_DATA, experiment_folder_name), 'r')
        with zip_file.open('PAMoDFleet/print_log.txt', 'r') as f:
            df_row = get_data_from_log_file(io.TextIOWrapper(f, encoding='utf-8'), experiment_name, experiment_label)
    else:
        log_file = os.path.join(PATH_TO_RESULTS_DATA, experiment_name, 'PAMoDFleet', 'print_log.txt')
        with open(log_file, 'r') as f:
            df_row = get_data_from_log_file(f, experiment_name, experiment_label)

    return pd.concat([df, df_row], ignore_index=True)


def get_data_from_log_file(log_file: io.TextIOWrapper, experiment_name: str, label: str) -> pd.DataFrame:
    lines = log_file.readlines()
    data = {}
    data['name'] = experiment_name
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


if __name__ == "__main__":
    df = pd.DataFrame(columns=['name'])

    sections_to_plot = [
        "sec2",
        "sec3",
        "sup1",
    ]

    if "sec2" in sections_to_plot:
        sec2_experiment_folder_names_to_label = {
            'dacia_spring_electric' : 'Crossover City Car',
            'chevrolet_bolt_ev.zip': 'Subcompact Hatchback',
            'hyundai_ioniq_electric.zip': 'Compact Liftback',
            'tesla_model_3.zip': 'Mid-size Sedan',
            'hyundai_ioniq_5.zip': 'Compact Crossover SUV',
            'jaguar_ipace.zip': 'Crossover SUV',
        }
        for experiment_folder_name, label in sec2_experiment_folder_names_to_label.items():
            df = add_experiment_to_df(df, experiment_folder_name, label)
        boxplot_stackedbar(df, sec2_experiment_folder_names_to_label, 'sec2')

    if "sec3" in sections_to_plot:
        sec3_experiment_folder_names_to_label = {
            'dacia_spring_electric' : 'Crossover City Car: Jointly Optimized Infrastructure',
            'dacia_spring_electric_use_baseline': 'Crossover City Car: Baseline Infrastructure',
        }
        for experiment_folder_name, label in sec3_experiment_folder_names_to_label.items():
            df = add_experiment_to_df(df, experiment_folder_name, label)


    if "sup1" in sections_to_plot:
        sup1_experiment_folder_names_to_label = {
            'dacia_spring_electric_b20.zip' : 'Crossover City Car',
            'chevrolet_bolt_ev_b20.zip': 'Subcompact Hatchback',
            'hyundai_ioniq_electric_b20.zip': 'Compact Liftback',
            'tesla_model_3_b20.zip': 'Mid-size Sedan',
            'hyundai_ioniq_5_b20.zip': 'Compact Crossover SUV',
            'jaguar_ipace_b20.zip': 'Crossover SUV',
        }
        for experiment_folder_name, label in sup1_experiment_folder_names_to_label.items():
            df = add_experiment_to_df(df, experiment_folder_name, label)
        boxplot_stackedbar(df, sup1_experiment_folder_names_to_label, 'sup1')