import matplotlib.pyplot as plt
import numpy as np

experiments = {
                  "dacia_spring_sherlock": {
                      "energy_consumption_Wh_per_km": 139,
                      "battery_size_kWh": 26.8,
                      "effective_range_km": 149.96,
                      "total_cost": 5991439.32481007,
                      "energy_cost": 960296.6161605711,
                      "demand_cost": 18793.23460428,
                      "infra_cost": 80889.83320893366,
                      "rebal_cost": 75981.98830131195,
                      "fleet_cost": 2812654.9519490437
                  },
                  "eff_140_batt_26.8": {
                        "energy_consumption_Wh_per_km": 140,
                        "battery_size_kWh": 26.8,
                        "effective_range_km": 148.89,
                        "total_cost": 5992973.45659605,
                        "energy_cost": 961749.2104114273,
                        "demand_cost": 18816.86860192,
                        "infra_cost": 80991.78265909581,
                        "rebal_cost": 75937.91772022752,
                        "fleet_cost": 2812654.976617341
                  },
                  "eff_155_batt_26.8": {
                        "energy_consumption_Wh_per_km": 155,
                        "battery_size_kWh": 26.8,
                        "effective_range_km": 134.48,
                        "total_cost": 6088359.06513101,
                        "energy_cost": 1044203.6245816405,
                        "demand_cost": 21060.52561973,
                        "infra_cost": 90648.75721431604,
                        "rebal_cost": 76968.51212350973,
                        "fleet_cost": 2812654.945005825
                  },
                  "eff_170_batt_26.8": {
                      "energy_consumption_Wh_per_km": 170,
                      "battery_size_kWh": 26.8,
                      "effective_range_km": 122.61,
                      "total_cost": 6210170.21347768,
                      "energy_cost": 1146335.3894898747,
                      "demand_cost": 24455.69815851,
                      "infra_cost": 105442.76006355147,
                      "rebal_cost": 78458.71929131876,
                      "fleet_cost": 2812654.945888446
                  },
                  "eff_185_batt_26.8": {
                      "energy_consumption_Wh_per_km": 185,
                      "battery_size_kWh": 26.8,
                      "effective_range_km": 112.67,
                      "total_cost": 6345473.43668692,
                      "energy_cost": 1256092.420256431,
                      "demand_cost": 27575.31640787,
                      "infra_cost": 122280.98519030922,
                      "rebal_cost": 84047.03960764378,
                      "fleet_cost": 2812654.974638858
                  },
                  "eff_200_batt_26.8": {
                      "energy_consumption_Wh_per_km": 200,
                      "battery_size_kWh": 26.8,
                      "effective_range_km": 104.22,
                      "total_cost": 6503021.61560038,
                      "energy_cost": 1398496.558661335,
                      "demand_cost": 28838.98257214,
                      "infra_cost": 135484.8770219479,
                      "rebal_cost": 84723.5337954046,
                      "fleet_cost": 2812654.962964963
                  },
                  "eff_215_batt_26.8": {
                      "energy_consumption_Wh_per_km": 215,
                      "battery_size_kWh": 26.8,
                      "effective_range_km": 96.95,
                      "total_cost": 6752625.48953828,
                      "energy_cost": 1530189.3128282682,
                      "demand_cost": 30330.46468593,
                      "infra_cost": 138983.18029340595,
                      "rebal_cost": 78832.3985893879,
                      "fleet_cost": 2931467.4325553253
                  },
                  "eff_185_batt_23.06": {
                      "energy_consumption_Wh_per_km": 185,
                      "battery_size_kWh": 23.06,
                      "effective_range_km": 96.95,
                      "total_cost": 6533545.27591163,
                      "energy_cost": 1337973.0568375762,
                      "demand_cost": 26474.18098734,
                      "infra_cost": 117508.87454606232,
                      "rebal_cost": 77299.03980554825,
                      "fleet_cost": 2931467.4231491596
                  },
                  "eff_185_batt_24.79": {
                      "energy_consumption_Wh_per_km": 185,
                      "battery_size_kWh": 24.79,
                      "effective_range_km": 104.22,
                      "total_cost": 6399209.59146982,
                      "energy_cost": 1310921.617550546,
                      "demand_cost": 27279.00927929,
                      "infra_cost": 121931.94870084692,
                      "rebal_cost": 83599.31675201772,
                      "fleet_cost": 2812654.9986011526
                  },
                  "eff_185_batt_29.16": {
                      "energy_consumption_Wh_per_km": 185,
                      "battery_size_kWh": 29.16,
                      "effective_range_km": 122.61,
                      "total_cost": 6317902.05933443,
                      "energy_cost": 1242767.8914980325,
                      "demand_cost": 26548.90264073,
                      "infra_cost": 114278.24457187949,
                      "rebal_cost": 78829.08796625343,
                      "fleet_cost": 2812655.232071445
                  },
                  "eff_185_batt_31.99": {
                      "energy_consumption_Wh_per_km": 185,
                      "battery_size_kWh": 31.99,
                      "effective_range_km": 134.48,
                      "total_cost": 6314624.35005589,
                      "energy_cost": 1245639.4977102103,
                      "demand_cost": 25342.8751024,
                      "infra_cost": 109143.8580316274,
                      "rebal_cost": 79020.44277460076,
                      "fleet_cost": 2812654.975851188
                  },
                  "eff_185_batt_35.41": {
                      "energy_consumption_Wh_per_km": 185,
                      "battery_size_kWh": 35.41,
                      "effective_range_km": 148.89,
                      "total_cost": 6309347.57620087,
                      "energy_cost": 1246007.790914496,
                      "demand_cost": 24645.45563869,
                      "infra_cost": 106078.91088220768,
                      "rebal_cost": 77137.76917430629,
                      "fleet_cost": 2812654.9490052625
                  }
}

efficiency_experiments_names = ['dacia_spring_sherlock', 'eff_140_batt_26.8', 'eff_155_batt_26.8', 'eff_170_batt_26.8',
                                'eff_185_batt_26.8', 'eff_200_batt_26.8', 'eff_215_batt_26.8']
efficiency = [experiments[name]["energy_consumption_Wh_per_km"] for name in efficiency_experiments_names]
ranges_efficiency = [experiments[name]["effective_range_km"] for name in efficiency_experiments_names]
labels_efficiency = ["{} Wh/km".format(experiments[name]["energy_consumption_Wh_per_km"]) for name in efficiency_experiments_names]
labels_efficiency[0] += "\n(Dacia Spring)"
total_cost_efficiency = [experiments[name]["total_cost"] for name in efficiency_experiments_names]
energy_efficiency = [experiments[name]["energy_cost"] for name in efficiency_experiments_names]
demand_efficiency = [experiments[name]["demand_cost"] for name in efficiency_experiments_names]
infra_efficiency = [experiments[name]["infra_cost"] for name in efficiency_experiments_names]
rebal_efficiency = [experiments[name]["rebal_cost"] for name in efficiency_experiments_names]
fleet_efficiency = [experiments[name]["fleet_cost"] for name in efficiency_experiments_names]

battery_experiments_names = ['eff_185_batt_23.06', 'eff_185_batt_24.79', 'eff_185_batt_26.8',
                                'eff_185_batt_29.16', 'eff_185_batt_31.99', 'eff_185_batt_35.41']
battery = [experiments[name]["battery_size_kWh"] for name in battery_experiments_names]
ranges_battery = [experiments[name]["effective_range_km"] for name in battery_experiments_names]
labels_battery = ["{} kWh".format(experiments[name]["battery_size_kWh"]) for name in battery_experiments_names]
total_cost_battery = [experiments[name]["total_cost"] for name in battery_experiments_names]
energy_battery = [experiments[name]["energy_cost"] for name in battery_experiments_names]
demand_battery = [experiments[name]["demand_cost"] for name in battery_experiments_names]
infra_battery = [experiments[name]["infra_cost"] for name in battery_experiments_names]
rebal_battery = [experiments[name]["rebal_cost"] for name in battery_experiments_names]
fleet_battery = [experiments[name]["fleet_cost"] for name in battery_experiments_names]

plt.figure(figsize=(8, 6), dpi=200)
plt.plot(ranges_efficiency, total_cost_efficiency, 'o--', label="Energy Consumption (battery size = 26.8 kWh)")
for i, (x, y, label) in enumerate(zip(ranges_efficiency, total_cost_efficiency, labels_efficiency)):
    if i == 0:
        plt.annotate(label, (x, y),
                     xycoords="data",
                     textcoords="offset points",
                     xytext=(10, -10), ha="center")
    else:
        plt.annotate(label, (x, y),
                     xycoords="data",
                     textcoords="offset points",
                     xytext=(0, 10), ha="center")
plt.plot(ranges_battery, total_cost_battery, 'x--', label="Battery Size (energy consumption = 185 Wh/km")
for x, y, label in zip(ranges_battery, total_cost_battery, labels_battery):
    plt.annotate(label, (x, y),
                 xycoords="data",
                 textcoords="offset points",
                 xytext=(0, -10), ha="center")
plt.xlabel("Effective Range [km]")
plt.ylabel("Total Cost [$USD]")
plt.grid()
plt.legend()
plt.show()


fig = plt.figure(figsize=(8, 6), dpi=200)
bottom = np.zeros(len(efficiency))
for i, (cost_label, data) in enumerate(zip(["Electricity (energy)", "Electricity (demand)", "Infrastructure", "Rebalancing", "Fleet"], [energy_efficiency, demand_efficiency, infra_efficiency, rebal_efficiency, fleet_efficiency])):
    data = np.array(data)
    data -= data[0]
    if i == 0:
        plt.bar(labels_efficiency, data, label=cost_label)
    else:
        plt.bar(labels_efficiency, data, bottom=bottom, label=cost_label)
    bottom += np.array(data)
plt.legend()
plt.ylabel("Cost relative to 139 Wh/km, 26.8 kWh vehicle (Dacia Spring) [$USD]")
plt.show()

fig = plt.figure(figsize=(8, 6), dpi=200)
bottom = np.zeros(len(battery))
for i, (cost_label, data) in enumerate(zip(["Electricity (energy)", "Electricity (demand)", "Infrastructure", "Rebalancing", "Fleet"], [energy_battery, demand_battery, infra_battery, rebal_battery, fleet_battery])):
    data = np.array(data)
    data -= data[-1]
    if i == 0:
        plt.bar(labels_battery, data, label=cost_label)
    else:
        plt.bar(labels_battery, data, bottom=bottom, label=cost_label)
    bottom += np.array(data)
plt.legend()
plt.ylabel("Cost relative to 185 kWh/km, 35.41 kWh vehicle [$USD]")
plt.show()

fig = plt.figure(figsize=(8, 6), dpi=200)
for cost_label, data in zip(["Electricity Cost (Energy) [$USD]", "Electricity Cost (Demand) [$USD]", "Infrastructure Cost [$USD]", "Rebalancing Cost [$USD]", "Fleet Cost [$USD]"], [energy_efficiency, demand_efficiency, infra_efficiency, rebal_efficiency, fleet_efficiency]):
    plt.plot(labels_efficiency, (np.array(data) - data[0]) / data[0], 'o--', label=cost_label)
plt.plot(labels_efficiency, (np.array(efficiency) - efficiency[0]) / efficiency[0], 'kx--', label="Energy Consumption [Wh/km]")
plt.ylabel("Deviation compared to 139 Wh/km (Dacia Spring)")
plt.legend()
plt.show()

fig = plt.figure(figsize=(8, 6), dpi=200)
for cost_label, data in zip(["Electricity Cost (Energy) [$USD]", "Electricity Cost (Demand) [$USD]", "Infrastructure Cost [$USD]", "Rebalancing Cost [$USD]", "Fleet Cost [$USD]"], [energy_battery, demand_battery, infra_battery, rebal_battery, fleet_battery]):
    plt.plot(labels_battery, (np.array(data) - data[0]) / data[0], 'o--', label=cost_label)
plt.plot(labels_battery, (np.array(battery) - battery[0]) / battery[0], 'kx--', label="Battery Size [kWh]")
plt.legend()
plt.ylabel("Deviation compared to 23.06 kWh vehicle")
plt.show()
