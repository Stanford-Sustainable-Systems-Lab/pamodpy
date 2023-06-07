import pamodpy
import pickle
import os
import config

def main():
    new_experiment_config_jsons = [
        os.path.join('experiment_configs', 'hyundai_ioniq_electric_sherlock.json')
        # os.path.join('experiment_configs', 'efficiency_and_battery', 'eff_170_batt_26.8.json'),
        # os.path.join('experiment_configs', 'efficiency_and_battery', 'eff_155_batt_26.8.json'),
        # os.path.join('experiment_configs', 'efficiency_and_battery', 'eff_140_batt_26.8.json'),
        # os.path.join('experiment_configs', 'efficiency_and_battery', 'eff_185_batt_23.06.json'),
        # os.path.join('experiment_configs', 'efficiency_and_battery', 'eff_185_batt_24.79.json'),
        # os.path.join('experiment_configs', 'efficiency_and_battery', 'eff_185_batt_29.16.json'),
        # os.path.join('experiment_configs', 'efficiency_and_battery', 'eff_185_batt_31.99.json'),
        # os.path.join('experiment_configs', 'efficiency_and_battery', 'eff_185_batt_35.41.json'),
        # os.path.join('experiment_configs', 'chevrolet_bolt_sherlock.json'),
        # os.path.join('experiment_configs', 'ioniq_electric_hybrid_pcarb0_price31_sherlock.json'),
        # os.path.join('experiment_configs', 'ioniq_electric_hybrid_pcarb0_price0_sherlock.json'),
        # os.path.join('experiment_configs', 'ioniq_electric_hybrid_pcarb185_price31_sherlock.json'),
        # os.path.join('experiment_configs', 'ioniq_electric_hybrid_pcarb185_price0_sherlock.json')
        # os.path.join('experiment_configs', 'nyc_3hr_base_infra.json'),
        # os.path.join('experiment_configs', 'nyc_3hr_opt_infra.json')
    ]

    for new_experiment_config_json in new_experiment_config_jsons:
        experiment = pamodpy.load_experiment(new_experiment_config_json)
        print("Running experiment {}...".format(experiment.name))
        experiment.run()
        experiment.save()
        experiment.plot()


    past_experiment_paths = [
        # 'results/NYC_manh/nyc_3hr_opt_infra/PAMoDFleet/nyc_3hr_opt_infra.p'
        # os.path.join('experiment_configs', 'efficiency_and_battery', 'eff_185_batt_26.8.json'),
        ]

    for past_experiment_path in past_experiment_paths:
        with open(past_experiment_path, 'rb') as f:
            experiment = pickle.load(f)
        experiment.plot(plot_graphs=True, plot_anim=['numVeh', 'chargingPower'])
        experiment.plot()

if __name__ == "__main__":
    main()
