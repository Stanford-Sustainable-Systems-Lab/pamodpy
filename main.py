import pamodpy
import pickle
import os
import config

def main():
    new_experiment_config_jsons = [
        # os.path.join('experiment_configs', 'efficiency_and_battery', 'eff_185_batt_29.16.json'),
        # os.path.join('experiment_configs', 'efficiency_and_battery', 'eff_185_batt_31.99.json'),
        # os.path.join('experiment_configs', 'efficiency_and_battery', 'eff_185_batt_35.41.json')
        # os.path.join('experiment_configs', 'sample_sherlock.json'),
        # os.path.join('experiment_configs', 'nyc_3hr_base_infra.json'),
        os.path.join('experiment_configs', 'nyc_3hr_opt_infra.json')
    ]

    for new_experiment_config_json in new_experiment_config_jsons:
        experiment = pamodpy.load_experiment(new_experiment_config_json)
        print("Running experiment {}...".format(experiment.name))
        experiment.run()
        experiment.save()
        experiment.plot()


    past_experiment_paths = [
        'results/NYC_manh/nyc_3hr_opt_infra/PAMoDFleet/nyc_3hr_opt_infra.p'
        ]

    # for past_experiment_path in past_experiment_paths:
    #     with open(past_experiment_path, 'rb') as f:
    #         experiment = pickle.load(f)
    #     experiment.plot(plot_graphs=True, plot_anim=['numVeh', 'chargingPower'])
    #     experiment.plot()

if __name__ == "__main__":
    main()
