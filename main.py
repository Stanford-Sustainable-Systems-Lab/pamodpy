import pickle
import os
import argparse

import pamodpy

def main(json_file, opt_time_limit, threads):
    if json_file is None:
        new_experiment_config_jsons = [
            os.path.join('experiment_configs', 'sample.json')
            # os.path.join('experiment_configs', 'dacia_spring_sherlock_SC9III_incentive.json')
            # os.path.join('experiment_configs', 'efficiency_and_battery', 'eff_170_batt_26.8.json'),
            # os.path.join('experiment_configs', 'efficiency_and_battery', 'eff_155_batt_26.8.json'),
            # os.path.join('experiment_configs', 'efficiency_and_battery', 'eff_140_batt_26.8.json'),
            # os.path.join('experiment_configs', 'efficiency_and_battery', 'eff_185_batt_23.06.json'),
            # os.path.join('experiment_configs', 'efficiency_and_battery', 'eff_185_batt_24.79.json'),
            # os.path.join('experiment_configs', 'efficiency_and_battery', 'eff_185_batt_29.16.json'),
            # os.path.join('experiment_configs', 'efficiency_and_battery', 'eff_185_batt_31.99.json'),
            # os.path.join('experiment_configs', 'efficiency_and_battery', 'eff_185_batt_35.41.json'),
            # os.path.join('experiment_configs', 'ioniq_electric_hybrid_pcarb0_price31_sherlock.json'),
            # os.path.join('experiment_configs', 'ioniq_electric_hybrid_pcarb0_price16_sherlock.json'),
            # os.path.join('experiment_configs', 'ioniq_electric_hybrid_pcarb0_price0_sherlock.json'),
            # os.path.join('experiment_configs', 'ioniq_electric_hybrid_pcarb51_price31_sherlock.json'),
            # os.path.join('experiment_configs', 'ioniq_electric_hybrid_pcarb51_price16_sherlock.json'),
            # os.path.join('experiment_configs', 'ioniq_electric_hybrid_pcarb51_price0_sherlock.json'),
            # os.path.join('experiment_configs', 'ioniq_electric_hybrid_pcarb185_price31_sherlock.json'),
            # os.path.join('experiment_configs', 'ioniq_electric_hybrid_pcarb185_price16_sherlock.json'),
            # os.path.join('experiment_configs', 'ioniq_electric_hybrid_pcarb185_price0_sherlock.json')
            # os.path.join('experiment_configs', 'nyc_3hr_base_infra.json'),
            # os.path.join('experiment_configs', 'nyc_3hr_opt_infra.json')
        ]
    else:
        new_experiment_config_jsons = [os.path.join('experiment_configs', json_file),]

    for new_experiment_config_json in new_experiment_config_jsons:
        experiment = pamodpy.load_experiment(new_experiment_config_json)
        print("Running experiment {}...".format(experiment.name))
        experiment.run(opt_time_limit=opt_time_limit, threads=threads)
        experiment.save()
        experiment.plot()


    past_experiment_paths = [
        # 'results/NYC_manh/nyc_3hr_opt_infra/PAMoDFleet/nyc_3hr_opt_infra.p'
        # os.path.join('experiment_configs', 'efficiency_and_battery', 'eff_185_batt_26.8.json'),
        ]

    for past_experiment_path in past_experiment_paths:
        with open(past_experiment_path, 'rb') as f:
            experiment = pickle.load(f)
        # experiment.plot(plot_graphs=False, plot_anim=['numVeh', 'chargingPower'])
        experiment.plot()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run P-AMoD Optimization")
    parser.add_argument("--json_file", type=str, default=None, help="The name of the json file to run")
    parser.add_argument("--opt_time_limit", type=int, default=60*60*24*6,
                        help="The time limit for the optimization in seconds")
    parser.add_argument("--threads", type=int, default=30, help="The number of CPU threads to use for the optimization")
    args = parser.parse_args()
    main(args.json_file, args.opt_time_limit, args.threads)
