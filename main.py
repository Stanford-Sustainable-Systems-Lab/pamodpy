import pamodpy
import pickle
import os

def main():
    new_experiment_config_jsons = [
            os.path.join('experiment_configs', 'sample.json'),
    ]

    for new_experiment_config_json in new_experiment_config_jsons:
        experiment = pamodpy.load_experiment(new_experiment_config_json)
        print("Running experiment {}...".format(experiment.name))
        experiment.run()
        experiment.save()
        experiment.plot()


    past_experiment_paths = [
        ]

    for past_experiment_path in past_experiment_paths:
        with open(past_experiment_path, 'rb') as f:
            experiment = pickle.load(f)
        # experiment.plot(plot_graphs=False, plot_anim=['numVeh', 'chargingPower'])
        experiment.plot()

if __name__ == "__main__":
    main()
