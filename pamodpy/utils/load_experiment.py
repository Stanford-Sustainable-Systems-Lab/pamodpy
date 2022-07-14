import json
from pathlib import Path

import config

def load_experiment(json_fname):
    with open(json_fname, "r") as f:
        experiment_config = json.load(f)

    name = Path(json_fname).stem
    experiment_config['name'] = name
    experiment_type = experiment_config['experiment_type']
    algorithm = experiment_config['algorithm']
    region = experiment_config['region']
    config.current_experiment_region = region

    if experiment_type == 'PAMoDFleet':
        from pamodpy.experiments.PAMoDFleet import PAMoDFleet
        return PAMoDFleet(experiment_config)
    else:
        raise ValueError('"{}" is not a valid experiment_type.'.format(experiment_type))
