import json
from pathlib import Path

from pamodpy.experiments.PAMoDFleet import PAMoDFleet
from pamodpy.experiments.Experiment import SF_25


def load_experiment(json_fname):
    with open(json_fname, "r") as f:
        config = json.load(f)

    name = Path(json_fname).stem
    config['name'] = name
    experiment_type = config['experiment_type']
    algorithm = config['algorithm']
    region = config['region']
    experiment = None

    if experiment_type == 'PAMoDFleet':
        if region == 'SF_25':
            experiment = PAMoDFleet(config)
        else:
            raise ValueError('"{}" is not a valid region.'.format(region))
        return experiment
    else:
        raise ValueError('"{}" is not a valid experiment_type.'.format(experiment_type))
