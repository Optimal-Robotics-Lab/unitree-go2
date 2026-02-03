from absl import app, flags

import itertools

import jax
from ml_collections import ConfigDict

from regression import train
from utilities.config import get_default_config

flags.DEFINE_string('dataset_name', None, 'The dataset to use for all experiments.', required=True)
flags.DEFINE_string('scene_file', None, 'The scene to use for all experiments.', required=True)

def generate_experiments(keys, min_size=2, max_size=None):
    if max_size is None:
        max_size = len(keys)
        
    experiments = []

    for r in range(min_size, max_size + 1):
        for combination in itertools.combinations(keys, r):
            name_suffix = "_".join(combination)
            experiment_name = f"experiment_{name_suffix}"
            
            experiment_config = {
                "name": experiment_name,
                "regress_keys": list(combination),
            }
            
            experiments.append(experiment_config)
            
    return experiments

def main(argv):
    summary_results = []

    # Create all combinations of the keys:
    keys = ["friction", "damping", "armature", "qpos0"]
    experiments = generate_experiments(keys)

    dataset_name = flags.FLAGS.dataset_name
    scene_file = flags.FLAGS.scene_file

    for i, exp in enumerate(experiments):
        print(f"Running Experiment : {exp['name']}")

        config = get_default_config()

        config.dataset_name = dataset_name
        config.scene_file = scene_file

        config.wandb.group = "Manual_Sweep_002"
        config.wandb.project = "Parameter-Regression-Sweep-Unitree-Go2"
        
        full_spec = config.regression.to_dict()
        filtered_spec = {k: v for k, v in full_spec.items() if k in exp['regress_keys']}
        config.regression = ConfigDict(filtered_spec)

        jax.clear_caches()
        
        _ = train(config)


if __name__ == "__main__":
    app.run(main)
