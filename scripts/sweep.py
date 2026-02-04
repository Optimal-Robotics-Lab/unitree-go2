from absl import app, flags

import itertools

import jax
from ml_collections import ConfigDict

from regression import train
from utilities.config import get_default_config

flags.DEFINE_list('datasets', None, 'The dataset/datasets to use for all experiments.', required=True)
flags.DEFINE_list('scene_files', None, 'The scene/scenes to use for all experiments.', required=True)
flags.DEFINE_string('group', None, 'The group to use for all experiments.', required=True)

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
    # keys = ["friction", "damping", "armature", "qpos0"]
    # experiments = generate_experiments(keys)

    # Manually run these experiments:
    experiments = [
        {
            "name": 'experiment_friction_armature_qpos0',
            "regress_keys": ['friction', 'armature', 'qpos0'],
        },
        {
            "name": 'experiment_damping_armature_qpos0',
            "regress_keys": ['damping', 'armature', 'qpos0'],
        },
        {
            "name": 'experiment_friction_damping_armature_qpos0',
            "regress_keys": ["friction", "damping", "armature", "qpos0"],
        },
    ]

    datasets = flags.FLAGS.datasets
    scene_files = flags.FLAGS.scene_files

    for scene_file in scene_files:
        for i, exp in enumerate(experiments):
            print(f"Running Experiment : {exp['name']}")

            config = get_default_config()

            config.dataset_directories = datasets
            config.scene_file = scene_file

            config.wandb.group = flags.FLAGS.group
            config.wandb.project = "Parameter-Regression-Sweep-Unitree-Go2"
            
            full_spec = config.regression.to_dict()
            filtered_spec = {k: v for k, v in full_spec.items() if k in exp['regress_keys']}
            config.regression = ConfigDict(filtered_spec)

            jax.clear_caches()
            
            _ = train(config)


if __name__ == "__main__":
    app.run(main)
