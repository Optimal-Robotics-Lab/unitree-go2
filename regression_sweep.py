from absl import app, flags

import itertools

import jax
from ml_collections import ConfigDict

from regression.regression import train
from regression.utilities.config import get_default_config

flags.DEFINE_list('datasets', None, 'The data/datasets to use for all experiments.', required=True)
flags.DEFINE_string('evaluation_dataset', None, 'The data/dataset to use for evaluation.', required=True)
flags.DEFINE_list('scene_files', None, 'The scene/scenes to use for all experiments.', required=True)
flags.DEFINE_string('group', None, 'The group to use for all experiments.', required=True)
flags.DEFINE_integer('seed', 42, 'JAX rng seed.')


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
    # Create all combinations of the keys:
    keys = ["dof_frictionloss", "dof_damping", "dof_armature", "log_cholesky_inertia"]
    experiments = generate_experiments(keys, min_size=4, max_size=len(keys))


    datasets = flags.FLAGS.datasets
    evaluation_dataset = flags.FLAGS.evaluation_dataset
    scene_files = flags.FLAGS.scene_files

    for scene_file in scene_files:
        for i, exp in enumerate(experiments):
            print(f"Running Experiment : {exp['name']}")

            config = get_default_config()

            config.training.seed = flags.FLAGS.seed
            config.training.num_epochs = 20

            config.loss.type = 'mse'

            config.datasets = datasets
            config.evaluation_dataset = evaluation_dataset
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
