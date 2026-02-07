import functools
from absl import app, flags

from pathlib import Path
import pickle

import jax
import jax.numpy as jnp

import mujoco
from mujoco import mjx

from ..utilities import evaluation
from ..utilities.data_utilities import chunk_and_flatten_dataset, shuffle_data
from ..utilities.model_utilities import hydrate_model
from ..utilities.loss_utilities import loss_function
from ..utilities.evaluation import init_function, step_function
from ..utilities.typedefs import Dataset


FLAGS = flags.FLAGS
flags.DEFINE_string('parameter_checkpoint', None, 'Path to the directory containing the regressed parameters and config.pkl', required=True)


def analyze_parameter_coupling(loss_fn, params, batch):
    """
        Computes the Correlation Matrix of the parameters using the Hessian.
    """
    # Flatten the Params:
    flat_params, unflatten_fn = jax.flatten_util.ravel_pytree(params)

    def flat_loss_fn(theta):
        p = unflatten_fn(theta)
        return loss_fn(p, batch)

    H = jax.hessian(flat_loss_fn)(flat_params)

    # Invert to get Covariance (Sigma)
    # Add small epsilon for numerical stability if H is singular
    Sigma = jnp.linalg.inv(H + jnp.eye(H.shape[0]) * 1e-6)

    # Normalize to get Correlation Matrix
    # R_ij = Sigma_ij / sqrt(Sigma_ii * Sigma_jj)
    diag = jnp.diag(Sigma)
    std_devs = jnp.sqrt(diag)
    outer_std = jnp.outer(std_devs, std_devs)

    Correlation = Sigma / outer_std

    return Correlation, unflatten_fn


def main(argv=None):
    # Set up paths:
    package_root = Path(__file__).resolve().parent.parent

    # Load Regressed Parameters and Config:
    parameter_checkpoint_path = Path(FLAGS.parameter_checkpoint) / 'regressed_params.pkl'
    with open(parameter_checkpoint_path, 'rb') as f:
        params = pickle.load(f)

    config_path = Path(FLAGS.parameter_checkpoint) / 'config.pkl'
    with open(config_path, 'rb') as f:
        config = pickle.load(f)

    # Clean up params:
    rename_map = {
        'armature': 'dof_armature',
        'friction': 'dof_frictionloss',
        'damping': 'dof_damping',
    }

    params = {
        rename_map.get(k, k): v
        for k, v in params.items()
        if not k.startswith('initial_')
    }

    # Load and Hydrate MuJoCo Model:
    mj_model_filepath = (package_root / "mjcf/scene_mjx_vendor.xml").resolve()
    mj_model = mujoco.MjModel.from_xml_path(str(mj_model_filepath))
    mj_model = hydrate_model(params, mj_model, config.regression.to_dict())

    # Create Static MJX Model:
    mjx_model_static = mjx.put_model(mj_model, impl="jax")
    n_substeps = int(config.physics.control_rate / mj_model.opt.timestep)

    # Wrap Step Function:
    init_fn = evaluation.init_function
    step_fn = functools.partial(
        evaluation.step_function,
        n_substeps=n_substeps,
    )

    # Load Data:
    datasets = []
    dataset_directories = config.dataset_directories if isinstance(config.dataset_directories, tuple) else (config.dataset_directories,)
    for directory_name in dataset_directories:
        data_path = Path(__file__).resolve().parent / directory_name / 'processed_data.pkl'
        if not data_path.exists():
            raise FileNotFoundError(f"{data_path} not found")

        print(f"Loading dataset: {directory_name}")
        with open(data_path, 'rb') as f:
            data_dict = pickle.load(f)

        effective_window = config.training.window_length + 1
        ds_chunked = chunk_and_flatten_dataset(
            jnp.array(data_dict['qpos']),
            jnp.array(data_dict['qvel']),
            jnp.array(data_dict['actuator_force']),
            jnp.array(data_dict['ctrl']),
            effective_window,
        )
        datasets.append(ds_chunked)

    dataset = jax.tree_util.tree_map(
        lambda *arrays: jnp.concatenate(arrays, axis=0),
        *datasets
    )

    # Sample a Random Batch
    key = jax.random.key(config.training.seed)
    key, subkey = jax.random.split(key)
    sample_fn = functools.partial(shuffle_data, dataset=dataset, batch_size=config.training.minibatch_size)
    batch = sample_fn(subkey)

    # Loss Function:
    loss_fn = functools.partial(
        loss_function,
        model_static=mjx_model_static,
        init_function=init_fn,
        step_function=step_fn,
        objective_function=evaluation.get_objective_fn(config.loss.type),
        objective_weights=config.loss.weights.to_dict(),
        regression_spec=config.regression.to_dict(),
    )

    # Compute Correlation Matrix:
    correlation_matrix, unflatten_fn = analyze_parameter_coupling(
        loss_fn,
        params,
        batch,
    )


if __name__ == '__main__':
    app.run(main)
