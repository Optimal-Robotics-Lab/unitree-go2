from absl import app

from typing import Tuple

from pathlib import Path
import pickle
import functools
import time

import jax
import jax.numpy as jnp

import numpy as np

import optax

import wandb

import mujoco
from mujoco import mjx

from ml_collections import config_flags, ConfigDict

from regression.utilities import model_utilities
from regression.utilities import transforms
from regression.utilities import logging_utils

from regression.utilities.config import get_default_config, process_regression_spec, build_parameter_scale
from regression.utilities.typedefs import Dataset, TrainState
from regression.utilities.constants import JOINT_NAMES
from regression.utilities.data_utilities import chunk_and_flatten_dataset, shuffle_data
from regression.utilities.factories import create_optimizer, get_objective_fn
from regression.utilities.autodiff import forward_mode_value_and_grad
from regression.utilities.loss_utilities import loss_function
from regression.utilities.mjx_utilities import init_function, step_function
from regression.utilities.evaluation import evaluate


_CONFIG = config_flags.DEFINE_config_dict('config', get_default_config())


def train(config: ConfigDict) -> Tuple[TrainState, np.ndarray]:
    # Load MuJoCo Model
    directory = Path(__file__).resolve().parent
    filepath = Path(config.scene_file).resolve()
    mj_model = mujoco.MjModel.from_xml_path(str(filepath))

    # Override Physics Settings:
    solver_map = {
        'newton': mujoco.mjtSolver.mjSOL_NEWTON,
        'cg': mujoco.mjtSolver.mjSOL_CG,
        'pgs': mujoco.mjtSolver.mjSOL_PGS,
    }
    if config.physics.solver in solver_map:
        mj_model.opt.solver = solver_map[config.physics.solver]
    mj_model.opt.iterations = config.physics.iterations
    mj_model.opt.ls_iterations = config.physics.ls_iterations
    mj_model.opt.timestep = config.physics.timestep

    # Create Static MJX Model:
    mjx_model_static = mjx.put_model(mj_model, impl="jax")
    n_substeps = int(config.physics.control_rate / mj_model.opt.timestep)

    # Dataset Directories:
    datasets = []
    training_dataset_directories = config.datasets if isinstance(config.datasets, tuple) else (config.datasets,)
    evaluation_dataset_directory = config.evaluation_dataset

    # Load Training Datasets:
    datasets = []
    for directory_name in training_dataset_directories:
        data_path = Path(directory_name) / 'processed_data.pkl'
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

    # Load Evaluation Data:
    evaluation_dataset_path = Path(evaluation_dataset_directory) / 'processed_data.pkl'
    if not evaluation_dataset_path.exists():
        raise FileNotFoundError(f"{evaluation_dataset_path} not found")

    print(f"Loading evaluation dataset: {evaluation_dataset_directory}")
    with open(evaluation_dataset_path, 'rb') as f:
        evaluation_data_dict = pickle.load(f)

    # Combine Training Datasets:
    dataset = jax.tree_util.tree_map(
        lambda *arrays: jnp.concatenate(arrays, axis=0),
        *datasets
    )

    total_samples = dataset.ctrl.shape[0]
    steps_per_epoch = total_samples // config.training.minibatch_size
    total_steps = steps_per_epoch * config.training.num_epochs

    # Initialize Parameters:
    params, regression_dict = process_regression_spec(mj_model, config.regression)
    parameter_scale = build_parameter_scale(regression_dict)

    # Initialize Optimizer and State:
    initial_params = params.copy()
    params = {k: jnp.zeros_like(v) for k, v in params.items()}
    optimizer = create_optimizer(config, total_steps)
    opt_state = optimizer.init(params)

    # Define Objective Function
    objective_metric = get_objective_fn(config.loss.type)
    objective_weights = config.loss.weights.to_dict()
    regularization_weights = config.loss.regularization_weights.to_dict()

    # Wrap Step Function:
    init_fn = init_function
    step_fn = functools.partial(
        step_function,
        n_substeps=n_substeps,
    )

    # Wrap Rehydrate Model Function and Transform Parameters Function:
    rehydrate_model_function = functools.partial(
        model_utilities.rehydrate_model,
        mj_model=mjx_model_static,
        regression_spec=regression_dict,
    )
    transform_parameters_function = functools.partial(
        transforms.transform_parameters,
        regression_spec=regression_dict,
        parameter_scale=parameter_scale,
    )

    # Loss Function:
    loss_fn = functools.partial(
        loss_function,
        init_function=init_fn,
        step_function=step_fn,
        rehydrate_model_function=rehydrate_model_function,
        transform_parameters_function=transform_parameters_function,
        objective_function=objective_metric,
        objective_weights=objective_weights,
        regularization_weights=regularization_weights,
        nominal_params=initial_params,
    )

    if config.physics.use_reverse_mode:
        # Reverse Mode:
        value_and_grad_fn = jax.value_and_grad(loss_fn)
    else:
        value_and_grad_fn = forward_mode_value_and_grad(loss_fn)

    # Training Step:
    @jax.jit
    def train_step(
        state: TrainState,
        batch: Dataset,
    ) -> Tuple[TrainState, jax.Array]:
        params, opt_state = state

        loss, grads = value_and_grad_fn(
            params, batch,
        )

        grad_norms = jax.tree.map(jnp.linalg.norm, grads)
        grad_maxes = jax.tree.map(lambda x: jnp.max(jnp.abs(x)), grads)

        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        metrics = {
            'loss': loss,
            'grad_norms': grad_norms,
            'grad_maxes': grad_maxes
        }

        return (params, opt_state), metrics

    # Initialize Weights and Biases Logging:
    wand_run = wandb.init(
        project=config.wandb.project,
        group=config.wandb.group,
        config=config.to_dict()
    )

    # Training Loop:
    key = jax.random.key(config.training.seed)
    state = (params, opt_state)
    sample_fn = functools.partial(shuffle_data, dataset=dataset, batch_size=config.training.minibatch_size)
    loss_history = []
    wallclock = time.time()
    for epoch in range(config.training.num_epochs):
        start_time = time.time()

        # Sample Shuffled Data:
        key, subkey = jax.random.split(key)
        shuffled_data = sample_fn(
            subkey,
        )

        # Train Epoch:
        state, batch_metrics = jax.lax.scan(train_step, state, shuffled_data)
        elapsed_time = time.time() - start_time

        current_params = jax.device_get(state[0])

        # Transform Optimizer Parameters to Physical Parameters:
        physical_params = transform_parameters_function(current_params, initial_params)

        epoch_metrics_cpu = jax.device_get(
            jax.tree.map(lambda x: jnp.mean(x, axis=0), batch_metrics)
        )
        avg_loss = float(epoch_metrics_cpu['loss'])
        loss_history.append(avg_loss)

        # Console: loss + physical-space parameter table (decoded from theta).
        print(f"Epoch {epoch} | Loss: {avg_loss:.6f} | Time: {elapsed_time:.2f}s")
        print(logging_utils.format_param_summary(
            physical_params, initial_params, regression_dict, JOINT_NAMES,
        ))

        # wandb: loss, gradient norms, and physical-space parameters + diagnostics.
        log_dict = {
            'loss': avg_loss,
            'wall_time': time.time() - wallclock,
        }
        for param_name, norm_val in epoch_metrics_cpu['grad_norms'].items():
            log_dict[f'grads/norm/{param_name}'] = float(norm_val)
        for param_name, max_val in epoch_metrics_cpu['grad_maxes'].items():
            log_dict[f'grads/max/{param_name}'] = float(max_val)
        log_dict.update(
            logging_utils.build_param_log(physical_params, regression_dict, JOINT_NAMES)
        )

        wandb.log(log_dict)

    # Save Initial / Regressed Parameters and Loss History:
    output_params_dict = {}
    for k, v in physical_params.items():
        output_params_dict[f'{k}'] = np.array(v)

    initial_params_dict = {}
    for k, v in initial_params.items():
        initial_params_dict[f'initial_{k}'] = np.array(v)

    loss_history = np.array(loss_history)

    # Make Output Directory and Save Regressed Parameters, Loss History, and Config:
    output_directory = directory / 'checkpoints' / wand_run.name
    output_directory.mkdir(parents=True, exist_ok=True)

    checkpoint_payload = {
        'parameters': output_params_dict,
        'spec': regression_dict,
    }

    with open(output_directory / 'checkpoint.pkl', 'wb') as file:
        pickle.dump(checkpoint_payload, file)

    with open(output_directory / 'initial_parameters.pkl', 'wb') as file:
        pickle.dump(initial_params_dict, file)

    with open(output_directory / 'loss_history.pkl', 'wb') as file:
        pickle.dump(loss_history, file)

    with open(output_directory / 'config.pkl', 'wb') as file:
        pickle.dump(config.to_dict(), file)

    with open(output_directory / 'config.yaml', 'w') as file:
        file.write(config.to_yaml())

    # TODO: Pass Physical or Opt Parameters to Evaluation Function?
    # Evaluate and Plot Trajectory Comparison:
    evaluate(
        key,
        mjx_model_static,
        initial_params,
        physical_params,
        evaluation_data_dict,
        config,
        regression_dict,
        wand_run,
    )

    wand_run.finish()

    return jax.device_get(state), loss_history


def main(argv):
    train(_CONFIG.value)


if __name__ == '__main__':
    app.run(main)
