from absl import app

from typing import Dict, Tuple

from pathlib import Path
import pickle
import functools
import time

import jax
import jax.numpy as jnp

import numpy as np

import flax.struct
import optax

import wandb

import mujoco
from mujoco import mjx

from ml_collections import config_flags, ConfigDict

from utilities.config import get_default_config
from utilities.typedefs import Dataset, ObjectiveFunction, TrainState
from utilities.constants import JOINT_NAMES
from utilities.data_utils import chunk_and_flatten_dataset, shuffle_data
from utilities.factories import create_optimizer, get_objective_fn
from utilities.autodiff import forward_mode_value_and_grad
import utilities.evaluation as evaluation

jax.config.update('jax_enable_x64', True)

_CONFIG = config_flags.DEFINE_config_dict('config', get_default_config())


def train(config: ConfigDict) -> Tuple[TrainState, np.ndarray]:
    # Load MuJoCo Model
    directory = Path(__file__).resolve().parent
    filepath = (directory / config.scene_file).resolve()
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

    # Load Data:
    data_path = directory / config.dataset_directory / config.dataset_name / 'processed_data.pkl'
    if not data_path.exists():
        raise FileNotFoundError(f"{data_path} not found")
        
    with open(data_path, 'rb') as f:
        data_dict = pickle.load(f)

    effective_window = config.training.window_length + 1
    
    dataset = chunk_and_flatten_dataset(
        jnp.array(data_dict['qpos']),
        jnp.array(data_dict['qvel']),
        jnp.array(data_dict['actuator_force']),
        jnp.array(data_dict['ctrl']),
        effective_window,
    )
    
    total_samples = dataset.ctrl.shape[0]
    steps_per_epoch = total_samples // config.training.minibatch_size
    total_steps = steps_per_epoch * config.training.num_epochs

    # Initialize Parameters:
    params = {}
    regression_spec = config.regression.to_dict()
    
    for name, spec in regression_spec.items():
        val = getattr(mjx_model_static, spec['field'])
        if 'column' in spec:
            val = val[:, spec['column']]
        params[name] = val

    # Initialize Optimizer and State:
    initial_params = params.copy()
    optimizer = create_optimizer(config, total_steps)
    opt_state = optimizer.init(params)

    # Define Objective Function
    objective_metric = get_objective_fn(config.loss.type)
    objective_weights = config.loss.weights.to_dict()

    # Wrap Step Function:
    init_function = evaluation.init_function
    step_function = functools.partial(
        evaluation.step_function,
        n_substeps=n_substeps,
    )

    # Loss Function:
    def loss_fn(
        params: Dict[str, jax.Array],
        model_static: mjx.Model,
        batch: Dataset,
        objective_function: ObjectiveFunction,
        objective_weights: Dict[str, float],
    ) -> jax.Array:
        # Rehydrate the model with new parameters:
        replace_kwargs = {}
        for name, value in params.items():
            spec = regression_spec[name]
            field = spec['field']
            if 'column' in spec:
                col_idx = spec['column']
                original_array = getattr(model_static, field)
                new_array = original_array.at[:, col_idx].set(value)
                replace_kwargs[field] = new_array
            else:
                replace_kwargs[field] = value
            
        model_dynamic = model_static.replace(**replace_kwargs)

        # Rollout Trajectory:
        def rollout(setpoints, qpos_init, qvel_init):
            d = init_function(
                model_dynamic, qpos_init, qvel_init, setpoints[0],
            )

            def step(carry, xs):
                d = step_function(model_dynamic, carry, xs)
                return d, (d.qpos, d.qvel, d.actuator_force)

            _, (qpos, qvel, actuator_force) = jax.lax.scan(step, d, setpoints)
            return qpos, qvel, actuator_force

        # Extract Initial States and Setpoints:
        initial_qpos = batch.qpos[:, 0]
        initial_qvel = batch.qvel[:, 0]
        setpoints = batch.ctrl[:, :-1]

        qpos_targets = batch.qpos[:, 1:]
        qvel_targets = batch.qvel[:, 1:]
        actuator_force_targets = batch.actuator_force[:, 1:]

        qpos_prediction, qvel_prediction, actuator_force_prediction = jax.vmap(rollout)(
            setpoints,
            initial_qpos,
            initial_qvel
        )

        losses = {
            'position': objective_function(
                qpos_prediction, qpos_targets,
            ),
            'velocity': objective_function(
                qvel_prediction, qvel_targets,
            ),
            'actuator_force': objective_function(
                actuator_force_prediction, actuator_force_targets,
            ),
        }

        losses = {
            k: v * objective_weights[k] for k, v in losses.items()
        }

        loss = sum(losses.values())

        return loss

    # Select Gradient Mode:
    loss_fn = functools.partial(
        loss_fn,
        objective_function=objective_metric,
        objective_weights=objective_weights,
    )

    if config.physics.use_reverse_mode:
        # Reverse Mode:
        value_and_grad_fn = jax.value_and_grad(loss_fn)
    else:
        # Forward Mode:
        # def fwd_value_and_grad_fn(
        #     params: Dict[str, jax.Array],
        #     mjx_model_static: mjx.Model,
        #     batch: Dataset,
        # ) -> Tuple[jax.Array, Dict[str, jax.Array]]:
        #     l = loss_fn(params, mjx_model_static, batch)
        #     g = jax.jacfwd(loss_fn, argnums=0)(params, mjx_model_static, batch)
        #     return l, g
        # value_and_grad_fn = fwd_value_and_grad_fn

        value_and_grad_fn = forward_mode_value_and_grad(loss_fn)

    # Training Step:
    @jax.jit
    def train_step(
        state: TrainState,
        batch: Dataset,
    ) -> Tuple[TrainState, jax.Array]:
        params, opt_state = state

        loss, grads = value_and_grad_fn(
            params, mjx_model_static, batch
        )

        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        clipped_params = {}
        for name, value in params.items():
            bounds = regression_spec[name].get('bounds')
            if bounds is not None:
                min_val, max_val = bounds
                value = jnp.clip(value, min_val, max_val)
            clipped_params[name] = value

        return (clipped_params, opt_state), loss

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
        state, batch_losses = jax.lax.scan(train_step, state, shuffled_data)
        elapsed_time = time.time() - start_time
        
        current_params = jax.device_get(state[0])
        avg_loss = float(jax.device_get(jnp.mean(batch_losses)))

        loss_history.append(avg_loss)

        print(
            f"Epoch {epoch} | "
            f"Loss: {avg_loss:.6f} | "
            f"Time: {elapsed_time:.2f}s"
        )
        with jnp.printoptions(precision=4, suppress=True, linewidth=200):
            for k, v in current_params.items():
                print(f"\t {k}:\t {v}")

        log_dict = {
            'loss': avg_loss,
            'wall_time': time.time() - wallclock,
        }
        for k, v in current_params.items():
            for i, name in enumerate(JOINT_NAMES):
                log_dict[f'params/{k}/{name}'] = float(v[i])

        wandb.log(log_dict)

    # Save Initial / Regressed Parameters and Loss History:
    output_params = {}
    for k, v in current_params.items():
        output_params[f'{k}'] = np.array(v)

    for k, v in initial_params.items():
        output_params[f'initial_{k}'] = np.array(v)
    
    loss_history = np.array(loss_history)

    # Make Output Directory and Save Regressed Parameters:
    output_directory = directory / config.dataset_directory / config.dataset_name / wand_run.name
    output_directory.mkdir(parents=True, exist_ok=True)
    with open(output_directory / 'regressed_params.pkl', 'wb') as file:
        pickle.dump(output_params, file)

    with open(output_directory / 'loss_history.pkl', 'wb') as file:
        pickle.dump(loss_history, file)

    # Evaluate and Plot Trajectory Comparison:
    evaluation.evaluate(
        key,
        mjx_model_static,
        initial_params,
        current_params,
        data_dict,
        config,
        wand_run,
    )

    wand_run.finish()

    return jax.device_get(state), loss_history


def main(argv):
    train(_CONFIG.value)


if __name__ == '__main__':
    app.run(main)
