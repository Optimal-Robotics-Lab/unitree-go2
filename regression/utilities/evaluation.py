from typing import Dict, Any
from ml_collections import ConfigDict

import jax
import jax.numpy as jnp
import numpy as np

import mujoco
from mujoco import mjx

import matplotlib.pyplot as plt

import wandb

from utilities.constants import JOINT_NAMES
from utilities.typedefs import ObjectiveFunction
from utilities.factories import get_objective_fn


def init_function(
    model: mjx.Model,
    qpos: jax.Array,
    qvel: jax.Array,
    ctrl: jax.Array,
) -> mjx.Data:
    data = mjx.make_data(model)
    data = data.replace(qpos=qpos, qvel=qvel, ctrl=ctrl)
    data = mjx.forward(model, data)
    return data

def step_function(
    model: mjx.Model,
    data: mjx.Data,
    ctrl: jax.Array,
    n_substeps: int,
) -> mjx.Data:
    data = data.replace(ctrl=ctrl)

    def loop(carry: mjx.Data, unused_t):
        return mjx.step(model, carry), None

    data, _ = jax.lax.scan(loop, data, None, length=n_substeps)
    return data

def evaluate(
    key: jax.Array,
    mjx_model_static: mjx.Model,
    initial_params: Dict[str, jax.Array],
    optimized_params: Dict[str, jax.Array],
    dataset: Dict[str, jax.Array],
    config: ConfigDict,
    wandb_run: Any,
):  

    # Get Config Settings:
    regression_spec = config.regression.to_dict()
    n_substeps = int(config.physics.control_rate / config.physics.timestep)
    objective_metric = get_objective_fn(config.loss.type)
    objective_weights = config.loss.weights.to_dict()

    # Sample a Random Trajectory
    n_trials, n_time, _ = dataset['ctrl'].shape
    trial_idx = jax.random.randint(key, (), 0, n_trials)
    
    # Set Targets and Initial State:
    qpos = jnp.array(dataset['qpos'][trial_idx])
    qvel = jnp.array(dataset['qvel'][trial_idx])
    actuator_force = jnp.array(dataset['actuator_force'][trial_idx])
    ctrl_setpoints = jnp.array(dataset['ctrl'][trial_idx])[:-1]

    qpos_init, qvel_init = qpos[0], qvel[0]
    qpos_target = qpos[1:]
    qvel_target = qvel[1:]
    actuator_force_target = actuator_force[1:]

    def run_trajectory(params_dict):
        # Rehydrate model
        replace_kwargs = {}
        for name, value in params_dict.items():
            spec = regression_spec[name]
            field = spec['field']
            if 'column' in spec:
                col_idx = spec['column']
                original_array = getattr(mjx_model_static, field)
                new_array = original_array.at[:, col_idx].set(value)
                replace_kwargs[field] = new_array
            else:
                replace_kwargs[field] = value
            
        model_dynamic = mjx_model_static.replace(**replace_kwargs)

        def rollout(setpoints, qpos_init, qvel_init):
            d = init_function(
                model_dynamic, qpos_init, qvel_init, setpoints[0],
            )

            def step(carry, xs):
                d = step_function(model_dynamic, carry, xs, n_substeps)
                return d, (d.qpos, d.qvel, d.actuator_force)

            _, (qpos, qvel, actuator_force) = jax.lax.scan(step, d, setpoints)
            return qpos, qvel, actuator_force

        return rollout(ctrl_setpoints, qpos_init, qvel_init)

    rollout_fn = jax.jit(run_trajectory)
    
    # Rollout Baseline vs Optimized Model:
    base_qpos, base_qvel, base_actuator_force = rollout_fn(initial_params)
    opt_qpos, opt_qvel, opt_actuator_force = rollout_fn(optimized_params)
    
    # Calculate Metrics:
    def compute_weighted_loss(qpos_prediction, qvel_prediction, actuator_force_prediction):
        losses = {
            'position': objective_metric(qpos_prediction, qpos_target),
            'velocity': objective_metric(qvel_prediction, qvel_target),
            'actuator_force': objective_metric(actuator_force_prediction, actuator_force_target),
        }
        losses = {k: v * objective_weights[k] for k, v in losses.items()}
        return losses, sum(losses.values())

    base_losses, base_loss = compute_weighted_loss(base_qpos, base_qvel, base_actuator_force)
    opt_losses, opt_loss = compute_weighted_loss(opt_qpos, opt_qvel, opt_actuator_force)
    
    metrics = {
        "eval/baseline_pos_loss": base_losses['position'],
        "eval/baseline_vel_loss": base_losses['velocity'],
        "eval/baseline_force_loss": base_losses['actuator_force'],
        "eval/baseline_loss": base_loss,
        "eval/optimized_pos_loss": opt_losses['position'],
        "eval/optimized_vel_loss": opt_losses['velocity'],
        "eval/optimized_force_loss": opt_losses['actuator_force'],
        "eval/optimized_loss": opt_loss,
    }
    
    # Generate Plots
    t = np.arange(1, n_time) * config.physics.control_rate
    
    fig, axes = plt.subplots(12, 3, figsize=(15, 30), sharex=True)
    fig.suptitle(f'Trajectory Comparison (Trial {trial_idx})', fontsize=16)
    
    def to_np(x): return np.array(x)

    # Rows: Joints | Columns: Position, Velocity, Force
    for i, joint_name in enumerate(JOINT_NAMES):
        # Position
        ax = axes[i, 0]
        ax.plot(t, to_np(qpos_target[:, i]), 'k-', alpha=0.6, label='Ground Truth')
        ax.plot(t, to_np(base_qpos[:, i]), 'r--', label='Baseline')
        ax.plot(t, to_np(opt_qpos[:, i]), 'b-', label='Optimized')
        ax.set_ylabel(f'{joint_name}\nPos (rad)')
        if i == 0: ax.set_title("Position")
        if i == 11: ax.legend()

        # Velocity
        ax = axes[i, 1]
        ax.plot(t, to_np(qvel_target[:, i]), 'k-', alpha=0.6)
        ax.plot(t, to_np(base_qvel[:, i]), 'r--')
        ax.plot(t, to_np(opt_qvel[:, i]), 'b-')
        ax.set_ylabel('Vel (rad/s)')
        if i == 0: ax.set_title("Velocity")

        # Force
        ax = axes[i, 2]
        ax.plot(t, to_np(actuator_force_target[:, i]), 'k-', alpha=0.6)
        ax.plot(t, to_np(base_actuator_force[:, i]), 'r--')
        ax.plot(t, to_np(opt_actuator_force[:, i]), 'b-')
        ax.set_ylabel('Force (Nm)')
        if i == 0: ax.set_title("Actuator Force")

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    metrics["eval/trajectory"] = wandb.Image(fig)
    wandb_run.log(metrics)
    
    plt.close(fig)
