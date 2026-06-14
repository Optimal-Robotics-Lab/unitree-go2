from typing import Dict, Callable, Any

import jax
import jax.numpy as jnp

from mujoco import mjx

from regression.utilities.model_utilities import log_cholesky_to_mujoco
from regression.utilities.typedefs import Dataset, ObjectiveFunction
from regression.utilities.decorators import force_static_args


@force_static_args(
    "model_static",
    "init_function",
    "step_function",
    "objective_function",
    "objective_weights",
    "regularization_weights",
    "regression_spec",
    "baseline_params"
)
def loss_function(
    opt_params: Dict[str, jax.Array],
    batch: Dataset,
    *,
    model_static: mjx.Model,
    init_function: Callable,
    step_function: Callable,
    objective_function: ObjectiveFunction,
    objective_weights: Dict[str, float],
    regularization_weights: Dict[str, float],
    regression_spec: Dict[str, Dict[str, Any]],
    baseline_params: Dict[str, jax.Array],
) -> jax.Array:
    # Map optimizer parameters to physical parameters and update the model:
    bound_deltas = {k: (v['bounds'][1] - v['bounds'][0]) / 2.0 for k, v in regression_spec.items()}
    params = transform_to_physical(opt_params, baseline_params, bound_deltas)

    # Rehydrate the model with new parameters:
    replace_kwargs = {}
    for name, value in params.items():
        spec = regression_spec[name]
        field = spec['field']
        if field == 'log_cholesky_inertia':
            body_ids = spec['body_ids']
            b_mass, b_ipos, b_inertia, b_iquat = jax.vmap(log_cholesky_to_mujoco)(value, baseline_params['log_cholesky_inertia'])
            replace_kwargs['body_mass'] = model_static.body_mass.at[body_ids].set(b_mass)
            replace_kwargs['body_ipos'] = model_static.body_ipos.at[body_ids, :].set(b_ipos)
            replace_kwargs['body_inertia'] = model_static.body_inertia.at[body_ids, :].set(b_inertia)
            replace_kwargs['body_iquat'] = model_static.body_iquat.at[body_ids, :].set(b_iquat)

        elif 'column' in spec:
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

    # Objective Losses:
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

    # Regularization Losses:
    regularization_losses = {
        k: regularization_weights[k] * jnp.mean(opt_params[k] ** 2) 
        for k in regularization_weights.keys() if k in opt_params
    }
    regularization_loss = sum(regularization_losses.values())

    losses = {
        k: v * objective_weights[k] for k, v in losses.items()
    }
    loss = sum(losses.values())

    return loss + regularization_loss


def transform_to_physical(opt_params: dict, nominal_parameters: dict, bound_deltas: dict) -> dict:
    """
        Maps optimizer parameters to the physical parameters.
    """
    params = {}
    
    for name, theta_opt in opt_params.items():
        baseline = nominal_parameters[name]
        delta = bound_deltas[name]

        squashed_opt = jnp.tanh(theta_opt)
        
        params[name] = baseline + (squashed_opt * delta)
        
    return params