from typing import Dict, Callable

import jax
import jax.numpy as jnp

from regression.utilities.typedefs import Dataset, ObjectiveFunction
from regression.utilities.decorators import force_static_args


@force_static_args(
    "init_function",
    "step_function",
    "rehydrate_model_function",
    "transform_parameters_function",
    "objective_function",
    "objective_weights",
    "regularization_weights",
    "nominal_params"
)
def loss_function(
    opt_params: Dict[str, jax.Array],
    batch: Dataset,
    *,
    init_function: Callable,
    step_function: Callable,
    rehydrate_model_function: Callable,
    transform_parameters_function: Callable,
    objective_function: ObjectiveFunction,
    objective_weights: Dict[str, float],
    regularization_weights: Dict[str, float],
    nominal_params: Dict[str, jax.Array],
) -> jax.Array:
    # Transform Optimizer Parameters to Physical Parameters:
    params = transform_parameters_function(opt_params, nominal_params)

    # Rehydrate Model with Parameters:
    model_dynamic = rehydrate_model_function(params, nominal_params)

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
