from typing import Dict, Callable, Any

import jax

from mujoco import mjx

from regression.utilities.typedefs import Dataset, ObjectiveFunction
from regression.utilities.decorators import force_static_args


@force_static_args(
    "model_static",
    "init_function",
    "step_function",
    "objective_function",
    "objective_weights",
    "regression_spec"
)
def loss_function(
    params: Dict[str, jax.Array],
    batch: Dataset,
    *,
    model_static: mjx.Model,
    init_function: Callable,
    step_function: Callable,
    objective_function: ObjectiveFunction,
    objective_weights: Dict[str, float],
    regression_spec: Dict[str, Dict[str, Any]],
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
