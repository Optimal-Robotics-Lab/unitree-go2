from absl import app, flags

from collections.abc import Callable
from typing import Dict, Tuple

from pathlib import Path
import pickle
import functools
import time

import jax
import jax.numpy as jnp

import flax.struct
import optax

import mujoco
from mujoco import mjx


jax.config.update('jax_enable_x64', True)

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'directory_name',
    None,
    'Directory containing the hardware data to process.',
    required=True,
    short_name='d',
)

flags.DEFINE_integer(
    'minibatch_size',
    25,
    'Size of each minibatch for training.',
    required=False,
    short_name='b',
)

flags.DEFINE_integer(
    'window_length',
    25,
    'Length of the window for each training sample.',
    required=False,
    short_name='w',
)


@flax.struct.dataclass
class Dataset:
    qpos: jax.Array
    qvel: jax.Array
    actuator_force: jax.Array
    ctrl: jax.Array


ObjectiveFunction = Callable[[jax.Array, jax.Array], jax.Array]


def main(argv=None):
    filename = 'mjcf/scene_mjx.xml'
    filepath = Path(__file__).resolve().parent / filename

    mj_model = mujoco.MjModel.from_xml_path(
        str(filepath),
    )
    mjx_model_static = mjx.put_model(mj_model, impl="jax")

    control_rate = 0.02
    n_substeps = int(control_rate / mj_model.opt.timestep)

    # Load Data
    base_directory = Path(__file__).resolve().parent
    directory = base_directory / 'data' / FLAGS.directory_name

    # Verify Directory Exists:
    if not directory.exists():
        raise FileNotFoundError(
            f'Directory {directory} does not exist.',
        )

    with open(directory / 'processed_data.pkl', 'rb') as file:
        data_dict = pickle.load(file)

    dataset = Dataset(
        qpos=jnp.array(data_dict['qpos']),
        qvel=jnp.array(data_dict['qvel']),
        actuator_force=jnp.array(data_dict['actuator_force']),
        ctrl=jnp.array(data_dict['ctrl']),
    )

    # VMAP and Jit the init and step functions:
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
    ) -> mjx.Data:
        data = data.replace(ctrl=ctrl)

        def loop(carry: mjx.Data, unused_t):
            return mjx.step(model, carry), None

        data, _ = jax.lax.scan(loop, data, None, length=n_substeps)
        return data

    # Initialize the parameter and optimizer:
    num_epochs = 50
    optimizer = optax.adam(learning_rate=1e-2)

    # Parameters to Regress:
    """
        MuJoCo Joint Parameters:
        - Friction Loss
        - Armature
        - Damping
        - Reference
        Actuator Parameters:
        - Time Constant
    """
    dof_frictionloss_params = mjx_model_static.dof_frictionloss
    dof_armature_params = mjx_model_static.dof_armature
    dof_damping_params = mjx_model_static.dof_damping
    dof_qpos0_params = mjx_model_static.qpos0
    actuator_timeconst_params = mjx_model_static.actuator_dynprm[:, 0]
    params = {
        'frictionloss': dof_frictionloss_params,
        'armature': dof_armature_params,
        'damping': dof_damping_params,
        'qpos0': dof_qpos0_params,
        'timeconst': actuator_timeconst_params,
    }

    def loss_fn(
        params: Dict[str, jax.Array],
        model_static: mjx.Model,
        batch: Dataset,
        objective_function: ObjectiveFunction,
        objective_weights: Dict[str, float],
    ) -> jax.Array:
        # Rehydrate the model with new parameters:
        model_dynamic = model_static.replace(
            dof_frictionloss=params['frictionloss'],
            dof_armature=params['armature'],
            dof_damping=params['damping'],
            qpos0=params['qpos0'],
            actuator_dynprm=model_static.actuator_dynprm.at[:, 0].set(params['timeconst'])
        )

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
        setpoints = batch.ctrl

        qpos_targets = batch.qpos
        qvel_targets = batch.qvel
        actuator_force_targets = batch.actuator_force

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

    objective_function = optax.l2_loss
    objective_weights = {
        'position': 1.0,
        'velocity': 1.0,
        'actuator_force': 1.0,
    }
    loss_fn = functools.partial(
        loss_fn,
        objective_function=objective_function,
        objective_weights=objective_weights,
    )

    @jax.jit
    def train_step(
        state: Tuple[Dict[str, jax.Array], optax.OptState], batch: Dataset,
    ) -> Tuple[Tuple[Dict[str, jax.Array], optax.OptState], jax.Array]:
        params, opt_state = state

        loss, grads = jax.value_and_grad(loss_fn)(
            params, mjx_model_static, batch
        )

        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        safe_keys = ['frictionloss', 'armature', 'damping', 'timeconst']
        params = {
            k: jnp.clip(v, 1e-4, 1e2) if k in safe_keys else v
            for k, v in params.items()
        }

        return (params, opt_state), loss

    def sample_random_windows(
        key: jax.Array, dataset: Dataset, batch_size: int, window_length: int,
    ) -> Dataset:
        num_trials, num_time_steps, dims = dataset.ctrl.shape

        # Calculate valid start indices:
        max_start_index = num_time_steps - window_length

        if max_start_index <= 0:
            raise ValueError(
                f"Window length {window_length} is larger than trajectory length {num_time_steps}"
            )

        # Generate random start indices:
        total_windows_needed = (num_trials * num_time_steps) // window_length
        num_batches = total_windows_needed // batch_size
        total_samples = num_batches * batch_size

        # Split key for trials and time steps
        key, trial_key, time_key = jax.random.split(key, 3)

        # Pick random trials
        trial_indices = jax.random.randint(
            trial_key, (total_samples,), 0, num_trials,
        )

        # Pick random start times within those trials
        time_indices = jax.random.randint(
            time_key, (total_samples,), 0, max_start_index + 1
        )

        # Generate random windows of data:
        def get_window(trial_idx, start_time, array):
            # Slice: array[trial_idx, start_time : start_time + window_len, :]
            return jax.lax.dynamic_slice(
                array,
                (trial_idx, start_time, 0),             # Start indices
                (1, window_length, array.shape[2])      # Slice sizes
            ).reshape(window_length, array.shape[2])    # Remove trial dim

        def slice_dataset(field):
            return jax.vmap(get_window, in_axes=(0, 0, None))(
                trial_indices, time_indices, field
            )

        windowed_data = jax.tree.map(slice_dataset, dataset)

        # Reshape into batches: (total_samples, ...) -> (num_batches, batch_size, ...)
        batched_data = jax.tree.map(
            lambda x: x.reshape(num_batches, batch_size, *x.shape[1:]),
            windowed_data
        )

        return batched_data

    sample_data_fn = functools.partial(
        sample_random_windows,
        dataset=dataset,
        batch_size=FLAGS.minibatch_size,
        window_length=FLAGS.window_length,
    )

    # Training Loop:
    key = jax.random.PRNGKey(42)
    state = (params, optimizer.init(params))
    for epoch in range(num_epochs):
        start_time = time.time()

        key, subkey = jax.random.split(key)
        shuffled_data = sample_data_fn(
            subkey,
        )
        state, batch_losses = jax.lax.scan(train_step, state, shuffled_data)

        elapsed_time = time.time() - start_time

        print(
            f"Epoch {epoch:02d} | "
            f"Loss: {jnp.mean(batch_losses):.6f} | "
            f"Time: {elapsed_time:.2f}s"
        )
        print(
            f"\t frictionloss: {state[0]['frictionloss']} \n"
            f"\t armature: {state[0]['armature']} \n"
            f"\t damping: {state[0]['damping']} \n"
            f"\t timeconst: {state[0]['timeconst']} \n"
        )


if __name__ == '__main__':
    app.run(main)
