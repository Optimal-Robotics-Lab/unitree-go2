import jax
import jax.numpy as jnp

import numpy as np

from regression.utilities.typedefs import Dataset
from regression.utilities import resampling


def build_dataset(
    qpos: np.ndarray,
    qvel: np.ndarray,
    actuator_force: np.ndarray,
    ctrl: np.ndarray,
    *,
    state_dt: float,
    control_dt: float,
    observation_dt: float,
    window_length: int,
) -> Dataset:
    """Resample native-rate logs onto the observation grid and chunk into windows.

    States (qpos/qvel/actuator_force, at ``state_dt``) are linearly interpolated
    to ``observation_dt``; control (at ``control_dt``) is zero-order held to the
    same grid, so every observation step carries the command active during it.
    The states define the horizon; control is aligned to the same length.

    Args:
        qpos, qvel, actuator_force: ``(trials, time, dims)`` at ``state_dt``.
        ctrl: ``(trials, time, dims)`` at ``control_dt``.
        state_dt, control_dt, observation_dt: sample periods (seconds).
        window_length: chunk length in observation steps.

    Returns:
        A chunked :class:`Dataset` on the observation grid.
    """
    n_obs = resampling.num_resampled_steps(qpos.shape[1], state_dt, observation_dt)
    q = resampling.resample_linear(qpos, state_dt, observation_dt, n_dst=n_obs)
    v = resampling.resample_linear(qvel, state_dt, observation_dt, n_dst=n_obs)
    f = resampling.resample_linear(actuator_force, state_dt, observation_dt, n_dst=n_obs)
    u = resampling.resample_zoh(ctrl, control_dt, observation_dt, n_dst=n_obs)

    return chunk_and_flatten_dataset(
        jnp.asarray(q), jnp.asarray(v), jnp.asarray(f), jnp.asarray(u), window_length,
    )


def chunk_and_flatten_dataset(
    qpos: jax.Array,
    qvel: jax.Array,
    actuator_force: jax.Array,
    ctrl: jax.Array,
    window_len: int
) -> Dataset:
    """
        Turns (Trials, Time, Dims) -> (Total_Samples, Window_Len, Dims)
    """
    n_trials, n_time, dims = ctrl.shape
    n_chunks = n_time // window_len
    cutoff = n_chunks * window_len

    # Truncate & Reshape: (Trials, Chunks, Window, Dims)
    def reshape_fn(x):
        return x[:, :cutoff, :].reshape(n_trials, n_chunks, window_len, -1)

    def flatten_fn(x):
        x_reshaped = reshape_fn(x)
        return x_reshaped.reshape(-1, window_len, x.shape[-1])

    return Dataset(
        qpos=flatten_fn(qpos),
        qvel=flatten_fn(qvel),
        actuator_force=flatten_fn(actuator_force),
        ctrl=flatten_fn(ctrl),
    )


def shuffle_data(key, dataset: Dataset, batch_size: int) -> Dataset:
    num_samples = dataset.ctrl.shape[0]
    indices = jax.random.permutation(key, num_samples)
    shuffled_dataset = jax.tree_util.tree_map(lambda x: x[indices], dataset)

    num_batches = num_samples // batch_size
    cutoff = num_batches * batch_size

    batched = jax.tree_util.tree_map(
        lambda x: x[:cutoff].reshape(num_batches, batch_size, *x.shape[1:]),
        shuffled_dataset
    )
    return batched


def sample_random_windows(
    key: jax.Array,
    dataset: Dataset,
    batch_size: int,
    window_length: int,
    num_batches_per_epoch: int,
):
    num_trials, num_time_steps, dims = dataset.ctrl.shape
    max_start_index = num_time_steps - window_length

    total_samples = num_batches_per_epoch * batch_size

    key, trial_key, time_key = jax.random.split(key, 3)

    trial_indices = jax.random.randint(
        trial_key, (total_samples,), 0, num_trials
    )
    time_indices = jax.random.randint(
        time_key, (total_samples,), 0, max_start_index + 1
    )

    def get_window(trial_idx, start_time, array):
        return jax.lax.dynamic_slice(
            array, (trial_idx, start_time, 0), (1, window_length, array.shape[2])
        ).reshape(window_length, array.shape[2])

    def slice_fn(field):
        return jax.vmap(
            get_window, in_axes=(0, 0, None),
        )(trial_indices, time_indices, field)

    windowed_data = jax.tree_util.tree_map(slice_fn, dataset)

    # Shape: (num_batches_per_epoch, batch_size, window_length, dims)
    batched_data = jax.tree_util.tree_map(
        lambda x: x.reshape(num_batches_per_epoch, batch_size, *x.shape[1:]),
        windowed_data
    )

    return batched_data
