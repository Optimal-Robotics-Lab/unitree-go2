import jax
import jax.numpy as jnp

from regression.utilities.typedefs import Dataset


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
