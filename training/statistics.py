"""
    Need to check this implementation carefully.
"""


import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing import Optional, Any, Tuple, Union

# Modern JAX tree syntax
Nest = Any


class RunningStatistics(nnx.Module):
    """
    Computes running statistics (mean, std) for nested data structures.
    Supports distributed computation (pmap/sharding) and Welford's algorithm.
    """
    def __init__(
        self,
        example_input: Nest,
        mode: str = 'welford',
        epsilon: float = 1e-4,
        clip_max: Optional[float] = None,
        pmap_axis_name: Optional[str] = None,
    ):
        self.mode = mode
        self.epsilon = epsilon
        self.clip_max = clip_max
        self.pmap_axis_name = pmap_axis_name

        # Get statistics shape (non-batch dims) from the example input
        self.structure_def = jax.tree.structure(example_input)

        def get_stat_shape(x):
            return x.shape

        # Initialize State (Mean = 0, Var = 1 approx, Count = 0)
        self.mean = jax.tree.map(
            lambda x: nnx.BatchStat(jnp.zeros_like(x)), example_input
        )

        # In Welford: stores Sum of Squares of Differences (M2)
        # In EMA: stores Running Variance
        self.var_sum = jax.tree.map(
            lambda x: nnx.BatchStat(jnp.ones_like(x)), example_input
        )

        # Count needs to be high precision for long training runs
        self.count = nnx.BatchStat(jnp.array(1e-4, dtype=jnp.float64))

    def __call__(
        self,
        x: Nest,
        update_stats: bool = True,
        validate_shapes: bool = True
    ) -> Nest:
        if update_stats:
            self.update(x, validate_shapes=validate_shapes)

        return self.normalize(x)

    def update(self, batch: Nest, validate_shapes: bool = True):
        """Updates statistics using the current batch, supporting distributed sync."""

        if validate_shapes:
            self._validate_batch(batch)

        # 2. Get Flattened Leaves for Processing
        batch_leaves = jax.tree.leaves(batch)
        mean_leaves = jax.tree.leaves(self.mean)
        var_leaves = jax.tree.leaves(self.var_sum)

        # Determine Batch Size (and sync across devices if distributed)
        local_batch_size = batch_leaves[0].shape[0]
        step_increment = jnp.array(local_batch_size, dtype=jnp.float64)

        if self.pmap_axis_name is not None:
            step_increment = jax.lax.psum(step_increment, axis_name=self.pmap_axis_name)

        # Current Global Count
        count_float = self.count.value
        new_count = count_float + step_increment

        # 3. Calculate Updates (Welford's Algorithm)
        # We stick to Welford as it is the default in Brax and numerically stable.
        if self.mode == 'welford':
            # We iterate purely over leaves (flattened), JAX handles the rest
            for i, (batch_data, mean_stat, var_stat) in enumerate(zip(batch_leaves, mean_leaves, var_leaves)):

                old_mean = mean_stat.value
                old_m2 = var_stat.value

                # A. Update Mean
                # delta = x - mean
                # mean_new = mean_old + sum(delta) / total_count

                # We sum over the LOCAL batch dimension (axis 0)
                diff_to_old_mean = batch_data - old_mean
                mean_update_sum = jnp.sum(diff_to_old_mean, axis=0)

                # Sync sum across devices
                if self.pmap_axis_name is not None:
                    mean_update_sum = jax.lax.psum(mean_update_sum, axis_name=self.pmap_axis_name)

                mean_update = mean_update_sum / new_count
                new_mean = old_mean + mean_update

                # B. Update Variance (M2)
                # M2_new = M2_old + sum( (x - mean_old) * (x - mean_new) )

                diff_to_new_mean = batch_data - new_mean

                # Element-wise multiplication, then sum over batch
                variance_update_local = jnp.sum(diff_to_old_mean * diff_to_new_mean, axis=0)

                # Sync variance sum across devices
                if self.pmap_axis_name is not None:
                    variance_update_local = jax.lax.psum(variance_update_local, axis_name=self.pmap_axis_name)

                new_m2 = old_m2 + variance_update_local

                # Apply in-place updates to BatchStat
                mean_stat.value = new_mean
                var_stat.value = new_m2

            self.count.value = new_count

        elif self.mode == 'ema':
            # EMA Logic (Simplified for brevity, but follows similar sync pattern)
            rate = step_increment / new_count
            for i, (batch_data, mean_stat, var_stat) in enumerate(zip(batch_leaves, mean_leaves, var_leaves)):
                # Calculate local stats
                batch_mean = jnp.mean(batch_data, axis=0)
                batch_var = jnp.var(batch_data, axis=0)

                # Sync if distributed
                if self.pmap_axis_name is not None:
                    batch_mean = jax.lax.pmean(batch_mean, axis_name=self.pmap_axis_name)
                    batch_var = jax.lax.pmean(batch_var, axis_name=self.pmap_axis_name)

                # Standard EMA Update
                delta = batch_mean - mean_stat.value
                new_mean = mean_stat.value + rate * delta
                new_var = var_stat.value + rate * (
                    batch_var - var_stat.value + delta * (batch_mean - new_mean)
                )

                mean_stat.value = new_mean
                var_stat.value = new_var

            self.count.value = new_count

    def normalize(self, x: Nest) -> Nest:
        """Applies normalization using current statistics."""
        
        # 1. Compute Standard Deviation
        # std = sqrt(M2 / count + epsilon)
        def compute_std(var_stat_leaf):
            # var_stat is M2 in Welford, or actual Var in EMA
            divisor = self.count.value if self.mode == 'welford' else 1.0

            # Guard against division by zero or negative variance due to float errors
            variance = jnp.maximum(var_stat_leaf / divisor, 0.0)
            return jnp.sqrt(variance + self.epsilon)

        std_tree = jax.tree.map(
            lambda s: compute_std(s.value),
            self.var_sum
        )

        def apply_norm(x_leaf, mean_stat, std_leaf):
            if not jnp.issubdtype(x_leaf.dtype, jnp.inexact):
                return x_leaf

            norm_val = (x_leaf - mean_stat.value) / std_leaf

            if self.clip_max is not None:
                norm_val = jnp.clip(norm_val, -self.clip_max, self.clip_max)
            return norm_val

        return jax.tree.map(
            apply_norm,
            x,
            self.mean,
            std_tree,
        )

    def _validate_batch(self, batch: Nest):
        batch_struct = jax.tree.structure(batch)
        if batch_struct != self.structure_def:
            raise ValueError(
                f"Input structure mismatch.\n"
                f"Expected: {self.structure_def}\n"
                f"Received: {batch_struct}"
            )

        batch_leaves = jax.tree.leaves(batch)
        stat_leaves = jax.tree.leaves(self.mean)

        for i, (b_leaf, s_leaf) in enumerate(zip(batch_leaves, stat_leaves)):
            expected_feature_shape = s_leaf.value.shape
            actual_feature_shape = b_leaf.shape[1:]

            if actual_feature_shape != expected_feature_shape:
                raise ValueError(
                    f"Shape mismatch at leaf {i}.\n"
                    f"Expected features: {expected_feature_shape}\n"
                    f"Received batch: {b_leaf.shape} (Features: {actual_feature_shape})"
                )
