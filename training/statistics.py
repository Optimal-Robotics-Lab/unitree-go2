"""
NNX Module to compute running statistics.
This file is adapted from Brax's running statistics implementation:
https://github.com/google/brax/blob/main/brax/training/acme/running_statistics.py
"""

from typing import Optional, Any, Tuple
import enum

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from training.module_types import NestedArray


if not jax.config.read("jax_enable_x64"):
    raise RuntimeError(
        "JAX float64 support is disabled. This module requires high precision. "
        "Please enable it by adding `jax.config.update('jax_enable_x64', True)` before importing this module."
    )


class NormalizationMode(enum.IntEnum):
    WELFORD = 0
    EMA = 1


def _mode_from_string(mode: str) -> int:
    if mode == 'ema':
        return NormalizationMode.EMA
    elif mode == 'welford':
        return NormalizationMode.WELFORD
    else:
        raise ValueError(f'Unknown normalization mode: {mode}')


def _to_nnx(node: Any) -> Any:
    """Recursively wraps python containers in nnx.Dict/nnx.List."""
    if isinstance(node, dict):
        return nnx.Dict({k: _to_nnx(v) for k, v in node.items()})
    elif isinstance(node, (list, tuple)):
        return nnx.List([_to_nnx(v) for v in node])
    else:
        return node


class RunningStatistics(nnx.Module):
    """
    Computes running statistics (mean, std) for nested data structures.
    Supports distributed computation (pmap/sharding) and Welford's algorithm.
    """
    def __init__(
        self,
        reference_input: NestedArray,
        mode: str = 'welford',
        epsilon: float = 1e-4,
        clip: Optional[float] = None,
        std_min: float = 1e-6,
        std_max: float = 1e6,
        pmap_axis_name: Optional[str] = None,
    ):
        self.mode: int = _mode_from_string(mode)
        self.epsilon = epsilon
        self.clip = clip
        self.std_min = std_min
        self.std_max = std_max
        self.pmap_axis_name = pmap_axis_name

        # Get statistics shape from the reference input
        self.structure_def = jax.tree.structure(reference_input)

        # Upgrade to nnx containers:
        reference_input = _to_nnx(reference_input)

        # Initialize State (Mean = 0, Var = 0, Std = 1, Count = 0)
        self.mean = jax.tree.map(
            lambda x: nnx.BatchStat(jnp.zeros_like(x)), reference_input
        )

        self.var = jax.tree.map(
            lambda x: nnx.BatchStat(jnp.zeros_like(x)), reference_input
        )

        self.std = jax.tree.map(
            lambda x: nnx.BatchStat(jnp.ones_like(x)), reference_input
        )

        # Count needs to be high precision for long training runs
        self.count = nnx.BatchStat(jnp.array(0.0, dtype=jnp.float64))

    def __call__(
        self,
        x: NestedArray,
    ) -> NestedArray:
        """ Alias for normalize """
        return self.normalize(x)

    def update(
        self,
        batch: NestedArray,
        *,
        weights: Optional[jnp.ndarray] = None,
        validate_shapes: bool = True,
    ):
        """
            Updates running statistics
        """
        assert jax.tree.structure(batch) == self.structure_def, \
            f"Structure mismatch: expected {self.structure_def}, got {jax.tree.structure(batch)}"

        # Update batch to nnx containers:
        batch = _to_nnx(batch)

        batch_leaves = jax.tree.leaves(batch)
        statistics_leaves = jax.tree.leaves(self.mean)

        if not batch_leaves:
            return

        first_batch_leaf = batch_leaves[0]
        first_statistics_leaf = statistics_leaves[0]

        batch_rank = first_batch_leaf.ndim - first_statistics_leaf.ndim

        if batch_rank < 0:
            raise ValueError(f"Input batch has fewer dimensions ({first_batch_leaf.ndim}) than stored statistics ({first_statistics_leaf.ndim}).")

        batch_dims = first_batch_leaf.shape[:batch_rank]
        batch_axis = tuple(range(batch_rank))

        # Validate shapes:
        if validate_shapes:
            if weights is not None:
                if weights.shape != batch_dims:
                    raise ValueError(f"Weights shape {weights.shape} must match batch dims {batch_dims}")
            self._validate_batch_shape(batch, batch_dims)

        # Increment count:
        if weights is not None:
            step_increment = jnp.sum(weights).astype(jnp.float64)
        else:
            step_increment = jnp.prod(
                jnp.array(batch_dims),
            ).astype(jnp.float64)

        if self.pmap_axis_name:
            step_increment = jax.lax.psum(
                step_increment, axis_name=self.pmap_axis_name,
            )

        count = self.count[...] + step_increment

        def compute_node_statistics(
            batch_leaf: jnp.ndarray,
            mean_statistics: jnp.ndarray,
            variance_statistics: jnp.ndarray,
        ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            if weights is not None:
                feature_ndim = batch_leaf.ndim - len(batch_dims)
                expanded_weights = weights.reshape(
                    weights.shape + (1,) * feature_ndim,
                )
            else:
                expanded_weights = jnp.array(1.0, dtype=batch_leaf.dtype)

            if self.mode == NormalizationMode.WELFORD:
                diff_to_old_mean = (
                    batch_leaf - mean_statistics
                ) * expanded_weights

                mean_update = jnp.sum(
                    diff_to_old_mean, axis=batch_axis,
                ) / count

                if self.pmap_axis_name:
                    mean_update = jax.lax.psum(
                        mean_update, axis_name=self.pmap_axis_name,
                    )

                mean = mean_statistics + mean_update

                diff_to_new_mean = batch_leaf - mean

                # Variance Update
                variance_update = jnp.sum(
                    diff_to_old_mean * diff_to_new_mean, axis=batch_axis,
                )

                if self.pmap_axis_name:
                    variance_update = jax.lax.psum(
                        variance_update, axis_name=self.pmap_axis_name,
                    )

                variance = variance_statistics + variance_update

                std = jnp.sqrt(jnp.maximum(variance, 0) / count + self.epsilon)
                std = jnp.clip(std, self.std_min, self.std_max)

                return mean, variance, std

            elif self.mode == NormalizationMode.EMA:
                rate = jnp.float32(step_increment) / count

                if weights is not None:
                    weight_sum = jnp.sum(expanded_weights)
                    weight_sum = jnp.maximum(weight_sum, 1e-6)

                    # Weighted Mean
                    batch_mean = jnp.sum(
                        batch_leaf * expanded_weights, axis=batch_axis,
                    ) / weight_sum

                    # Weighted Variance
                    diff = batch_leaf - batch_mean
                    batch_variance = jnp.sum(
                        expanded_weights * (diff ** 2), axis=batch_axis,
                    ) / weight_sum
                else:
                    batch_mean = jnp.mean(batch_leaf, axis=batch_axis)
                    batch_variance = jnp.var(batch_leaf, axis=batch_axis)

                if self.pmap_axis_name:
                    batch_mean = jax.lax.pmean(
                        batch_mean, axis_name=self.pmap_axis_name,
                    )
                    batch_variance = jax.lax.pmean(
                        batch_variance, axis_name=self.pmap_axis_name,
                    )

                delta = batch_mean - mean_statistics
                mean = mean_statistics + rate * delta
                variance = variance_statistics + rate * (
                    batch_variance - variance_statistics
                    + delta * (batch_mean - mean)
                )

                std = jnp.sqrt(jnp.maximum(variance, 0) + self.epsilon)
                std = jnp.clip(std, self.std_min, self.std_max)

                return mean, variance, std

            else:
                raise ValueError(f"Unknown normalization mode: {self.mode}")

        # Compute updates
        updated_statistics = jax.tree.map(
            compute_node_statistics,
            batch,
            self.mean,
            self.var,
        )

        def apply_updates(
            mean: nnx.BatchStat,
            variance: nnx.BatchStat,
            std: nnx.BatchStat,
            update_tuple: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
        ) -> None:
            mean[...] = update_tuple[0]
            variance[...] = update_tuple[1]
            std[...] = update_tuple[2]

        jax.tree.map(
            apply_updates,
            self.mean,
            self.var,
            self.std,
            updated_statistics,
            is_leaf=lambda x: isinstance(x, nnx.BatchStat)
        )

        self.count[...] = count

    def normalize(
        self,
        batch: NestedArray,
    ) -> NestedArray:
        """Normalizes data using running statistics."""
        batch = _to_nnx(batch)

        def normalize_leaf(
            data: jnp.ndarray,
            mean: jnp.ndarray,
            std: jnp.ndarray,
        ) -> jnp.ndarray:
            if not jnp.issubdtype(data.dtype, jnp.inexact):
                return data

            data = (data - mean) / std

            if self.clip is not None:
                data = jnp.clip(data, -self.clip, +self.clip)
            return data

        result = jax.tree.map(
            normalize_leaf, batch, self.mean, self.std,
        )

        return jax.tree.unflatten(self.structure_def, jax.tree.leaves(result))

    def denormalize(
        self,
        batch: NestedArray,
    ) -> NestedArray:
        """Denormalizes values in a nested structure using the given mean/std.

        Only values of inexact types are denormalized.
        See https://numpy.org/doc/stable/_images/dtype-hierarchy.png
        for Numpy type hierarchy.

        Args:
            batch: a nested structure containing batch of data.

        Returns:
            NestedArray structure with denormalized values.
        """
        batch = _to_nnx(batch)

        def denormalize_leaf(
            data: jnp.ndarray,
            mean: jnp.ndarray,
            std: jnp.ndarray,
        ) -> jnp.ndarray:
            # Only denormalize inexact
            if not jnp.issubdtype(data.dtype, jnp.inexact):
                return data
            return data * std + mean

        result = jax.tree.map(
            denormalize_leaf, batch, self.mean, self.std,
        )

        return jax.tree.unflatten(self.structure_def, jax.tree.leaves(result))

    def _validate_batch_shape(
        self,
        batch: NestedArray,
        batch_dims: Tuple[int, ...],
    ) -> None:
        """
            Validates that all leaves in 'batch' match the structural
            definition and have consistent batch dimensions.
        """

        def validate_leaf(
            batch_leaf: jnp.ndarray,
            statistics_leaf: nnx.BatchStat,
        ) -> None:
            expected_shape = batch_dims + statistics_leaf[...].shape
            assert batch_leaf.shape == expected_shape, \
                f"Shape mismatch: {batch_leaf.shape} != {expected_shape} (Batch dims: {batch_dims})"

        jax.tree.map(validate_leaf, batch, self.mean, is_leaf=lambda x: isinstance(x, nnx.BatchStat))
