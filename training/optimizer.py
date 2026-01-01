from typing import NamedTuple

import jax
import jax.numpy as jnp

import optax


def ignore_extra_args(
    optimizer: optax.GradientTransformation
) -> optax.GradientTransformationExtraArgs:
    """Wraps an optimizer to silently ignore any extra keyword arguments."""

    def init_fn(params):
        return optimizer.init(params)

    def update_fn(updates, state, params=None, **extra_args):
        return optimizer.update(updates, state, params)

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)


class AdaptiveKLState(NamedTuple):
    """State for the Adaptive KL scheduler."""
    learning_rate: jax.Array


def adaptive_kl_scheduler(
    init_lr: float,
    desired_kl: float,
    min_lr: float = 1e-5,
    max_lr: float = 1e-2,
    adjustment_factor: float = 1.5,
) -> optax.GradientTransformationExtraArgs:
    """Adaptive Learning Rate based on KL Divergence.

    Scales updates by a learning rate that adapts based on the `kl_mean` metric.
    To use this, you must pass `kl_mean` as a keyword argument to `optimizer.update`.

    Args:
        init_lr: Initial learning rate.
        desired_kl: The target KL divergence.
        min_lr: Minimum allowed learning rate.
        max_lr: Maximum allowed learning rate.
        adjustment_factor: Factor by which to increase/decrease the LR (default 1.5).

    Returns:
        An optax.GradientTransformationExtraArgs.
    """

    def init_fn(params) -> AdaptiveKLState:
        del params  # Unused
        return AdaptiveKLState(learning_rate=jnp.array(init_lr, dtype=jnp.float32))

    def update_fn(
        updates: optax.Updates,
        state: AdaptiveKLState,
        params=None,
        *,
        kl_mean: float,
        **extra_args,
    ) -> tuple[optax.Params, AdaptiveKLState]:
        del params, extra_args

        # Stop gradient on kl_mean to prevent gradients flowing back.
        kl_mean = jax.lax.stop_gradient(kl_mean)

        lr = state.learning_rate

        lr_decreased = jnp.maximum(min_lr, lr / adjustment_factor)
        lr = jnp.where(kl_mean > desired_kl * 2.0, lr_decreased, lr)
        lr_increased = jnp.minimum(max_lr, lr * adjustment_factor)
        should_increase = (kl_mean < desired_kl / 2.0) & (kl_mean > 0.0)
        lr = jnp.where(should_increase, lr_increased, lr)

        # Update state:
        new_state = AdaptiveKLState(learning_rate=lr)

        # Scale updates by the new learning rate:
        updates = jax.tree.map(lambda g: g * lr, updates)

        return updates, new_state

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)
