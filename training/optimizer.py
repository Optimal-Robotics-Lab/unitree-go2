from typing import NamedTuple
import dataclasses

import jax
import jax.numpy as jnp

import optax


@dataclasses.dataclass
class OptimizerConfig:
    learning_rate: float = 3e-4
    grad_clip_norm: float = 1.0
    desired_kl: float | None = None
    min_learning_rate: float = 1e-5
    max_learning_rate: float = 1e-2
    kl_adjustment_factor: float = 1.5


def create_optimizer(
    optimizer_config: OptimizerConfig = OptimizerConfig(),
) -> optax.GradientTransformation:
    """Creates an optimizer from OptimizerConfig.

    Args:
        optimizer_config: An OptimizerConfig instance.

    Returns:
        An optax.GradientTransformation.
    """
    if optimizer_config.desired_kl is None:
        optimizer = optax.chain(
            optax.clip_by_global_norm(optimizer_config.grad_clip_norm),
            optax.adam(learning_rate=optimizer_config.learning_rate),
        )
        return optimizer
    elif optimizer_config.desired_kl is not None:
        optimizer = optax.chain(
            optax.clip_by_global_norm(optimizer_config.grad_clip_norm),
            optax.scale_by_adam(eps=1e-5),
            adaptive_kl_scheduler(
                init_lr=optimizer_config.learning_rate,
                desired_kl=optimizer_config.desired_kl,
                min_lr=optimizer_config.min_learning_rate,
                max_lr=optimizer_config.max_learning_rate,
                adjustment_factor=optimizer_config.kl_adjustment_factor,
            ),
            optax.scale(-1),
        )
        return optimizer
    else:
        raise ValueError("Invalid OptimizerConfig.")


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
