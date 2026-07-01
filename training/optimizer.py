from typing import NamedTuple, Any
import dataclasses

import jax
import jax.numpy as jnp

import optax


@dataclasses.dataclass
class OptimizerConfig:
    optimizer_type: str = "adam"
    scheduler_type: str = "constant_schedule"
    optimizer_params: dict[str, Any] = dataclasses.field(default_factory=dict)
    scheduler_params: dict[str, Any] = dataclasses.field(default_factory=dict)
    grad_clip_norm: float = 1.0


def create_optimizer(
    optimizer_config: OptimizerConfig = OptimizerConfig(),
) -> optax.GradientTransformation:
    """Creates an optimizer from OptimizerConfig.

    Args:
        optimizer_config: An OptimizerConfig instance.

    Returns:
        An optax.GradientTransformation.
    """

    components = []
    if optimizer_config.grad_clip_norm > 0:
        components.append(optax.clip_by_global_norm(optimizer_config.grad_clip_norm))

    # Check for KL-based scheduler:
    if optimizer_config.scheduler_type == "adaptive_kl_schedule":
        try:
            scale_by_cls = getattr(optax, optimizer_config.optimizer_type)
        except AttributeError:
            raise ValueError(
                f"Unsupported optimizer type: '{optimizer_config.optimizer_type}'. "
                "Using the adaptive KL scheduler requires a scale_by_* transformation. "
                "Please check the optax documentation for valid names."
            )

        components.append(scale_by_cls(**optimizer_config.optimizer_params))
        components.append(adaptive_kl_schedule(**optimizer_config.scheduler_params))
        components.append(optax.scale(-1))
    else:
        try:
            scheduler_cls = getattr(optax, optimizer_config.scheduler_type)
        except AttributeError:
            raise ValueError(
                f"Unsupported scheduler type: '{optimizer_config.scheduler_type}'. "
                "Please check the optax documentation for valid names."
            )

        scheduler = scheduler_cls(**optimizer_config.scheduler_params)

        try:
            optimizer_cls = getattr(optax, optimizer_config.optimizer_type)
        except AttributeError:
            raise ValueError(
                f"Unsupported optimizer type: '{optimizer_config.optimizer_type}'. "
                "Please check the optax documentation for valid names."
            )

        # Check and remove learning rate in optimizer params:
        opt_params = optimizer_config.optimizer_params.copy()
        if "learning_rate" in opt_params:
            opt_params.pop("learning_rate")
            print("Warning: 'learning_rate' should not be specified in optimizer_params when using a scheduler. ")

        components.append(optimizer_cls(learning_rate=scheduler, **opt_params))

    return optax.chain(*components)


class AdaptiveKLState(NamedTuple):
    """State for the Adaptive KL scheduler."""
    learning_rate: jax.Array


def adaptive_kl_schedule(
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
