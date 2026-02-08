import optax
import jax.numpy as jnp
from ml_collections import ConfigDict

from regression.utilities.typedefs import ObjectiveFunction


def create_optimizer(cfg: ConfigDict, total_steps: int) -> optax.GradientTransformation:
    """Creates the optimizer chain based on config."""
    warmup_steps = int(cfg.optimizer.warmup_pct * total_steps)
    decay_steps = total_steps - warmup_steps
    
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=cfg.optimizer.lr_init,
        peak_value=cfg.optimizer.lr_peak,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        end_value=cfg.optimizer.lr_end,
    )
    
    return optax.chain(
        optax.clip_by_global_norm(cfg.optimizer.clip_norm),
        optax.adamw(learning_rate=schedule, weight_decay=cfg.optimizer.weight_decay)
    )


def get_objective_fn(loss_type: str) -> ObjectiveFunction:
    """Returns the JAX-compatible objective function."""
    if loss_type == 'rmse':
        return lambda p, t: jnp.sqrt(jnp.mean((p - t) ** 2))
    elif loss_type == 'mse':
        return lambda p, t: jnp.mean((p - t) ** 2)
    elif loss_type == 'mae':
        return lambda p, t: jnp.mean(jnp.abs(p - t))
    elif loss_type == 'huber':
        return lambda p, t: jnp.mean(optax.huber_loss(p, t, delta=1.0))
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
