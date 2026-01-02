from typing import Tuple

import jax
import jax.numpy as jnp

from training.algorithms.ppo import agent
from training import module_types as types


def calculate_gae(
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    bootstrap_value: jnp.ndarray,
    truncation_mask: jnp.ndarray,
    termination_mask: jnp.ndarray,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Calculates the Generalized Advantage Estimation."""
    values_ = jnp.concatenate(
        [values, jnp.expand_dims(bootstrap_value, axis=0)], axis=0,
    )
    deltas = rewards + gamma * termination_mask * values_[1:] - values_[:-1]
    deltas *= truncation_mask

    initial_gae = jnp.zeros_like(bootstrap_value)

    def scan_loop(carry, xs):
        gae = carry
        truncation_mask, termination_mask, delta = xs
        gae = (
            delta
            + gamma * gae_lambda * termination_mask * truncation_mask * gae
        )
        return gae, gae

    _, vs = jax.lax.scan(
        scan_loop,
        initial_gae,
        (truncation_mask, termination_mask, deltas),
        length=int(truncation_mask.shape[0]),
        reverse=True,
    )

    vs = jnp.add(vs, values)
    vs_ = jnp.concatenate(
        [vs[1:], jnp.expand_dims(bootstrap_value, axis=0)], axis=0,
    )
    advantages = (
        rewards
        + gamma * termination_mask * vs_ - values
    ) * truncation_mask
    return jax.lax.stop_gradient(vs), jax.lax.stop_gradient(advantages)


def loss_function(
    agent: agent.Agent,
    data: types.Transition,
    rng_key: types.PRNGKey,
    policy_clip_coef: float = 0.2,
    value_clip_coef: float | None = None,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    normalize_advantages: bool = True,
) -> Tuple[jnp.ndarray, types.Metrics]:
    # Unpack action distribution:
    action_distribution = agent.action_distribution

    # Reorder data: (B, T, ...) -> (T, B, ...)
    data = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), data)

    # Policy returns logits (raw network output)
    logits = agent.policy(data.observation)

    # Value returns scalar values
    values = agent.value(data.observation)

    terminal_observation = jax.tree.map(lambda x: x[-1], data.next_observation)
    bootstrap_values = agent.value(terminal_observation)

    # Create masks for truncation and termination:
    rewards = data.reward
    truncation_mask = 1 - data.extras['state_data']['truncation']
    termination_mask = 1 - data.termination

    # Calculate GAE:
    vs, advantages = calculate_gae(
        rewards=rewards,
        values=values,
        bootstrap_value=bootstrap_values,
        truncation_mask=truncation_mask,
        termination_mask=termination_mask,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )

    if normalize_advantages:
        advantages = (
            (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)
        )

    # Calculate ratios:
    log_prob = action_distribution.log_prob(
        logits,
        data.extras['policy_data']['raw_action'],
    )
    previous_log_prob = data.extras['policy_data']['log_prob']
    log_ratios = log_prob - previous_log_prob
    ratios = jnp.exp(log_ratios)

    # Policy Loss:
    unclipped_loss = ratios * advantages
    clipped_loss = advantages * jnp.clip(
        ratios,
        1.0 - policy_clip_coef,
        1.0 + policy_clip_coef,
    )
    policy_loss = -jnp.mean(jnp.minimum(unclipped_loss, clipped_loss))

    # Value Loss:
    value_error = vs - values
    value_loss = value_error * value_error
    if value_clip_coef is not None:
        old_values = data.extras['policy_data']['value']
        value_clipped = old_values + jnp.clip(
            values - old_values,
            -value_clip_coef,
            value_clip_coef,
        )
        value_loss_clipped = (vs - value_clipped) ** 2
        value_loss = jnp.maximum(value_loss, value_loss_clipped)
    value_loss = jnp.mean(value_loss) * 0.5 * value_coef

    # Entropy Loss:
    entropy = action_distribution.entropy(
        logits,
        rng_key,
    )
    entropy_loss = -entropy_coef * jnp.mean(
        entropy,
    )

    loss = policy_loss + value_loss + entropy_loss

    # Calculate KL Divergence Metric:
    if hasattr(action_distribution, 'kl_divergence'):
        old_distribution_logits = data.extras['policy_data']['logits']
        kl_values = action_distribution.kl_divergence(
            logits, old_distribution_logits,
        )
        kl_mean = jnp.mean(kl_values)
    else:
        kl_mean = jnp.array(0.0)

    return loss, {
        "loss": loss,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy_loss": entropy_loss,
        "kl_mean": kl_mean,
    }
