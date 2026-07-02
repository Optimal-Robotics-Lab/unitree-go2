from __future__ import annotations

from typing import Callable, Any

import jax
import jax.numpy as jnp
import jax.flatten_util

import optax


def forward_mode_value_and_grad(loss_fn: Callable):
    """
    Returns a function that computes (value, grad) using forward-mode AD.
    """
    def value_and_grad_fwd(params, *args, **kwargs):
        loss_val, jvp_fn = jax.linearize(
            lambda p: loss_fn(p, *args, **kwargs), params
        )

        flat_params, unflatten = jax.flatten_util.ravel_pytree(params)
        basis = jnp.eye(len(flat_params))

        flat_grads = jax.vmap(lambda v: jvp_fn(unflatten(v)))(basis)

        grads = unflatten(flat_grads)

        return loss_val, grads

    return value_and_grad_fwd


def value_and_grad_from_state_using(value_and_grad_fn: Callable):
    """Reuse the (value, grad) cached in a line-search state, recomputing with
    ``value_and_grad_fn`` when the cache is invalid (first step / failed search).

    Generic over the AD mode of ``value_and_grad_fn`` -- pass e.g.
    ``forward_mode_value_and_grad(loss_fn)`` or a chunked accumulator from
    ``batching.chunked_value_and_grad`` so the recompute path has the same
    memory profile as the rest of the pipeline.
    """

    def _value_and_grad(
        params: optax.Params,
        *fn_args: Any,
        state: optax.OptState,
        **fn_kwargs: dict[str, Any],
    ):
        value = optax.tree.get(state, "value")
        grad = optax.tree.get(state, "grad")

        if (value is None) or (grad is None):
            raise ValueError(
                "Value or gradient not found in the state. "
                "Make sure that these values are stored in the state by the "
                "optimizer."
            )

        value, grad = jax.lax.cond(
            (~jnp.isinf(value)) & (~jnp.isnan(value)),
            lambda *_: (value, grad),
            lambda p, a, kwa: value_and_grad_fn(p, *a, **kwa),
            params,
            fn_args,
            fn_kwargs,
        )
        return value, grad

    return _value_and_grad


def forward_mode_value_and_grad_from_state(loss_fn: Callable):
    """
        Returns a function that computes (value, grad) from state using forward-mode AD.
    """
    return value_and_grad_from_state_using(forward_mode_value_and_grad(loss_fn))
