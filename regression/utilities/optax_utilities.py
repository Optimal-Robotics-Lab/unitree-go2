from __future__ import annotations

from typing import Callable, Any, Optional

import functools

import jax
import jax.numpy as jnp

import optax

from regression.utilities.autodiff import forward_mode_value_and_grad


def forward_mode_scale_by_zoom_linesearch(
    max_linesearch_steps: jax.typing.ArrayLike,  # int
    max_learning_rate: Optional[jax.typing.ArrayLike] = None,  # float
    tol: jax.typing.ArrayLike = 0.0,
    increase_factor: jax.typing.ArrayLike = 2.0,
    slope_rtol: jax.typing.ArrayLike = 1e-4,
    curv_rtol: jax.typing.ArrayLike = 0.9,
    approx_dec_rtol: Optional[jax.typing.ArrayLike] = 1e-6,
    stepsize_precision: jax.typing.ArrayLike = 1e-5,
    initial_guess_strategy: str = "keep",
    verbose: bool = False,
    value_and_grad_fn: Optional[Callable] = None,
) -> optax.GradientTransformationExtraArgs:
    # Forward-mode Compatibility version of scale_by_zoom_linesearch.
    #
    # Forked from optax 0.2.8 scale_by_zoom_linesearch; sole functional changes:
    #   * internal gradients use forward-mode AD instead of jax.value_and_grad
    #   * optional `value_and_grad_fn` override: a callable `(params, **fn_kwargs)
    #     -> (value, grad)` used for the line search's internal evaluations. Pass
    #     a chunked accumulator (batching.chunked_value_and_grad) for full-batch
    #     objectives so each evaluation stays memory-bounded. MUST compute the
    #     same function as `value_fn` -- consistency is the caller's
    #     responsibility. Default (None) linearizes `value_fn` directly.

    # Instantiates the linesearch with the given arguments.
    init_ls, step_ls, cond_step_ls = optax._src.linesearch.zoom_linesearch(
        max_linesearch_steps=max_linesearch_steps,
        max_stepsize=max_learning_rate,
        tol=tol,
        increase_factor=increase_factor,
        slope_rtol=slope_rtol,
        curv_rtol=curv_rtol,
        approx_dec_rtol=approx_dec_rtol,
        interval_threshold=stepsize_precision,
        verbose=verbose,
    )

    def init_fn(params: optax.Params) -> optax.ScaleByZoomLinesearchState:
        """Initializes state of scale_by_zoom_linesearch."""
        placeholder = jnp.empty((), jax.tree.leaves(params)[0].dtype)
        val_dtype = jnp.real(placeholder).dtype
        return optax.ScaleByZoomLinesearchState(
            learning_rate=jnp.asarray(1.0, dtype=val_dtype),
            value=jnp.asarray(jnp.inf, dtype=val_dtype),
            grad=optax.tree.zeros_like(params),
            info=optax.ZoomLinesearchInfo(
                num_linesearch_steps=jnp.asarray(0),
                decrease_error=jnp.asarray(jnp.inf),
                curvature_error=jnp.asarray(jnp.inf),
            ),
        )

    def update_fn(
        updates: optax.Updates,
        state: optax.ScaleByZoomLinesearchState,
        params: optax.Params,
        *,
        value: jax.typing.ArrayLike,
        grad: optax.Updates,
        value_fn: Callable[..., tuple[jax.typing.ArrayLike, optax.Updates]],
        **extra_args: dict[str, Any],
    ) -> tuple[optax.Updates, optax.ScaleByZoomLinesearchState]:
        (fn_kwargs,), remaining_kwargs = optax._src.linesearch._extract_fns_kwargs(
            (value_fn,), extra_args
        )
        if remaining_kwargs:
            raise TypeError(
                "Unexpected keyword arguments passed to "
                "`scale_by_zoom_linesearch.update`. "
                f"These arguments were not consumed by `value_fn`: "
                f"{sorted(remaining_kwargs.keys())}. "
                "Ensure that all extra keyword arguments are accepted "
                "by `value_fn`."
            )

        if value_and_grad_fn is None:
            ls_value_and_grad_fn = forward_mode_value_and_grad(value_fn)
        else:
            ls_value_and_grad_fn = value_and_grad_fn

        init_state = init_ls(
            updates,
            params,
            value=value,
            grad=grad,
            prev_stepsize=state.learning_rate,
            initial_guess_strategy=initial_guess_strategy,
        )

        final_state = jax.lax.while_loop(
            cond_step_ls,
            functools.partial(
                step_ls,
                value_and_grad_fn=ls_value_and_grad_fn,
                fn_kwargs=fn_kwargs,
            ),
            init_state,
        )
        learning_rate = final_state.stepsize
        scaled_updates = optax.tree.scale(learning_rate, updates)
        info_step = optax.ZoomLinesearchInfo(
            num_linesearch_steps=final_state.count,
            decrease_error=final_state.decrease_error,
            curvature_error=final_state.curvature_error,
        )
        new_state = optax.ScaleByZoomLinesearchState(
            learning_rate=learning_rate,
            value=final_state.value,
            grad=final_state.grad,
            info=info_step,
        )
        return scaled_updates, optax.tree.cast_like(new_state, other_tree=state)

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)
