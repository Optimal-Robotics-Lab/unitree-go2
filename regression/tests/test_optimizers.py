"""Tests for the optimizer factory, solver strategies, and forward-mode AD glue.

Covers the real usage pattern: value_and_grad functions are pre-wrapped
(functools.partial / closure) before being handed to the solver, including the
forward-mode variants from regression.utilities.autodiff.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from regression.utilities.autodiff import (
    forward_mode_value_and_grad,
    forward_mode_value_and_grad_from_state,
)
from regression.utilities.optax_utilities import forward_mode_scale_by_zoom_linesearch
from regression.utilities.optimizers import (
    OptimizerConfig,
    build_optimizer,
    make_solver,
    StochasticSolver,
    LineSearchSolver,
)


_TARGET = jnp.array([1.0, -2.0, 3.0])


def _loss(params, batch):
    return jnp.sum((params["x"] - batch) ** 2)


# ----------------------------------------------------------------------------
# Factory.
# ----------------------------------------------------------------------------
def test_build_optimizer_unknown_type_raises():
    cfg = OptimizerConfig(optimizer_type="not_a_real_optimizer")
    with pytest.raises(ValueError, match="Unknown optax attribute"):
        build_optimizer(cfg)


def test_build_optimizer_first_order_with_scheduler():
    cfg = OptimizerConfig(
        optimizer_type="adamw",
        scheduler_type="warmup_cosine_decay_schedule",
        scheduler_params={
            "init_value": 1e-5, "peak_value": 1e-2, "end_value": 1e-6,
            "warmup_steps": 100, "decay_steps": 900,
        },
    )
    build_optimizer(cfg).init({"x": jnp.zeros(3)})


def test_make_solver_line_search_requires_loss_fn():
    cfg = OptimizerConfig(optimizer_type="lbfgs")
    with pytest.raises(ValueError, match="require a loss_fn"):
        make_solver(cfg, build_optimizer(cfg), lambda p, b: (0.0, p))


# ----------------------------------------------------------------------------
# Forward-mode AD glue.
# ----------------------------------------------------------------------------
def test_forward_mode_matches_reverse_mode():
    params = {"x": jnp.array([0.3, -1.2, 2.2])}
    v_fwd, g_fwd = forward_mode_value_and_grad(_loss)(params, _TARGET)
    v_rev, g_rev = jax.value_and_grad(_loss)(params, _TARGET)
    np.testing.assert_allclose(float(v_fwd), float(v_rev), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(g_fwd["x"]), np.asarray(g_rev["x"]), rtol=1e-6)


def test_from_state_raises_without_stored_value():
    """A state that stores no value/grad (adamw) must fail loudly, not silently."""
    cfg = OptimizerConfig(
        optimizer_type="adamw", scheduler_type="constant_schedule",
        scheduler_params={"value": 0.1},
    )
    opt = build_optimizer(cfg)
    params = {"x": jnp.zeros(3)}
    state = opt.init(params)
    bound_loss = functools.partial(_loss, batch=_TARGET)
    with pytest.raises(ValueError, match="not found in the state"):
        forward_mode_value_and_grad_from_state(bound_loss)(params, state=state)


# ----------------------------------------------------------------------------
# Solvers (jitted, converging, with forward-mode wrappers as in real usage).
# ----------------------------------------------------------------------------
def test_stochastic_solver_forward_mode_minimizes():
    cfg = OptimizerConfig(
        optimizer_type="adamw", scheduler_type="constant_schedule",
        scheduler_params={"value": 0.1},
    )
    solver = make_solver(
        cfg, build_optimizer(cfg), forward_mode_value_and_grad(_loss)
    )
    assert isinstance(solver, StochasticSolver)

    params = {"x": jnp.zeros(3)}
    state = solver.init(params)
    step = jax.jit(solver.step)
    for _ in range(500):
        params, state, metrics = step(params, state, _TARGET)
    np.testing.assert_allclose(np.asarray(params["x"]), np.asarray(_TARGET), atol=1e-2)
    assert np.isfinite(float(metrics["loss"]))


def _run_lbfgs(optimizer_params, n_steps=30):
    cfg = OptimizerConfig(optimizer_type="lbfgs", optimizer_params=optimizer_params)
    bound_loss = functools.partial(_loss, batch=_TARGET)   # full batch pre-bound
    solver = make_solver(
        cfg,
        build_optimizer(cfg),
        forward_mode_value_and_grad_from_state(bound_loss),
        loss_fn=bound_loss,
    )
    assert isinstance(solver, LineSearchSolver)

    params = {"x": jnp.zeros(3)}
    state = solver.init(params)
    step = jax.jit(solver.step)
    for _ in range(n_steps):
        params, state, metrics = step(params, state, None)  # batch unused by design
    return params, metrics


def test_line_search_solver_default_zoom_minimizes():
    params, metrics = _run_lbfgs({})
    np.testing.assert_allclose(np.asarray(params["x"]), np.asarray(_TARGET), atol=1e-4)
    assert np.isfinite(float(metrics["loss"]))


def test_line_search_solver_backtracking_store_grad_minimizes():
    """The forward-mode-friendly configuration: backtracking linesearch computes
    its gradient once (linearize + transpose) at the accepted point instead of
    zoom's jax.value_and_grad at every line-search iterate."""
    linesearch = optax.scale_by_backtracking_linesearch(
        max_backtracking_steps=20, store_grad=True,
    )
    params, metrics = _run_lbfgs({"linesearch": linesearch}, n_steps=50)
    np.testing.assert_allclose(np.asarray(params["x"]), np.asarray(_TARGET), atol=1e-3)
    assert np.isfinite(float(metrics["loss"]))


def test_line_search_skips_grad_clip():
    """Clipping would corrupt L-BFGS curvature pairs, so it must be omitted."""
    assert OptimizerConfig(optimizer_type="lbfgs").is_line_search
    assert not OptimizerConfig(optimizer_type="adamw").is_line_search


# ----------------------------------------------------------------------------
# Forward-mode zoom linesearch (optax_utilities).
# ----------------------------------------------------------------------------
def _run_lbfgs_on(loss, optimizer_params, n_steps):
    """Drive LineSearchSolver on ``loss`` with an lbfgs configured by dict."""
    cfg = OptimizerConfig(optimizer_type="lbfgs", optimizer_params=optimizer_params)
    bound_loss = functools.partial(loss, batch=_TARGET)
    solver = make_solver(
        cfg,
        build_optimizer(cfg),
        forward_mode_value_and_grad_from_state(bound_loss),
        loss_fn=bound_loss,
    )
    params = {"x": jnp.zeros(3)}
    state = solver.init(params)
    step = jax.jit(solver.step)
    for _ in range(n_steps):
        params, state, metrics = step(params, state, None)
    return params, metrics


def _while_loop_loss(params, batch):
    """Objective reverse-mode AD cannot differentiate (while_loop has no
    transpose rule) but forward-mode can -- the MJX-solver situation."""
    def cond(carry):
        return carry[1] < 5

    def body(carry):
        x, i = carry
        return x + 0.5 * jnp.tanh(batch - x), i + 1

    x, _ = jax.lax.while_loop(cond, body, (params["x"], 0))
    return jnp.sum((x - batch) ** 2)


def test_while_loop_loss_is_reverse_mode_incompatible():
    """Probe validity: reverse-mode genuinely fails on this objective."""
    params = {"x": jnp.zeros(3)}
    with pytest.raises(ValueError, match="[Rr]everse-mode"):
        jax.value_and_grad(_while_loop_loss)(params, _TARGET)


def test_default_zoom_linesearch_uses_reverse_mode():
    """Documents the problem: optax's default zoom internally calls
    jax.value_and_grad(value_fn), so it fails on a forward-only objective."""
    with pytest.raises(ValueError, match="[Rr]everse-mode"):
        _run_lbfgs_on(_while_loop_loss, {}, n_steps=1)


def test_forward_zoom_linesearch_is_fully_forward_mode():
    """The fix: with forward_mode_scale_by_zoom_linesearch, the entire L-BFGS
    step (outer grad + line search) differentiates a reverse-incompatible
    objective and still minimizes it."""
    linesearch = forward_mode_scale_by_zoom_linesearch(max_linesearch_steps=15)
    params, metrics = _run_lbfgs_on(
        _while_loop_loss, {"linesearch": linesearch}, n_steps=25
    )
    # Minimum is at x = target (fixed point of the inner loop with zero loss).
    np.testing.assert_allclose(np.asarray(params["x"]), np.asarray(_TARGET), atol=1e-3)
    assert np.isfinite(float(metrics["loss"]))


def test_forward_zoom_matches_default_zoom_on_smooth_objective():
    """Same algorithm, different AD mode: on a reverse-friendly objective the
    two zoom variants must produce (numerically) identical iterates."""
    default_zoom = {"linesearch": optax.scale_by_zoom_linesearch(max_linesearch_steps=15)}
    forward_zoom = {"linesearch": forward_mode_scale_by_zoom_linesearch(max_linesearch_steps=15)}
    p_ref, _ = _run_lbfgs_on(_loss, default_zoom, n_steps=15)
    p_fwd, _ = _run_lbfgs_on(_loss, forward_zoom, n_steps=15)
    np.testing.assert_allclose(
        np.asarray(p_fwd["x"]), np.asarray(p_ref["x"]), atol=1e-5
    )
