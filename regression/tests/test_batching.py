"""Tests for chunked full-batch evaluation (regression/utilities/batching.py).

The design claim under test: chunking changes the memory profile, never the
math. Chunked value/grad must equal the unchunked ones exactly (up to fp
reassociation), and a fully-wired chunked L-BFGS must reach the same optimum
as an unchunked one -- here verified against the analytic solution.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from regression.utilities.autodiff import (
    forward_mode_value_and_grad,
    value_and_grad_from_state_using,
)
from regression.utilities.batching import (
    chunk_dataset,
    chunked_value,
    chunked_value_and_grad,
)
from regression.utilities.optax_utilities import forward_mode_scale_by_zoom_linesearch
from regression.utilities.optimizers import OptimizerConfig, build_optimizer, make_solver


_REG = 1e-3


def _lsq_loss(params, batch):
    """Regularized least squares: data mean + a params-only reg term.

    The reg term deliberately mirrors the real loss structure (data_mean + reg)
    to prove the chunked mean counts regularization exactly once.
    """
    residual = batch["x"] @ params["w"] - batch["y"]
    return jnp.mean(residual**2) + _REG * jnp.sum(params["w"] ** 2)


def _make_data(n=64, d=3, seed=0):
    rng = np.random.default_rng(seed)
    x = jnp.asarray(rng.standard_normal((n, d)))
    w_true = jnp.asarray(rng.standard_normal(d))
    y = x @ w_true + 0.05 * jnp.asarray(rng.standard_normal(n))
    return {"x": x, "y": y}


def _analytic_solution(data):
    """argmin of mean((Xw - y)^2) + REG * ||w||^2."""
    x = np.asarray(data["x"]); y = np.asarray(data["y"])
    n = x.shape[0]
    lhs = x.T @ x / n + _REG * np.eye(x.shape[1])
    return np.linalg.solve(lhs, x.T @ y / n)


# ----------------------------------------------------------------------------
# chunk_dataset
# ----------------------------------------------------------------------------
def test_chunk_dataset_shapes():
    data = _make_data(n=64)
    chunked = chunk_dataset(data, 8)
    assert chunked["x"].shape == (8, 8, 3)
    assert chunked["y"].shape == (8, 8)


def test_chunk_dataset_warns_on_truncation(capsys):
    data = _make_data(n=10)
    chunked = chunk_dataset(data, 4)
    assert chunked["x"].shape == (2, 4, 3)
    assert "dropping 2 of 10" in capsys.readouterr().out


def test_chunk_dataset_rejects_oversized_chunk():
    with pytest.raises(ValueError, match="exceeds the number of samples"):
        chunk_dataset(_make_data(n=4), 8)


# ----------------------------------------------------------------------------
# Exactness: chunked == unchunked.
# ----------------------------------------------------------------------------
def test_chunked_value_matches_direct():
    data = _make_data()
    params = {"w": jnp.array([0.5, -1.0, 2.0])}
    direct = _lsq_loss(params, data)
    chunked = chunked_value(_lsq_loss)(params, chunk_dataset(data, 8))
    np.testing.assert_allclose(float(chunked), float(direct), rtol=1e-6)


def test_chunked_value_and_grad_matches_direct():
    data = _make_data()
    params = {"w": jnp.array([0.5, -1.0, 2.0])}

    v_direct, g_direct = jax.value_and_grad(_lsq_loss)(params, data)
    v_chunked, g_chunked = chunked_value_and_grad(
        forward_mode_value_and_grad(_lsq_loss)
    )(params, chunk_dataset(data, 8))

    np.testing.assert_allclose(float(v_chunked), float(v_direct), rtol=1e-6)
    np.testing.assert_allclose(
        np.asarray(g_chunked["w"]), np.asarray(g_direct["w"]), rtol=1e-5
    )


# ----------------------------------------------------------------------------
# End-to-end: chunked full-batch L-BFGS, fully forward-mode.
# ----------------------------------------------------------------------------
def _run_chunked_lbfgs(data, chunk_size, n_steps=30):
    chunked = chunk_dataset(data, chunk_size)

    # Per-chunk forward-mode AD, accumulated across chunks: only one chunk's
    # linearization is alive at a time.
    full_vg = chunked_value_and_grad(forward_mode_value_and_grad(_lsq_loss))
    bound_vg = lambda p, **kw: full_vg(p, chunked)          # noqa: E731
    bound_value = lambda p, **kw: chunked_value(_lsq_loss)(p, chunked)  # noqa: E731

    linesearch = forward_mode_scale_by_zoom_linesearch(
        max_linesearch_steps=15, value_and_grad_fn=bound_vg,
    )
    cfg = OptimizerConfig(
        optimizer_type="lbfgs", optimizer_params={"linesearch": linesearch}
    )
    solver = make_solver(
        cfg,
        build_optimizer(cfg),
        value_and_grad_from_state_using(bound_vg),
        loss_fn=bound_value,
    )

    params = {"w": jnp.zeros(3)}
    state = solver.init(params)
    step = jax.jit(solver.step)
    for _ in range(n_steps):
        params, state, metrics = step(params, state, None)
    return params, metrics


def test_chunked_full_batch_lbfgs_reaches_analytic_solution():
    data = _make_data()
    params, metrics = _run_chunked_lbfgs(data, chunk_size=8)
    np.testing.assert_allclose(
        np.asarray(params["w"]), _analytic_solution(data), atol=1e-4
    )
    assert np.isfinite(float(metrics["loss"]))


def test_chunked_and_single_chunk_lbfgs_agree():
    """chunk_size = N (one chunk) is the unchunked objective; results must match."""
    data = _make_data()
    p_chunked, _ = _run_chunked_lbfgs(data, chunk_size=8)
    p_single, _ = _run_chunked_lbfgs(data, chunk_size=64)
    np.testing.assert_allclose(
        np.asarray(p_chunked["w"]), np.asarray(p_single["w"]), atol=1e-5
    )
