"""Tests for the log-Cholesky inertia parameterization and its eigh gradient path.

These exercise the real Unitree Go2 model and validate the theories discussed
around `log_cholesky_to_mujoco`:

  1. The forward map `theta_to_pi` is the exact inverse of `get_nominal_theta`
     (the U @ U.T convention fix).
  2. The Go2 thighs/calves are near inertially degenerate (rod-like), so their
     principal-moment gaps are small.
  3. Gradients through `eigh -> mjx.forward` are finite at nominal and match
     finite differences.
  4. The eigenvector sensitivity (hence the eigh gradient) scales like 1 / gap,
     which is the mechanism that can destabilize training as those links are
     pushed toward axisymmetry.
  5. The JIT-compatible diagnostics run under jit / grad.

Run: `PYTHONPATH=$PWD ./env/bin/python -m pytest regression/tests/test_inertia_gradients.py -v`
"""

from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import pytest

from regression.utilities.math import matrix_to_quaternion
from regression.utilities.model_utilities import (
    get_nominal_theta,
    log_cholesky_to_mujoco,
    log_cholesky_conditioning,
)
from regression.utilities.diagnostics import principal_moment_gaps


SCENE = "regression/mjcf/scene_mjx_transparent.xml"

ESTIMATED_BODIES = [
    "front_right_hip", "front_right_thigh", "front_right_calf",
    "front_left_hip", "front_left_thigh", "front_left_calf",
    "hind_right_hip", "hind_right_thigh", "hind_right_calf",
    "hind_left_hip", "hind_left_thigh", "hind_left_calf",
]

# Bodies whose top two principal moments are nearly equal (rod-like about the
# long axis) — the near-degenerate, gradient-risky links.
DEGENERATE = {b for b in ESTIMATED_BODIES if b.endswith(("thigh", "calf"))}
WELL_SEPARATED = {b for b in ESTIMATED_BODIES if b.endswith("hip")}


def _repo_root() -> Path:
    # regression/tests/ -> repo root is two levels up.
    return Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def mj_model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(_repo_root() / SCENE))


@pytest.fixture(scope="module")
def body_ids(mj_model) -> np.ndarray:
    ids = [
        mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, b)
        for b in ESTIMATED_BODIES
    ]
    assert -1 not in ids, "an estimated body is missing from the model"
    return np.array(ids, dtype=np.int64)


@pytest.fixture(scope="module")
def nominal_theta(mj_model, body_ids) -> jnp.ndarray:
    return jnp.array(
        np.stack([get_nominal_theta(mj_model, int(b)) for b in body_ids])
    )


# ----------------------------------------------------------------------------
# 1. Convention: theta_to_pi (U U^T) is the exact inverse of get_nominal_theta.
# ----------------------------------------------------------------------------
def test_convention_roundtrip(body_ids, nominal_theta):
    """theta -> log_cholesky_to_mujoco -> MjModel -> get_nominal_theta == theta."""
    # Pass the nominal so the regularization is 1e-5 * I_nominal (relative), as in
    # production `rehydrate_model`; the default diag([1e-6..]) is ~2% of a calf's
    # smallest principal moment and would dominate the residual.
    mass, ipos, inertia, iquat = jax.device_get(
        jax.vmap(log_cholesky_to_mujoco)(nominal_theta, nominal_theta)
    )

    # Fresh model so we never mutate the shared module fixture.
    model = mujoco.MjModel.from_xml_path(str(_repo_root() / SCENE))
    for k, b in enumerate(body_ids):
        model.body_mass[b] = mass[k]
        model.body_ipos[b] = ipos[k]
        model.body_inertia[b] = inertia[k]
        model.body_iquat[b] = iquat[k]
    recovered = np.stack([get_nominal_theta(model, int(b)) for b in body_ids])

    # Residual is set by float32 + the relative eigh regularization (~1e-5), not
    # the convention. Before the U U^T fix this was O(1).
    np.testing.assert_allclose(recovered, np.asarray(nominal_theta), atol=2e-3)


def test_physical_interpretation_of_theta(nominal_theta):
    """Under U U^T: mass == exp(2*alpha) and COM == [t1, t2, t3]."""
    from regression.utilities.model_utilities import theta_to_pi, get_icom_from_pi

    for th in nominal_theta:
        pi = theta_to_pi(th)
        mass, ipos, _ = get_icom_from_pi(pi)
        assert np.isclose(float(mass), float(np.exp(2 * th[0])), rtol=1e-5)
        np.testing.assert_allclose(np.asarray(ipos), np.asarray(th[7:10]), atol=1e-6)


# ----------------------------------------------------------------------------
# 2. Near-degeneracy of the real Go2 links.
# ----------------------------------------------------------------------------
def test_principal_moment_gaps(nominal_theta):
    """Report per-body inertial degeneracy; thighs/calves should be tightest."""
    metrics = log_cholesky_conditioning(nominal_theta)
    gaps = np.asarray(metrics["min_principal_gap"])

    assert np.all(np.isfinite(gaps))
    assert np.all(gaps > 0.0)

    gap_by_body = dict(zip(ESTIMATED_BODIES, gaps))
    worst = min(gap_by_body, key=gap_by_body.get)
    assert worst in DEGENERATE, f"expected a rod-like link to be tightest, got {worst}"

    # The rod-like links are meaningfully closer to degeneracy than the hips.
    worst_degenerate = max(gap_by_body[b] for b in DEGENERATE)
    best_hip = min(gap_by_body[b] for b in WELL_SEPARATED)
    assert worst_degenerate < best_hip


# ----------------------------------------------------------------------------
# Physics loss through the real MJX inertia pipeline (com_pos consumes the eigh
# outputs: ximat @ diag(body_inertia) @ ximat.T).
# ----------------------------------------------------------------------------
def _make_physics_loss(mj_model, body_ids):
    mx = mjx.put_model(mj_model, impl="jax")
    ids = jnp.array(body_ids)
    qpos = mx.qpos0 + 0.1  # generic (non-aligned) link orientations

    def loss(theta_batch):
        mass, ipos, inertia, quat = jax.vmap(log_cholesky_to_mujoco)(theta_batch)
        model = mx.replace(
            body_mass=mx.body_mass.at[ids].set(mass),
            body_ipos=mx.body_ipos.at[ids].set(ipos),
            body_inertia=mx.body_inertia.at[ids].set(inertia),
            body_iquat=mx.body_iquat.at[ids].set(quat),
        )
        d = mjx.make_data(model).replace(qpos=qpos)
        d = mjx.forward(model, d)
        # Only the estimated bodies' cinert, excluding the constant mass column
        # (index 9): summing the whole model adds a large theta-independent
        # baseline (trunk mass) whose float32 ULP would quantize away the signal
        # in finite differences.
        return jnp.sum(d._impl.cinert[ids][:, :9] ** 2)

    return loss


def test_gradients_finite_at_nominal(mj_model, body_ids, nominal_theta):
    """Gradient of a real MJX physics loss w.r.t. theta is finite at nominal."""
    loss = jax.jit(_make_physics_loss(mj_model, body_ids))
    grads = jax.grad(loss)(nominal_theta)
    grads = np.asarray(grads)
    assert np.all(np.isfinite(grads)), "non-finite gradient at nominal parameters"


def test_autodiff_matches_finite_difference(mj_model, body_ids):
    """AD gradient matches central finite differences through mjx.forward.

    The composite theta -> loss is smooth (the eigh round-trips inside MJX), so
    away from exact degeneracy AD and FD must agree even though AD routes through
    the ill-conditioned eigenvector VJP. Run in float64 so the finite-difference
    reference is trustworthy (float32 central differences catastrophically cancel
    for the small-gradient components).
    """
    jax.config.update("jax_enable_x64", True)
    try:
        nominal = jnp.asarray(
            np.stack([get_nominal_theta(mj_model, int(b)) for b in body_ids]),
            dtype=jnp.float64,
        )
        loss = jax.jit(_make_physics_loss(mj_model, body_ids))
        g_ad = np.asarray(jax.grad(loss)(nominal))

        theta0 = np.asarray(nominal)
        eps = 1e-6
        # Representative components: alpha (mass), d1 (stretch), s12 (shear), t1 (COM).
        components = [0, 1, 4, 7]

        compared = 0
        for k in range(theta0.shape[0]):
            floor = 1e-8 * max(np.linalg.norm(g_ad[k]), 1e-12)
            for c in components:
                pert = theta0.copy()
                pert[k, c] += eps
                lp = float(loss(jnp.array(pert)))
                pert[k, c] -= 2 * eps
                lm = float(loss(jnp.array(pert)))
                g_fd = (lp - lm) / (2 * eps)
                g = g_ad[k, c]
                if max(abs(g), abs(g_fd)) < floor:
                    continue
                denom = max(abs(g_fd), abs(g))
                assert abs(g - g_fd) / denom < 1e-3, (
                    f"AD vs FD mismatch at body {ESTIMATED_BODIES[k]} comp {c}: "
                    f"AD={g:.6e} FD={g_fd:.6e}"
                )
                compared += 1

        assert compared > 0, "no components had resolvable finite-difference gradients"
    finally:
        jax.config.update("jax_enable_x64", False)


# ----------------------------------------------------------------------------
# 4. Mechanism: eigenvector sensitivity ~ 1 / (principal-moment gap).
# ----------------------------------------------------------------------------
def _quat_of_inertia(inertia):
    _, V = jnp.linalg.eigh(inertia)
    det = jnp.linalg.det(V)
    parity = jax.lax.stop_gradient(jnp.where(det < 0.0, -1.0, 1.0))
    V = V.at[:, 2].multiply(parity)
    return matrix_to_quaternion(V)


def test_eigenvector_sensitivity_scales_inverse_gap(mj_model, body_ids, nominal_theta):
    """Seed eigenvectors from a real calf, drive its top two moments together,
    and confirm d(iquat)/d(inertia) grows like 1 / gap."""
    from regression.utilities.model_utilities import theta_to_pi, get_icom_from_pi

    calf_idx = ESTIMATED_BODIES.index("front_right_calf")
    _, _, i_com = get_icom_from_pi(theta_to_pi(nominal_theta[calf_idx]))
    w, V = np.linalg.eigh(np.asarray(i_com))  # ascending; w[1], w[2] are the close pair
    w_bar = 0.5 * (w[1] + w[2])
    V = jnp.array(V)

    def inertia_with_gap(gap):
        eig = jnp.array([w[0], w_bar - 0.5 * gap, w_bar + 0.5 * gap])
        return V @ jnp.diag(eig) @ V.T

    sens_fn = jax.jit(lambda I: jnp.linalg.norm(jax.jacobian(_quat_of_inertia)(I)))

    gaps = np.array([1e-2, 1e-3, 1e-4, 1e-5, 1e-6])
    sens = np.array([float(sens_fn(inertia_with_gap(g))) for g in gaps])

    assert np.all(np.isfinite(sens))
    # Monotone growth as the gap shrinks.
    assert np.all(np.diff(sens) > 0)
    # sens * gap is ~constant (the 1/gap law) across four decades.
    products = sens * gaps
    assert products.max() / products.min() < 10.0
    # Dramatic conditioning swing across the sampled gaps.
    assert sens[-1] / sens[0] > 1e3


# ----------------------------------------------------------------------------
# 5. Diagnostics are jit / grad compatible.
# ----------------------------------------------------------------------------
def test_debug_hook_runs_under_jit(nominal_theta, capsys):
    """log_cholesky_to_mujoco(debug=True) prints diagnostics and stays finite."""
    fn = jax.jit(lambda th: log_cholesky_to_mujoco(th, debug=True)[2])

    # Forward: min-gap print fires.
    out = fn(nominal_theta[0])
    jax.block_until_ready(out)
    assert np.all(np.isfinite(np.asarray(out)))

    # Backward: grad_probe prints the eigh-input cotangent.
    g = jax.grad(lambda th: jnp.sum(log_cholesky_to_mujoco(th, debug=True)[2]))(
        nominal_theta[0]
    )
    jax.block_until_ready(g)
    assert np.all(np.isfinite(np.asarray(g)))

    printed = capsys.readouterr().out
    assert "[eigh] min_principal_gap" in printed
    assert "[grad_probe:eigh_in]" in printed


def test_conditioning_is_jittable(nominal_theta):
    metrics = jax.jit(log_cholesky_conditioning)(nominal_theta)
    gaps = np.asarray(metrics["min_principal_gap"])
    assert gaps.shape == (len(ESTIMATED_BODIES),)
    assert np.all(np.isfinite(gaps)) and np.all(gaps > 0.0)
