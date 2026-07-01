"""Tests for parameter reparameterizations (regression/utilities/transforms.py).

Covers the math of each transform and the end-to-end config wiring, including
the key property motivating log_exp for the dof parameters: strictly-positive
physical values (the nominal-centered additive transform could go negative).
"""

from pathlib import Path

import numpy as np
import jax.numpy as jnp
import mujoco
import pytest

from regression.utilities import transforms
from regression.utilities.config import (
    get_default_config,
    process_regression_spec,
    build_parameter_scale,
)


SCENE = "regression/mjcf/scene_mjx_transparent.xml"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


# ----------------------------------------------------------------------------
# Per-transform math.
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("name", list(transforms.TRANSFORMS))
def test_nominal_at_zero(name):
    """theta = 0 must map to the nominal value for every transform."""
    t = transforms.get_transform(name)
    nominal = jnp.array([0.1, 0.5, 1.3])
    scale = jnp.array([0.2, 0.2, 0.2])
    physical = t.forward(jnp.zeros(3), nominal, scale)
    np.testing.assert_allclose(np.asarray(physical), np.asarray(nominal), atol=1e-7)


@pytest.mark.parametrize("name", list(transforms.TRANSFORMS))
def test_roundtrip(name):
    """inverse(forward(theta)) == theta within each transform's domain."""
    t = transforms.get_transform(name)
    nominal = jnp.array([0.05, 0.5, 2.0])
    scale = jnp.array([0.4, 0.4, 0.4])
    # Keep |theta| modest so bounded (tanh) transforms stay off saturation.
    theta = jnp.array([-1.2, 0.3, 1.1])
    recovered = t.inverse(t.forward(theta, nominal, scale), nominal, scale)
    np.testing.assert_allclose(np.asarray(recovered), np.asarray(theta), atol=1e-5)


def test_log_exp_is_strictly_positive():
    """log_exp cannot produce a non-positive value, even for large |theta|."""
    t = transforms.get_transform("log_exp")
    nominal = jnp.array([1e-3, 0.1, 1.0])  # small nominal, the risky case
    theta = jnp.linspace(-20.0, 20.0, 41)[:, None]  # sweep well past the bounds
    physical = t.forward(theta, nominal, None)
    assert np.all(np.asarray(physical) > 0.0)


def test_log_exp_is_multiplicative():
    """log_exp is scale-invariant: scaling nominal scales the output identically."""
    t = transforms.get_transform("log_exp")
    theta = jnp.array([-0.7, 0.0, 0.7])
    base = jnp.array([0.2, 0.2, 0.2])
    np.testing.assert_allclose(
        np.asarray(t.forward(theta, 10.0 * base, None)),
        np.asarray(10.0 * t.forward(theta, base, None)),
        rtol=1e-6,
    )


def test_affine_tanh_can_go_negative():
    """Documents the motivation for log_exp: the additive transform used for a
    nominal below its bound-center produces negative (unphysical) values."""
    t = transforms.get_transform("affine_tanh")
    nominal = jnp.array([0.1])       # below bound center
    scale = jnp.array([0.5])         # half-width larger than nominal
    physical = t.forward(jnp.array([-5.0]), nominal, scale)  # tanh -> ~-1
    assert float(physical[0]) < 0.0


def test_unknown_transform_raises():
    with pytest.raises(KeyError):
        transforms.get_transform("does_not_exist")


# ----------------------------------------------------------------------------
# End-to-end config wiring.
# ----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def wired():
    config = get_default_config()
    mj_model = mujoco.MjModel.from_xml_path(str(_repo_root() / SCENE))
    nominal, spec = process_regression_spec(mj_model, config.regression)
    scale = build_parameter_scale(spec)
    return nominal, spec, scale


def test_every_regressed_param_declares_a_valid_transform(wired):
    _, spec, _ = wired
    for name, s in spec.items():
        assert "transform" in s, f"{name} is missing a 'transform' key"
        assert s["transform"] in transforms.TRANSFORMS


def test_zero_maps_to_nominal_end_to_end(wired):
    """The optimizer initializes theta = 0; that must reproduce the model."""
    nominal, spec, scale = wired
    opt_params = {k: jnp.zeros_like(v) for k, v in nominal.items()}
    physical = transforms.transform_parameters(
        opt_params, nominal, regression_spec=spec, parameter_scale=scale
    )
    for k in nominal:
        np.testing.assert_allclose(
            np.asarray(physical[k]), np.asarray(nominal[k]), rtol=1e-6, atol=1e-7
        )


def test_dof_stay_positive_under_large_updates(wired):
    """With log_exp, no optimizer step can drive a dof parameter non-positive."""
    nominal, spec, scale = wired
    opt_params = {
        k: -8.0 * jnp.ones_like(v)  # a large negative step
        for k, v in nominal.items()
        if spec[k]["field"].startswith("dof_")
    }
    physical = transforms.transform_parameters(
        opt_params,
        {k: nominal[k] for k in opt_params},
        regression_spec=spec,
        parameter_scale=scale,
    )
    for k, v in physical.items():
        assert np.all(np.asarray(v) > 0.0), f"{k} went non-positive"


def test_dof_anchor_is_reference_not_model(wired):
    """dof params anchor to the config reference, not the (unusable) model value
    (damping = 0, friction/armature = 0.001 placeholder)."""
    nominal, spec, _ = wired
    np.testing.assert_allclose(np.asarray(nominal["dof_armature"]), 0.01)
    np.testing.assert_allclose(np.asarray(nominal["dof_damping"]), 0.1)
    # Every anchor is strictly positive so the log transform is well-defined.
    for name, s in spec.items():
        if s["field"].startswith("dof_"):
            assert np.all(np.asarray(nominal[name]) > 0.0)


def test_log_exp_reaches_values_far_from_reference(wired):
    """An imperfect reference must not restrict the search: from armature ref
    0.01 the transform still reaches the user's expected ~0.025, at a finite,
    O(1) theta."""
    nominal, spec, scale = wired
    t = transforms.get_transform("log_exp")
    ref = nominal["dof_armature"]
    target = 0.025 * jnp.ones_like(ref)
    theta = t.inverse(target, ref, scale.get("dof_armature"))
    assert np.all(np.isfinite(np.asarray(theta)))
    assert np.max(np.abs(np.asarray(theta))) < 3.0  # well-conditioned
    np.testing.assert_allclose(
        np.asarray(t.forward(theta, ref, None)), np.asarray(target), rtol=1e-5
    )


# ----------------------------------------------------------------------------
# Defensive validation.
# ----------------------------------------------------------------------------
def test_validate_missing_field_raises():
    with pytest.raises(ValueError, match="missing required key 'field'"):
        transforms.validate_transform_spec("p", {"transform": "log_exp", "reference": 0.1})


def test_validate_unknown_transform_raises():
    with pytest.raises(KeyError):
        transforms.validate_transform_spec("p", {"field": "x", "transform": "nope"})


def test_validate_scale_using_transform_requires_scale():
    with pytest.raises(ValueError, match="requires a 'scale'"):
        transforms.validate_transform_spec("p", {"field": "x", "transform": "affine_tanh"})


def test_validate_scale_free_transform_rejects_scale():
    with pytest.raises(ValueError, match="scale-free"):
        transforms.validate_transform_spec(
            "p", {"field": "x", "transform": "log_exp", "reference": 0.1, "scale": (0.1,)}
        )


def test_validate_returns_resolved_transform():
    t = transforms.validate_transform_spec(
        "p", {"field": "x", "transform": "log_exp", "reference": 0.1}
    )
    assert t is transforms.get_transform("log_exp")


def test_build_parameter_scale_rejects_nonpositive_scale():
    bad = {"p": {"field": "x", "transform": "affine_tanh", "scale": (0.3, -0.1, 0.2)}}
    with pytest.raises(ValueError, match="finite and strictly positive"):
        build_parameter_scale(bad)


def test_process_regression_spec_rejects_nonpositive_anchor():
    config = get_default_config()
    config.regression.dof_damping.reference = 0.0  # invalid multiplicative anchor
    mj_model = mujoco.MjModel.from_xml_path(str(_repo_root() / SCENE))
    with pytest.raises(ValueError, match="strictly-positive anchor"):
        process_regression_spec(mj_model, config.regression)


def test_config_roundtrip(wired):
    """untransform ∘ transform is identity on a modest random theta."""
    nominal, spec, scale = wired
    rng = np.random.default_rng(0)
    opt_params = {k: jnp.asarray(rng.uniform(-1.0, 1.0, v.shape)) for k, v in nominal.items()}
    physical = transforms.transform_parameters(
        opt_params, nominal, regression_spec=spec, parameter_scale=scale
    )
    recovered = transforms.untransform_parameters(
        physical, nominal, regression_spec=spec, parameter_scale=scale
    )
    for k in nominal:
        np.testing.assert_allclose(
            np.asarray(recovered[k]), np.asarray(opt_params[k]), atol=1e-4
        )
