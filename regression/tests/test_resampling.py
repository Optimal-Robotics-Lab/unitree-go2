"""Tests for observation resampling, rate validation, and dataset building."""

import numpy as np
import pytest

from regression.utilities import resampling
from regression.utilities.data_utilities import build_dataset
from regression.utilities.config import validate_rates


# ----------------------------------------------------------------------------
# resample_linear
# ----------------------------------------------------------------------------
def test_num_resampled_steps():
    # 500 Hz over 1000 samples (1.998 s) decimated to 250 Hz -> 500 samples.
    assert resampling.num_resampled_steps(1000, 0.002, 0.004) == 500
    # Native rate is identity.
    assert resampling.num_resampled_steps(1000, 0.002, 0.002) == 1000


def test_resample_linear_exact_decimation():
    """dst_dt an integer multiple of src_dt lands on source samples (no interp)."""
    rng = np.random.default_rng(0)
    data = rng.standard_normal((3, 11, 4))
    out = resampling.resample_linear(data, src_dt=0.002, dst_dt=0.004)
    np.testing.assert_allclose(out, data[:, ::2, :])


def test_resample_linear_interpolates_ramp():
    """On a linear ramp, upsampling hits the exact interpolated values."""
    # data[t] = sample index, broadcast over trials/dims.
    idx = np.arange(6.0)
    data = np.tile(idx[None, :, None], (2, 1, 3))
    out = resampling.resample_linear(data, src_dt=0.004, dst_dt=0.002)
    # dst positions in source-index units: 0, 0.5, 1.0, 1.5, ...
    expected_positions = (np.arange(out.shape[1]) * 0.002) / 0.004
    np.testing.assert_allclose(out[0, :, 0], expected_positions, atol=1e-9)


def test_resample_linear_native_is_identity():
    rng = np.random.default_rng(1)
    data = rng.standard_normal((2, 50, 12))
    out = resampling.resample_linear(data, src_dt=0.002, dst_dt=0.002)
    np.testing.assert_allclose(out, data)


# ----------------------------------------------------------------------------
# resample_zoh
# ----------------------------------------------------------------------------
def test_resample_zoh_holds_command():
    """A 50 Hz command upsampled to 250 Hz repeats each value 5x (ZOH)."""
    ctrl = np.array([[[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]]])  # (1, 3, 2) @ 0.02
    out = resampling.resample_zoh(ctrl, src_dt=0.02, dst_dt=0.004)
    held = out[0, :, 0]
    assert np.all(held[:5] == 10.0)
    assert np.all(held[5:10] == 20.0)
    assert held[-1] == 30.0  # last obs step lands on the final command


def test_resample_rejects_bad_shape():
    with pytest.raises(ValueError, match="trials, time, dims"):
        resampling.resample_linear(np.zeros((5, 5)), 0.002, 0.004)


# ----------------------------------------------------------------------------
# resample_from_timestamps (non-uniform / ROS2 ingestion)
# ----------------------------------------------------------------------------
def test_resample_from_timestamps_linear_exact():
    """A linear function is recovered exactly on the uniform grid, despite jitter."""
    times = np.array([0.0, 0.0021, 0.0039, 0.0061, 0.0080, 0.0102])  # jittered ~500 Hz
    data = (3.0 * times + 1.0)[:, None] * np.ones((1, 3))            # linear in t
    grid, out = resampling.resample_from_timestamps(times, data, target_dt=0.002)
    np.testing.assert_allclose(out, (3.0 * grid + 1.0)[:, None] * np.ones((1, 3)), atol=1e-9)
    # Grid is uniform and never extrapolates past the last timestamp.
    np.testing.assert_allclose(np.diff(grid), 0.002)
    assert grid[-1] <= times[-1] + 1e-9


def test_resample_from_timestamps_rejects_nonmonotonic():
    times = np.array([0.0, 0.002, 0.001, 0.003])
    data = np.zeros((4, 2))
    with pytest.raises(ValueError, match="strictly increasing"):
        resampling.resample_from_timestamps(times, data, target_dt=0.002)


# ----------------------------------------------------------------------------
# validate_rates
# ----------------------------------------------------------------------------
def test_validate_rates_valid_hierarchy():
    n_sim_per_obs, n_obs_per_ctrl = validate_rates(
        sim_dt=0.002, observation_dt=0.004, control_dt=0.02, state_dt=0.002
    )
    assert n_sim_per_obs == 2
    assert n_obs_per_ctrl == 5


def test_validate_rates_observation_finer_than_sim_raises():
    with pytest.raises(ValueError, match="integer multiple"):
        validate_rates(sim_dt=0.004, observation_dt=0.002, control_dt=0.02, state_dt=0.002)


def test_validate_rates_non_integer_multiple_raises():
    with pytest.raises(ValueError, match="integer multiple"):
        validate_rates(sim_dt=0.003, observation_dt=0.004, control_dt=0.02, state_dt=0.002)


def test_validate_rates_observe_faster_than_logged_raises():
    with pytest.raises(ValueError, match="observe faster than"):
        validate_rates(sim_dt=0.001, observation_dt=0.001, control_dt=0.02, state_dt=0.002)


def test_validate_rates_rejects_nonpositive():
    with pytest.raises(ValueError, match="must be positive"):
        validate_rates(sim_dt=0.0, observation_dt=0.004, control_dt=0.02, state_dt=0.002)


# ----------------------------------------------------------------------------
# build_dataset
# ----------------------------------------------------------------------------
def test_build_dataset_aligns_and_chunks():
    # 500 Hz states, 50 Hz control, both spanning 0.2 s; observe at 250 Hz.
    trials, nu = 2, 12
    qpos = np.random.default_rng(0).standard_normal((trials, 101, nu))
    qvel = np.random.default_rng(1).standard_normal((trials, 101, nu))
    force = np.random.default_rng(2).standard_normal((trials, 101, nu))
    ctrl = np.random.default_rng(3).standard_normal((trials, 11, nu))

    ds = build_dataset(
        qpos, qvel, force, ctrl,
        state_dt=0.002, control_dt=0.02, observation_dt=0.004, window_length=10,
    )
    # n_obs = 51 -> 5 chunks of 10 (last obs step dropped), x2 trials = 10 samples.
    assert ds.qpos.shape == (10, 10, nu)
    assert ds.ctrl.shape == (10, 10, nu)
    assert ds.qvel.shape == (10, 10, nu)
    assert ds.actuator_force.shape == (10, 10, nu)
    assert np.all(np.isfinite(np.asarray(ds.qpos)))
