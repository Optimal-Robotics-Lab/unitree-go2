"""Tests for ROS2 timestamped-log ingestion (regression/tools/process.py)."""

import numpy as np
import pytest

from regression.tools.process import process_trial, load_timestamped_csv


# Epoch-scale nanoseconds (~1.7e18) exceed float64's exact-integer range (2**53),
# so these tests would fail if timestamps were handled as float rather than int64.
_EPOCH_NS = 1_700_000_000_000_000_000


def _jittered_ns(origin, n, nominal_ns, rng):
    """Strictly-increasing int64 timestamps around a nominal period, with jitter."""
    intervals = nominal_ns + rng.integers(-nominal_ns // 4, nominal_ns // 4, n - 1)
    return origin + np.concatenate([[0], np.cumsum(intervals)]).astype(np.int64)


def test_process_trial_regularizes_splits_and_preserves_timing():
    rng = np.random.default_rng(0)
    nj = 3
    state_dt, control_dt = 0.002, 0.02

    # States @ ~500 Hz, commands @ ~50 Hz, both jittered, over ~0.2 s.
    state_ns = _jittered_ns(_EPOCH_NS, 101, 2_000_000, rng)
    cmd_ns = _jittered_ns(_EPOCH_NS, 11, 20_000_000, rng)

    # Linear-in-true-time content so interpolation is exactly recoverable.
    state_t = (state_ns - _EPOCH_NS).astype(np.float64) * 1e-9
    cmd_t = (cmd_ns - _EPOCH_NS).astype(np.float64) * 1e-9
    slopes = np.arange(1, 3 * nj + 1)                       # distinct per column
    state_data = state_t[:, None] * slopes[None, :]         # (n, 3*nj)
    cmd_data = cmd_t[:, None] * np.arange(1, nj + 1)[None, :]

    qpos, qvel, torque, ctrl = process_trial(
        state_ns, state_data, cmd_ns, cmd_data, state_dt, control_dt
    )

    # Split is correct (qpos|qvel|torque), each nj wide.
    assert qpos.shape[1] == nj and qvel.shape[1] == nj and torque.shape[1] == nj
    assert ctrl.shape[1] == nj

    # Each column is linear on the uniform grid: constant slope = a_j * dt.
    # (float64 mishandling of the 1.7e18 origin would corrupt this slope.)
    for j in range(nj):
        np.testing.assert_allclose(np.diff(qpos[:, j]), slopes[j] * state_dt, atol=1e-9)
        np.testing.assert_allclose(np.diff(qvel[:, j]), slopes[nj + j] * state_dt, atol=1e-9)
        np.testing.assert_allclose(np.diff(torque[:, j]), slopes[2 * nj + j] * state_dt, atol=1e-9)
    np.testing.assert_allclose(np.diff(ctrl[:, 0]), 1.0 * control_dt, atol=1e-9)


def test_process_trial_rejects_mismatched_joint_counts():
    state = np.zeros((10, 7))     # 7 is not 3 * nj for the 2-joint command below
    cmd = np.zeros((5, 2))
    with pytest.raises(ValueError, match="expected 3"):
        process_trial(
            np.arange(10, dtype=np.int64), state,
            np.arange(5, dtype=np.int64) * 4, cmd,
            0.002, 0.02,
        )


def test_load_timestamped_csv_preserves_int64_and_header(tmp_path):
    ns = np.array([_EPOCH_NS, _EPOCH_NS + 2_000_000, _EPOCH_NS + 4_000_001], dtype=np.int64)
    data = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])

    for header in ("", "stamp,a,b\n"):
        path = tmp_path / f"log_{bool(header)}.csv"
        with open(path, "w") as fh:
            fh.write(header)
            for t, row in zip(ns, data):
                fh.write(f"{t},{row[0]},{row[1]}\n")

        times, values = load_timestamped_csv(path)
        assert times.dtype == np.int64
        np.testing.assert_array_equal(times, ns)   # exact, incl. the +1 ns
        np.testing.assert_allclose(values, data)
