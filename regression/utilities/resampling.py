"""Resampling measured trajectories onto a target observation grid.

Measured signals arrive at their own native rates -- states at the logging rate
(e.g. 500 Hz), control at the command rate (e.g. 50 Hz). Before chunking they are
resampled onto a common observation grid:

  * states  -> linear interpolation (continuous signals),
  * control -> zero-order hold (piecewise-constant between commands).

All functions operate on ``(trials, time, dims)`` arrays with a scalar sample
period ``dt`` (seconds) and return ``(trials, n_dst, dims)``. Resampling is a
one-time preprocessing step, so this is plain NumPy.
"""

import numpy as np


def num_resampled_steps(n_src: int, src_dt: float, dst_dt: float) -> int:
    """Number of destination samples covering the same span as ``n_src`` sources.

    The destination grid starts at t=0 and never extrapolates past the last
    source sample: ``(n_dst - 1) * dst_dt <= (n_src - 1) * src_dt``.
    """
    duration = (n_src - 1) * src_dt
    return int(np.floor(duration / dst_dt + 1e-9)) + 1


def resample_linear(
    data: np.ndarray, src_dt: float, dst_dt: float, n_dst: int | None = None
) -> np.ndarray:
    """Linearly interpolate a ``(trials, time, dims)`` signal to ``dst_dt``.

    An exact decimation (``dst_dt`` an integer multiple of ``src_dt``) lands on
    source samples exactly (no interpolation error). Upsampling interpolates.

    Note: linear interpolation does not anti-alias. When *decimating* a signal
    with meaningful energy above the destination Nyquist, low-pass first.
    """
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 3:
        raise ValueError(f"expected (trials, time, dims), got shape {data.shape}")
    if src_dt <= 0 or dst_dt <= 0:
        raise ValueError(f"sample periods must be positive: src_dt={src_dt}, dst_dt={dst_dt}")

    n_src = data.shape[1]
    if n_dst is None:
        n_dst = num_resampled_steps(n_src, src_dt, dst_dt)

    src_pos = (np.arange(n_dst) * dst_dt) / src_dt
    lo = np.clip(np.floor(src_pos + 1e-9).astype(int), 0, n_src - 2)
    frac = np.clip(src_pos - lo, 0.0, 1.0)[None, :, None]
    return data[:, lo, :] + (data[:, lo + 1, :] - data[:, lo, :]) * frac


def resample_zoh(
    data: np.ndarray, src_dt: float, dst_dt: float, n_dst: int | None = None
) -> np.ndarray:
    """Zero-order-hold resample of a ``(trials, time, dims)`` signal to ``dst_dt``.

    Each destination sample takes the most recent source value (held constant),
    which is the correct model for a command applied through a ZOH.
    """
    data = np.asarray(data)
    if data.ndim != 3:
        raise ValueError(f"expected (trials, time, dims), got shape {data.shape}")
    if src_dt <= 0 or dst_dt <= 0:
        raise ValueError(f"sample periods must be positive: src_dt={src_dt}, dst_dt={dst_dt}")

    n_src = data.shape[1]
    if n_dst is None:
        n_dst = num_resampled_steps(n_src, src_dt, dst_dt)

    dst_t = np.arange(n_dst) * dst_dt
    idx = np.clip(np.floor(dst_t / src_dt + 1e-9).astype(int), 0, n_src - 1)
    return data[:, idx, :]


def resample_from_timestamps(
    times: np.ndarray,
    data: np.ndarray,
    target_dt: float,
    t0: float | None = None,
    t_end: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Regularize a non-uniformly sampled signal onto a uniform grid.

    Timestamped logs (e.g. ROS2) have sample jitter, so the source grid is not
    uniform. This linearly interpolates ``data`` (sampled at the monotonic
    ``times``) onto a uniform grid at ``target_dt``, which is how raw logs should
    be regularized at ingestion before the rest of the (uniform-rate) pipeline.

    Args:
        times: ``(n,)`` strictly-increasing timestamps in seconds.
        data: ``(n, dims)`` samples aligned to ``times``.
        target_dt: uniform output period (seconds).
        t0: grid start time; defaults to ``times[0]``.
        t_end: grid end (inclusive bound); defaults to ``times[-1]``. Keep within
            ``[times[0], times[-1]]`` to avoid clamped extrapolation.

    Returns:
        ``(grid_times (n_dst,), resampled (n_dst, dims))``.
    """
    times = np.asarray(times, dtype=np.float64)
    data = np.asarray(data, dtype=np.float64)
    if times.ndim != 1:
        raise ValueError(f"times must be 1-D, got shape {times.shape}")
    if data.ndim != 2 or data.shape[0] != times.shape[0]:
        raise ValueError(
            f"data must be (n, dims) aligned to times ({times.shape[0]}), got {data.shape}"
        )
    if target_dt <= 0:
        raise ValueError(f"target_dt must be positive, got {target_dt}")
    if np.any(np.diff(times) <= 0):
        raise ValueError("times must be strictly increasing")

    if t0 is None:
        t0 = float(times[0])
    if t_end is None:
        t_end = float(times[-1])
    if t_end <= t0:
        raise ValueError(f"t_end ({t_end}) must be greater than t0 ({t0})")

    duration = t_end - t0
    n_dst = int(np.floor(duration / target_dt + 1e-9)) + 1
    grid = t0 + np.arange(n_dst) * target_dt

    resampled = np.empty((n_dst, data.shape[1]))
    for d in range(data.shape[1]):
        resampled[:, d] = np.interp(grid, times, data[:, d])
    return grid, resampled
