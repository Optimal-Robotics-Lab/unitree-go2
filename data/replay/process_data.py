import numpy as np
import pandas as pd
import scipy


def preprocess_outliers(
    data: np.ndarray,
    threshold_sigma: float = 3.0
) -> np.ndarray:
    """
        Median Absolute Deviation Thresholding:
        Detects outliers using a rolling difference metric,
        replaces them with NaN, and linearly interpolates the gaps.
    """
    clean_data = data.copy()

    diffs = np.diff(clean_data, axis=0, prepend=clean_data[0:1])
    dists = np.linalg.norm(diffs, axis=1)

    median_jump = np.nanmedian(dists)
    mad_jump = np.nanmedian(np.abs(dists - median_jump))

    # Avoid division by zero if data is perfectly smooth
    if mad_jump == 0:
        return clean_data

    cutoff = median_jump + (threshold_sigma * 1.4826 * mad_jump)

    outlier_mask = dists > cutoff
    clean_data[outlier_mask] = np.nan

    df = pd.DataFrame(clean_data)
    df = df.interpolate(method='linear', axis=0, limit_direction='both')

    return df.to_numpy()


def filter_data(
    data: np.ndarray,
    time: np.ndarray,
    frequency: float = 100.0,
    window_size: int = 5,
    order: int = 2,
    derivative: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """ Smooth and Differentiate Data using Savitzky-Golay Filter """
    dt = 1.0 / frequency
    fixed_time = np.arange(time[0], time[-1], dt)

    interpolation_fn = scipy.interpolate.interp1d(
        time,
        data,
        kind='linear',
        axis=0,
        fill_value="extrapolate"
    )
    resampled_data = interpolation_fn(fixed_time)

    result = scipy.signal.savgol_filter(
        resampled_data,
        window_length=window_size,
        polyorder=order,
        deriv=derivative,
        delta=dt,
        axis=0,
    )

    return result, fixed_time


def process_data(
    command_history: np.ndarray,
    state_history: np.ndarray,
    imu_history: np.ndarray,
    policy_command_history: np.ndarray,
    vicon_history: np.ndarray,
    trim_time: float = 5.0,
    sample_frequency: float = 100.0,
) -> dict[str, np.ndarray]:
    """ Process data from CSV files """

    # Find first Command Signal:
    start_idx = np.where(np.any(policy_command_history[:, 1:] != 0, axis=1))[0][0]
    start_time = policy_command_history[start_idx, 0]

    # Find the corresponding Low Level Command Timestamp:
    command_idx = np.where(state_history[:, 0] >= start_time)[0][0]
    command_start_time = command_history[command_idx, 0]

    # Find the corresponding State Timestamp:
    state_idx = np.where(state_history[:, 0] >= start_time)[0][0]
    state_start_time = state_history[state_idx, 0]

    # Find the corresponding IMU Timestamp:
    imu_idx = np.where(imu_history[:, 0] >= start_time)[0][0]
    imu_start_time = imu_history[imu_idx, 0]

    # Find the corresponding Vicon Timestamp:
    vicon_idx = np.where(vicon_history[:, 0] >= start_time)[0][0]
    vicon_start_time = vicon_history[vicon_idx, 0]

    # Align time:
    command_history = command_history[command_idx:, :]
    state_history = state_history[state_idx:, :]
    imu_history = imu_history[imu_idx:, :]
    vicon_history = vicon_history[vicon_idx:, :]

    # Adjust time to start at zero
    policy_command_history[:, 0] -= start_time
    command_history[:, 0] -= command_start_time
    state_history[:, 0] -= state_start_time
    imu_history[:, 0] -= imu_start_time
    vicon_history[:, 0] -= vicon_start_time

    # Set Timescale to Seconds:
    command_history[:, 0] *= 1e-9
    state_history[:, 0] *= 1e-9
    imu_history[:, 0] *= 1e-9
    policy_command_history[:, 0] *= 1e-9
    vicon_history[:, 0] *= 1e-9

    # Calculate Velocity from Vicon Data:
    positions = vicon_history[:, 1:4] * 1e-3    # Convert mm to m
    positions = preprocess_outliers(positions)
    velocity, sample_time = filter_data(
        data=positions,
        time=vicon_history[:, 0],
        frequency=sample_frequency,
    )

    # Resample Data:
    dt = 1.0 / sample_frequency
    time_points = np.arange(0, trim_time, dt)

    state_interpolation_function = scipy.interpolate.interp1d(
        x=state_history[:, 0],
        y=state_history[:, 1:],
        axis=0,
        kind='nearest',
    )

    command_interpolation_function = scipy.interpolate.interp1d(
        x=command_history[:, 0],
        y=command_history[:, 1:],
        axis=0,
        kind='nearest',
    )

    imu_interpolation_function = scipy.interpolate.interp1d(
        x=imu_history[:, 0],
        y=imu_history[:, 1:],
        axis=0,
        kind='nearest',
    )

    policy_command_interpolation_function = scipy.interpolate.interp1d(
        x=policy_command_history[:, 0],
        y=policy_command_history[:, 1:],
        axis=0,
        kind='nearest',
    )

    vicon_interpolation_function = scipy.interpolate.interp1d(
        x=vicon_history[:, 0],
        y=vicon_history[:, 1:],
        axis=0,
        kind='nearest',
    )

    # Align Filtered Data to Resampled Timepoints:
    filtered_positions_function = scipy.interpolate.interp1d(
        x=vicon_history[:, 0],
        y=positions,
        axis=0,
        kind='nearest',
    )
    filtered_velocity_function = scipy.interpolate.interp1d(
        x=sample_time,
        y=velocity,
        axis=0,
        kind='nearest',
    )

    state_history_sampled = state_interpolation_function(
        time_points
    )
    state_history = np.hstack((
        time_points[:, np.newaxis],
        state_history_sampled
    ))

    command_history_sampled = command_interpolation_function(
        time_points
    )
    command_history = np.hstack((
        time_points[:, np.newaxis],
        command_history_sampled
    ))

    imu_history_sampled = imu_interpolation_function(
        time_points
    )
    imu_history = np.hstack((
        time_points[:, np.newaxis],
        imu_history_sampled
    ))

    policy_command_history_sampled = policy_command_interpolation_function(
        time_points
    )
    policy_command_history = np.hstack((
        time_points[:, np.newaxis],
        policy_command_history_sampled
    ))

    vicon_history_sampled = vicon_interpolation_function(
        time_points
    )
    vicon_history = np.hstack((
        time_points[:, np.newaxis],
        vicon_history_sampled
    ))

    positions_sampled = filtered_positions_function(
        time_points
    )
    velocity_sampled = filtered_velocity_function(
        time_points
    )
    filtered_history = np.hstack((
        time_points[:, np.newaxis],
        positions_sampled,
        velocity_sampled,
    ))

    processed_data = {
        "command_history": command_history,
        "state_history": state_history,
        "imu_history": imu_history,
        "policy_command_history": policy_command_history,
        "vicon_history": vicon_history,
        "filtered_history": filtered_history,
    }

    return processed_data
