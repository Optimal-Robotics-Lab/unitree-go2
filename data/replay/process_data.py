import numpy as np
import scipy


def process_data(
    command_history: np.ndarray,
    state_history: np.ndarray,
    imu_history: np.ndarray,
    vicon_history: np.ndarray,
    metadata: dict,
    trim_time: float = 5.0,
    sample_rate: float = 0.02,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """ Process data from CSV files """
    start_time = metadata['rosbag2_bagfile_information']['starting_time']['nanoseconds_since_epoch']

    # Adjust time to seconds:
    command_history[:, 0] = (command_history[:, 0] - start_time) * 1e-9
    state_history[:, 0] = (state_history[:, 0] - start_time) * 1e-9
    imu_history[:, 0] = (imu_history[:, 0] - start_time) * 1e-9
    vicon_history[:, 0] = (vicon_history[:, 0] - start_time) * 1e-9

    # Find first Command Signal:
    zero_command = 2.146e+09    # Only for Vendor Controller
    # zero_command = 0.0          # Only for Custom Controller

    command_idx = np.where(command_history[:, 1] != zero_command)[0][0]
    command_start_time = command_history[command_idx, 0]

    # Find the corresponding State Timestamp:
    state_idx = np.where(state_history[:, 0] >= command_start_time)[0][0]
    state_start_time = state_history[state_idx, 0]

    # Find the corresponding IMU Timestamp:
    imu_idx = np.where(imu_history[:, 0] >= command_start_time)[0][0]
    imu_start_time = imu_history[imu_idx, 0]

    # Find the corresponding Vicon Timestamp:
    vicon_idx = np.where(vicon_history[:, 0] >= command_start_time)[0][0]
    vicon_start_time = vicon_history[vicon_idx, 0]

    # Align time:
    command_history = command_history[command_idx:, :]
    state_history = state_history[state_idx:, :]
    imu_history = imu_history[imu_idx:, :]
    vicon_history = vicon_history[vicon_idx:, :]

    # Adjust time to start at zero
    command_history[:, 0] -= command_start_time
    state_history[:, 0] -= state_start_time
    imu_history[:, 0] -= imu_start_time
    vicon_history[:, 0] -= vicon_start_time

    # Interpolate at fixed time steps:
    time_points = np.arange(0, trim_time, sample_rate)

    state_interpolation_function = scipy.interpolate.interp1d(
        x=state_history[:, 0],
        y=state_history[:, 1:],
        axis=0,
        kind='linear',
    )

    command_interpolation_function = scipy.interpolate.interp1d(
        x=command_history[:, 0],
        y=command_history[:, 1:],
        axis=0,
        kind='linear',
    )

    # Cannot interpolate data due to rotations:
    imu_interpolation_function = scipy.interpolate.interp1d(
        x=imu_history[:, 0],
        y=imu_history[:, 1:],
        axis=0,
        kind='nearest',
    )

    vicon_interpolation_function = scipy.interpolate.interp1d(
        x=vicon_history[:, 0],
        y=vicon_history[:, 1:],
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

    vicon_history_sampled = vicon_interpolation_function(
        time_points
    )
    vicon_history = np.hstack((
        time_points[:, np.newaxis],
        vicon_history_sampled
    ))

    return command_history, state_history, imu_history, vicon_history


def process_hardware_data(
    command_history: np.ndarray,
    state_history: np.ndarray,
    metadata: dict,
    sample_rate: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    """ Process data from CSV files """
    start_time = metadata['rosbag2_bagfile_information']['starting_time']['nanoseconds_since_epoch']
    command_history[:, 0] = (command_history[:, 0] - start_time) * 1e-9
    state_history[:, 0] = (state_history[:, 0] - start_time) * 1e-9

    # Find first Command Signal:
    command_idx = np.where(command_history[:, 1] != 0)[0][0]
    command_start_time = command_history[command_idx, 0]

    # Find the corresponding State Signal:
    state_idx = np.where(state_history[:, 0] >= command_start_time)[0][0]
    state_start_time = state_history[state_idx, 0]

    # Align time:
    command_history = command_history[command_idx:, :]
    state_history = state_history[state_idx:, :]
    command_history[:, 0] -= command_start_time
    state_history[:, 0] -= state_start_time

    # Trim Time Length: X seconds
    trim_time = 5.0
    state_idx = np.where(
        state_history[:, 0] >= trim_time
    )[0][0]
    command_idx = np.where(
        command_history[:, 0] >= trim_time
    )[0][0]
    state_history = state_history[:state_idx, :]
    command_history = command_history[:command_idx, :]

    # Interpolate to fixed time steps:
    x = np.arange(0, trim_time, sample_rate)
    state_interpolation_function = scipy.interpolate.interp1d(
        x=state_history[:, 0],
        y=state_history[:, 1:],
        axis=0,
        kind='linear',
    )
    state_history_sampled = state_interpolation_function(
        x
    )
    state_history = np.hstack((
        x,
        state_history_sampled
    ))

    command_interpolation_function = scipy.interpolate.interp1d(
        x=command_history[:, 0],
        y=command_history[:, 1:],
        axis=0,
        kind='linear',
    )
    command_history_sampled = command_interpolation_function(
        x
    )
    command_history = np.hstack((
        x,
        command_history_sampled
    ))

    return command_history, state_history
