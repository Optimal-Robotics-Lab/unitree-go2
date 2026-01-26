import numpy as np
import scipy


def process_data(
    command_history: np.ndarray,
    state_history: np.ndarray,
    simulation_history: np.ndarray,
    metadata: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    # Trim to simulation length:
    state_idx = np.where(
        state_history[:, 0] >= simulation_history[-1, 0]
    )[0][0]
    command_idx = np.where(
        command_history[:, 0] >= simulation_history[-1, 0]
    )[0][0]
    state_history = state_history[:state_idx, :]
    command_history = command_history[:command_idx, :]

    # Find when test ends:

    
    state_idx = np.where(
        state_history[:, 0] >= trim_time
    )[0][0]
    command_idx = np.where(
        command_history[:, 0] >= trim_time
    )[0][0]
    simulation_idx = np.where(
        simulation_history[:, 0] >= trim_time
    )[0][0]
    state_history = state_history[:state_idx, :]
    command_history = command_history[:command_idx, :]
    simulation_history = simulation_history[:simulation_idx, :]

    # Interpolate to fixed time steps:
    state_interpolation_function = scipy.interpolate.interp1d(
        x=state_history[:, 0],
        y=state_history[:, 1:],
        axis=0,
        kind='linear',
    )
    state_history_sampled = state_interpolation_function(
        simulation_history[:, 0]
    )
    state_history = np.hstack((
        simulation_history[:, 0:1],
        state_history_sampled
    ))

    command_interpolation_function = scipy.interpolate.interp1d(
        x=command_history[:, 0],
        y=command_history[:, 1:],
        axis=0,
        kind='linear',
    )
    command_history_sampled = command_interpolation_function(
        simulation_history[:, 0]
    )
    command_history = np.hstack((
        simulation_history[:, 0:1],
        command_history_sampled
    ))

    return command_history, state_history, simulation_history


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
