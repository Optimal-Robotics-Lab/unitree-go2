# Inside postprocess.py
import sys
import pathlib
import numpy as np

def run_postprocess(directory: pathlib.Path, time_window: tuple = None, treadmill_rpm: float = None) -> bool:
    """Reads preprocessed CSVs, applies window/rpm adjustments, saves as postprocessed."""
    
    command_history = directory / "preprocessed_command_history.csv"
    policy_command_history = directory / "preprocessed_policy_command_history.csv"
    state_history = directory / "preprocessed_state_history.csv"
    imu_history = directory / "preprocessed_imu_history.csv"
    vicon_history = directory / "preprocessed_vicon_history.csv"
    filtered_vicon_history = directory / "preprocessed_filtered_history.csv"
    contact_history = directory / "preprocessed_contact_history.csv"

    files_exist = all([
        command_history.exists(), policy_command_history.exists(), state_history.exists(),
        imu_history.exists(), vicon_history.exists(), filtered_vicon_history.exists(), contact_history.exists(),
    ])

    if not files_exist:
        print(f"Error: Preprocessed files not found in {directory}.", file=sys.stderr)
        return False

    command_history = np.loadtxt(
        command_history, delimiter=',',
    )
    policy_command_history = np.loadtxt(
        policy_command_history, delimiter=',',
    )
    state_history = np.loadtxt(
        state_history, delimiter=',',
    )
    imu_history = np.loadtxt(
        imu_history, delimiter=',',
    )
    vicon_history = np.loadtxt(
        vicon_history, delimiter=',',
    )
    filtered_vicon_history = np.loadtxt(
        filtered_vicon_history, delimiter=',',
    )
    contact_history = np.loadtxt(
        contact_history, delimiter=',',
    )

    if time_window is not None:
        if len(time_window) != 2:
            print("Error: Invalid time window.", file=sys.stderr)
            return

        start_time, end_time = time_window
    else:
        start_time, end_time = state_history[:, 0][0], state_history[:, 0][-1]

    nearest_start_idx = np.argmin(np.abs(state_history[:, 0] - start_time))
    nearest_end_idx = np.argmin(np.abs(state_history[:, 0] - end_time)) + 1

    command_history = command_history[nearest_start_idx:nearest_end_idx, :]
    state_history = state_history[nearest_start_idx:nearest_end_idx, :]
    imu_history = imu_history[nearest_start_idx:nearest_end_idx, :]
    vicon_history = vicon_history[nearest_start_idx:nearest_end_idx, :]
    filtered_vicon_history = filtered_vicon_history[nearest_start_idx:nearest_end_idx, :]
    policy_command_history = policy_command_history[nearest_start_idx:nearest_end_idx, :]
    contact_history = contact_history[nearest_start_idx:nearest_end_idx, :]

    if treadmill_rpm is not None:
        treadmill_speed = 0.0108 * treadmill_rpm + -0.0056
        filtered_vicon_history[:, 4] += treadmill_speed
    
    np.savetxt(directory / "postprocessed_command_history.csv", command_history, delimiter=',')
    np.savetxt(directory / "postprocessed_state_history.csv", state_history, delimiter=',')
    np.savetxt(directory / "postprocessed_imu_history.csv", imu_history, delimiter=',')
    np.savetxt(directory / "postprocessed_vicon_history.csv", vicon_history, delimiter=',')
    np.savetxt(directory / "postprocessed_filtered_vicon_history.csv", filtered_vicon_history, delimiter=',')
    np.savetxt(directory / "postprocessed_policy_command_history.csv", policy_command_history, delimiter=',')
    np.savetxt(directory / "postprocessed_contact_history.csv", contact_history, delimiter=',')

    return True