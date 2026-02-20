# Inside preprocess.py
import sys
import pathlib
import numpy as np

from process_data import process_data

def run_preprocess(input_directory: pathlib.Path, output_directory: pathlib.Path) -> bool:
    """Reads raw CSVs from input_directory, processes them, and saves to output_directory."""
    
    command_history = input_directory / "command_history.csv"
    state_history = input_directory / "state_history.csv"
    imu_history = input_directory / "imu_history.csv"
    policy_command_history = input_directory / "policy_command_history.csv"
    vicon_history = input_directory / "vicon_history.csv"
    contact_history = input_directory / "contact_history.csv"

    files_exist = all([
        command_history.exists(), state_history.exists(), imu_history.exists(),
        policy_command_history.exists(), vicon_history.exists(), contact_history.exists(),
    ])

    if not files_exist:
        print(f"Error: Raw files not found in {input_directory}.", file=sys.stderr)
        return False

    # Format Data:
    command_data_columns = 37
    state_data_columns = 49
    command_history = np.loadtxt(
        command_history, delimiter=',',
    ).reshape(-1, command_data_columns)
    state_history = np.loadtxt(
        state_history, delimiter=',',
    ).reshape(-1, state_data_columns)

    imu_data_columns = 11
    imu_history = np.loadtxt(
        imu_history, delimiter=',',
    ).reshape(-1, imu_data_columns)

    policy_command_data_columns = 4
    policy_command_history = np.loadtxt(
        policy_command_history, delimiter=',',
    ).reshape(-1, policy_command_data_columns)

    vicon_data_columns = 8
    vicon_history = np.loadtxt(
        vicon_history, delimiter=',',
    ).reshape(-1, vicon_data_columns)

    contact_history_data_columns = 9
    contact_history = np.loadtxt(
        contact_history, delimiter=',',
    ).reshape(-1, contact_history_data_columns)

    data_dictionary = process_data(
        command_history,
        state_history,
        imu_history,
        policy_command_history,
        vicon_history,
        contact_history,
        sample_frequency=50.0,
    )

    output_directory.mkdir(parents=True, exist_ok=True)

    np.savetxt(output_directory / "preprocessed_command_history.csv", data_dictionary["command_history"], delimiter=',')
    np.savetxt(output_directory / "preprocessed_state_history.csv", data_dictionary["state_history"], delimiter=',')
    np.savetxt(output_directory / "preprocessed_imu_history.csv", data_dictionary["imu_history"], delimiter=',')
    np.savetxt(output_directory / "preprocessed_policy_command_history.csv", data_dictionary["policy_command_history"], delimiter=',')
    np.savetxt(output_directory / "preprocessed_vicon_history.csv", data_dictionary["vicon_history"], delimiter=',')
    np.savetxt(output_directory / "preprocessed_filtered_history.csv", data_dictionary["filtered_history"], delimiter=',')
    np.savetxt(output_directory / "preprocessed_contact_history.csv", data_dictionary["contact_history"], delimiter=',')

    return True
