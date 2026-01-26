import sys
import pathlib

from absl import app, flags

import numpy as np


from process_data import process_data


FLAGS = flags.FLAGS
flags.DEFINE_string(
    'directory_name', None, 'Desired checkpoint folder name to load.', short_name='d',
)


def main(argv=None):
    # Load Data:
    data_directory = pathlib.Path(__file__).parent.parent

    command_history = data_directory / f"bags/{FLAGS.directory_name}/command_history.csv"
    state_history = data_directory / f"bags/{FLAGS.directory_name}/state_history.csv"
    imu_history = data_directory / f"bags/{FLAGS.directory_name}/imu_history.csv"
    policy_command_history = data_directory / f"bags/{FLAGS.directory_name}/policy_command_history.csv"
    vicon_history = data_directory / f"bags/{FLAGS.directory_name}/vicon_history.csv"

    files_exist = all([
        command_history.exists(),
        state_history.exists(),
        imu_history.exists(),
        policy_command_history.exists(),
        vicon_history.exists(),
    ])

    if not files_exist:
        print("Error: Files not found.", file=sys.stderr)
        print("Please put the 'command_history.csv' and 'state_history.csv' in the data directory.", file=sys.stderr)
        return

    # Shape Data:
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

    data_dictionary = process_data(
        command_history,
        state_history,
        imu_history,
        policy_command_history,
        vicon_history,
        sample_frequency=50.0,
    )

    command_history = data_dictionary["command_history"]
    state_history = data_dictionary["state_history"]
    imu_history = data_dictionary["imu_history"]
    policy_command_history = data_dictionary["policy_command_history"]
    vicon_history = data_dictionary["vicon_history"]
    filtered_history = data_dictionary["filtered_history"]

    # Save Processed Data:
    output_directory_name = FLAGS.directory_name.replace('_', '-')
    output_directory = data_directory / f"processed/{output_directory_name}"
    output_directory.mkdir(parents=True, exist_ok=True)

    np.savetxt(
        output_directory / "preprocessed_command_history.csv",
        command_history,
        delimiter=',',
    )
    np.savetxt(
        output_directory / "preprocessed_state_history.csv",
        state_history,
        delimiter=',',
    )
    np.savetxt(
        output_directory / "preprocessed_imu_history.csv",
        imu_history,
        delimiter=',',
    )
    np.savetxt(
        output_directory / "preprocessed_policy_command_history.csv",
        policy_command_history,
        delimiter=',',
    )
    np.savetxt(
        output_directory / "preprocessed_vicon_history.csv",
        vicon_history,
        delimiter=',',
    )
    np.savetxt(
        output_directory / "preprocessed_filtered_history.csv",
        filtered_history,
        delimiter=',',
    )


if __name__ == "__main__":
    app.run(main)
