import sys
import pathlib
import yaml

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
    vicon_history = data_directory / f"bags/{FLAGS.directory_name}/vicon_history.csv"

    metadata_file = data_directory / f"bags/{FLAGS.directory_name}/metadata.yaml"
    if not metadata_file.exists():
        print(f"Error: '{metadata_file}' not found.", file=sys.stderr)
        print("Please put the 'metadata.yaml' in the data directory.", file=sys.stderr)
        return

    # Load metadata:
    with metadata_file.open('r', encoding='utf-8') as f:
        metadata = yaml.safe_load(f)

    files_exist = all([
        command_history.exists(),
        state_history.exists(),
        imu_history.exists(),
        vicon_history.exists(),
    ])

    if not files_exist:
        print(f"Error: Files not found.", file=sys.stderr)
        print("Please put the 'command_history.csv' and 'state_history.csv' in the data directory.", file=sys.stderr)
        return

    # Shape Data:
    robot_data_columns = 37
    command_history = np.loadtxt(
        command_history, delimiter=',',
    ).reshape(-1, robot_data_columns)
    state_history = np.loadtxt(
        state_history, delimiter=',',
    ).reshape(-1, robot_data_columns)

    imu_data_columns = 11
    imu_history = np.loadtxt(
        imu_history, delimiter=',',
    ).reshape(-1, imu_data_columns)

    vicon_data_columns = 8
    vicon_history = np.loadtxt(
        vicon_history, delimiter=',',
    ).reshape(-1, vicon_data_columns)

    command_history, state_history, imu_history, vicon_history = process_data(
        command_history, state_history, imu_history, vicon_history, metadata,
    )

    # Save Processed Data:
    output_directory = data_directory / f"processed/{FLAGS.directory_name}"
    output_directory.mkdir(parents=True, exist_ok=True)

    np.savetxt(
        output_directory / "command_history.csv",
        command_history,
        delimiter=',',
    )
    np.savetxt(
        output_directory / "state_history.csv",
        state_history,
        delimiter=',',
    )
    np.savetxt(
        output_directory / "imu_history.csv",
        imu_history,
        delimiter=',',
    )
    np.savetxt(
        output_directory / "vicon_history.csv",
        vicon_history,
        delimiter=',',
    )


if __name__ == "__main__":
    app.run(main)
