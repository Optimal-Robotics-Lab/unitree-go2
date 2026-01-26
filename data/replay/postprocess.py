import sys
import pathlib

from absl import app, flags

import numpy as np


FLAGS = flags.FLAGS
flags.DEFINE_string(
    'directory_name', None, 'Desired checkpoint folder name to load.', short_name='d',
)
flags.DEFINE_multi_float(
    'time_window', None, 'Start and end time for data point window in seconds.', short_name='t',
)
flags.DEFINE_float(
    'treadmill_rpm', None, 'Treadmill speed in RPM.', short_name='r',
)


def main(argv=None):
    # Load Data:
    prefix = FLAGS.directory_name.split('-')[0]
    if prefix not in ['standard', 'transparent']:
        print("Error: Invalid directory name prefix.", file=sys.stderr)
        return

    data_directory = pathlib.Path(__file__).parent.parent

    command_history = data_directory / f"processed/{FLAGS.directory_name}/preprocessed_command_history.csv"
    policy_command_history = data_directory / f"processed/{FLAGS.directory_name}/preprocessed_policy_command_history.csv"
    state_history = data_directory / f"processed/{FLAGS.directory_name}/preprocessed_state_history.csv"
    imu_history = data_directory / f"processed/{FLAGS.directory_name}/preprocessed_imu_history.csv"
    vicon_history = data_directory / f"processed/{FLAGS.directory_name}/preprocessed_vicon_history.csv"
    filtered_vicon_history = data_directory / f"processed/{FLAGS.directory_name}/preprocessed_filtered_history.csv"
    files_exist = all([
        command_history.exists(),
        policy_command_history.exists(),
        state_history.exists(),
        imu_history.exists(),
        vicon_history.exists(),
        filtered_vicon_history.exists(),
    ])

    if not files_exist:
        print("Error: Files not found.", file=sys.stderr)
        return

    time_window = FLAGS.time_window
    if len(time_window) != 2:
        print("Error: Invalid time window.", file=sys.stderr)
        return

    start_time, end_time = time_window

    # Load Data:
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

    nearest_start_idx = np.argmin(np.abs(state_history[:, 0] - start_time))
    nearest_end_idx = np.argmin(np.abs(state_history[:, 0] - end_time)) + 1

    command_history = command_history[nearest_start_idx:nearest_end_idx, :]
    state_history = state_history[nearest_start_idx:nearest_end_idx, :]
    imu_history = imu_history[nearest_start_idx:nearest_end_idx, :]
    vicon_history = vicon_history[nearest_start_idx:nearest_end_idx, :]
    filtered_vicon_history = filtered_vicon_history[nearest_start_idx:nearest_end_idx, :]
    policy_command_history = policy_command_history[nearest_start_idx:nearest_end_idx, :]

    # Subtract treadmill speed from forward velocity:
    if FLAGS.treadmill_rpm is not None:
        treadmill_speed = 0.0108 * FLAGS.treadmill_rpm + -0.0056
        filtered_vicon_history[:, 4] += treadmill_speed

    output_directory = data_directory / f"processed/{FLAGS.directory_name}"
    np.savetxt(
        output_directory / "postprocessed_command_history.csv",
        command_history,
        delimiter=',',
    )
    np.savetxt(
        output_directory / "postprocessed_state_history.csv",
        state_history,
        delimiter=',',
    )
    np.savetxt(
        output_directory / "postprocessed_imu_history.csv",
        imu_history,
        delimiter=',',
    )
    np.savetxt(
        output_directory / "postprocessed_vicon_history.csv",
        vicon_history,
        delimiter=',',
    )
    np.savetxt(
        output_directory / "postprocessed_filtered_vicon_history.csv",
        filtered_vicon_history,
        delimiter=',',
    )
    np.savetxt(
        output_directory / "postprocessed_policy_command_history.csv",
        policy_command_history,
        delimiter=',',
    )


if __name__ == '__main__':
    app.run(main)
