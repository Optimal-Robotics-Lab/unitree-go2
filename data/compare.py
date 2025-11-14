import sys
import pathlib
import yaml

import absl.app as app
import absl.flags as flags

import numpy as np

import matplotlib
import matplotlib.pyplot as plt


FLAGS = flags.FLAGS
flags.DEFINE_string(
    'directory_name', None, 'Desired checkpoint folder name to load.', short_name='d',
)

def main(argv=None):
    # Current directory
    current_directory = pathlib.Path(__file__).parent

    command_history = current_directory / f"bags/{FLAGS.directory_name}/command_history.csv"
    state_history = current_directory / f"bags/{FLAGS.directory_name}/state_history.csv"
    simulation_history = current_directory / "simulation_history.csv"

    metadata_file = current_directory / f"bags/{FLAGS.directory_name}/metadata.yaml"
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
        simulation_history.exists(),
    ])

    if not files_exist:
        print(f"Error: '{command_history}' or '{state_history}' or '{simulation_history}' not found.", file=sys.stderr)
        print("Please put the 'command_history.csv', 'state_history.csv', and 'simulation_history.csv' in the data directory.", file=sys.stderr)
        return

    # Load data:
    num_columns = 37
    command_history = np.loadtxt(
        command_history, delimiter=',',
    ).reshape(-1, num_columns)
    state_history = np.loadtxt(
        state_history, delimiter=',',
    ).reshape(-1, num_columns)
    simulation_history = np.loadtxt(
        simulation_history, delimiter=',',
    ).reshape(-1, num_columns + 12)

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

    # Joint ID Map:
    joints = {
        'front_right_abduction': 0,
        'front_right_hip': 1,
        'front_right_knee': 2,
        'front_left_abduction': 3,
        'front_left_hip': 4,
        'front_left_knee': 5,
        'rear_right_abduction': 6,
        'rear_right_hip': 7,
        'rear_right_knee': 8,
        'rear_left_abduction': 9,
        'rear_left_hip': 10,
        'rear_left_knee': 11,
    }

    joints_of_interest = ['front_right_knee']

    # Plot data:
    fig, axs = plt.subplots(3, 1, figsize=(10, 8))
    plt.suptitle("Unitree State vs Command: Front Right Leg")

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    # Position Plot:
    joints_to_plot = [joints[joint_name] for joint_name in joints_of_interest]
    for i in joints_to_plot:
        axs[0].plot(
            state_history[:, 0],
            state_history[:, 1+i],
            label="State",
            color=colors[i],
        )
        axs[0].plot(
            command_history[:, 0],
            command_history[:, 1+i],
            '--',
            label="Command",
            color=colors[i],
        )
        axs[0].plot(
            simulation_history[:, 0],
            simulation_history[:, 1+i],
            ':',
            label="Simulation",
            color=colors[i],
        )

    axs[0].set_ylabel("Position (rad)")
    axs[0].legend()

    # Velocity Plot:
    for i in joints_to_plot:
        axs[1].plot(
            state_history[:, 0],
            state_history[:, 13+i],
            label="State",
            color=colors[i],
        )
        axs[1].plot(
            command_history[:, 0],
            command_history[:, 13+i],
            '--',
            label="Command",
            color=colors[i],
        )
        axs[1].plot(
            simulation_history[:, 0],
            simulation_history[:, 13+i],
            ':',
            label="Simulation",
            color=colors[i],
        )

    axs[1].set_ylabel("Velocity (rad/s)")
    axs[1].legend()

    # Torque Plot:
    for i in joints_to_plot:
        axs[2].plot(
            state_history[:, 0],
            state_history[:, 25+i],
            label="State",
            color=colors[i],
        )
        axs[2].plot(
            command_history[:, 0],
            command_history[:, 25+i],
            '--',
            label="Command",
            color=colors[i],
        )
        axs[2].plot(
            simulation_history[:, 0],
            simulation_history[:, 25+i],
            ':',
            label="Simulation",
            color=colors[i],
        )

    axs[2].set_ylabel("Torque (Nm)")
    axs[2].legend()
    axs[2].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()

    output_filename = current_directory / f"plots/{FLAGS.directory_name}_comparison_plot.png"
    plt.savefig(output_filename)
    print(f"Plot saved successfully to {output_filename}")


if __name__ == "__main__":
    app.run(main)
