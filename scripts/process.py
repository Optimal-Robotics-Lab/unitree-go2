from absl import app, flags

from pathlib import Path
import pickle

import numpy as np


FLAGS = flags.FLAGS
flags.DEFINE_string(
    'directory_name',
    None,
    'Directory containing the hardware data to process.',
    required=True,
    short_name='d',
)


def main(argv=None):
    # Load Data
    base_directory = Path(__file__).resolve().parent
    directory = base_directory / 'data' / FLAGS.directory_name

    # Verify Directory Exists:
    if not directory.exists():
        raise FileNotFoundError(
            f'Directory {directory} does not exist.',
        )

    # Load Setpoints:
    with open(directory / 'trajectories.csv', 'r') as f:
        header = f.readline().strip().replace('SHAPE:', '')
        shape = tuple(map(int, header.split(',')))
        setpoints = np.loadtxt(f, delimiter=',').reshape(shape)

    # Unpack shape:
    num_trials, num_time_steps, num_setpoints = shape
    if (num_setpoints != 12):
        raise ValueError(
            f'Expected 12 setpoints, got {num_setpoints} setpoints instead.',
        )

    # Load Command and State History and Reshape:
    command_history = np.loadtxt(
        directory / 'command_history.csv', delimiter=','
    )[:, 1:num_setpoints+1].reshape(
        num_trials,
        num_time_steps,
        num_setpoints,
    )
    state_history = np.loadtxt(
        directory / 'state_history.csv', delimiter=','
    )[:, 1:3*num_setpoints+1].reshape(
        num_trials,
        num_time_steps,
        3 * num_setpoints,
    )

    # Pickle Data:
    ctrl = command_history
    qpos = state_history[:, :, :num_setpoints]
    qvel = state_history[:, :, num_setpoints:2*num_setpoints]
    actuator_force = state_history[:, :, 2*num_setpoints:3*num_setpoints]

    data = {
        'shape': shape,
        'ctrl': ctrl,
        'qpos': qpos,
        'qvel': qvel,
        'actuator_force': actuator_force,
    }

    with open(directory / 'processed_data.pkl', 'wb') as f:
        pickle.dump(data, f, protocol=-1)


if __name__ == '__main__':
    app.run(main)
