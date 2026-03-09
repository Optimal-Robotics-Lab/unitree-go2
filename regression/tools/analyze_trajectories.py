from absl import app, flags

from pathlib import Path

import numpy as np
from scipy import signal

import matplotlib.pyplot as plt


FLAGS = flags.FLAGS
flags.DEFINE_string(
    'filename',
    None,
    'CSV file to load the generated trajectories from.',
    required=True,
)


def main(argv=None):
    directory = Path(__file__).resolve().parent
    filepath = directory / 'data' / 'generated_trajectories' / f"{FLAGS.filename}.csv"

    if not filepath.exists():
        raise FileNotFoundError(f'File {filepath} not found.')

    # Load Data
    # Shape: (Num_Trajectories, Time_Steps, Num_Joints)
    with open(filepath, 'r') as f:
        header = f.readline().strip().replace('SHAPE:', '')
        shape = tuple(map(int, header.split(',')))
        data_pos = np.loadtxt(f, delimiter=',').reshape(shape)

    # Calculate Velocity:
    dt = 0.02
    data_vel = np.gradient(data_pos, axis=1) / dt

    # Flatten trajectories for scatter plotting
    # (Total_Samples, Num_Joints)
    flat_pos = data_pos.reshape(-1, shape[2])
    flat_vel = data_vel.reshape(-1, shape[2])

    # Plotting Setup
    joint_names = ['Hip', 'Thigh', 'Calf']
    front_right_leg_indices = [0, 1, 2]
    front_left_leg_indices = [3, 4, 5]
    hind_right_leg_indices = [6, 7, 8]
    hind_left_leg_indices = [9, 10, 11]

    leg_map = {
        'Front Right Leg': front_right_leg_indices,
        'Front Left Leg': front_left_leg_indices,
        'Hind Right Leg': hind_right_leg_indices,
        'Hind Left Leg': hind_left_leg_indices,
    }

    # Phase Plots:
    for i, (leg_name, leg_ids) in enumerate(leg_map.items()):

        fig = plt.figure(figsize=(16, 10))
        fig.suptitle(f'SysID Space Coverage Analysis ({leg_name})', fontsize=16)

        for j, idx in enumerate(leg_ids):
            ax = fig.add_subplot(3, 3, j + 1)

            ax.hexbin(flat_pos[:, idx], flat_vel[:, idx], gridsize=50, cmap='inferno', mincnt=1)

            ax.set_title(f'{joint_names[j]} Phase Plane')
            ax.set_xlabel('Position (rad)')
            ax.set_ylabel('Velocity (rad/s)')
            ax.grid(True, alpha=0.3)

        # Spatial Coupling Plots:
        # Plot 1: Hip vs Thigh
        ax4 = fig.add_subplot(3, 3, 4)
        ax4.plot(flat_pos[:, leg_ids[0]], flat_pos[:, leg_ids[1]], '.', markersize=1, alpha=0.1)
        ax4.set_title('Hip vs Thigh Position')
        ax4.set_xlabel('Hip (rad)')
        ax4.set_ylabel('Thigh (rad)')

        # Plot 2: Thigh vs Calf
        ax5 = fig.add_subplot(3, 3, 5)
        ax5.plot(flat_pos[:, leg_ids[1]], flat_pos[:, leg_ids[2]], '.', markersize=1, alpha=0.1)
        ax5.set_title('Thigh vs Calf Position')
        ax5.set_xlabel('Thigh (rad)')
        ax5.set_ylabel('Calf (rad)')

        # Frequency Analysis:
        ax7 = fig.add_subplot(3, 1, 3)

        # Analyze the Thigh joint:
        sample_signal = data_pos[:, :, leg_ids[1]].flatten()

        f, t, Sxx = signal.spectrogram(sample_signal, fs=1.0/dt, nperseg=256)
        ax7.pcolormesh(t, f, np.log10(Sxx + 1e-10), shading='gouraud', cmap='viridis')
        ax7.set_ylabel('Frequency [Hz]')
        ax7.set_xlabel('Time [s]')
        ax7.set_title('Spectrogram')
        ax7.set_ylim(0, 10)

        plt.tight_layout()
        plt.show()

        fig.savefig(directory / 'data' / 'generated_trajectories' / f"{FLAGS.filename}_{leg_name.replace(' ', '_')}_analysis.png")

        fig.clear()


if __name__ == "__main__":
    app.run(main)
