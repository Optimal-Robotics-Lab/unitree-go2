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
    # We will focus on the first leg (3 joints) for clarity: Hip, Thigh, Calf
    joint_names = ['Hip', 'Thigh', 'Calf']
    leg_offset = 0  # 0 for Front Right

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f'SysID Space Coverage Analysis (Front Right Leg)', fontsize=16)

    # --- ROW 1: PHASE PLOTS (Pos vs Vel) ---
    # Goal: Check if we separate stiffness (pos) from damping (vel)
    for i in range(3):
        ax = fig.add_subplot(3, 3, i + 1)
        j_idx = leg_offset + i

        # Plot a subset of points to save render time, or use hexbin for density
        ax.hexbin(flat_pos[:, j_idx], flat_vel[:, j_idx], gridsize=50, cmap='inferno', mincnt=1)

        ax.set_title(f'{joint_names[i]} Phase Plane')
        ax.set_xlabel('Position (rad)')
        ax.set_ylabel('Velocity (rad/s)')
        ax.grid(True, alpha=0.3)

    # --- ROW 2: SPATIAL COUPLING (Joint vs Joint) ---
    # Goal: Check if IK is restricting us to a manifold (bad) or exploring volume (good)

    # Plot 1: Hip vs Thigh
    ax4 = fig.add_subplot(3, 3, 4)
    ax4.plot(flat_pos[:, 0], flat_pos[:, 1], '.', markersize=1, alpha=0.1)
    ax4.set_title('Hip vs Thigh Position')
    ax4.set_xlabel('Hip (rad)')
    ax4.set_ylabel('Thigh (rad)')

    # Plot 2: Thigh vs Calf
    ax5 = fig.add_subplot(3, 3, 5)
    ax5.plot(flat_pos[:, 1], flat_pos[:, 2], '.', markersize=1, alpha=0.1)
    ax5.set_title('Thigh vs Calf Position')
    ax5.set_xlabel('Thigh (rad)')
    ax5.set_ylabel('Calf (rad)')

    # --- ROW 3: FREQUENCY ANALYSIS (Spectrogram) ---
    # Goal: Verify the Chirp actually output linear frequency growth
    # We take the mean across trials for one joint to see the signal structure
    ax7 = fig.add_subplot(3, 1, 3)

    # Analyze the Thigh joint (usually most active)
    # Concatenate first 5 trajectories to see the pattern over time
    sample_signal = data_pos[:5, :, 1].flatten()

    f, t, Sxx = signal.spectrogram(sample_signal, fs=1.0/dt, nperseg=256)
    ax7.pcolormesh(t, f, np.log10(Sxx + 1e-10), shading='gouraud', cmap='viridis')
    ax7.set_ylabel('Frequency [Hz]')
    ax7.set_xlabel('Time [s]')
    ax7.set_title('Spectrogram (Check for Rising Chirp Lines)')
    ax7.set_ylim(0, 10)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    app.run(main)
