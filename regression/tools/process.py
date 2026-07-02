"""Process timestamped ROS2 logs into a uniform-rate training pickle.

Each trial is one bag, exported to a trial subdirectory containing two
independently-timestamped CSVs:

  * state CSV   : col 0 = timestamp (int nanoseconds), then qpos[nj], qvel[nj],
                  torque[nj]  (3*nj data columns)
  * command CSV : col 0 = timestamp (int nanoseconds), then ctrl[nj]

The topics are recorded with jitter, so each is regularized onto a uniform grid
by linear interpolation from its actual timestamps -- states at ``state_rate``,
commands at ``control_rate`` -- both from a common per-trial time origin so the
downstream command-to-state alignment is preserved. Trials are truncated to a
common length and stacked. The native rates are stored in the pickle so the
training pipeline can cross-check them against its config.

Usage:
    python -m regression.tools.process -d <dataset_dir> \
        [--state_file states.csv --command_file commands.csv \
         --state_rate 0.002 --control_rate 0.02]
"""

from absl import app, flags

from pathlib import Path
import pickle

import numpy as np

from regression.utilities import resampling


FLAGS = flags.FLAGS
flags.DEFINE_string(
    'directory_name', None,
    'Dataset directory (under regression/data) with one subdirectory per trial.',
    required=True, short_name='d',
)
flags.DEFINE_string('state_file', 'states.csv', 'State CSV filename within each trial dir.')
flags.DEFINE_string('command_file', 'commands.csv', 'Command CSV filename within each trial dir.')
flags.DEFINE_float('state_rate', 0.002, 'Uniform state grid period in seconds (500 Hz).')
flags.DEFINE_float('control_rate', 0.02, 'Uniform command grid period in seconds (50 Hz).')


def load_timestamped_csv(path: Path, has_header: bool | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Load ``(timestamps_ns int64, data float64)`` from a ROS2 CSV.

    The timestamp column is read as ``int64`` (epoch nanoseconds exceed float64's
    exact-integer range) and kept separate from the float data columns. A header
    row is auto-detected if ``has_header`` is None.
    """
    if has_header is None:
        with open(path) as handle:
            first = handle.readline().split(',')
        try:
            int(first[0])
            has_header = False
        except ValueError:
            has_header = True
    skiprows = 1 if has_header else 0

    times_ns = np.loadtxt(path, delimiter=',', usecols=(0,), skiprows=skiprows, dtype=np.int64)
    data = np.loadtxt(path, delimiter=',', skiprows=skiprows, dtype=np.float64)[:, 1:]
    return np.atleast_1d(times_ns), np.atleast_2d(data)


def process_trial(
    state_times_ns: np.ndarray,
    state_data: np.ndarray,
    command_times_ns: np.ndarray,
    command_data: np.ndarray,
    state_dt: float,
    control_dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Regularize one trial's state/command topics onto uniform grids.

    Returns ``(qpos, qvel, torque, ctrl)`` -- states at ``state_dt`` and control
    at ``control_dt``, both starting at the common overlapping time origin.
    """
    n_joints = command_data.shape[1]
    if state_data.shape[1] != 3 * n_joints:
        raise ValueError(
            f"state has {state_data.shape[1]} data columns; expected 3*{n_joints} "
            f"(qpos, qvel, torque) to match the command joint count."
        )

    # Common origin via integer-ns subtraction (preserves sub-microsecond timing).
    origin = min(int(state_times_ns[0]), int(command_times_ns[0]))
    state_t = (state_times_ns - origin).astype(np.float64) * 1e-9
    command_t = (command_times_ns - origin).astype(np.float64) * 1e-9

    # Overlapping window covered by both topics.
    t0 = max(state_t[0], command_t[0])
    t_end = min(state_t[-1], command_t[-1])
    if t_end <= t0:
        raise ValueError("state and command logs have no overlapping time window.")

    _, states = resampling.resample_from_timestamps(state_t, state_data, state_dt, t0=t0, t_end=t_end)
    _, ctrl = resampling.resample_from_timestamps(command_t, command_data, control_dt, t0=t0, t_end=t_end)

    qpos = states[:, :n_joints]
    qvel = states[:, n_joints:2 * n_joints]
    torque = states[:, 2 * n_joints:3 * n_joints]
    return qpos, qvel, torque, ctrl


def main(argv=None):
    package_directory = Path(__file__).resolve().parent.parent
    directory = package_directory / 'data' / FLAGS.directory_name
    if not directory.exists():
        raise FileNotFoundError(f'Directory {directory} does not exist.')

    trial_dirs = sorted(p for p in directory.iterdir() if p.is_dir())
    if not trial_dirs:
        raise FileNotFoundError(f'No trial subdirectories found in {directory}.')

    qpos_list, qvel_list, force_list, ctrl_list = [], [], [], []
    for trial_dir in trial_dirs:
        state_ts, state_data = load_timestamped_csv(trial_dir / FLAGS.state_file)
        command_ts, command_data = load_timestamped_csv(trial_dir / FLAGS.command_file)
        q, v, f, u = process_trial(
            state_ts, state_data, command_ts, command_data,
            FLAGS.state_rate, FLAGS.control_rate,
        )
        qpos_list.append(q)
        qvel_list.append(v)
        force_list.append(f)
        ctrl_list.append(u)

    # Truncate to common lengths so trials stack into rectangular arrays.
    n_state = min(a.shape[0] for a in qpos_list)
    n_ctrl = min(a.shape[0] for a in ctrl_list)
    data = {
        'qpos': np.stack([a[:n_state] for a in qpos_list]),
        'qvel': np.stack([a[:n_state] for a in qvel_list]),
        'actuator_force': np.stack([a[:n_state] for a in force_list]),
        'ctrl': np.stack([a[:n_ctrl] for a in ctrl_list]),
        'state_rate': FLAGS.state_rate,
        'control_rate': FLAGS.control_rate,
    }

    out_path = directory / 'processed_data.pkl'
    with open(out_path, 'wb') as handle:
        pickle.dump(data, handle, protocol=-1)
    print(
        f"Wrote {out_path}: {len(trial_dirs)} trials | "
        f"states {data['qpos'].shape} @ {FLAGS.state_rate}s | "
        f"ctrl {data['ctrl'].shape} @ {FLAGS.control_rate}s"
    )


if __name__ == '__main__':
    app.run(main)
