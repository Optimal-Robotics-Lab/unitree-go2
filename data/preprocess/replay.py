"""
    We want to replay hardware states and inputs on the simulation to compare the distributions.
"""
import sys
import pathlib
import yaml

from absl import app, flags

import numpy as np


from process_data import process_hardware_data


FLAGS = flags.FLAGS
flags.DEFINE_string(
    'directory_name', None, 'Desired checkpoint folder name to load.', short_name='d',
)


def main(argv=None):
    # Load Replay Data:
    current_directory = pathlib.Path(__file__).parent

    command_history = current_directory / f"bags/{FLAGS.directory_name}/command_history.csv"
    state_history = current_directory / f"bags/{FLAGS.directory_name}/state_history.csv"

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
    ])

    if not files_exist:
        print(f"Error: '{command_history}' or '{state_history}' not found.", file=sys.stderr)
        print("Please put the 'command_history.csv' and 'state_history.csv' in the data directory.", file=sys.stderr)
        return

    num_columns = 37
    command_history = np.loadtxt(
        command_history, delimiter=',',
    ).reshape(-1, num_columns)
    state_history = np.loadtxt(
        state_history, delimiter=',',
    ).reshape(-1, num_columns)

    command_history, state_history = process_hardware_data(
        command_history, state_history, metadata,
    )



if __name__ == "__main__":
    app.run(main)
