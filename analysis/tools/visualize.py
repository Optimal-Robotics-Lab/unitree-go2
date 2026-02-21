from absl import app, flags

import pathlib
from visualizer import visualize_run

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'data_directory', None, 'Path to the data directory.', short_name='d', required=True,
)


def main(argv=None):
    data_dir = pathlib.Path(FLAGS.data_directory)

    # Find all subdirectories in the processed folder:
    run_folders = [d.name for d in data_dir.iterdir() if d.is_dir()]

    for folder_name in sorted(run_folders):
        success = visualize_run(
            directory_name=folder_name,
            data_directory=data_dir
        )
        if not success:
            print(f"Failed to visualize {folder_name}.")


if __name__ == "__main__":
    app.run(main)
