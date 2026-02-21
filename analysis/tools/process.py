from absl import app, flags
import pathlib

from preprocess import run_preprocess
from postprocess import run_postprocess
from compute_rewards import compute_rewards

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'directory_name', None, 'Target bag folder name to process.', short_name='d', required=True,
)
flags.DEFINE_multi_float(
    'time_window', None, 'Start and end time for data point window in seconds.', short_name='t',
)
flags.DEFINE_float(
    'treadmill_rpm', None, 'Treadmill speed in RPM.', short_name='r',
)


def process_single_run(raw_input_directory: pathlib.Path, processed_output_directory: pathlib.Path):
    """Executes the full pipeline for a single directory."""

    print(f"\n--- Processing: {raw_input_directory.name} ---")

    print("Running Preprocess...")
    success = run_preprocess(raw_input_directory, processed_output_directory)
    if not success:
        print(f"Skipping {raw_input_directory.name} due to preprocessing failure.")
        return

    print("Running Postprocess...")
    success = run_postprocess(processed_output_directory, FLAGS.time_window, FLAGS.treadmill_rpm)
    if not success:
        print(f"Skipping {raw_input_directory.name} due to postprocessing failure.")
        return

    print("Computing Rewards...")
    compute_rewards(str(processed_output_directory))
    print(f"Finished: {raw_input_directory.name}")


def main(argv=None):
    base_directory = pathlib.Path(__file__).resolve().parent.parent

    target_bag_directory = base_directory / "bags" / FLAGS.directory_name

    if not target_bag_directory.exists() or not target_bag_directory.is_dir():
        raise FileNotFoundError(f"Raw data directory not found at {target_bag_directory}")

    subdirectories = [p for p in target_bag_directory.iterdir() if p.is_dir()]

    if not subdirectories:
        print(f"No subdirectories found. Running pipeline on main directory: {target_bag_directory.name}")

        output_name = target_bag_directory.name.replace('_', '-')
        processed_directory = base_directory / "processed" / output_name

        process_single_run(target_bag_directory, processed_directory)
    else:
        print(f"Found {len(subdirectories)} subdirectories. Running pipeline for each...")
        for subdir in sorted(subdirectories):
            output_name = subdir.name.replace('_', '-')
            processed_directory = base_directory / "processed" / FLAGS.directory_name.replace('_', '-') / output_name
            process_single_run(subdir, processed_directory)


if __name__ == "__main__":
    app.run(main)
