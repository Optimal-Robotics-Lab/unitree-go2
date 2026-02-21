from absl import app, flags

import shutil
from pathlib import Path

FLAGS = flags.FLAGS

flags.DEFINE_string("directory", None, "Base directory containing the CSV folders (e.g parsed_csvs)")
flags.mark_flag_as_required("directory")


def main(argv):
    del argv

    base_path = Path(FLAGS.directory)

    if not base_path.exists() or not base_path.is_dir():
        print(f"Error: Directory {FLAGS.directory} does not exist.")
        return

    # Find all subdirectories
    directories = [d for d in base_path.iterdir() if d.is_dir()]

    for dir_path in directories:
        dir_name = dir_path.name

        # Check if it's a split directory
        if "-robot-" in dir_name or "-vicon-" in dir_name:
            # Generate the new merged name
            merged_name = dir_name.replace("-robot-", "-").replace("-vicon-", "-")
            merged_path = base_path / merged_name

            # Create the target directory if it doesn't exist yet
            merged_path.mkdir(parents=True, exist_ok=True)

            # Move all CSV files
            for csv_file in dir_path.glob("*.csv"):
                target_file = merged_path / csv_file.name
                shutil.move(str(csv_file), str(target_file))
                print(f"Moved: {csv_file.name} -> {merged_name}/")

            # Clean up the original directory if it's now empty
            if not any(dir_path.iterdir()):
                dir_path.rmdir()
                print(f"Cleaned up empty directory: {dir_name}\n")
            else:
                print(f"Warning: {dir_name} is not empty after moving CSVs.\n")


if __name__ == "__main__":
    app.run(main)
