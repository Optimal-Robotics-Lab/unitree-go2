from absl import app, flags
import pathlib

from compute_rewards import compute_rewards

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'directory_name', None, 'Desired checkpoint folder name to load.', short_name='d', required=True,
)

def main(argv=None):
    directory_path = pathlib.Path(FLAGS.directory_name).resolve()

    if not directory_path.exists():
        raise FileNotFoundError(f"Directory not found at {directory_path}")
    
    if not directory_path.is_dir():
        raise NotADirectoryError(f"The path provided is a file, not a directory: {directory_path}")
    
    subdirectories = [p for p in directory_path.iterdir() if p.is_dir()]
    
    if not subdirectories:
        print(f"No subdirectories found. Computing rewards for main directory: {directory_path.name}")
        compute_rewards(directory_path)
        
    else:
        print(f"Found {len(subdirectories)} subdirectories. Computing rewards for each...")
        for subdir in sorted(subdirectories):
            print(f"Processing: {subdir.name}")
            compute_rewards(subdir)

if __name__ == "__main__":
    app.run(main)