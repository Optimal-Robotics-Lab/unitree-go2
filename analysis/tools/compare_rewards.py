from absl import app, flags

from pathlib import Path
import yaml

import numpy as np

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'directory_name', None, 'Desired data folder.', short_name='d', required=True,
)


def main(argv=None):
    base_directory = Path(__file__).resolve().parent.parent
    data_directory = base_directory / "processed" / FLAGS.directory_name

    if not data_directory.exists():
        print(f"Error: Directory not found at {data_directory}")
        return

    raw_configs = {}
    extracted_rewards = {}

    for yaml_path in data_directory.rglob("*.yaml"):
        if yaml_path.name == "reward_comparison.yaml":
            continue

        with yaml_path.open('r') as file:
            try:
                data = yaml.safe_load(file)
                run_name = yaml_path.parent.name
                raw_configs[run_name] = data
                
                if isinstance(data, dict) and 'sum' in data:
                    extracted_rewards[run_name] = float(data['sum'])
                else:
                    print(f"Warning: '{run_name}' has no 'sum' key. Skipping reward calc.")
                    
            except yaml.YAMLError as exc:
                print(f"Error parsing {yaml_path.name}: {exc}")

    if not extracted_rewards:
        print("No rewards found in the specified directory. Exiting.")
        return

    reward_array = np.array(list(extracted_rewards.values()), dtype=np.float64)
    
    global_mean = np.mean(reward_array).item()
    global_std = np.std(reward_array).item()

    output_data = {
        "global_metrics": {
            "average": global_mean,
            "std_deviation": global_std,
        },
        "runs": {}
    }

    for run_name, reward in extracted_rewards.items():
        if global_std > 0.0:
            z_score = (reward - global_mean) / global_std
        else:
            z_score = 0.0

        print(f"{run_name} | Reward: {reward:.2f} | Z-Score: {z_score:.2f}")

        output_data["runs"][run_name] = {
            "reward": reward,
            "z_score": float(z_score),
            "original_config": raw_configs[run_name] 
        }

    output_path = data_directory / "reward_comparison.yaml"
    with output_path.open('w') as file:
        yaml.dump(output_data, file, default_flow_style=False)

    print(f"\nSuccessfully saved comparison to: {output_path.name}")

if __name__ == '__main__':
    app.run(main)