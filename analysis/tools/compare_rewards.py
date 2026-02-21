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

    # Grouped Runs:
    system_grouped_rewards = {}
    for run_name, reward in extracted_rewards.items():
        # Split on the last hyphen to drop the run number and get the system prefix
        system_name = run_name.rsplit('-', 1)[0]

        if system_name not in system_grouped_rewards:
            system_grouped_rewards[system_name] = []
        system_grouped_rewards[system_name].append(reward)

    # Grouped Run Metrics:
    system_metrics = {}
    for system_name, rewards in system_grouped_rewards.items():
        sys_array = np.array(rewards, dtype=np.float64)
        system_average = np.mean(sys_array).item()

        if global_std > 0.0:
            system_z_score = (system_average - global_mean) / global_std
        else:
            system_z_score = 0.0

        system_metrics[system_name] = {
            "average": system_average,
            "std_deviation": np.std(sys_array).item(),
            "runs_counted": len(rewards),
            "z_score": float(system_z_score)
        }

    # Output structure for YAML:
    output_data = {
        "global_metrics": {
            "average": global_mean,
            "std_deviation": global_std,
        },
        "grouped_metrics": system_metrics,
        "runs": {}
    }

    # Individual run metrics:
    for run_name, reward in extracted_rewards.items():
        if global_std > 0.0:
            run_z_score = (reward - global_mean) / global_std
        else:
            run_z_score = 0.0

        print(f"{run_name} | Reward: {reward:.2f} | Z-Score: {run_z_score:.2f}")

        output_data["runs"][run_name] = {
            "reward": reward,
            "z_score": float(run_z_score),
            "original_config": raw_configs[run_name]
        }

    output_path = data_directory / "reward_comparison.yaml"
    with output_path.open('w') as file:
        yaml.dump(output_data, file, default_flow_style=False)

    print(f"\nSuccessfully saved comparison to: {output_path.name}")


if __name__ == '__main__':
    app.run(main)
