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
    episodic_rewards = {}

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

                if 'episodic_reward' in data:
                    episodic_rewards[run_name] = float(data['episodic_reward'])
                else:
                    episodic_rewards[run_name] = 0.0

            except yaml.YAMLError as exc:
                print(f"Error parsing {yaml_path.name}: {exc}")

    if not extracted_rewards:
        print("No rewards found in the specified directory. Exiting.")
        return

    reward_array = np.array(list(extracted_rewards.values()), dtype=np.float64)
    global_mean = np.mean(reward_array).item()
    global_std = np.std(reward_array).item()

    ep_array = np.array(list(episodic_rewards.values()), dtype=np.float64)
    global_episodic_mean = np.mean(ep_array).item()
    global_episodic_std = np.std(ep_array).item()

    # Grouped Runs:
    system_grouped_rewards = {}
    system_grouped_episodic = {}
    for run_name in extracted_rewards.keys():
        # Split on the last hyphen to drop the run number and get the system prefix
        system_name = run_name.rsplit('-', 1)[0]

        if system_name not in system_grouped_rewards:
            system_grouped_rewards[system_name] = []
            system_grouped_episodic[system_name] = []

        system_grouped_rewards[system_name].append(extracted_rewards[run_name])
        system_grouped_episodic[system_name].append(episodic_rewards[run_name])

    # Grouped Run Metrics:
    system_metrics = {}
    for system_name in system_grouped_rewards.keys():
        sys_sum_array = np.array(system_grouped_rewards[system_name], dtype=np.float64)
        sys_sum_avg = np.mean(sys_sum_array).item()
        sys_sum_std = np.std(sys_sum_array).item()

        sys_ep_array = np.array(system_grouped_episodic[system_name], dtype=np.float64)
        sys_ep_avg = np.mean(sys_ep_array).item()
        sys_ep_std = np.std(sys_ep_array).item()

        sys_sum_z = (sys_sum_avg - global_mean) / global_std if global_std > 0.0 else 0.0
        sys_ep_z = (sys_ep_avg - global_episodic_mean) / global_episodic_std if global_episodic_std > 0.0 else 0.0

        system_metrics[system_name] = {
            "average": sys_sum_avg,
            "std_deviation": sys_sum_std,
            "z_score": float(sys_sum_z),
            "episodic_average": sys_ep_avg,
            "episodic_std_deviation": sys_ep_std,
            "episodic_z_score": float(sys_ep_z),
            "runs_counted": len(sys_sum_array)
        }

    # Output structure for YAML:
    output_data = {
        "global_metrics": {
            "average": global_mean,
            "std_deviation": global_std,
            "episodic_average": global_episodic_mean,
            "episodic_std_deviation": global_episodic_std,
        },
        "grouped_metrics": system_metrics,
        "runs": {}
    }

    # Individual run metrics:
    for run_name, reward in extracted_rewards.items():
        ep_val = episodic_rewards[run_name]
        run_sum_z = (reward - global_mean) / global_std if global_std > 0.0 else 0.0
        run_ep_z = (ep_val - global_episodic_mean) / global_episodic_std if global_episodic_std > 0.0 else 0.0

        print(f"{run_name} | Reward: {reward:.2f} | Z-Score: {run_sum_z:.2f}")

        output_data["runs"][run_name] = {
            "reward": reward,
            "z_score": float(run_sum_z),
            "episodic_reward": ep_val,
            "episodic_z_score": float(run_ep_z),
            "original_config": raw_configs[run_name]
        }

    output_path = data_directory / "reward_comparison.yaml"
    with output_path.open('w') as file:
        yaml.dump(output_data, file, default_flow_style=False)

    print(f"\nSuccessfully saved comparison to: {output_path.name}")


if __name__ == '__main__':
    app.run(main)
