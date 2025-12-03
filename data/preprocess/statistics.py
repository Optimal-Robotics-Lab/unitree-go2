import sys
import pathlib
import yaml

from absl import app, flags

import matplotlib.pyplot as plt

import numpy as np
import scipy.stats


from process_data import process_data


FLAGS = flags.FLAGS
flags.DEFINE_string(
    'directory_name', None, 'Desired checkpoint folder name to load.', short_name='d',
)


def main(argv=None):
    # Current directory
    current_directory = pathlib.Path(__file__).parent

    command_history = current_directory / f"bags/{FLAGS.directory_name}/command_history.csv"
    state_history = current_directory / f"bags/{FLAGS.directory_name}/state_history.csv"
    simulation_history = current_directory / "simulation_history.csv"

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
        simulation_history.exists(),
    ])

    if not files_exist:
        print(f"Error: '{command_history}' or '{state_history}' or '{simulation_history}' not found.", file=sys.stderr)
        print("Please put the 'command_history.csv', 'state_history.csv', and 'simulation_history.csv' in the data directory.", file=sys.stderr)
        return

    # Load data:
    num_columns = 37
    command_history = np.loadtxt(
        command_history, delimiter=',',
    ).reshape(-1, num_columns)
    state_history = np.loadtxt(
        state_history, delimiter=',',
    ).reshape(-1, num_columns)
    simulation_history = np.loadtxt(
        simulation_history, delimiter=',',
    ).reshape(-1, num_columns + 12)

    # Process data
    command_history, state_history, simulation_history = process_data(
        command_history,
        state_history,
        simulation_history,
        metadata
    )

    # Create an empirical distribution for each state:
    joint_names = [
        'front_right_abduction',
        'front_right_hip',
        'front_right_knee',
        'front_left_abduction',
        'front_left_hip',
        'front_left_knee',
        'hind_right_abduction',
        'hind_right_hip',
        'hind_right_knee',
        'hind_left_abduction',
        'hind_left_hip',
        'hind_left_knee',
    ]

    def get_empirical_distribution(data: np.ndarray) -> dict:
        states = {}
        q_offset, qd_offset, torque_offset = 1, 13, 25

        for i, joint in enumerate(joint_names):
            states[f'{joint}_q'] = data[:, i + q_offset]
            states[f'{joint}_qd'] = data[:, i + qd_offset]
            states[f'{joint}_torque'] = data[:, i + torque_offset]

        frozen_states = states.copy()
        for key, value in frozen_states.items():
            counts, bins = np.histogram(value, bins='auto', density=True)
            states[f'histogram_{key}'] = (counts, bins)

            histogram_distribution = scipy.stats.rv_histogram((counts, bins))
            states[f'histogram_distribution_{key}'] = histogram_distribution

            unique_values, counts = np.unique(value, return_counts=True)
            probabilities = counts / len(value)

            discrete_distribution = scipy.stats.rv_discrete(
                values=(unique_values, probabilities)
            )
            states[f'discrete_distribution_{key}'] = discrete_distribution
            states[f'pmf_{key}'] = discrete_distribution.pmf(value)
            states[f'unique_values_{key}'] = unique_values
            states[f'probabilities_{key}'] = probabilities

        return states

    hardware_distributions = get_empirical_distribution(state_history)
    simulation_distributions = get_empirical_distribution(simulation_history)

    # Calculate KL Divergence between hardware and simulation distributions:
    kl_divergence = {}
    for joint in joint_names:
        kl_divergence[joint] = {}
        for key in ['q', 'qd', 'torque']:
            hardware_data = hardware_distributions[f'{joint}_{key}']
            simulation_data = simulation_distributions[f'{joint}_{key}']

            combined_data = np.concatenate([hardware_data, simulation_data])
            bins = np.histogram_bin_edges(combined_data, bins='auto')
            hardware_probabilities, _ = np.histogram(
                hardware_data, bins=bins, density=True,
            )
            simulation_probabilities, _ = np.histogram(
                simulation_data, bins=bins, density=True,
            )

            bin_widths = np.diff(bins)
            pmf_hardware = hardware_probabilities * bin_widths
            pmf_simulation = simulation_probabilities * bin_widths

            epsilon = 1e-10
            entropy = scipy.stats.entropy(
                    pmf_hardware + epsilon,
                    pmf_simulation + epsilon,
                )
            kl_divergence[joint][key] = float(entropy)

    # # Calculate KL Divergence between hardware and simulation distributions:
    # kl_divergence = {}
    # for joint in joint_names:
    #     kl_divergence[joint] = {}
    #     for key in ['q', 'qd', 'torque']:
    #         epsilon = 1e-10
    #         entropy = scipy.stats.entropy(
    #             hardware_distributions[f'pmf_{joint}_{key}'],
    #             simulation_distributions[f'pmf_{joint}_{key}'],
    #         )
    #         kl_divergence[joint][key] = float(entropy)

    yaml_file = current_directory / f'plots/{FLAGS.directory_name}/kl_divergence_{FLAGS.directory_name}.yaml'
    with open(yaml_file, "w") as file:
        yaml.dump(kl_divergence, file, default_flow_style=False)

    for joint in joint_names:
        fig, axs = plt.subplots(3, 1, figsize=(10, 15))
        fig.suptitle(
            f"State Distributions Comparison: {joint.replace('_', ' ').title()}"
        )

        axs[0].set_title('Joint Position Distributions')
        axs[0].hist(
            hardware_distributions[f'histogram_{joint}_q'][1][:-1],
            hardware_distributions[f'histogram_{joint}_q'][1],
            weights=hardware_distributions[f'histogram_{joint}_q'][0],
            alpha=0.7, label='Hardware',
        )
        axs[0].hist(
            simulation_distributions[f'histogram_{joint}_q'][1][:-1],
            simulation_distributions[f'histogram_{joint}_q'][1],
            weights=simulation_distributions[f'histogram_{joint}_q'][0],
            alpha=0.7, label='Simulation'
        )
        axs[0].legend()

        axs[1].set_title('Joint Velocity Distributions')
        axs[1].hist(
            hardware_distributions[f'histogram_{joint}_qd'][1][:-1],
            hardware_distributions[f'histogram_{joint}_qd'][1],
            weights=hardware_distributions[f'histogram_{joint}_qd'][0],
            alpha=0.7, label='Hardware'
        )
        axs[1].hist(
            simulation_distributions[f'histogram_{joint}_qd'][1][:-1],
            simulation_distributions[f'histogram_{joint}_qd'][1],
            weights=simulation_distributions[f'histogram_{joint}_qd'][0],
            alpha=0.7, label='Simulation'
        )
        axs[1].legend()

        axs[2].set_title('Joint Torque Distributions')
        axs[2].hist(
            hardware_distributions[f'histogram_{joint}_torque'][1][:-1],
            hardware_distributions[f'histogram_{joint}_torque'][1],
            weights=hardware_distributions[f'histogram_{joint}_torque'][0],
            alpha=0.7, label='Hardware'
        )
        axs[2].hist(
            simulation_distributions[f'histogram_{joint}_torque'][1][:-1],
            simulation_distributions[f'histogram_{joint}_torque'][1],
            weights=simulation_distributions[f'histogram_{joint}_torque'][0],
            alpha=0.7, label='Simulation'
        )
        axs[2].legend()

        fig.tight_layout()
        fig.show()

        fig.savefig(
            current_directory / f'plots/{FLAGS.directory_name}/state_distributions_{joint}_{FLAGS.directory_name}.png'
        )


if __name__ == '__main__':
    app.run(main)
