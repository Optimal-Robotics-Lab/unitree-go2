import pathlib
import yaml

from absl import app, flags

import numpy as np

import pandas as pd
import plotly.express as px


FLAGS = flags.FLAGS
flags.DEFINE_string(
    'joint_name', 'front_right_hip', 'Joint name to visualize.', short_name='j',
)


def main(argv=None):
    # Load yaml:
    data_directory = pathlib.Path(__file__).parent.parent
    directories = [
        data_directory / f"processed/standard-replay-position",
        data_directory / f"processed/transparent-replay-position",
    ]

    stats = {}
    for directory in directories:
        prefix = directory.name.split('-')[0]
        with open(directory / "stats.yaml", 'r') as f:
            stats[prefix] = yaml.safe_load(f)

    name_to_index = {
        'front_right_hip': 0,
        'front_right_thigh': 1,
        'front_right_calf': 2,
        'front_left_hip': 3,
        'front_left_thigh': 4,
        'front_left_calf': 5,
        'hind_right_hip': 6,
        'hind_right_thigh': 7,
        'hind_right_calf': 8,
        'hind_left_hip': 9,
        'hind_left_thigh': 10,
        'hind_left_calf': 11,
    }

    name = FLAGS.joint_name
    if name not in name_to_index:
        raise ValueError(f"Invalid joint name: {name}")

    # Calculate errors for both models:
    position_errors_standard = np.array(stats['standard']['position_error'])[:, 7:]
    position_errors_transparent = np.array(stats['transparent']['position_error'])[:, 7:]
    velocity_errors_standard = np.array(stats['standard']['velocity_error'])[:, 6:]
    velocity_errors_transparent = np.array(stats['transparent']['velocity_error'])[:, 6:]
    torque_errors_standard = np.array(stats['standard']['torque_error'])
    torque_errors_transparent = np.array(stats['transparent']['torque_error'])

    position_errors_standard_abs = np.abs(position_errors_standard[:, name_to_index[name]])
    position_errors_transparent_abs = np.abs(position_errors_transparent[:, name_to_index[name]])
    velocity_errors_standard_abs = np.abs(velocity_errors_standard[:, name_to_index[name]])
    velocity_errors_transparent_abs = np.abs(velocity_errors_transparent[:, name_to_index[name]])
    torque_errors_standard_abs = np.abs(torque_errors_standard[:, name_to_index[name]])
    torque_errors_transparent_abs = np.abs(torque_errors_transparent[:, name_to_index[name]])

    data = []

    def add_data(standard_errors, transparent_errors, error_name):
        for val in standard_errors:
            data.append({'Metric': error_name, 'Model': 'Standard', 'Error': val})
        for val in transparent_errors:
            data.append({'Metric': error_name, 'Model': 'Transparent', 'Error': val})

    add_data(position_errors_standard_abs, position_errors_transparent_abs, 'Position Error (m)')
    add_data(velocity_errors_standard_abs, velocity_errors_transparent_abs, 'Velocity Error (m/s)')
    add_data(torque_errors_standard_abs, torque_errors_transparent_abs, 'Torque Error (Nm)')

    df = pd.DataFrame(data)

    fig = px.box(
        df,
        x="Model",
        y="Error",
        color="Model",
        facet_col="Metric",
        title="Prediction Accuracy Comparison",
        color_discrete_map={'Standard': 'royalblue', 'Transparent': 'orange'},
        points="outliers",
    )

    fig.update_yaxes(matches=None)
    fig.update_xaxes(matches=None)

    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_layout(showlegend=False)

    fig.show()


if __name__ == '__main__':
    app.run(main)
