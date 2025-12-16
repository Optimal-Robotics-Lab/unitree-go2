import pathlib
import yaml

from absl import app, flags

import numpy as np
import plotly.graph_objects as go


FLAGS = flags.FLAGS
flags.DEFINE_string(
    'directory_name', None, 'Desired checkpoint folder name to load.', short_name='d',
)

flags.DEFINE_boolean(
    'plot', False, 'Plot the comparison results.', short_name='p',
)


def main(argv=None):
    # Load yaml:
    data_directory = pathlib.Path(__file__).parent.parent
    yaml_directory = data_directory / f"processed/{FLAGS.directory_name}"

    with open(yaml_directory / "inputs.yaml", 'r') as f:
        inputs = yaml.safe_load(f)

    with open(yaml_directory / "outputs_hardware.yaml", 'r') as f:
        outputs_hardware = yaml.safe_load(f)

    with open(yaml_directory / "outputs_simulation.yaml", 'r') as f:
        outputs_simulation = yaml.safe_load(f)

    # Compare Outputs:
    position_error = np.asarray(outputs_hardware['qpos']) - np.array(outputs_simulation['qpos'])
    position_error_rmse = np.sqrt(np.mean(np.square(position_error), axis=0))
    velocity_error = np.asarray(outputs_hardware['qvel']) - np.array(outputs_simulation['qvel'])
    velocity_error_rmse = np.sqrt(np.mean(np.square(velocity_error), axis=0))
    torque_error = np.asarray(outputs_hardware['torque']) - np.array(outputs_simulation['torque'])
    torque_error_rmse = np.sqrt(np.mean(np.square(np.asarray(outputs_hardware['torque']) - np.array(outputs_simulation['torque'])), axis=0))
    print("Position Error (RMSE):", np.mean(position_error_rmse[7:]))
    print("Velocity Error (RMSE):", np.mean(velocity_error_rmse[6:]))
    print("Torque Error (RMSE):", np.mean(torque_error_rmse))

    stats = {
        'position_error': position_error.tolist(),
        'velocity_error': velocity_error.tolist(),
        'torque_error': torque_error.tolist(),
        'position_error_rmse': position_error_rmse[7:].tolist(),
        'velocity_error_rmse': velocity_error_rmse[6:].tolist(),
        'torque_error_rmse': torque_error_rmse.tolist(),
        'position_error_rmse_mean': float(np.mean(position_error_rmse[7:])),
        'velocity_error_rmse_mean': float(np.mean(velocity_error_rmse[6:])),
        'torque_error_rmse_mean': float(np.mean(torque_error_rmse)),
    }

    with open(yaml_directory / "stats.yaml", 'w') as f:
        yaml.dump(stats, f)

    if not FLAGS.plot:
        joint_positions_hardware = np.array(outputs_hardware['qpos'])[:, 7:]
        joint_positions_simulation = np.array(outputs_simulation['qpos'])[:, 7:]
        joint_velocities_hardware = np.array(outputs_hardware['qvel'])[:, 6:]
        joint_velocities_simulation = np.array(outputs_simulation['qvel'])[:, 6:]
        joint_torques_hardware = np.array(outputs_hardware['torque'])
        joint_torques_simulation = np.array(outputs_simulation['torque'])

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

        name = 'front_right_hip'
        x = np.arange(len(outputs_hardware['qpos']))

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x,
            y=joint_positions_hardware[:, name_to_index[name]],
            name='Hardware',
            mode='lines',
            line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=x,
            y=joint_positions_simulation[:, name_to_index[name]],
            name='Simulation',
            mode='lines',
            line=dict(color='red'),
            fill='tonexty',
            fillcolor='rgba(255, 0, 0, 0.2)'
        ))

        fig.update_layout(title="Hardware vs Simulation: Position",)
        fig.show()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x,
            y=joint_velocities_hardware[:, name_to_index[name]],
            name='Hardware',
            mode='lines',
            line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=x,
            y=joint_velocities_simulation[:, name_to_index[name]],
            name='Simulation',
            mode='lines',
            line=dict(color='red'),
            fill='tonexty',
            fillcolor='rgba(255, 0, 0, 0.2)'
        ))

        fig.update_layout(title="Hardware vs Simulation: Velocity",)
        fig.show()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x,
            y=joint_torques_hardware[:, name_to_index[name]],
            name='Hardware',
            mode='lines',
            line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=x,
            y=joint_torques_simulation[:, name_to_index[name]],
            name='Simulation',
            mode='lines',
            line=dict(color='red'),
            fill='tonexty',
            fillcolor='rgba(255, 0, 0, 0.2)'
        ))

        fig.update_layout(title="Hardware vs Simulation: Torque",)
        fig.show()


if __name__ == '__main__':
    app.run(main)
