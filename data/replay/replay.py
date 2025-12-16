import os
import sys
import pathlib
import time

from collections import defaultdict
import yaml

from absl import app, flags

import numpy as np

import mujoco
import mujoco.viewer


FLAGS = flags.FLAGS
flags.DEFINE_string(
    'directory_name', None, 'Desired checkpoint folder name to load.', short_name='d',
)


def main(argv=None):
    # Load Data:
    prefix = FLAGS.directory_name.split('-')[0]
    if prefix not in ['standard', 'transparent']:
        print("Error: Invalid directory name prefix.", file=sys.stderr)
        return

    data_directory = pathlib.Path(__file__).parent.parent

    command_history = data_directory / f"processed/{FLAGS.directory_name}/command_history.csv"
    state_history = data_directory / f"processed/{FLAGS.directory_name}/state_history.csv"
    imu_history = data_directory / f"processed/{FLAGS.directory_name}/imu_history.csv"
    vicon_history = data_directory / f"processed/{FLAGS.directory_name}/vicon_history.csv"
    filtered_vicon_history = data_directory / f"processed/{FLAGS.directory_name}/filtered_history.csv"

    files_exist = all([
        command_history.exists(),
        state_history.exists(),
        imu_history.exists(),
        vicon_history.exists(),
        filtered_vicon_history.exists(),
    ])

    if not files_exist:
        print("Error: Files not found.", file=sys.stderr)
        return

    # Load Data:
    command_history = np.loadtxt(
        command_history, delimiter=',',
    )
    state_history = np.loadtxt(
        state_history, delimiter=',',
    )
    imu_history = np.loadtxt(
        imu_history, delimiter=',',
    )
    vicon_history = np.loadtxt(
        vicon_history, delimiter=',',
    )
    filtered_vicon_history = np.loadtxt(
        filtered_vicon_history, delimiter=',',
    )

    replay_data = zip(
        command_history,
        state_history,
        imu_history,
        vicon_history,
        filtered_vicon_history,
    )
    replay_data_list = list(replay_data)

    # Create Simulation:
    path = pathlib.Path(__file__).parent.parent.parent
    if prefix == 'standard':
        model_path = path / "training/envs/unitree_go2/mjcf/scene_mjx.xml"
    elif prefix == 'transparent':
        model_path = path / "training/envs/unitree_go2/mjcf/scene_mjx_transparent.xml"
    else:
        print("Error: Invalid directory name prefix.", file=sys.stderr)
        return

    model = mujoco.MjModel.from_xml_path(
        model_path.as_posix(),
    )
    model.opt.timestep = 0.004

    data = mujoco.MjData(model)

    # Initial Pose:
    qpos = np.array(model.keyframe('home').qpos)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)

    control_rate = 0.02
    num_steps = int(control_rate / model.opt.timestep)

    termination_flag = False

    # Get Vicon Marker to Body Offset:
    marker_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE.value, 'vicon_marker')
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY.value, 'base_link')

    base_position = data.xpos[base_id]
    marker_position = data.site_xpos[marker_id]
    vicon_offset = marker_position - base_position

    # Precalculate Hardware Outputs:
    outputs_hardware = []
    for outputs in replay_data_list[1:]:
        command, state, imu, vicon, filtered_vicon = outputs

        # Joint States:
        qpos = state[1:13]
        qvel = state[13:25]
        torque = state[25:37]

        # Body Pose and Orientation:
        vicon_position = vicon[1:4] * 1e-3  # Convert mm to m
        vicon_orientation = vicon[4:]

        # Filtered Vicon Position and Velocity:
        position = filtered_vicon[1:4]
        velocity = filtered_vicon[4:7]

        # Adjust Vicon Position by Marker Offset:
        # rotation = scipy.spatial.transform.Rotation.from_quat(vicon_orientation)
        # rotated_offset = rotation.apply(vicon_offset)
        # body_position = vicon_position - rotated_offset

        # Body Angular Velocity:
        imu_orientation = imu[1:5]
        angular_velocity = imu[5:8]
        linear_acceleration = imu[8:]

        # Set Body Position, Velocity, and Orientation:
        body_position = position - vicon_offset
        body_velocity = velocity
        body_orientation = imu_orientation

        output_dictionary = {}
        output_dictionary['qpos'] = np.concatenate([body_position, body_orientation, qpos]).tolist()
        output_dictionary['qvel'] = np.concatenate([body_velocity, angular_velocity, qvel]).tolist()
        output_dictionary['torque'] = np.array(torque).tolist()
        output_dictionary['ctrl'] = np.array(command[1:13]).tolist()
        outputs_hardware.append(output_dictionary)

    inputs = []
    outputs_simulation = []
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.trackbodyid = 1
        viewer.cam.distance = 5

        time.sleep(5.0)

        while viewer.is_running() and not termination_flag:
            for command, state, imu, vicon, filtered_vicon in replay_data_list[:-1]:
                # Joint States:
                qpos = state[1:13]
                qvel = state[13:25]
                torque = state[25:37]

                # Body Pose and Orientation:
                vicon_position = vicon[1:4] * 1e-3  # Convert mm to m
                vicon_orientation = vicon[4:]

                # Filtered Vicon Position and Velocity:
                position = filtered_vicon[1:4]
                velocity = filtered_vicon[4:7]

                # Adjust Vicon Position by Marker Offset:
                # rotation = scipy.spatial.transform.Rotation.from_quat(vicon_orientation)
                # rotated_offset = rotation.apply(vicon_offset)
                # body_position = vicon_position - rotated_offset

                # Body Angular Velocity:
                imu_orientation = imu[1:5]
                angular_velocity = imu[5:8]
                linear_acceleration = imu[8:]

                # Set Body Position, Velocity, and Orientation:
                body_position = position - vicon_offset
                body_velocity = velocity
                body_orientation = imu_orientation

                # Set Body Position States:
                data.qpos[:3] = body_position
                data.qpos[3:7] = body_orientation

                # Set Body Velocity States: (Get Linear Velocity from Vicon)
                data.qvel[:3] = body_velocity
                data.qvel[3:6] = angular_velocity

                # Set Joint States:
                data.qpos[7:19] = qpos
                data.qvel[6:18] = qvel

                # Set Control Inputs:
                data.ctrl[:] = command[1:13]

                mujoco.mj_forward(model, data)

                # Append Data to Inputs Dictionary:
                input_dictionary = {}
                input_dictionary['qpos'] = data.qpos.tolist().copy()
                input_dictionary['qvel'] = data.qvel.tolist().copy()
                input_dictionary['torque'] = data.actuator_force.tolist().copy()
                input_dictionary['ctrl'] = data.ctrl.tolist().copy()

                for _ in range(num_steps):
                    mujoco.mj_step(model, data)  # type: ignore

                # Append Data to Outputs Dictionary:
                output_dictionary = {}
                output_dictionary['qpos'] = data.qpos.tolist().copy()
                output_dictionary['qvel'] = data.qvel.tolist().copy()
                output_dictionary['torque'] = data.actuator_force.tolist().copy()
                output_dictionary['ctrl'] = data.ctrl.tolist().copy()

                # Append Dictionaries:
                inputs.append(input_dictionary)
                outputs_simulation.append(output_dictionary)

                viewer.sync()

                time.sleep(0.02)

            termination_flag = True

    # Save Data to YAML:
    save_directory = data_directory / f"processed/{FLAGS.directory_name}"
    save_directory.mkdir(parents=True, exist_ok=True)

    input_history = defaultdict(list)
    hardware_output_history = defaultdict(list)
    simulation_output_history = defaultdict(list)

    for entry in inputs:
        for key, value in entry.items():
            input_history[key].append(value)

    for entry in outputs_hardware:
        for key, value in entry.items():
            hardware_output_history[key].append(value)

    for entry in outputs_simulation:
        for key, value in entry.items():
            simulation_output_history[key].append(value)

    with open(save_directory / "inputs.yaml", 'w') as file:
        yaml.dump(dict(input_history), file)

    with open(save_directory / "outputs_hardware.yaml", 'w') as file:
        yaml.dump(dict(hardware_output_history), file)

    with open(save_directory / "outputs_simulation.yaml", 'w') as file:
        yaml.dump(dict(simulation_output_history), file)


if __name__ == '__main__':
    app.run(main)
