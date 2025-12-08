import os
import sys
import pathlib
import time

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
    data_directory = pathlib.Path(__file__).parent.parent

    command_history = data_directory / f"processed/{FLAGS.directory_name}/command_history.csv"
    state_history = data_directory / f"processed/{FLAGS.directory_name}/state_history.csv"
    imu_history = data_directory / f"processed/{FLAGS.directory_name}/imu_history.csv"
    vicon_history = data_directory / f"processed/{FLAGS.directory_name}/vicon_history.csv"

    files_exist = all([
        command_history.exists(),
        state_history.exists(),
        imu_history.exists(),
        vicon_history.exists(),
    ])

    if not files_exist:
        print(f"Error: Files not found.", file=sys.stderr)
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

    replay_data = zip(
        command_history,
        state_history,
        imu_history,
        vicon_history,
    )

    # Create Simulation:
    path = pathlib.Path(__file__).parent.parent.parent
    model_path = path / "training/envs/unitree_go2/mjcf/scene_mjx.xml"
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

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.trackbodyid = 1
        viewer.cam.distance = 5

        time.sleep(10.0)

        while viewer.is_running() and not termination_flag:
            for command, state, imu, vicon in replay_data:
                # Joint States:
                qpos = state[1:13]
                qvel = state[13:25]
                torque = state[25:]

                # Body Pose and Orientation:
                vicon_position = vicon[1:4] * 1e-3  # Convert mm to m
                vicon_orientation = vicon[4:]

                # Adjust Vicon Position by Marker Offset:
                # rotation = scipy.spatial.transform.Rotation.from_quat(vicon_orientation)
                # rotated_offset = rotation.apply(vicon_offset)
                # body_position = vicon_position - rotated_offset

                body_position = vicon_position - vicon_offset

                # Body Angular Velocity:
                imu_orientation = imu[1:5]
                angular_velocity = imu[5:8]
                linear_acceleration = imu[8:]

                # body_orientation = imu_orientation
                body_orientation = vicon_orientation

                # Set Body Position States:
                data.qpos[:3] = body_position
                data.qpos[3:7] = body_orientation

                # Set Body Velocity States: (Get Linear Velocity from Vicon)
                data.qvel[:3] = np.array([0.0, 0.0, 0.0])
                data.qvel[3:6] = angular_velocity

                # Set Joint States:
                data.qpos[7:19] = qpos
                data.qvel[6:18] = qvel

                mujoco.mj_forward(model, data)

                # for _ in range(num_steps):
                #     mujoco.mj_step(model, data)  # type: ignore

                viewer.sync()

                time.sleep(0.02)

            termination_flag = True


if __name__ == '__main__':
    app.run(main)
