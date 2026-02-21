import sys
import pathlib
import time

import numpy as np
import mujoco
import mujoco.viewer


def visualize_run(directory_name: str, data_directory: pathlib.Path = None, model_path: pathlib.Path = None):
    """
    Loads and visualizes a single MuJoCo run from CSV files.
    """
    if data_directory is None:
        data_directory = pathlib.Path(__file__).parent.parent

    if model_path is None:
        base_path = pathlib.Path(__file__).parent.parent.parent
        model_path = base_path / "training/envs/unitree_go2/mjcf/scene_mjx_standard_position.xml"

    command_file = data_directory / f"{directory_name}/postprocessed_command_history.csv"
    state_file = data_directory / f"{directory_name}/postprocessed_state_history.csv"
    imu_file = data_directory / f"{directory_name}/postprocessed_imu_history.csv"
    vicon_file = data_directory / f"{directory_name}/postprocessed_vicon_history.csv"

    files_exist = all([
        command_file.exists(),
        state_file.exists(),
        imu_file.exists(),
        vicon_file.exists(),
    ])

    if not files_exist:
        print(f"Error: Missing CSV files for run '{directory_name}'. Skipping.", file=sys.stderr)
        return False

    print(f"Visualizing run: {directory_name}")

    command_history = np.loadtxt(command_file, delimiter=',')
    state_history = np.loadtxt(state_file, delimiter=',')
    imu_history = np.loadtxt(imu_file, delimiter=',')
    vicon_history = np.loadtxt(vicon_file, delimiter=',')

    replay_data = zip(command_history, state_history, imu_history, vicon_history)

    model = mujoco.MjModel.from_xml_path(model_path.as_posix())
    model.opt.timestep = 0.004

    data = mujoco.MjData(model)

    # Initial Pose
    qpos = np.array(model.keyframe('home').qpos)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)

    termination_flag = False

    # Get Vicon Marker to Body Offset
    marker_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE.value, 'vicon_marker')
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY.value, 'base_link')

    base_position = data.xpos[base_id]
    marker_position = data.site_xpos[marker_id]
    vicon_offset = marker_position - base_position

    # 5. Launch Viewer Loop
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.trackbodyid = 1
        viewer.cam.distance = 5

        time.sleep(0.5)

        while viewer.is_running() and not termination_flag:
            for command, state, imu, vicon in replay_data:
                # Joint States
                qpos = state[1:13]
                qvel = state[13:25]

                # Body Pose and Orientation
                vicon_position = vicon[1:4] * 1e-3  # Convert mm to m

                body_position = vicon_position - vicon_offset

                # Body Angular Velocity
                imu_orientation = imu[1:5]
                angular_velocity = imu[5:8]

                body_orientation = imu_orientation

                # Set Body Position States
                data.qpos[:3] = body_position
                data.qpos[3:7] = body_orientation

                # Set Body Velocity States (Get Linear Velocity from Vicon)
                data.qvel[:3] = np.array([0.0, 0.0, 0.0])
                data.qvel[3:6] = angular_velocity

                # Set Joint States
                data.qpos[7:19] = qpos
                data.qvel[6:18] = qvel

                mujoco.mj_forward(model, data)

                viewer.sync()
                time.sleep(0.02)

            # End of CSV data
            termination_flag = True

    return True
