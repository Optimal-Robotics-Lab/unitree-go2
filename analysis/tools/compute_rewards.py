import sys
import pathlib
import yaml
import copy

import mujoco

import numpy as np
import scipy


reward_config = {
    # Rewards:
    'tracking_linear_velocity': 1.5,
    'tracking_angular_velocity': 0.75,
    # Orientation Regularization Terms:
    'orientation_regularization': -5.0,
    'linear_z_velocity': -2.0,
    'angular_xy_velocity': -0.05,
    # Energy Regularization Terms:
    'torque': -2e-4,
    'action_rate': -0.1,
    # Auxilary Terms:
    'stand_still': -1.0,
    # Gait Terms:
    'foot_slip': -0.5,
    'air_time': 0.75,
    'foot_clearance': 0.5,
    'gait_variance': -1.0,
}


def compute_rewards(directory_name: str) -> None:
    """ Load Data """
    filepath = pathlib.Path(__file__).resolve()
    base_directory = filepath.parent.parent

    directory_path = pathlib.Path(directory_name)
    # Robot Command, State, Contact, IMU Data:
    command_history = directory_path / "postprocessed_command_history.csv"
    state_history = directory_path / "postprocessed_state_history.csv"
    imu_history = directory_path / "postprocessed_imu_history.csv"

    # Velocity Command:
    policy_command_history = directory_path / "postprocessed_policy_command_history.csv"

    # Vicon and Filtered Vicon Data:
    vicon_history = directory_path / "postprocessed_vicon_history.csv"
    filtered_history = directory_path / "postprocessed_filtered_vicon_history.csv"

    # Contact Data:
    contact_history = directory_path / "postprocessed_contact_history.csv"

    files_exist = all([
        command_history.exists(),
        state_history.exists(),
        imu_history.exists(),
        vicon_history.exists(),
        filtered_history.exists(),
        policy_command_history.exists(),
        contact_history.exists(),
    ])

    if not files_exist:
        print("Error: Files not found.", file=sys.stderr)
        return

    # Load  Robot Data:
    command_history = np.loadtxt(
        command_history, delimiter=',',
    )
    state_history = np.loadtxt(
        state_history, delimiter=',',
    )
    imu_history = np.loadtxt(
        imu_history, delimiter=',',
    )

    # Load Policy Command Data:
    policy_command_history = np.loadtxt(
        policy_command_history, delimiter=',',
    )

    # Load Vicon and Filtered Data:
    vicon_history = np.loadtxt(
        vicon_history, delimiter=',',
    )
    filtered_history = np.loadtxt(
        filtered_history, delimiter=',',
    )

    # Load Contact Data:
    contact_history = np.loadtxt(
        contact_history, delimiter=',',
    )

    # Assert all time columns are the same
    time_stamps = [
        command_history[:, 0],
        state_history[:, 0],
        contact_history[:, 0],
        imu_history[:, 0],
        policy_command_history[:, 0],
        vicon_history[:, 0],
        filtered_history[:, 0],
    ]

    reference = time_stamps[0]
    names = ["command", "state", "contact", "imu", "policy", "vicon", "filtered"]

    for i, current in enumerate(time_stamps[1:], start=1):
        # Check shape first
        assert reference.shape == current.shape, \
            f"Shape mismatch: '{names[0]}' is {reference.shape}, but '{names[i]}' is {current.shape}"

        # Check values with tolerance (atol=1e-6 is usually safe for seconds)
        assert np.allclose(reference, current, atol=1e-6), \
            f"Time drift detected in '{names[i]}' compared to reference."

    # Environemt Constants:
    action_rate = 0.5
    default_pose = np.array([0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8])
    dt = 0.02

    # Unpack Command Data:
    time_stamps = command_history[:, 0]
    command_positions = command_history[:, 1:13]
    command_velocities = command_history[:, 13:25]
    command_torques = command_history[:, 25:]

    # Unpack State Data:
    state_positions = state_history[:, 1:13]
    state_velocities = state_history[:, 13:25]
    state_torques = state_history[:, 25:]

    # Unpack IMU Data:
    imu_orientation = imu_history[:, 1:5]
    imu_angular_velocity = imu_history[:, 5:8]
    imu_linear_acceleration = imu_history[:, 8:]

    # Unpack Vicon Data:
    vicon_positions = vicon_history[:, 1:4] * 1e-3  # Convert mm to m
    vicon_orientation = vicon_history[:, 4:]

    # Unpack Filtered Data:
    global_position = filtered_history[:, 1:4]
    global_velocity = filtered_history[:, 4:7]

    # Unpack Policy Command Data:
    command = policy_command_history[:, 1:4]

    # Unpack Contact Data:
    contact_forces = contact_history[:, 1:]

    # Calculate Local Frame Data:
    body_orientation = imu_orientation
    rotations = scipy.spatial.transform.Rotation.from_quat(body_orientation, scalar_first=True)
    local_linear_velocity = rotations.inv().apply(global_velocity)
    local_angular_velocity = imu_angular_velocity
    global_anglular_velocity = rotations.apply(local_angular_velocity)
    up_vector = rotations.apply(np.array([0.0, 0.0, 1.0]))

    # Back Calculate the Actions from the Policy:
    actions = (command_positions - default_pose[np.newaxis, :]) / action_rate
    previous_actions = np.vstack((
        np.zeros((1, actions.shape[1])),
        actions[:-1, :],
    ))

    # Setup MuJoCo Model and Data for Reward Calculation:
    mj_model_path = base_directory / "replay/mjcf/scene_mjx_vendor_position.xml"
    mj_model = mujoco.MjModel.from_xml_path(
        str(mj_model_path)
    )
    feet_site = [
        'front_right_foot',
        'front_left_foot',
        'hind_right_foot',
        'hind_left_foot',
    ]
    feet_site_idx = [
        mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE.value, f)
        for f in feet_site
    ]
    assert not any(id_ == -1 for id_ in feet_site_idx), 'Site not found.'
    feet_site_idx = np.array(feet_site_idx)

    # Calculate MuJoCo Data for all States:
    base_positions = np.concatenate([vicon_positions, body_orientation], axis=1)
    base_velocities = np.concatenate([global_velocity, global_anglular_velocity], axis=1)
    joint_positions = state_positions
    joint_velocities = state_velocities
    mj_datas = []
    for base_position, base_velocity, joint_position, joint_velocity in zip(
        base_positions, base_velocities, joint_positions, joint_velocities
    ):
        mj_data = mujoco.MjData(mj_model)
        mj_data.qpos = np.concatenate([base_position, joint_position])
        mj_data.qvel = np.concatenate([base_velocity, joint_velocity])
        mujoco.mj_forward(mj_model, mj_data)
        mj_datas.append(copy.deepcopy(mj_data))

    # Calculate Gait Timings:
    previous_air_times = np.zeros(contact_forces.shape)
    previous_contact_times = np.zeros(contact_forces.shape)
    for i, contact in enumerate(contact_forces):
        if i == 0:
            previous_air_times[i] = np.where(
                contact == True, 0.0, dt,
            )
            previous_contact_times[i] = np.where(
                contact == False, 0.0, dt,
            )
        else:
            previous_air_times[i] = np.where(
                contact == True, 0.0, previous_air_times[i-1] + dt,
            )
            previous_contact_times[i] = np.where(
                contact == False, 0.0, previous_contact_times[i-1] + dt,
            )

    def calculate_reward(
        reward_data: dict[str, np.ndarray]
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        rewards = {
            'tracking_linear_velocity': (
                _reward_tracking_velocity(reward_data['command'][:2], reward_data['local_linear_velocity'][:2])
            ),
            'tracking_angular_velocity': (
                _reward_tracking_yaw_rate(reward_data['command'][-1], reward_data['local_angular_velocity'][-1])
            ),
            'linear_z_velocity': _cost_vertical_velocity(
                reward_data['global_velocity'][-1],
            ),
            'angular_xy_velocity': _cost_angular_velocity(
                reward_data['global_angular_velocity'][:2],
            ),
            'orientation_regularization': _cost_orientation_regularization(
                reward_data['up_vector'][:2],
            ),
            'torque': _cost_torques(reward_data['torques']),
            'action_rate': _cost_action_rate(reward_data['action'], reward_data['previous_action']),
            'stand_still': _cost_stand_still(
                reward_data['command'], reward_data['joint_positions'], reward_data['default_pose'],
            ),
            'foot_slip': _cost_foot_slip(
                reward_data['mj_model'],
                reward_data['mj_data'],
                reward_data['contact'],
                reward_data['command'],
            ),
            'air_time': _reward_air_time(
                reward_data['previous_air_time'],
                reward_data['previous_contact_time'],
                reward_data['command'],
                reward_data['global_velocity'],
                mode_time=0.2,
                command_threshold=0.0,
                velocity_threshold=0.5,
            ),
            'gait_variance': _cost_gait_variance(
                reward_data['previous_air_time'],
                reward_data['previous_contact_time'],
            ),
            'foot_clearance': _reward_foot_clearance(
                reward_data['mj_model'],
                reward_data['mj_data'],
                reward_data['feet_site_idx'],
                target_foot_height=0.125,
                velocity_scale=2.0,
                sigma=0.05,
            ),
        }
        rewards = {
            k: v * reward_config[k] for k, v in rewards.items()
        }
        reward = np.clip(sum(rewards.values()) * dt, 0.0, 10000.0)
        return reward, rewards

    rewards = []
    reward_dicts = []
    for i in range(command.shape[0]):
        reward_data = {
            'command': command[i],
            'local_linear_velocity': local_linear_velocity[i],
            'local_angular_velocity': local_angular_velocity[i],
            'global_velocity': global_velocity[i],
            'global_angular_velocity': global_anglular_velocity[i],
            'up_vector': up_vector[i],
            'torques': state_torques[i],
            'action': actions[i],
            'previous_action': previous_actions[i],
            'joint_positions': state_positions[i],
            'default_pose': default_pose,
            'mj_data': mj_datas[i],
            'mj_model': mj_model,
            'feet_site_idx': feet_site_idx,
            'contact': contact_forces[i],
            'previous_air_time': previous_air_times[i],
            'previous_contact_time': previous_contact_times[i],
        }
        reward, reward_dict = calculate_reward(reward_data)
        rewards.append(reward)
        reward_dicts.append(reward_dict)

    print("Average Reward:", np.mean(rewards))

    mean_rewards = {}
    for key in reward_config.keys():
        values = list(map(lambda rd: rd[key].item(), reward_dicts))
        avg = np.mean(values).item()
        mean_rewards[key] = avg
    mean_rewards['sum'] = sum(mean_rewards.values())

    with open(directory_path / "reward_summary.yaml", 'w') as f:
        yaml.dump(mean_rewards, f, sort_keys=False)


# Rewards and Costs:
def _reward_tracking_velocity(
    desired_xy_velocity: np.ndarray,
    local_xy_velocity: np.ndarray,
    kernel_sigma: float = 0.5,
) -> np.ndarray:
    velocity_error = np.sum(np.square(desired_xy_velocity - local_xy_velocity))
    return np.exp(-velocity_error / kernel_sigma)


def _reward_tracking_yaw_rate(
    desired_yaw_rate: float,
    local_yaw_rate: float,
    kernel_sigma: float = 0.5,
) -> np.ndarray:
    yaw_rate_error = np.square(desired_yaw_rate - local_yaw_rate)
    return np.exp(-yaw_rate_error / kernel_sigma)


def _cost_vertical_velocity(
    global_z_velocity: np.ndarray,
) -> np.ndarray:
    return np.square(global_z_velocity)


def _cost_angular_velocity(
    global_xy_angular_velocity: np.ndarray,
) -> np.ndarray:
    return np.sum(np.square(global_xy_angular_velocity))


def _cost_orientation_regularization(
    base_z_axis: np.ndarray,
) -> np.ndarray:
    return np.sum(np.square(base_z_axis))


def _cost_torques(
    torques: np.ndarray,
) -> np.ndarray:
    return np.sqrt(np.sum(np.square(torques))) + np.sum(np.abs(torques))


def _cost_action_rate(
    action: np.ndarray,
    previous_action: np.ndarray,
) -> np.ndarray:
    return np.sqrt(np.sum(np.square(action - previous_action)))


def _cost_acceleration(
    joint_accelerations: np.ndarray,
) -> np.ndarray:
    return np.sqrt(np.sum(np.square(joint_accelerations)))


def _cost_stand_still(
    commands: np.ndarray,
    joint_positions: np.ndarray,
    default_pose: np.ndarray,
) -> np.ndarray:
    command_norm = np.linalg.norm(commands)
    return np.sum(np.abs(joint_positions - default_pose)) * (command_norm < 0.1)


def _reward_air_time(
    air_time: np.ndarray,
    contact_time: np.ndarray,
    commands: np.ndarray,
    body_velocity: np.ndarray,
    mode_time: float = 0.3,
    command_threshold: float = 0.0,
    velocity_threshold: float = 0.5,
) -> np.ndarray:
    # Calculate Mode Timing Reward
    t_max = np.maximum(air_time, contact_time)
    t_min = np.clip(t_max, max=mode_time)
    stance_reward = np.clip(contact_time - air_time, min=-mode_time, max=mode_time)
    # Command and Body Velocity:
    command_norm = np.linalg.norm(commands)
    velocity_norm = np.linalg.norm(body_velocity)
    # Reward:
    reward = np.where(
        (command_norm > command_threshold) | (velocity_norm > velocity_threshold),
        np.where(t_max < mode_time, t_min, 0.0),
        stance_reward,
    )
    return np.sum(reward)


def _cost_gait_variance(
    previous_air_time: np.ndarray,
    previous_contact_time: np.ndarray,
) -> np.ndarray:
    # Penalize variance in gait timing
    air_time_variance = np.var(
        np.clip(previous_air_time, max=0.5),
    )
    contact_time_variance = np.var(
        np.clip(previous_contact_time, max=0.5),
    )
    return air_time_variance + contact_time_variance


def _reward_foot_clearance(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    feet_site_idx: np.ndarray,
    target_foot_height: float = 0.1,
    velocity_scale: float = 2.0,
    sigma: float = 0.05,
) -> np.ndarray:
    foot_position = data.site_xpos[feet_site_idx]
    foot_height = foot_position[..., -1]
    foot_error = np.square(foot_height - target_foot_height)
    foot_velocity = get_feet_velocity(model, data)[..., :2]
    foot_velocity_norm = np.linalg.norm(foot_velocity)
    foot_velocity_tanh = np.tanh(velocity_scale * foot_velocity_norm)
    error = np.sum(foot_error * foot_velocity_tanh)
    return np.exp(-error / sigma)


def _cost_foot_slip(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    contact: np.ndarray,
    commands: np.ndarray,
) -> np.ndarray:
    # Penalize foot slip
    command_norm = np.linalg.norm(commands)
    foot_velocity = get_feet_velocity(model, data)
    foot_velocity_xy = foot_velocity[..., :2]
    velocity_xy_sq = np.sum(np.square(foot_velocity_xy), axis=-1)
    return np.sum(velocity_xy_sq * contact) * (command_norm > 0.1)


# Reward depends on information we dont have.
def _cost_unwanted_contact(
    unwanted_contacts: any,
) -> np.ndarray:
    # Unwanted Contact Penalty
    return np.sum(unwanted_contacts)


# Training only reward. Do not need.
def _cost_termination(done: np.ndarray) -> np.ndarray:
    return done

def get_sensor_data(
    model: mujoco.MjModel, data: mujoco.MjData, sensor_name: str
) -> np.ndarray:
    """Gets sensor data given sensor name."""
    sensor_id = model.sensor(sensor_name).id
    sensor_adr = model.sensor_adr[sensor_id]
    sensor_dim = model.sensor_dim[sensor_id]
    return data.sensordata[sensor_adr: sensor_adr + sensor_dim]

def get_feet_position(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    feet_position_sensor = [
        "front_right_position",
        "front_left_position",
        "hind_right_position",
        "hind_left_position",
    ]
    return np.vstack([
        get_sensor_data(model, data, sensor_name)
        for sensor_name in feet_position_sensor
    ])

def get_feet_velocity(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    feet_linear_velocity_sensor = [
        "front_right_global_linear_velocity",
        "front_left_global_linear_velocity",
        "hind_right_global_linear_velocity",
        "hind_left_global_linear_velocity",
    ]
    return np.vstack([
        get_sensor_data(model, data, sensor_name)
        for sensor_name in feet_linear_velocity_sensor
    ])
