import sys
import pathlib

from absl import app, flags

import numpy as np
import scipy


FLAGS = flags.FLAGS
flags.DEFINE_string(
    'directory_name', None, 'Desired checkpoint folder name to load.', short_name='d',
)


reward_config = {
    # Rewards:
    'tracking_linear_velocity': 1.5,
    'tracking_angular_velocity': 0.75,
    # Orientation Regularization Terms:
    'orientation_regularization': -2.5,
    'linear_z_velocity': -2.0,
    'angular_xy_velocity': -0.05,
    # Energy Regularization Terms:
    'torque': -2e-4,
    'action_rate': -0.01,
    # Auxilary Terms:
    'stand_still': -1.0,
}


def main(argv=None):
    """ Load Data """
    data_directory = pathlib.Path(__file__).parent.parent

    # Robot Command, State, Contact, IMU Data:
    command_history = data_directory / f"processed/{FLAGS.directory_name}/command_history.csv"
    state_history = data_directory / f"processed/{FLAGS.directory_name}/state_history.csv"
    # contact_history = data_directory / f"processed/{FLAGS.directory_name}/contact_history.csv"
    imu_history = data_directory / f"processed/{FLAGS.directory_name}/imu_history.csv"

    # Velocity Command:
    policy_command_history = data_directory / f"processed/{FLAGS.directory_name}/policy_command_history.csv"

    # Vicon and Filtered Vicon Data:
    vicon_history = data_directory / f"processed/{FLAGS.directory_name}/vicon_history.csv"
    filtered_history = data_directory / f"processed/{FLAGS.directory_name}/filtered_history.csv"

    files_exist = all([
        command_history.exists(),
        state_history.exists(),
        imu_history.exists(),
        vicon_history.exists(),
        filtered_history.exists(),
        # contact_history.exists(),
        policy_command_history.exists(),
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
    # contact_history = np.loadtxt(
    #     contact_history, delimiter=',',
    # )
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

    # Assert all time columns are the same
    time_stamps = [
        command_history[:, 0],
        state_history[:, 0],
        # contact_history[:, 0],
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

    # Unpack Contact Data:
    # contact_forces = contact_history[:, 1:5]
    # estimated_contact_forces = contact_history[:, 5:9]

    # Unpack Vicon Data:
    vicon_positions = vicon_history[:, 1:4] * 1e-3  # Convert mm to m
    vicon_orientation = vicon_history[:, 4:]

    # Unpack Filtered Data:
    global_position = filtered_history[:, 1:4]
    global_velocity = filtered_history[:, 4:7]

    # Unpack Policy Command Data:
    command = policy_command_history[:, 1:4]

    # Calculate Local Frame Data:
    rotations = scipy.spatial.transform.Rotation.from_quat(imu_orientation)
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
        }
        reward, reward_dict = calculate_reward(reward_data)
        rewards.append(reward)
        reward_dicts.append(reward_dict)

    print("Average Reward:", np.mean(rewards))


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


# Check to see if we have joint acceleration data
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
    t_min = np.clip(t_max, a_max=mode_time)
    stance_reward = np.clip(contact_time - air_time, a_min=-mode_time, a_max=mode_time)
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
        np.clip(previous_air_time, a_max=0.5),
    )
    contact_time_variance = np.var(
        np.clip(previous_contact_time, a_max=0.5),
    )
    return air_time_variance + contact_time_variance


# Reward depends on information we dont have.
def _cost_foot_slip(
    pipeline_state: any,
    target_foot_height: float = 0.1,
    decay_rate: float = 0.95,
) -> np.ndarray:
    # Penalizes foot slip velocity at contact to encourage ground speed matching.
    if not (0.0 < decay_rate <= 1.0):
        raise ValueError("Decay rate must be between 0 and 1.")

    # Foot velocities and foot heights
    foot_velocity = self.get_feet_velocity(pipeline_state)
    foot_velocity_xy = foot_velocity[..., :2]
    foot_position = pipeline_state.site_xpos[self.feet_site_idx]
    foot_height = foot_position[..., -1]

    velocity_xy_sq = np.sum(np.square(foot_velocity_xy), axis=-1)

    scale_factor = -target_foot_height / np.log(1.0 - decay_rate)
    height_gate = np.exp(-foot_height / scale_factor)

    return np.sum(velocity_xy_sq * height_gate)


# Reward depends on information we dont have.
def _cost_unwanted_contact(
    unwanted_contacts: any,
) -> np.ndarray:
    # Unwanted Contact Penalty
    return np.sum(unwanted_contacts)


# Training only reward. Do not need.
def _cost_termination(done: np.ndarray) -> np.ndarray:
    return done


if __name__ == "__main__":
    app.run(main)
