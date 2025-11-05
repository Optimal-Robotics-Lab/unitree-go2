"""
    Unitree Go2 Environment:
"""

from typing import Any, Dict, TypeAlias
from absl import app
import os

import flax.serialization
import jax
import jax.numpy as jnp

import numpy as np

import flax.serialization

from brax import base
from brax import envs
from brax import math
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf, html

import mujoco
from mujoco.mjx._src import math as mjx_math

from training.envs.unitree_go2_handstand.config import (
    RewardConfig,
    NoiseConfig,
    DisturbanceConfig,
    EnvironmentConfig,
)

# Types:
PRNGKey: TypeAlias = jax.Array


class UnitreeGo2Env(PipelineEnv):
    """Environment for training the Unitree Go2 quadruped joystick policy in MJX."""

    def __init__(
        self,
        env_config: EnvironmentConfig = EnvironmentConfig(),
        reward_config: RewardConfig = RewardConfig(),
        noise_config: NoiseConfig = NoiseConfig(),
        disturbance_config: DisturbanceConfig = DisturbanceConfig(),
        **kwargs,
    ):
        self.filename = f'mjcf/{env_config.filename}'
        self.filepath = os.path.join(
            os.path.dirname(__file__),
            self.filename,
        )
        sys = mjcf.load(self.filepath)

        self.step_dt = env_config.control_timestep
        sys = sys.tree_replace({'opt.timestep': env_config.optimizer_timestep})

        n_frames = kwargs.pop('n_frames', int(self.step_dt / sys.opt.timestep))
        super().__init__(sys, backend='mjx', n_frames=n_frames)

        if env_config.control_type == "position":
            self.control_type = "position"
        elif env_config.control_type == "velocity":
            self.control_type = "velocity"
        else:
            raise ValueError(f"Unknown control type: {env_config.control_type}")

        self.base_sigma = reward_config.base_sigma
        self.orientation_sigma = reward_config.orientation_sigma
        self.pose_sigma = reward_config.pose_sigma
        reward_config_dict = flax.serialization.to_state_dict(reward_config)
        del reward_config_dict['base_sigma']
        del reward_config_dict['orientation_sigma']
        del reward_config_dict['pose_sigma']
        self.reward_config = reward_config_dict

        self.noise_config = noise_config
        self.disturbance_config = disturbance_config

        self.floor_geom_idx = self.sys.mj_model.geom('floor').id
        self.base_idx = mujoco.mj_name2id(
            sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, 'base_link'
        )
        self.base_link_mass = self.sys.mj_model.body_subtreemass[self.base_idx]

        self.action_scale = env_config.action_scale
        self.init_q = jnp.array(sys.mj_model.keyframe('home').qpos)
        self.init_qd = jnp.zeros(sys.nv)
        self.default_pose = jnp.array(sys.mj_model.keyframe('home').qpos[7:])
        self.default_ctrl = jnp.array(sys.mj_model.keyframe('home').ctrl)
        self.joint_lb, self.joint_ub = self.sys.mj_model.jnt_range[1:].T

        # Soft Joint Limits:
        c = (self.joint_lb + self.joint_ub) / 2.0
        r = self.joint_ub - self.joint_lb
        self.soft_joint_limit_factor = 0.9
        self.soft_lb = c - 0.5 * r * self.soft_joint_limit_factor
        self.soft_ub = c + 0.5 * r * self.soft_joint_limit_factor

        self.num_joints = self.sys.nv - 6
        self.nu = self.sys.nu

        # Task Pose:
        self.footstand_q = jnp.array(sys.mj_model.keyframe('footstand').qpos)
        self.footstand_pose = jnp.array(sys.mj_model.keyframe('footstand').qpos[7:])
        self.desired_height = 0.53
        self.tracking_vector = jnp.array([0.0, 0.0, 1.0])
        self.target_pose_threshold = 0.9

        # Sites and Bodies:
        feet_site = [
            'front_right_foot',
            'front_left_foot',
            'hind_right_foot',
            'hind_left_foot',
        ]
        feet_site_idx = [
            mujoco.mj_name2id(sys.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, f)
            for f in feet_site
        ]
        assert not any(id_ == -1 for id_ in feet_site_idx), 'Site not found.'
        self.feet_site_idx = np.array(feet_site_idx)
        calf_body = [
            'front_right_calf',
            'front_left_calf',
            'hind_right_calf',
            'hind_left_calf',
        ]
        calf_body_idx = [
            mujoco.mj_name2id(sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, c)
            for c in calf_body
        ]
        assert not any(id_ == -1 for id_ in calf_body_idx), 'Body not found.'
        self.calf_body_idx = np.array(calf_body_idx)
        imu_site_idx = mujoco.mj_name2id(
            sys.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, 'imu'
        )
        assert not any(id_ == -1 for id_ in [imu_site_idx]), 'IMU site not found.'
        self.imu_site_idx = np.array(imu_site_idx)

        calf_joint = [
            "front_right_calf_joint",
            "front_left_calf_joint",
            "hind_right_calf_joint",
            "hind_left_calf_joint",
        ]
        calf_joint_idx = [
            mujoco.mj_name2id(sys.mj_model, mujoco.mjtObj.mjOBJ_JOINT.value, j)
            for j in calf_joint
        ]
        self.calf_joint_idx = np.array(calf_joint_idx)

        # Contact Sensors:
        feet_sensor_names = [
            "front_right_foot_to_floor",
            "front_left_foot_to_floor",
            "hind_right_foot_to_floor",
            "hind_left_foot_to_floor",
        ]
        self.feet_contact_sensor = [
            self.sys.mj_model.sensor(f'{foot_sensor_name}').id
            for foot_sensor_name in feet_sensor_names
        ]

        termination_sensor_names = [
            "left_torso_to_floor",
            "right_torso_to_floor",
        ]
        termination_sensor_names.extend([
            "front_right_calf_upper_to_floor",
            "front_right_calf_lower_to_floor",
            "front_left_calf_upper_to_floor",
            "front_left_calf_lower_to_floor",
            "hind_right_calf_upper_to_floor",
            "hind_right_calf_lower_to_floor",
            "hind_left_calf_upper_to_floor",
            "hind_left_calf_lower_to_floor",
            "hind_right_thigh_to_floor",
            "hind_left_thigh_to_floor",
            "front_right_foot_to_hind_right_hip",
            "front_left_foot_to_hind_left_hip",
        ])
        self.termination_contact_sensor = [
            self.sys.mj_model.sensor(f'{termination_sensor_name}').id
            for termination_sensor_name in termination_sensor_names
        ]
        self.terminate_on_contact = env_config.terminate_on_contact

        unwanted_contact_sensor_names = [
            "front_right_foot_to_floor",
            "front_left_foot_to_floor",
            "front_right_calf_upper_to_floor",
            "front_right_calf_lower_to_floor",
            "hind_right_calf_upper_to_floor",
            "hind_right_calf_lower_to_floor",
            "hind_left_calf_upper_to_floor",
            "hind_left_calf_lower_to_floor",
            "hind_right_thigh_to_floor",
            "hind_left_thigh_to_floor",
            "front_right_foot_to_hind_right_hip",
            "front_left_foot_to_hind_left_hip",
        ]
        self.unwanted_contact_sensor = [
            self.sys.mj_model.sensor(f'{sensor_name}').id
            for sensor_name in unwanted_contact_sensor_names
        ]

        # Feet Sensors:
        self.feet_position_sensor = [
            "front_right_foot_position",
            "front_left_foot_position",
            "hind_right_foot_position",
            "hind_left_foot_position",
        ]
        self.feet_linear_velocity_sensor = [
            "front_right_foot_global_linear_velocity",
            "front_left_foot_global_linear_velocity",
            "hind_right_foot_global_linear_velocity",
            "hind_left_foot_global_linear_velocity",
        ]

        # Observation Size: 
        self.num_observations = 30 + self.nu
        self.num_privileged_observations = self.num_observations + 66

    def reset(self, rng: PRNGKey) -> State:  # pytype: disable=signature-mismatch
        # Initial Position:
        rng, key = jax.random.split(rng)
        delta = jax.random.uniform(
            key, shape=(2,), minval=-0.5, maxval=0.5,
        )
        qpos = self.init_q.at[0:2].set(self.init_q[0:2] + delta)

        # Yaw: Uniform [-pi, pi]
        rng, key = jax.random.split(rng)
        yaw = jax.random.uniform(key, (1,), minval=-jnp.pi, maxval=jnp.pi)
        rotation = mjx_math.axis_angle_to_quat(jnp.array([0, 0, 1]), yaw)
        quaternion = mjx_math.quat_mul(self.init_q[3:7], rotation)
        qpos = qpos.at[3:7].set(quaternion)

        # Initial Velocity: Normal STD Deviation 0.2 m/s
        rng, key = jax.random.split(rng)
        qvel = self.init_qd.at[0:6].set(
            jax.random.uniform(key, (6,), minval=-0.5, maxval=0.5)
        )

        qpos = self.init_q
        qvel = self.init_qd
        ctrl = self.default_ctrl

        # Initialize State:
        pipeline_state = self.pipeline_init(qpos, qvel, ctrl=ctrl)

        # Disturbance: (Force Based)
        rng, disturbance_time_key, disturbance_duration_key, disturbance_magnitude_key = jax.random.split(rng, 4)
        time_until_next_disturbance = jax.random.uniform(
            disturbance_time_key,
            minval=self.disturbance_config.wait_times[0],
            maxval=self.disturbance_config.wait_times[1],
        )
        steps_until_next_disturbance = jnp.round(
            time_until_next_disturbance / self.dt
        ).astype(jnp.int32)
        disturbance_duration = jax.random.uniform(
            disturbance_duration_key,
            minval=self.disturbance_config.durations[0],
            maxval=self.disturbance_config.durations[1],
        )
        disturbance_duration_steps = jnp.round(
            disturbance_duration / self.dt
        ).astype(jnp.int32)
        disturbance_magnitude = jax.random.uniform(
            disturbance_magnitude_key,
            minval=self.disturbance_config.magnitudes[0],
            maxval=self.disturbance_config.magnitudes[1],
        )
        
        feet_contacts = jnp.array([
            pipeline_state.sensordata[self.sys.sensor_adr[sensor_id]] > 0
            for sensor_id in self.feet_contact_sensor
        ])
        unwanted_contacts = jnp.array([
            pipeline_state.sensordata[self.sys.sensor_adr[sensor_id]] > 0
            for sensor_id in self.unwanted_contact_sensor
        ])

        state_info = {
            'rng': rng,
            'previous_action': jnp.zeros(self.nu),
            'previous_velocity': jnp.zeros(self.num_joints),
            'previous_feet_contact': feet_contacts,
            'previous_unwanted_contacts': unwanted_contacts,
            'previous_motor_targets': self.default_ctrl,
            'front_feet_air_time': 0.0,
            'first_contact': False,
            'rewards': {k: 0.0 for k in self.reward_config.keys()},
            'steps_until_next_disturbance': steps_until_next_disturbance,
            'disturbance_duration': disturbance_duration,
            'disturbance_duration_steps': disturbance_duration_steps,
            'steps_since_previous_disturbance': 0,
            'disturbance_step': 0,
            'disturbance_magnitude': disturbance_magnitude,
            'disturbance_direction': jnp.array([0.0, 0.0, 0.0]),
            'time': 0.0,
        }

        # Observation Initialization:
        observation = self.get_observation(
            pipeline_state, state_info,
        )

        reward, done = jnp.zeros(2)
        done = jnp.float64(done) if jax.config.x64_enabled else jnp.float32(done)

        metrics = {}
        for k in state_info['rewards']:
            metrics[k] = state_info['rewards'][k]

        state = State(
            pipeline_state=pipeline_state,
            obs=observation,
            reward=reward,
            done=done,
            metrics=metrics,
            info=state_info,
        )
        return state

    def step(self, state: State, action: jax.Array) -> State:  # pytype: disable=signature-mismatch
        rng, key = jax.random.split(state.info['rng'])

        # Disturbance: (Force based)
        state = self.maybe_apply_perturbation(state)

        # Physics step:
        if self.control_type == "position":
            motor_targets = state.info['previous_motor_targets'] + action * self.action_scale
            # motor_targets = self.default_ctrl + action * self.action_scale
        elif self.control_type == "velocity":
            position_actions, velocity_actions = jnp.split(action, 2)
            previous_position_targets, previous_velocity_targets = jnp.split(
                state.info['previous_motor_targets'], 2
            )
            default_position_ctrl, default_velocity_ctrl = jnp.split(self.default_ctrl, 2)
            position_targets = previous_position_targets + position_actions * self.action_scale
            # position_targets = default_position_ctrl + position_actions * self.action_scale
            velocity_targets = default_velocity_ctrl + velocity_actions * self.action_scale
            motor_targets = jnp.concatenate([position_targets, velocity_targets])
        else:
            raise ValueError(f"Unknown control type: {self.control_type}")

        pipeline_state = self.pipeline_step(
            state.pipeline_state, motor_targets,
        )

        imu_height = pipeline_state.site_xpos[self.imu_site_idx][2]
        joint_angles = pipeline_state.q[7:]
        joint_velocities = pipeline_state.qd[6:]
        base_velocity = pipeline_state.qd[:6]
        forward_vector = pipeline_state.site_xmat[self.imu_site_idx] @ jnp.array([1.0, 0.0, 0.0])

        # Sensor Contacts:
        feet_contacts = jnp.array([
            pipeline_state.sensordata[self.sys.sensor_adr[sensor_id]] > 0
            for sensor_id in self.feet_contact_sensor
        ])
        unwanted_contacts = jnp.array([
            pipeline_state.sensordata[self.sys.sensor_adr[sensor_id]] > 0
            for sensor_id in self.unwanted_contact_sensor
        ])
        termination_contacts = jnp.array([
            pipeline_state.sensordata[self.sys.sensor_adr[sensor_id]] > 0
            for sensor_id in self.termination_contact_sensor
        ])

        # Observation data:
        observation = self.get_observation(
            pipeline_state,
            state.info,
        )

        # Termination Condition:
        done = self._get_termination(
            pipeline_state,
            termination_contacts,
            terminate_on_contact=self.terminate_on_contact,
        )

        # Rewards:
        rewards = {
            'tracking_base_pose': (
                self._reward_tracking_base_pose(imu_height, self.base_sigma)
            ),
            'tracking_orientation': self._reward_tracking_orientation(
                forward_vector, self.orientation_sigma,
            ),
            'tracking_joint_pose': self._reward_tracking_joint_pose(
                joint_angles, self.pose_sigma,
            ),
            'torque': self._cost_torques(pipeline_state.actuator_force),
            'action_rate': self._cost_action_rate(action, state.info['previous_action']),
            'acceleration': self._cost_acceleration(
                pipeline_state.qacc,
            ),
            'base_velocity': self._cost_base_velocity(
                base_velocity,
            ),
            'stand_still': self._cost_stand_still(
                forward_vector, joint_angles,
            ),
            'feet_contact': self._cost_feet_contact(feet_contacts, alpha=1.0),
            'feet_slip': self._cost_feet_slip(
                pipeline_state, feet_contacts
            ),
            'unwanted_contact': self._cost_unwanted_contact(
                unwanted_contacts,
            ),
            'pose': self._cost_pose(joint_angles),
            'joint_limits': self._cost_joint_position_limits(
                joint_angles,
            ),
            'symmetry': self._cost_joint_symmetry(joint_angles),
            'termination': jnp.float64(
                self._cost_termination(done)
            ) if jax.config.x64_enabled else jnp.float32(
                self._cost_termination(done)
            ),
        }
        rewards = {
            k: v * self.reward_config[k] for k, v in rewards.items()
        }
        reward = jnp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)

        # State management
        state.info['rewards'] = rewards
        state.info['rng'] = rng
        state.metrics.update(state.info['rewards'])

        done = jnp.float64(done) if jax.config.x64_enabled else jnp.float32(done)

        # Auto Reset Wrapper does not reset some state fields:
        state.info['previous_action'] = jnp.where(done, jnp.zeros_like(action), action)
        state.info['previous_velocity'] = jnp.where(done, jnp.zeros_like(joint_velocities), joint_velocities)
        state.info['previous_feet_contact'] = jnp.where(done, jnp.zeros_like(feet_contacts), feet_contacts)
        state.info['previous_unwanted_contacts'] = jnp.where(done, jnp.zeros_like(unwanted_contacts), unwanted_contacts)
        state.info['previous_motor_targets'] = jnp.where(done, self.default_ctrl, motor_targets)
        state.info['time'] = jnp.where(done, 0.0, state.info['time'] + self.dt)

        state = state.replace(
            pipeline_state=pipeline_state,
            obs=observation,
            reward=reward,
            done=done,
        )

        return state

    def get_observation(
        self,
        pipeline_state: base.State,
        state_info: dict[str, Any],
    ) -> Dict[str, jax.Array]:
        """
            Observation: [
                gyroscope,
                projected_gravity,
                relative_motor_positions,
                motor_velocities,
                previous_action,
            ]
        """
        q = pipeline_state.q[7:]
        qd = pipeline_state.qd[6:]

        # Gyroscope Noise:
        gyroscope = self.get_gyro(pipeline_state)
        state_info['rng'], noise_key = jax.random.split(state_info['rng'])
        gyroscope_noise = jax.random.uniform(
            noise_key,
            shape=gyroscope.shape,
            minval=-self.noise_config.gyroscope,
            maxval=self.noise_config.gyroscope,
        )
        noisy_angular_rate = gyroscope + gyroscope_noise

        # Gravity noise:
        projected_gravity = self.get_gravity(pipeline_state)
        state_info['rng'], noise_key = jax.random.split(state_info['rng'])
        gravity_noise = jax.random.uniform(
            noise_key,
            shape=projected_gravity.shape,
            minval=-self.noise_config.gravity_vector,
            maxval=self.noise_config.gravity_vector,
        )
        noisy_projected_gravity = projected_gravity + gravity_noise

        # Joint position noise:
        state_info['rng'], noise_key = jax.random.split(state_info['rng'])
        joint_position_noise = jax.random.uniform(
            noise_key,
            shape=q.shape,
            minval=-self.noise_config.joint_position,
            maxval=self.noise_config.joint_position,
        )
        noisy_joint_positions = q + joint_position_noise

        # Joint velocity noise:
        state_info['rng'], noise_key = jax.random.split(state_info['rng'])
        joint_velocity_noise = jax.random.uniform(
            noise_key,
            shape=qd.shape,
            minval=-self.noise_config.joint_velocity,
            maxval=self.noise_config.joint_velocity,
        )
        noisy_joint_velocities = qd + joint_velocity_noise

        observation = jnp.concatenate([
            noisy_angular_rate,                         # 3
            noisy_projected_gravity,                    # 3
            noisy_joint_positions - self.default_pose,  # 12
            noisy_joint_velocities,                     # 12
            state_info['previous_action'],              # 12 or 24
        ]) # Size: 42 or 54

        accelerometer = self.get_accelerometer(pipeline_state)
        linear_velocity = self.get_local_linear_velocity(pipeline_state)
        global_angular_velocity = self.get_global_angular_velocity(pipeline_state)
        actuator_force = pipeline_state.actuator_force
        imu_height = jnp.asarray([pipeline_state.site_xpos[self.imu_site_idx][2]])

        privileged_observation = jnp.concatenate([
            observation,                                                                                # 42 or 54
            accelerometer,                                                                              # 3
            gyroscope,                                                                                  # 3
            projected_gravity,                                                                          # 3
            linear_velocity,                                                                            # 3
            global_angular_velocity,                                                                    # 3
            q - self.default_pose,                                                                      # 12
            qd,                                                                                         # 12
            actuator_force,                                                                             # 12
            imu_height,                                                                                 # 1
            state_info['previous_feet_contact'],                                                        # 4
            state_info['previous_unwanted_contacts'],                                                   # 10
        ])
        # Size: 66

        return {
            'state': observation,
            'privileged_state': privileged_observation,
        }

    def _get_termination(
        self,
        pipeline_state: base.State,
        termination_contacts: jax.Array,
        terminate_on_contact: bool = False,
    ) -> jax.Array:
        # Termination Condition:
        done = self.get_upvector(pipeline_state)[-1] < -0.25
        done |= terminate_on_contact * jnp.any(termination_contacts)
        return done

    def _reward_tracking_base_pose(
        self,
        pose: jax.Array,
        kernel_sigma: float = 0.25,
    ) -> jax.Array:
        # Reward Correct Height:
        base_height = jnp.min(jnp.array([pose, self.desired_height]))
        error = jnp.sum(jnp.square(base_height - self.desired_height))
        return jnp.exp(-error / kernel_sigma)
    
    def _reward_tracking_orientation(
        self,
        forward_vector: jax.Array,
        kernel_sigma: float = 0.25,
    ) -> jax.Array:
        '''
            Sigma tuning: Desired Allowed Angle Deviation to achieve 61% Reward
                sigma = 2 * (allowed_angle * pi / 180)^2

                Ex. Angle Deviation of 20 Degrees to achieve 61% Reward
                    sigma = 2 * (20 * pi / 180)^2 = 0.24
        '''
        # Reward Handstand/Footstand Orientation:
        dot_product = jnp.clip(
            jnp.dot(forward_vector, self.tracking_vector),
            -1.0,
            1.0,
        )
        error = jnp.square(dot_product - 1.0)
        return jnp.exp(-error / kernel_sigma)

    def _reward_tracking_joint_pose(
        self,
        qpos: jax.Array,
        kernel_sigma: float = 0.25,
    ) -> jax.Array:
        # Reward for Handstand/Footstand Pose:
        weight = jnp.array([
            1.0, 1.0, 1.0,
            1.0, 1.0, 1.0,
            1.0, 1.0, 1.0,
            1.0, 1.0, 1.0,
        ])
        weight = weight / jnp.sum(weight)
        error = jnp.sum(jnp.square(qpos - self.footstand_pose) * weight)
        return jnp.exp(-error / kernel_sigma)

    def _cost_torques(self, torques: jax.Array) -> jax.Array:
        # Penalize torques
        return jnp.sqrt(jnp.sum(jnp.square(torques))) + jnp.sum(jnp.abs(torques))

    def _cost_action_rate(
        self, action: jax.Array, previous_action: jax.Array
    ) -> jax.Array:
        # Penalize changes in actions
        return jnp.sum(jnp.square(action - previous_action))

    def _cost_acceleration(
        self, qacc: jax.Array,
    ) -> jax.Array:
        # Penalize Motor/Joint Acceleration
        return jnp.sqrt(jnp.sum(jnp.square(qacc)))

    def _cost_pose(
        self, qpos: jax.Array,
    ) -> jax.Array:
        front_joint_ids = jnp.array([0, 1, 2, 3, 4, 5])
        return jnp.sum(jnp.square(qpos[front_joint_ids] - self.default_pose[front_joint_ids]))
    
    def _cost_stand_still(
        self,
        forward_vector: jax.Array,
        joint_angles: jax.Array,
    ) -> jax.Array:
        # Penalize motion at target pose
        dot_product = jnp.dot(forward_vector, self.tracking_vector)
        target_pose = dot_product >= self.target_pose_threshold
        return jnp.sum(jnp.abs(joint_angles - self.footstand_pose)) * (target_pose)

    def _cost_base_velocity(
        self,
        base_velocity: jax.Array,
    ) -> jax.Array:
        # Penalize base velocity at target pose
        linear_velocity_xy_error = jnp.sum(jnp.square(base_velocity[:2]))
        angular_velocity_z_error = jnp.square(base_velocity[5])
        return (linear_velocity_xy_error + angular_velocity_z_error)
    
    def _cost_joint_position_limits(
        self, joint_angles: jax.Array,
    ) -> jax.Array:
        # Penalize joint angles outside of limits
        out_of_limits = -jnp.clip(joint_angles - self.soft_lb, None, 0.0)
        out_of_limits += jnp.clip(joint_angles - self.soft_ub, 0.0, None)
        return jnp.sum(out_of_limits)

    def _cost_feet_contact(
        self,
        contact: jax.Array,
        alpha: float = 1.0,
    ) -> jax.Array:
        # Reward Correct Feet Contact and Penalize Incorrect Feet Contact
        correct_contact = jnp.sum(jnp.array([0, 0, 1, 1]) * contact)  # Hind Feet in Contact
        incorrect_contact = jnp.sum(
            jnp.array([1, 1, 0, 0]) * contact       # Front Feet in Contact
            + jnp.array([0, 0, 1, 1]) * ~contact    # Hind Feet not in Contact
        )

        # Linear Formulation:
        reward = (correct_contact - incorrect_contact) / 2.0

        # Exponential Formulation:
        # reward = jnp.exp(alpha * (correct_contact - incorrect_contact)) - 1.0

        return reward

    def _cost_feet_slip(
        self,
        pipeline_state: base.State,
        contact: jax.Array,
    ) -> jax.Array:
        # Penalize foot slip
        foot_velocity = self.get_feet_velocity(pipeline_state)
        foot_velocity_xy = foot_velocity[..., :2]
        velocity_xy_sq = jnp.sum(jnp.square(foot_velocity_xy), axis=-1)
        return jnp.sum(velocity_xy_sq * contact)

    def _cost_unwanted_contact(
        self,
        unwanted_contacts: base.State,
    ) -> jax.Array:
        # Unwanted Contact Penalty
        return jnp.sum(unwanted_contacts)

    def _cost_joint_symmetry(
        self,
        joint_angles: jax.Array,
    ) -> jax.Array:
        # Penalize asymmetry between left and right joints
        symmetry_mask = jnp.array([-1, 1, 1] * 2)
        right_ids = jnp.array([0, 1, 2, 6, 7, 8])
        left_ids = jnp.array([3, 4, 5, 9, 10, 11])
        right_joints = joint_angles[right_ids]
        left_joints = joint_angles[left_ids] * symmetry_mask
        return jnp.sum(jnp.square(right_joints - left_joints))

    def _cost_termination(self, done: jax.Array) -> jax.Array:
        return done

    @staticmethod
    def get_sensor_data(
        model: mujoco.MjModel, pipeline_state: base.State, sensor_name: str
    ) -> jax.Array:
        """Gets sensor data given sensor name."""
        sensor_id = model.sensor(sensor_name).id
        sensor_adr = model.sensor_adr[sensor_id]
        sensor_dim = model.sensor_dim[sensor_id]
        return pipeline_state.sensordata[sensor_adr: sensor_adr + sensor_dim]

    def get_upvector(self, pipeline_state: base.State) -> jax.Array:
        return self.get_sensor_data(self.sys.mj_model, pipeline_state, "upvector")

    def get_gravity(self, pipeline_state: base.State) -> jax.Array:
        return pipeline_state.site_xmat[self.imu_site_idx].T @ jnp.array([0, 0, -1])

    def get_global_linear_velocity(self, pipeline_state: base.State) -> jax.Array:
        return self.get_sensor_data(
            self.sys.mj_model, pipeline_state, "global_linear_velocity"
        )

    def get_global_angular_velocity(self, pipeline_state: base.State) -> jax.Array:
        return self.get_sensor_data(
            self.sys.mj_model, pipeline_state, "global_angular_velocity"
        )

    def get_local_linear_velocity(self, pipeline_state: base.State) -> jax.Array:
        return self.get_sensor_data(
            self.sys.mj_model, pipeline_state, "local_linear_velocity"
        )

    def get_accelerometer(self, pipeline_state: base.State) -> jax.Array:
        return self.get_sensor_data(
            self.sys.mj_model, pipeline_state, "imu_acceleration"
        )

    def get_gyro(self, pipeline_state: base.State) -> jax.Array:
        return self.get_sensor_data(self.sys.mj_model, pipeline_state, "imu_gyro")

    def get_feet_position(self, pipeline_state: base.State) -> jax.Array:
        return jnp.vstack([
            self.get_sensor_data(self.sys.mj_model, pipeline_state, sensor_name)
            for sensor_name in self.feet_position_sensor
        ])

    def get_feet_velocity(self, pipeline_state: base.State) -> jax.Array:
        return jnp.vstack([
            self.get_sensor_data(self.sys.mj_model, pipeline_state, sensor_name)
            for sensor_name in self.feet_linear_velocity_sensor
        ])

    # Adapted from mujoco_playground:
    def maybe_apply_perturbation(self, state: State) -> State:
        def gen_dir(rng: jax.Array) -> jax.Array:
            angle = jax.random.uniform(rng, minval=0.0, maxval=jnp.pi * 2)
            return jnp.array([jnp.cos(angle), jnp.sin(angle), 0.0])

        def apply_perturbation(state: State) -> State:
            t = state.info["disturbance_step"] * self.dt
            u_t = 0.5 * jnp.sin(jnp.pi * t / state.info["disturbance_duration"])
            # kg * m/s * 1/s = m/s^2 = kg * m/s^2 (N).
            force = (
                u_t  # (unitless)
                * self.base_link_mass  # kg
                * state.info["disturbance_magnitude"]  # m/s
                / state.info["disturbance_duration"]  # 1/s
            )
            xfrc_applied = jnp.zeros((self.sys.nbody, 6))
            xfrc_applied = xfrc_applied.at[self.base_idx, :3].set(
                force * state.info["disturbance_direction"]
            )
            pipeline_state = state.pipeline_state.replace(xfrc_applied=xfrc_applied)
            state = state.replace(pipeline_state=pipeline_state)
            state.info["steps_since_previous_disturbance"] = jnp.where(
                state.info["disturbance_step"] >= state.info["disturbance_duration_steps"],
                0,
                state.info["steps_since_previous_disturbance"],
            )
            state.info["disturbance_step"] += 1
            return state

        def wait(state: State) -> State:
            state.info["rng"], rng = jax.random.split(state.info["rng"])
            state.info["steps_since_previous_disturbance"] += 1
            xfrc_applied = jnp.zeros((self.sys.mj_model.nbody, 6))
            pipeline_state = state.pipeline_state.replace(xfrc_applied=xfrc_applied)
            state.info["disturbance_step"] = jnp.where(
                state.info["steps_since_previous_disturbance"]
                >= state.info["steps_until_next_disturbance"],
                0,
                state.info["disturbance_step"],
            )
            state.info["disturbance_direction"] = jnp.where(
                state.info["steps_since_previous_disturbance"]
                >= state.info["steps_until_next_disturbance"],
                gen_dir(rng),
                state.info["disturbance_direction"],
            )
            return state.replace(pipeline_state=pipeline_state)

        return jax.lax.cond(
            state.info["steps_since_previous_disturbance"]
            >= state.info["steps_until_next_disturbance"],
            apply_perturbation,
            wait,
            state,
        )

    def np_observation(
        self,
        mj_data: mujoco.MjData,
        previous_action: np.ndarray,
        add_noise: bool = True,
    ) -> np.ndarray:
        # Numpy implementation of the observation function:
        def rotate(vec: np.ndarray, quat: np.ndarray) -> np.ndarray:
            if len(vec.shape) != 1:
                raise ValueError('vec must have no batch dimensions.')
            s, u = quat[0], quat[1:]
            r = 2 * (np.dot(u, vec) * u) + (s * s - np.dot(u, u)) * vec
            r = r + 2 * s * np.cross(u, vec)
            return r

        def quat_inv(q: np.ndarray) -> np.ndarray:
            return q * np.array([1, -1, -1, -1])

        base_w = mj_data.qpos[3:7]
        q = mj_data.qpos[7:]
        qd = mj_data.qvel[6:]

        gyroscope = self.get_gyro(mj_data)

        inverse_trunk_rotation = quat_inv(base_w)
        projected_gravity = rotate(
            np.array([0, 0, -1]), inverse_trunk_rotation,
        )

        if add_noise:
            gyroscope = gyroscope + np.random.uniform(
                low=-self.noise_config.gyroscope,
                high=self.noise_config.gyroscope,
                size=gyroscope.shape,
            )
            projected_gravity = projected_gravity + np.random.uniform(
                low=-self.noise_config.gravity_vector,
                high=self.noise_config.gravity_vector,
                size=projected_gravity.shape,
            )
            q = q + np.random.uniform(
                low=-self.noise_config.joint_position,
                high=self.noise_config.joint_position,
                size=q.shape,
            )
            qd = qd + np.random.uniform(
                low=-self.noise_config.joint_velocity,
                high=self.noise_config.joint_velocity,
                size=qd.shape,
            )

        observation = np.concatenate([
            gyroscope,
            projected_gravity,
            q - self.default_pose,
            qd,
            previous_action,
        ])

        return {
            'state': observation,
            'privileged_state': np.zeros((self.num_privileged_observations,)),
        }

envs.register_environment('unitree_go2', UnitreeGo2Env)


def main(argv=None):
    env = UnitreeGo2Env()
    rng = jax.random.PRNGKey(0)

    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)

    state = reset_fn(rng)

    num_steps = 100
    states = []
    for i in range(num_steps):
        print(f"Step: {i}")
        state = step_fn(state, jnp.zeros_like(env.default_ctrl))
        states.append(state.pipeline_state)

    html_string = html.render(
        sys=env.sys.tree_replace({'opt.timestep': env.step_dt}),
        states=states,
        height="100vh",
        colab=False,
    )
    html_path = os.path.join(
        os.path.join(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(__file__),
                ),
            ),
        ),
        "visualization/visualization.html",
    )

    with open(html_path, "w") as f:
        f.writelines(html_string)


if __name__ == '__main__':
    app.run(main)
