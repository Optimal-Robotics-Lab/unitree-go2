"""
    Unitree Go2 Environment:
"""

from typing import Any, Dict, TypeAlias
from absl import app
import os

import jax
import jax.numpy as jnp

import numpy as np

import mujoco
from mujoco import mjx
from mujoco.mjx._src import math as mjx_math
from mujoco_playground._src import mjx_env

from brax.io import html

from training.envs.unitree_go2 import base
from training.envs.unitree_go2.config import (
    RewardConfig,
    NoiseConfig,
    DisturbanceConfig,
    CommandConfig,
    EnvironmentConfig,
)

# Types:
PRNGKey: TypeAlias = jax.Array


class UnitreeGo2Env(base.UnitreeGo2Env):
    """Environment for training the Unitree Go2 quadruped joystick policy in MJX."""

    def __init__(
        self,
        environment_config: EnvironmentConfig = EnvironmentConfig(),
        reward_config: RewardConfig = RewardConfig(),
        noise_config: NoiseConfig = NoiseConfig(),
        disturbance_config: DisturbanceConfig = DisturbanceConfig(),
        command_config: CommandConfig = CommandConfig(),
        **kwargs,
    ) -> None:
        super().__init__(
            environment_config=environment_config,
            reward_config=reward_config,
            noise_config=noise_config,
            disturbance_config=disturbance_config,
            command_config=command_config,
            **kwargs,
        )

    def sample_command(
        self,
        rng: jax.Array,
    ) -> jax.Array:
        _, command_key, single_command_key, stand_still_key = jax.random.split(rng, 4)

        command = jax.random.uniform(
            command_key,
            shape=(3,),
            minval=-self.command_config.command_range,
            maxval=self.command_config.command_range,
        )
        single_command_mask = jax.random.choice(
            single_command_key,
            a=jnp.array([
                [1.0, 1.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
            ]),
            p=jnp.array([
                1.0 - self.command_config.single_command_probability,
                self.command_config.single_command_probability / 3.0,
                self.command_config.single_command_probability / 3.0,
                self.command_config.single_command_probability / 3.0
            ]),
        )
        stand_still_mask = jax.random.bernoulli(
            stand_still_key,
            p=self.command_config.command_mask_probability,
        )

        command = single_command_mask * command
        command = stand_still_mask * command

        return command

    def reset(self, rng: PRNGKey) -> mjx_env.State:  # pytype: disable=signature-mismatch
        # Choose Initial Position and Velocity:
        initial_qpos = self.home_qpos
        initial_qvel = self.home_qvel
        rotation_axis = jnp.array([0, 0, 1])

        # Initial Position:
        rng, key = jax.random.split(rng)
        delta = jax.random.uniform(
            key, shape=(2,), minval=-0.5, maxval=0.5,
        )
        qpos = initial_qpos.at[0:2].set(initial_qpos[0:2] + delta)

        # Yaw: Uniform [-pi, pi]
        rng, key = jax.random.split(rng)
        yaw = jax.random.uniform(key, (1,), minval=-jnp.pi, maxval=jnp.pi)
        rotation = mjx_math.axis_angle_to_quat(rotation_axis, yaw)
        quaternion = mjx_math.quat_mul(initial_qpos[3:7], rotation)
        qpos = qpos.at[3:7].set(quaternion)

        # Initial Velocity: Normal STD Deviation 0.2 m/s
        rng, key = jax.random.split(rng)
        qvel = initial_qvel.at[0:6].set(
            jax.random.uniform(key, (6,), minval=-0.5, maxval=0.5)
        )

        # Small Joint Perturbation:
        rng, key = jax.random.split(rng)
        delta = jax.random.uniform(
            key,
            shape=(self.num_joints,),
            minval=-0.1,
            maxval=0.1,
        )
        qpos = qpos.at[7:].set(qpos[7:] + delta)

        # Initialize State:
        ctrl = qpos[7:]

        data = mjx_env.make_data(
            self._mj_model,
            qpos=qpos,
            qvel=qvel,
            ctrl=ctrl,
            impl=self._mjx_model.impl.value,
            nconmax=self.environment_config.nconmax,
            njmax=self.environment_config.njmax,
        )
        data = mjx.forward(self._mjx_model, data)

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

        # Command Sampling:
        rng, command_sample_key, command_frequency_key = jax.random.split(rng, 3)
        seconds_until_next_command = jax.random.uniform(
            command_frequency_key,
            minval=self.command_config.command_frequency[0],
            maxval=self.command_config.command_frequency[1],
        )
        steps_until_next_command = jnp.round(
            seconds_until_next_command / self.dt
        ).astype(jnp.int32)
        command = self.sample_command(command_sample_key)

        # Foot Contacts:
        feet_contacts = jnp.array([
            data.sensordata[self._mj_model.sensor_adr[sensor_id]] > 0
            for sensor_id in self.feet_contact_sensor
        ])

        state_info = {
            'rng': rng,
            'previous_action': jnp.zeros(self.nu),
            'previous_joint_positions': jnp.zeros(self.num_joints),
            'previous_velocity': jnp.zeros(self.num_joints),
            'command': command,
            'steps_until_next_command': steps_until_next_command,
            'previous_contact': feet_contacts,
            'feet_air_time': jnp.zeros(4),
            'feet_contact_time': jnp.zeros(4),
            'previous_air_time': jnp.zeros(4),
            'previous_contact_time': jnp.zeros(4),
            'swing_peak': jnp.zeros(4),
            'rewards': {k: 0.0 for k in self.reward_config.keys()},
            'steps_until_next_disturbance': steps_until_next_disturbance,
            'disturbance_duration': disturbance_duration,
            'disturbance_duration_steps': disturbance_duration_steps,
            'steps_since_previous_disturbance': 0,
            'disturbance_step': 0,
            'disturbance_magnitude': disturbance_magnitude,
            'disturbance_direction': jnp.array([0.0, 0.0, 0.0]),
        }

        # Observation Initialization:
        observation = self.get_observation(
            data, state_info,
        )

        reward, done = jnp.zeros(2)
        done = jnp.float64(done) if jax.config.x64_enabled else jnp.float32(done)

        metrics = {}
        for k in state_info['rewards']:
            metrics[k] = state_info['rewards'][k]
        metrics['total_distance'] = 0.0
        metrics['swing_peak'] = jnp.zeros(())

        state = mjx_env.State(
            data=data,
            obs=observation,
            reward=reward,
            done=done,
            metrics=metrics,
            info=state_info,
        )
        return state

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:  # pytype: disable=signature-mismatch
        rng, cmd_key, cmd_frequency_key = jax.random.split(state.info['rng'], 3)

        # Disturbance: (Force based)
        if self.disturbance_config.magnitudes[1] > 0.0:
            state = self.maybe_apply_perturbation(state)

        # Physics step:
        motor_targets = self.default_ctrl + action * self.action_scale
        data = mjx_env.step(
            self._mjx_model, state.data, motor_targets, self._n_substeps,
        )

        imu_height = data.site_xpos[self.imu_site_idx][2]
        joint_angles = data.qpos[7:]
        joint_velocities = data.qvel[6:]

        # Sensor Contacts:
        feet_contacts = jnp.array([
            data.sensordata[self._mj_model.sensor_adr[sensor_id]] > 0
            for sensor_id in self.feet_contact_sensor
        ])
        unwanted_contacts = jnp.array([
            data.sensordata[self._mj_model.sensor_adr[sensor_id]] > 0
            for sensor_id in self.unwanted_contact_sensor
        ])

        # Feet Air and Contact Time:
        state.info['previous_air_time'] = jnp.where(
            feet_contacts, state.info['feet_air_time'], state.info['previous_air_time'],
        )
        state.info['previous_contact_time'] = jnp.where(
            ~feet_contacts, state.info['feet_contact_time'], state.info['previous_contact_time'],
        )

        state.info['feet_contact_time'] += self.dt
        state.info['feet_air_time'] += self.dt

        state.info['feet_air_time'] *= ~feet_contacts
        state.info['feet_contact_time'] *= feet_contacts

        # Foot Swing Peak Height:
        foot_position = data.site_xpos[self.feet_site_idx]
        foot_position_z = foot_position[..., -1]
        state.info['swing_peak'] = jnp.maximum(
            state.info['swing_peak'], foot_position_z,
        )

        # Body Velocity:
        global_body_velocity = self.get_global_linear_velocity(data)
        local_body_velocity = self.get_local_linear_velocity(data)

        # Observation data:
        observation = self.get_observation(
            data,
            state.info,
        )

        # Termination:
        done = self.get_termination(data)

        # Rewards:
        rewards = {
            'tracking_linear_velocity': (
                self._reward_tracking_velocity(state.info['command'], local_body_velocity)
            ),
            'tracking_angular_velocity': (
                self._reward_tracking_yaw_rate(state.info['command'], self.get_gyro(data))
            ),
            'linear_z_velocity': self._cost_vertical_velocity(
                global_body_velocity,
            ),
            'angular_xy_velocity': self._cost_angular_velocity(
                self.get_global_angular_velocity(data),
            ),
            'orientation_regularization': self._cost_orientation_regularization(
                self.get_upvector(data),
            ),
            'torque': self._cost_torques(data.actuator_force),
            'action_rate': self._cost_action_rate(action, state.info['previous_action']),
            'acceleration': self._cost_acceleration(
                data.qacc,
            ),
            'stand_still': self._cost_stand_still(
                state.info['command'], joint_angles,
            ),
            'foot_slip': self._cost_foot_slip(
                data, feet_contacts, state.info['command'],
            ),
            'air_time': self._reward_air_time(
                state.info['feet_air_time'],
                state.info['feet_contact_time'],
                state.info['command'],
                global_body_velocity,
                self.mode_time,
                self.command_threshold,
                self.velocity_threshold,
            ),
            'gait_variance': self._cost_gait_variance(
                state.info['previous_air_time'],
                state.info['previous_contact_time'],
            ),
            'foot_clearance': self._reward_foot_clearance(
                data,
                self.target_foot_height,
                self.foot_clearance_velocity_scale,
                self.foot_clearance_sigma,
            ),
            'unwanted_contact': self._cost_unwanted_contact(
                unwanted_contacts,
            ),
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
        state.info['previous_action'] = action
        state.info['previous_joint_positions'] = joint_angles
        state.info['previous_velocity'] = joint_velocities
        state.info['previous_contact'] = feet_contacts
        state.info['swing_peak'] *= ~feet_contacts
        state.info['rewards'] = rewards
        state.info['steps_until_next_command'] -= 1
        state.info['rng'] = rng

        # Command Sampling:
        state.info['command'] = jnp.where(
            state.info['steps_until_next_command'] <= 0,
            self.sample_command(cmd_key),
            state.info['command'],
        )

        # Randomize Command Interval:
        seconds_until_next_command = jax.random.uniform(
            cmd_frequency_key,
            minval=self.command_config.command_frequency[0],
            maxval=self.command_config.command_frequency[1],
        )
        state.info['steps_until_next_command'] = jnp.where(
            done | (state.info['steps_until_next_command'] <= 0),
            jnp.round(
                seconds_until_next_command / self.dt
            ).astype(jnp.int32),
            state.info['steps_until_next_command'],
        )

        # Proxy Metrics:
        state.metrics['total_distance'] = mjx_math.norm(
            data.xpos[self.base_idx - 1],
        )
        state.metrics['swing_peak'] = jnp.mean(
            state.info['swing_peak']
        )
        state.metrics.update(state.info['rewards'])

        done = jnp.float64(done) if jax.config.x64_enabled else jnp.float32(done)

        state = state.replace(
            data=data,
            obs=observation,
            reward=reward,
            done=done,
        )
        return state

    def get_termination(self, data: mjx.Data) -> jax.Array:
        joint_angles = data.qpos[7:]

        termination_contacts = jnp.array([
            data.sensordata[self._mj_model.sensor_adr[sensor_id]] > 0
            for sensor_id in self.termination_contact_sensor
        ])

        done = self.get_upvector(data)[-1] < -0.25
        done |= jnp.any(joint_angles < self.joint_lb)
        done |= jnp.any(joint_angles > self.joint_ub)
        done |= jnp.any(termination_contacts)
        return done

    def get_observation(
        self,
        data: mjx.Data,
        state_info: dict[str, Any],
    ) -> Dict[str, jax.Array]:
        """
            Observation: [
                gyroscope,
                projected_gravity,
                relative_motor_positions,
                motor_velocities,
                previous_action,
                command,
            ]
        """
        q = data.qpos[7:]
        qd = data.qvel[6:]

        # Gyroscope Noise:
        gyroscope = self.get_gyro(data)
        state_info['rng'], noise_key = jax.random.split(state_info['rng'])
        gyroscope_noise = jax.random.uniform(
            noise_key,
            shape=gyroscope.shape,
            minval=-self.noise_config.gyroscope,
            maxval=self.noise_config.gyroscope,
        )
        noisy_angular_rate = gyroscope + gyroscope_noise

        # Gravity noise:
        projected_gravity = self.get_gravity(data)
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
            state_info['command'],                      # 3
        ])

        accelerometer = self.get_accelerometer(data)
        linear_velocity = self.get_local_linear_velocity(data)
        global_angular_velocity = self.get_global_angular_velocity(data)
        actuator_force = data.actuator_force
        feet_velocity = self.get_feet_velocity(data).ravel()

        privileged_observation = jnp.concatenate([
            observation,                                                                                # 45 or 57
            accelerometer,                                                                              # 3
            gyroscope,                                                                                  # 3
            projected_gravity,                                                                          # 3
            linear_velocity,                                                                            # 3
            global_angular_velocity,                                                                    # 3
            q - self.default_pose,                                                                      # 12
            qd,                                                                                         # 12
            actuator_force,                                                                             # 12 or 24
            feet_velocity,                                                                              # 12
            state_info['previous_contact'],                                                             # 4
            state_info['feet_air_time'],                                                                # 4
            state_info['feet_contact_time'],                                                            # 4
            state_info['previous_air_time'],                                                            # 4
            state_info['previous_contact_time'],                                                        # 4
            state_info['swing_peak'],                                                                   # 4
            data.xfrc_applied[self.base_idx, :3],                                             # 3
            jnp.asarray([
                state_info['steps_since_previous_disturbance'] >= state_info['steps_until_next_disturbance']
            ]),                                                                                         # 1
        ])
        # Size: 91 or 115

        return {
            'state': observation,
            'privileged_state': privileged_observation,
        }

    def _reward_tracking_velocity(
        self, commands: jax.Array, local_velocity: jax.Array
    ) -> jax.Array:
        # Tracking of linear velocity commands (xy axes)
        error = jnp.sum(jnp.square(commands[:2] - local_velocity[:2]))
        return jnp.exp(-error / self.kernel_sigma)

    def _reward_tracking_yaw_rate(
        self, commands: jax.Array, x: jax.Array
    ) -> jax.Array:
        # Tracking of angular velocity commands (yaw)
        error = jnp.square(commands[2] - x[2])
        return jnp.exp(-error / self.kernel_sigma)

    def _cost_vertical_velocity(
        self, global_base_linvel: jax.Array
    ) -> jax.Array:
        # Penalize z axis base linear velocity
        return jnp.square(global_base_linvel[2])

    def _cost_angular_velocity(
        self, global_base_angvel: jax.Array,
    ) -> jax.Array:
        # Penalize xy axes base angular velocity
        return jnp.sum(jnp.square(global_base_angvel[:2]))

    def _cost_orientation_regularization(
        self, base_z_axis: jax.Array,
    ) -> jax.Array:
        # Penalize non flat base orientation
        return jnp.sum(jnp.square(base_z_axis[:2]))

    def _cost_pose_regularization(
        self, qpos: jax.Array,
    ) -> jax.Array:
        weight = jnp.array([1.0, 1.0, 0.1] * 4) / 12.0
        error = jnp.sum(jnp.square(qpos - self.default_pose) * weight)
        return jnp.exp(-error)

    def _cost_torques(self, torques: jax.Array) -> jax.Array:
        # Penalize torques
        return jnp.sqrt(jnp.sum(jnp.square(torques))) + jnp.sum(jnp.abs(torques))

    def _cost_action_rate(
        self, action: jax.Array, previous_action: jax.Array
    ) -> jax.Array:
        # Penalize changes in actions
        return jnp.sum(jnp.square(action - previous_action))

    def _cost_mechanical_power(
        self, qd: jax.Array, torques: jax.Array
    ) -> jax.Array:
        # Penalize mechanical power
        return jnp.sum(jnp.abs(torques) * jnp.abs(qd))

    def _cost_acceleration(
        self, qacc: jax.Array,
    ) -> jax.Array:
        # Penalize Motor/Joint Acceleration
        return jnp.sqrt(jnp.sum(jnp.square(qacc)))

    def _cost_stand_still(
        self,
        commands: jax.Array,
        joint_angles: jax.Array,
    ) -> jax.Array:
        # Penalize motion at zero commands
        command_norm = jnp.linalg.norm(commands)
        return jnp.sum(jnp.abs(joint_angles - self.default_pose)) * (command_norm < 0.1)

    # def _reward_air_time(
    #     self,
    #     air_time: jax.Array,
    #     first_contact: jax.Array,
    #     commands: jax.Array,
    # ) -> jax.Array:
    #     # Flight Phase Reward:
    #     command_norm = jnp.linalg.norm(commands)
    #     reward_air_time = jnp.sum((air_time - self.target_air_time) * first_contact)
    #     reward_air_time *= (
    #         command_norm > 0.1
    #     )
    #     return reward_air_time

    def _reward_air_time(
        self,
        air_time: jax.Array,
        contact_time: jax.Array,
        commands: jax.Array,
        body_velocity: jax.Array,
        mode_time: float = 0.3,
        command_threshold: float = 0.0,
        velocity_threshold: float = 0.5,
    ) -> jax.Array:
        # Calculate Mode Timing Reward
        t_max = jnp.maximum(air_time, contact_time)
        t_min = jnp.clip(t_max, max=mode_time)
        stance_reward = jnp.clip(contact_time - air_time, -mode_time, mode_time)
        # Command and Body Velocity:
        command_norm = jnp.linalg.norm(commands)
        velocity_norm = jnp.linalg.norm(body_velocity)
        # Reward:
        reward = jnp.where(
            (command_norm > command_threshold) | (velocity_norm > velocity_threshold),
            jnp.where(t_max < mode_time, t_min, 0.0),
            stance_reward,
        )
        return jnp.sum(reward)

    def _cost_gait_variance(
        self,
        previous_air_time: jax.Array,
        previous_contact_time: jax.Array,
    ) -> jax.Array:
        # Penalize variance in gait timing
        air_time_variance = jnp.var(
            jnp.clip(previous_air_time, max=0.5),
        )
        contact_time_variance = jnp.var(
            jnp.clip(previous_contact_time, max=0.5),
        )
        return air_time_variance + contact_time_variance

    def _reward_foot_clearance(
        self,
        data: mjx.Data,
        target_foot_height: float = 0.1,
        velocity_scale: float = 2.0,
        sigma: float = 0.05,
    ) -> jax.Array:
        foot_position = data.site_xpos[self.feet_site_idx]
        foot_height = foot_position[..., -1]
        foot_error = jnp.square(foot_height - target_foot_height)
        foot_velocity = self.get_feet_velocity(data)[..., :2]
        foot_velocity_norm = jnp.linalg.norm(foot_velocity)
        foot_velocity_tanh = jnp.tanh(velocity_scale * foot_velocity_norm)
        error = jnp.sum(foot_error * foot_velocity_tanh)
        return jnp.exp(-error / sigma)

    def _cost_foot_slip(
        self,
        data: mjx.Data,
        contact: jax.Array,
        commands: jax.Array,
    ) -> jax.Array:
        # Penalize foot slip
        command_norm = jnp.linalg.norm(commands)
        foot_velocity = self.get_feet_velocity(data)
        foot_velocity_xy = foot_velocity[..., :2]
        velocity_xy_sq = jnp.sum(jnp.square(foot_velocity_xy), axis=-1)
        return jnp.sum(velocity_xy_sq * contact) * (command_norm > 0.1)

    # def _cost_foot_slip(
    #     self,
    #     data: base.State,
    #     target_foot_height: float = 0.1,
    #     decay_rate: float = 0.95,
    # ) -> jax.Array:
    #     # Penalizes foot slip velocity at contact to encourage ground speed matching.
    #     if not (0.0 < decay_rate <= 1.0):
    #         raise ValueError("Decay rate must be between 0 and 1.")

    #     # Foot velocities and foot heights
    #     foot_velocity = self.get_feet_velocity(data)
    #     foot_velocity_xy = foot_velocity[..., :2]
    #     foot_position = data.site_xpos[self.feet_site_idx]
    #     foot_height = foot_position[..., -1]

    #     # Calculate velocity of each foot relative to the base
    #     velocity_xy_sq = jnp.sum(jnp.square(foot_velocity_xy), axis=-1)

    #     # Calculate scale factor to smoothly increase penalty as foot approaches target height
    #     scale_factor = -target_foot_height / jnp.log(1.0 - decay_rate)
    #     height_gate = jnp.exp(-foot_height / scale_factor)

    #     return jnp.sum(velocity_xy_sq * height_gate)

    def _cost_unwanted_contact(
        self,
        unwanted_contacts: jax.Array,
    ) -> jax.Array:
        # Unwanted Contact Penalty
        return jnp.sum(unwanted_contacts)

    def _cost_termination(self, done: jax.Array) -> jax.Array:
        return done

    def _cost_position_rate(
        self,
        joint_angles: jax.Array,
        previous_joint_angles: jax.Array,
    ) -> jax.Array:
        # Penalize large fast joint position changes
        return jnp.sum(jnp.square(joint_angles - previous_joint_angles))

    # Adapted from mujoco_playground:
    def maybe_apply_perturbation(self, state: mjx_env.State) -> mjx_env.State:
        def gen_dir(rng: jax.Array) -> jax.Array:
            angle = jax.random.uniform(rng, minval=0.0, maxval=jnp.pi * 2)
            return jnp.array([jnp.cos(angle), jnp.sin(angle), 0.0])

        def apply_perturbation(state: mjx_env.State) -> mjx_env.State:
            t = state.info["disturbance_step"] * self.dt
            u_t = 0.5 * jnp.sin(jnp.pi * t / state.info["disturbance_duration"])
            # kg * m/s * 1/s = m/s^2 = kg * m/s^2 (N).
            force = (
                u_t  # (unitless)
                * self.base_link_mass  # kg
                * state.info["disturbance_magnitude"]  # m/s
                / state.info["disturbance_duration"]  # 1/s
            )
            xfrc_applied = jnp.zeros((self._mj_model.nbody, 6))
            xfrc_applied = xfrc_applied.at[self.base_idx, :3].set(
                force * state.info["disturbance_direction"]
            )
            data = state.data.replace(xfrc_applied=xfrc_applied)
            state = state.replace(data=data)
            state.info["steps_since_previous_disturbance"] = jnp.where(
                state.info["disturbance_step"] >= state.info["disturbance_duration_steps"],
                0,
                state.info["steps_since_previous_disturbance"],
            )
            state.info["disturbance_step"] += 1
            return state

        def wait(state: mjx_env.State) -> mjx_env.State:
            state.info["rng"], rng = jax.random.split(state.info["rng"])
            state.info["steps_since_previous_disturbance"] += 1
            xfrc_applied = jnp.zeros((self._mj_model.nbody, 6))
            data = state.data.replace(xfrc_applied=xfrc_applied)
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
            return state.replace(data=data)

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
        command: np.ndarray,
        previous_action: np.ndarray,
        add_noise: bool = True,
    ) -> Dict[str, np.ndarray]:
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
            q - self.default_ctrl,
            qd,
            previous_action,
            command,
        ])

        return {
            'state': observation,
            'privileged_state': np.zeros((self.num_privileged_observations,)),
        }


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
        states.append(state.data)

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
