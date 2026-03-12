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

from training.envs.unitree_go2_minimal import base
from training.envs.unitree_go2_minimal.config import (
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
        ctrl = jnp.pad(
            qpos[7:],
            (0, self.nu - qpos[7:].shape[0]),
            mode='constant',
            constant_values=0,
        )

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
        feet_velocity = self.get_feet_velocity(data)

        state_info = {
            'rng': rng,
            'previous_action': jnp.zeros(self.nu),
            'previous_joint_positions': jnp.zeros(self.num_joints),
            'previous_velocity': jnp.zeros(self.num_joints),
            'previous_foot_velocity': feet_velocity,
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
            # New Power Terms:
            'velocity_ema': 0.0,
            'power_ema': 80.0,
            # Curriculum Terms:
            'global_step': jnp.zeros((), dtype=jnp.int32),
            'curriculum_fn_result': jnp.zeros((), dtype=jnp.float32),
        }

        # Observation Initialization:
        observation = self.get_observation(
            data, feet_contacts, state_info,
        )

        reward, done = jnp.zeros(2)
        done = jnp.float64(done) if jax.config.x64_enabled else jnp.float32(done)

        metrics = {}
        for k in state_info['rewards']:
            metrics[k] = state_info['rewards'][k]
        metrics['total_distance'] = 0.0
        metrics['swing_peak'] = jnp.zeros(())

        # Power Metrics:
        metrics['energy/power_ema'] = 0.0
        metrics['energy/total_power'] = 0.0
        metrics['energy/positive_mechanical_power'] = 0.0
        metrics['energy/negative_mechanical_power'] = 0.0
        metrics['energy/thermal_power'] = 0.0
        metrics['energy/static_power'] = 80.0
        metrics['energy/gravitational_power'] = 0.0
        metrics['energy/swing_power'] = 0.0

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
        data = self._step(state.data, action)

        imu_height = data.site_xpos[self.imu_site_idx][2]
        joint_angles = data.qpos[7:]
        joint_velocities = data.qvel[6:]

        # Sensor Contacts:
        feet_contacts = jnp.array([
            data.sensordata[self._mj_model.sensor_adr[sensor_id]] > 0
            for sensor_id in self.feet_contact_sensor
        ])
        foot_forces = jnp.array([
            data.sensordata[self._mj_model.sensor_adr[sensor_id] + 2]
            for sensor_id in self.feet_force_sensor
        ])
        unwanted_contacts = jnp.array([
            data.sensordata[self._mj_model.sensor_adr[sensor_id]] > 0
            for sensor_id in self.unwanted_contact_sensor
        ])

        # Feet Air and Contact Time:
        feet_velocity = self.get_feet_velocity(data)
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
        local_angular_velocity = self.get_gyro(data)

        # Power Calculations:
        power_metrics = self._compute_power_components(
            global_body_velocity,
            data.actuator_force,
            joint_velocities,
            feet_velocity,
            feet_contacts,
        )

        # Update EMAs:
        body_velocity = jnp.concatenate([local_body_velocity, local_angular_velocity])
        state.info['velocity_ema'] = self._update_velocity_ema(
            state.info['velocity_ema'],
            state.info['command'],
            body_velocity,
        )
        state.info['power_ema'] = self._update_power_ema(
            state.info['power_ema'],
            power_metrics['total_power'],
        )

        # Observation data:
        observation = self.get_observation(
            data,
            feet_contacts,
            state.info,
        )

        # Termination:
        done = self.get_termination(data, state.info)

        # Curriculum Reward Shaping:
        curriculum_factor = state.info['curriculum_fn_result']

        # Rewards:
        rewards = {
            # Tracking Rewards:
            'tracking_linear_velocity': (
                self._reward_tracking_velocity(state.info['command'], local_body_velocity)
            ),
            'tracking_angular_velocity': (
                self._reward_tracking_yaw_rate(state.info['command'], self.get_gyro(data))
            ),
            # Power Costs:
            'electrical_power': self._cost_electrical_power(
                power_metrics['positive_mechanical_power'],
                power_metrics['negative_mechanical_power'],
                power_metrics['thermal_power'],
            ) * curriculum_factor,
            'gravitational_power': self._cost_gravitational_power(
                power_metrics['gravitational_power'],
            ) * curriculum_factor,
            # Energy and Power Costs:
            'energy': self._cost_energy(
                state.info['power_ema'],
            ) * curriculum_factor,
            'action_rate': self._cost_action_rate(action, state.info['previous_action']) * curriculum_factor,
            'acceleration': self._cost_acceleration(
                data.qacc,
            ) * curriculum_factor,
            # Gait Costs:
            'impact': self._cost_impact(
                feet_contacts,
                state.info['previous_contact'],
                state.info['previous_foot_velocity'],
            ) * curriculum_factor,
            'foot_slip': self._cost_foot_slip(
                data,
                target_foot_height=0.05,
            ) * curriculum_factor,
            # Miscellaneous Costs:
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
        state.info['previous_foot_velocity'] = feet_velocity
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
            data.xpos[self.base_idx],
        )
        state.metrics['swing_peak'] = jnp.mean(
            state.info['swing_peak']
        )

        # Power Metrics:
        state.metrics['energy/power_ema'] = state.info['power_ema']
        state.metrics['energy/total_power'] = power_metrics['total_power']
        state.metrics['energy/positive_mechanical_power'] = power_metrics['positive_mechanical_power']
        state.metrics['energy/negative_mechanical_power'] = power_metrics['negative_mechanical_power']
        state.metrics['energy/thermal_power'] = power_metrics['thermal_power']
        state.metrics['energy/static_power'] = power_metrics['static_power']
        state.metrics['energy/gravitational_power'] = power_metrics['gravitational_power']
        state.metrics['energy/swing_power'] = power_metrics['swing_power']

        state.metrics.update(state.info['rewards'])

        done = jnp.float64(done) if jax.config.x64_enabled else jnp.float32(done)

        state = state.replace(
            data=data,
            obs=observation,
            reward=reward,
            done=done,
        )
        return state

    def get_termination(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        joint_angles = data.qpos[7:]

        termination_contacts = jnp.array([
            data.sensordata[self._mj_model.sensor_adr[sensor_id]] > 0
            for sensor_id in self.termination_contact_sensor
        ])

        max_power_budget = 800.0
        thermal_trip = info['power_ema'] > max_power_budget

        done = self.get_upvector(data)[-1] < -0.25
        done |= jnp.any(joint_angles < self.joint_lb)
        done |= jnp.any(joint_angles > self.joint_ub)
        done |= jnp.any(termination_contacts)
        done |= thermal_trip
        return done

    def get_observation(
        self,
        data: mjx.Data,
        contacts: jax.Array,
        state_info: dict[str, Any],
    ) -> Dict[str, jax.Array]:
        """
            Observation: [
                gyroscope,
                projected_gravity,
                relative_motor_positions,
                motor_velocities,
                contacts,
                previous_action,
                command,
            ]
        """
        q = data.qpos[7:]
        qd = data.qvel[6:]

        # Linear Velocity:
        linear_velocity = self.get_local_linear_velocity(data)
        state_info['rng'], noise_key = jax.random.split(state_info['rng'])
        linear_velocity_noise = jax.random.uniform(
            noise_key,
            shape=linear_velocity.shape,
            minval=-self.noise_config.linear_velocity,
            maxval=self.noise_config.linear_velocity,
        )
        noisy_linear_velocity = linear_velocity + linear_velocity_noise

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

        # Feet Contacts:
        state_info['rng'], noise_key = jax.random.split(state_info['rng'])
        dropout_mask = jax.random.bernoulli(
            noise_key,
            p=self.noise_config.contact_dropout,
            shape=(4,)
        )
        noisy_feet_contacts = contacts * dropout_mask

        observation = jnp.concatenate([
            noisy_linear_velocity,                      # 3
            noisy_angular_rate,                         # 3
            noisy_projected_gravity,                    # 3
            noisy_joint_positions - self.default_pose,  # 12
            noisy_joint_velocities,                     # 12
            noisy_feet_contacts,                        # 4
            state_info['previous_action'],              # 12 or 24
            state_info['command'],                      # 3
        ])

        accelerometer = self.get_accelerometer(data)
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
            data.xfrc_applied[self.base_idx, :3],                                                       # 3
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

    def _reward_cost_of_transport(
        self,
        velocity_ema: jax.Array,
        power_ema: jax.Array,
    ) -> jax.Array:
        # Cost of Transport:
        denominator = self.total_mass * 9.81 * jnp.maximum(velocity_ema, 1e-3)
        cot = power_ema / denominator
        return jnp.exp(-cot / 2.0)

    def _penalty_cost_of_transport(
        self,
        velocity_ema: jax.Array,
        power_ema: jax.Array,
    ) -> jax.Array:
        # Cost of Transport:
        denominator = self.total_mass * 9.81 * jnp.maximum(velocity_ema, 1e-3)
        return power_ema / denominator

    def _cost_energy(
        self,
        power_ema: jax.Array,
    ) -> jax.Array:
        # Penalize high power consumption to encourage energy efficiency.
        limit = 300.0
        overdraw = jnp.maximum(power_ema - limit, 0.0)
        return jnp.square(overdraw)

    def _cost_electrical_power(
        self,
        positive_mechanical_power: jax.Array | float,
        negative_mechanical_power: jax.Array | float,
        thermal_power: jax.Array | float,
    ) -> jax.Array | float:
        electrical_power = positive_mechanical_power + (negative_mechanical_power * self.motor_config.regen_efficiency) + thermal_power
        return electrical_power

    def _cost_mechanical_power(
        self,
        positive_mechanical_power: jax.Array | float,
        negative_mechanical_power: jax.Array | float,
    ) -> jax.Array | float:
        mechanical_power = positive_mechanical_power + (negative_mechanical_power * self.motor_config.regen_efficiency)
        return mechanical_power

    def _cost_thermal_power(
        self,
        thermal_power: jax.Array | float,
    ) -> jax.Array | float:
        return thermal_power

    def _cost_gravitational_power(
        self,
        gravitational_power: jax.Array | float,
    ) -> jax.Array | float:
        return gravitational_power

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

    def _cost_impact(
        self,
        contacts: jax.Array,
        previous_contacts: jax.Array,
        previous_foot_velocities: jax.Array,
    ) -> jax.Array:
        # Penalize high foot impact velocities
        just_landed = (contacts == 1.0) & (previous_contacts == 0.0)
        return jnp.sum(jnp.square(previous_foot_velocities[..., 2]) * just_landed)

    def _cost_foot_slip(
        self,
        data: mjx.Data,
        target_foot_height: float = 0.1,
        decay_rate: float = 0.95,
    ) -> jax.Array:
        # Penalizes foot slip velocity at contact to encourage ground speed matching.
        if not (0.0 < decay_rate <= 1.0):
            raise ValueError("Decay rate must be between 0 and 1.")

        # Foot velocities and foot heights
        foot_velocity = self.get_feet_velocity(data)
        foot_velocity_xy = foot_velocity[..., :2]
        foot_position = data.site_xpos[self.feet_site_idx]
        foot_height = foot_position[..., -1]

        # Calculate velocity of each foot relative to the base
        velocity_xy_sq = jnp.sum(jnp.square(foot_velocity_xy), axis=-1)

        # Calculate scale factor to smoothly increase penalty as foot approaches target height
        scale_factor = -target_foot_height / jnp.log(1.0 - decay_rate)
        height_gate = jnp.exp(-foot_height / scale_factor)

        return jnp.sum(velocity_xy_sq * height_gate)

    def _cost_unwanted_contact(
        self,
        unwanted_contacts: jax.Array,
    ) -> jax.Array:
        # Unwanted Contact Penalty
        return jnp.sum(unwanted_contacts)

    def _cost_termination(self, done: jax.Array) -> jax.Array:
        return done

    def _compute_power_components(
        self,
        global_body_velocity: jax.Array,
        torques: jax.Array,
        joint_velocities: jax.Array,
        foot_velocities: jax.Array,
        feet_contacts: jax.Array,
    ) -> dict[str, jax.Array | float]:
        # Electical and Mechanical Power Calculations:
        motor_torques = torques / self.motor_config.reduction_ratio
        current = jnp.abs(motor_torques) / self.motor_config.kt

        # Thermal Power
        thermal_power = jnp.sum(jnp.square(current) * self.motor_config.resistance)

        # Mechanical Power
        mechanical_power = torques * joint_velocities
        positive_mechanical_power = jnp.sum(jnp.maximum(mechanical_power, 0.0))
        negative_mechanical_power = jnp.sum(jnp.minimum(mechanical_power, 0.0))

        # Total Electrical Power:
        electrical_power = positive_mechanical_power + (negative_mechanical_power * self.motor_config.regen_efficiency) + thermal_power

        # Metabolic Power:
        static_power = 80.0

        # Gravitational Power:
        gravitational_power = self.total_mass * 9.81 * jnp.abs(global_body_velocity[2])

        # Biomechanical Swing Power (Kinetic penalty for leg swinging)
        is_swing = ~feet_contacts
        relative_foot_velocities = foot_velocities - global_body_velocity
        velocity_xy_sq = jnp.sum(jnp.square(relative_foot_velocities[..., :2]), axis=-1)
        swing_power = self.leg_mass * jnp.sum(velocity_xy_sq * is_swing)

        total_power = electrical_power + static_power + gravitational_power + swing_power

        return {
            'electrical_power': electrical_power,
            'thermal_power': thermal_power,
            'positive_mechanical_power': positive_mechanical_power,
            'negative_mechanical_power': negative_mechanical_power,
            'static_power': static_power,
            'gravitational_power': gravitational_power,
            'swing_power': swing_power,
            'total_power': total_power,
        }

    # EMA Calculations for Cost of Transport:
    def _update_velocity_ema(
        self,
        velocity_ema: jax.Array,
        commands: jax.Array,
        base_qvel: jax.Array,
        alpha: float = 0.1,
    ) -> jax.Array:
        # Base Velocity:
        velocity_xy = base_qvel[:2]
        yaw = base_qvel[-1]

        # Calculate Command Direction:
        command_xy = commands[:2]
        command_norm = jnp.maximum(jnp.linalg.norm(commands), 1e-6)
        command_direction = command_xy / command_norm

        # Project velocity onto command direction:
        velocity_xy_projected = jnp.dot(velocity_xy, command_direction) * (command_norm > 0.1)
        velocity_xy_projected = jnp.maximum(velocity_xy_projected, 0.0)

        # Project yaw onto command direction:
        base_to_foot_radius = 0.24
        command_yaw = commands[2]
        yaw = yaw * jnp.sign(command_yaw)
        yaw_projected = base_to_foot_radius * jnp.maximum(yaw, 0.0)

        # Velocity for Cost of Transport:
        velocity = velocity_xy_projected + yaw_projected

        # Exponential Moving Average for Velocity:
        velocity_ema = alpha * velocity + (1 - alpha) * velocity_ema

        return velocity_ema

    def _update_power_ema(
        self,
        power_ema: jax.Array | float,
        power: jax.Array | float,
        alpha: float = 0.01,
    ) -> jax.Array | float:
        # Exponential Moving Average for Power:
        power_ema = alpha * power + (1 - alpha) * power_ema
        return power_ema

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
