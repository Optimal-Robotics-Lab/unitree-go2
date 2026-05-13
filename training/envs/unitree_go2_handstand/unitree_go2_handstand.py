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

from training.envs.unitree_go2_handstand import base
from training.envs.unitree_go2_handstand.config import (
    RewardConfig,
    NoiseConfig,
    DisturbanceConfig,
    CommandConfig,
    EnvironmentConfig,
)

# Types:
PRNGKey: TypeAlias = jax.Array


class Handstand(base.UnitreeGo2Env):
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

        # Initial Velocity:
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
            naconmax=self.environment_config.nconmax,
            naccdmax=self.environment_config.naccdmax,
            njmax=self.environment_config.njmax,
        )

        if self.environment_config.impl == 'warp':
            data = self._to_f32(data)

        data = mjx.forward(self._mjx_model, data)

        if self.environment_config.impl == 'warp':
            data = self._to_f64(data)

        # Initialize Filter:
        filter_state = self.filter.init()

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
            'rewards': {k: 0.0 for k in self.reward_config.keys()},
            'steps_until_next_disturbance': steps_until_next_disturbance,
            'disturbance_duration': disturbance_duration,
            'disturbance_duration_steps': disturbance_duration_steps,
            'steps_since_previous_disturbance': 0,
            'disturbance_step': 0,
            'disturbance_magnitude': disturbance_magnitude,
            'disturbance_direction': jnp.array([0.0, 0.0, 0.0]),
            'filter_state': filter_state,
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
        # if self.disturbance_config.magnitudes[1] > 0.0:
        #     state = self.maybe_apply_perturbation(state)

        # Apply Action Filter:
        filtered_action, filter_state = self.filter.apply(
            action, state.info['filter_state']
        )
        state.info['filter_state'] = filter_state

        # Physics step:
        data = self._step(state.data, filtered_action)

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
        termination_contacts = jnp.array([
            data.sensordata[self._mj_model.sensor_adr[sensor_id]] > 0
            for sensor_id in self.termination_contact_sensor
        ])

        # Body Velocity:
        global_body_velocity = self.get_global_linear_velocity(data)
        global_angular_velocity = self.get_global_angular_velocity(data)
        local_body_velocity = self.get_local_linear_velocity(data)

        # Forward Vector:
        forward_vector = self.get_forwardvector(data)

        # Observation data:
        observation = self.get_observation(
            data,
            state.info,
        )

        # Termination:
        done = self._get_termination(
            data,
            feet_contacts,
            termination_contacts,
            unwanted_contacts,
            terminate_on_unwanted_contacts=False,
        )

        # Rewards:
        rewards = {
            'tracking_orientation': (
                self._reward_tracking_orientation(forward_vector, self.orientation_sigma)
            ),
            'tracking_pose': (
                self._reward_tracking_joint_pose(joint_angles, self.pose_sigma)
            ),
            'orientation_regularization': self._cost_orientation_regularization(
                forward_vector,
            ),
            'torque': self._cost_torques(data.actuator_force),
            'action_rate': self._cost_action_rate(action, state.info['previous_action']),
            'acceleration': self._cost_acceleration(
                data.qacc,
            ),
            'stand_still': self._cost_stand_still(
                global_body_velocity,
                global_angular_velocity,
            ),
            'feet_contact': self._cost_feet_contact(
                feet_contacts,
            ),
            'foot_slip': self._cost_foot_slip(
                data,
                feet_contacts,
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
        state.metrics.update(state.info['rewards'])

        done = jnp.float64(done) if jax.config.x64_enabled else jnp.float32(done)

        state = state.replace(
            data=data,
            obs=observation,
            reward=reward,
            done=done,
        )
        return state

    def _get_termination(
        self,
        data: mjx.Data,
        feet_contacts: jax.Array,
        termination_contacts: jax.Array,
        unwanted_contacts: jax.Array,
        terminate_on_unwanted_contacts: bool = False,
    ) -> jax.Array:
        # Termination Condition:
        done = jnp.any(termination_contacts)
        done |= terminate_on_unwanted_contacts * jnp.any(unwanted_contacts)
        done |= (data.time >= 2.0) & jnp.any(feet_contacts[:2])
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
                filter_observation,
            ]
        """
        q = data.qpos[7:]
        qd = data.qvel[6:]

        # Linear Velocity:
        linear_velocity = self.get_local_linear_velocity(data)

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

        # Filter State:
        filter_observation = self.filter.get_observation(state_info['filter_state'])

        observation = jnp.concatenate([
            noisy_angular_rate,                         # 3
            noisy_projected_gravity,                    # 3
            noisy_joint_positions - self.default_pose,  # 12
            noisy_joint_velocities,                     # 12
            state_info['previous_action'],              # 12 or 24
            filter_observation,                         # Based on filter
        ])

        accelerometer = self.get_accelerometer(data)
        global_angular_velocity = self.get_global_angular_velocity(data)
        actuator_force = data.actuator_force
        feet_velocity = self.get_feet_velocity(data).ravel()

        privileged_observation = jnp.concatenate([
            observation,                                                                                
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
            data.xfrc_applied[self.base_idx, :3],                                                       # 3
            jnp.asarray([
                state_info['steps_since_previous_disturbance'] >= state_info['steps_until_next_disturbance']
            ]),                                                                                         # 1
        ])

        return {
            'state': observation,
            'privileged_state': privileged_observation,
        }

    def _reward_tracking_orientation(
        self, forward_vector: jax.Array, up_vec: jax.Array
    ) -> jax.Array:
        cos_dist = jnp.dot(forward_vector, self.tracking_vector)
        normalized = 0.5 * cos_dist + 0.5
        return jnp.square(normalized)
        
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

    def _cost_orientation_regularization(
        self, forward_vector: jax.Array,
    ) -> jax.Array:
        # Orientation Penalty:
        dot_product = jnp.clip(
            jnp.dot(forward_vector, self.tracking_vector),
            -1.0,
            1.0,
        )
        error = jnp.square(dot_product - 1.0)
        return error

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

    def _cost_stand_still(
        self,
        global_base_linear_velocity: jax.Array,
        global_base_angular_velocity: jax.Array,
    ) -> jax.Array:
        # Penalize Base Velocity:
        linear_xy_error = jnp.sum(jnp.square(global_base_linear_velocity[:2]))
        angular_z_error = jnp.sum(jnp.square(global_base_angular_velocity[-1]))
        return linear_xy_error + angular_z_error

    def _cost_feet_contact(
        self,
        contact: jax.Array,
    ) -> jax.Array:
        # Reward Correct Feet Contact and Penalize Incorrect Feet Contact
        correct_contact = jnp.sum(jnp.array([0, 0, 1, 1]) * contact)
        incorrect_contact = jnp.sum(
            jnp.array([1, 1, 0, 0]) * contact       # Front Feet in Contact
            + jnp.array([0, 0, 1, 1]) * ~contact    # Hind Feet not in Contact
        )
        reward = (correct_contact - incorrect_contact) / 2.0
        return reward

    # Could be a possible reward term to try:
    # def _reward_feet_forces(
    #     self,
    #     foot_forces: jax.Array,
    #     force_kernel_sigma: float = 0.25,
    # ) -> jax.Array:
    #     # Penalize front feet forces:
    #     front_force_error = jnp.sum(jnp.square(foot_forces[:2]))

    #     # Reward maintaining a target force on the hind feet:
    #     target_force = (self.robot_mass * 9.81) / 2.0
    #     hind_force_error = jnp.sum(jnp.square(foot_forces[2:] - target_force))

    #     total_error = front_force_error + hind_force_error
    #     return jnp.exp(-total_error / force_kernel_sigma)

    # def _reward_front_clearance(
    #     self,
    #     data: mjx.Data,
    #     target_foot_height: float = 0.25,
    #     clearance_kernel_sigma: float = 0.1,
    # ) -> jax.Array:
    #     # Penalize the front feet for being below the target height:
    #     foot_position = data.site_xpos[self.feet_site_idx]
    #     foot_height = jnp.minimum(foot_position[..., -1], target_foot_height)[..., :2]
    #     foot_error = jnp.sum(jnp.square(foot_height - target_foot_height))
    #     return jnp.exp(-foot_error / clearance_kernel_sigma)

    def _cost_foot_slip(
        self,
        data: mjx.Data,
        contact: jax.Array,
    ) -> jax.Array:
        # Penalize foot slip
        foot_velocity = self.get_feet_velocity(data)
        foot_velocity_xy = foot_velocity[..., :2]
        velocity_xy_sq = jnp.sum(jnp.square(foot_velocity_xy), axis=-1)
        return jnp.sum(velocity_xy_sq * contact)

    def _cost_unwanted_contact(
        self,
        unwanted_contacts: jax.Array,
    ) -> jax.Array:
        # Unwanted Contact Penalty
        return jnp.sum(unwanted_contacts)

    def _cost_termination(self, done: jax.Array) -> jax.Array:
        return done

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
                * self.robot_mass  # kg
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
