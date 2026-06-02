"""
    Unitree Go2 Environment:
"""

from typing import Any, Dict, TypeAlias

import jax
import jax.numpy as jnp

import flax
import flax.serialization

import numpy as np

import mujoco
from mujoco import mjx
from mujoco.mjx._src import math as mjx_math
from mujoco_playground._src import mjx_env
import scipy

from training.envs.unitree_go2_backflip import base
from training.envs.unitree_go2_backflip.config import (
    RewardConfig,
    NoiseConfig,
    DisturbanceConfig,
    CommandConfig,
    EnvironmentConfig,
)

# Types:
PRNGKey: TypeAlias = jax.Array


class Backflip(base.UnitreeGo2Env):
    """Environment for training the Unitree Go2 quadruped."""

    def __init__(
        self,
        reward_config: RewardConfig = RewardConfig(),
        environment_config: EnvironmentConfig = EnvironmentConfig(),
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

        # Task Specific Constraints:
        if self.command_config.command_frequency[0] < 3.0:
            raise ValueError("Minimum command frequency must be at least 3.0 seconds for backflip environment.")

        # Task Specific Reward Implementation:
        self.height_sigma = reward_config.height_sigma
        self.brake_sigma = reward_config.brake_sigma
        reward_config_dict = flax.serialization.to_state_dict(reward_config)
        del reward_config_dict['height_sigma']
        del reward_config_dict['brake_sigma']
        self.reward_config = reward_config_dict
        
        # Task Specific Implementation Details: Generate Reference Frames
        # Backflip Time: 0.8 - 1.2s
        self.flip_duration_s = 0.8
        self.num_phase_steps = int(self.flip_duration_s / self.dt)
        self.phase_step_lookahead = 25
        
        # 1.2s Backflip Frames:
        # self.phase_frames = {
        #     'start': 0.0,
        #     'crouch': 0.1,
        #     'liftoff': 0.2,
        #     'apex': 0.6,
        #     'end': 1.0,
        # }
        # self.height_frames = {
        #     'start': 0.3,
        #     'crouch': 0.2,
        #     'liftoff': 0.4,
        #     'apex': 0.75,
        #     'end': 0.3,
        # }
        # self.pitch_frames = {
        #     'start': 0.0,
        #     'crouch': jnp.pi / 8,
        #     'liftoff': -jnp.pi / 4,
        #     'apex': -jnp.pi,
        #     'end': -2 * jnp.pi,
        # }

        # 0.8s Backflip Frames:
        self.phase_frames = {
            'start': 0.0,
            'crouch': 0.15,
            'liftoff': 0.30,
            'apex': 0.65,
            'end': 1.0,
        }
        self.height_frames = {
            'start': 0.3,
            'crouch': 0.2,
            'liftoff': 0.4,
            'apex': 0.78, 
            'end': 0.3,
        }
        self.pitch_frames = {
            'start': 0.0,
            'crouch': jnp.pi / 8, 
            'liftoff': -jnp.pi / 4,
            'apex': -jnp.pi,
            'end': -2 * jnp.pi,
        }

        assert all(k in self.phase_frames for k in self.height_frames)
        assert all(k in self.phase_frames for k in self.pitch_frames)
        
        self.height_reference_fn = scipy.interpolate.CubicSpline(
            x=list(self.phase_frames.values()),
            y=list(self.height_frames.values()),
            bc_type='clamped',
        )
        self.pitch_reference_fn = scipy.interpolate.CubicSpline(
            x=list(self.phase_frames.values()),
            y=list(self.pitch_frames.values()),
            bc_type='clamped',
        )
        self.pitch_rate_reference_fn = self.pitch_reference_fn.derivative()

        self.phase_reference = jnp.linspace(0.0, 1.0, self.num_phase_steps + 1)
        self.height_reference = jnp.asarray(self.height_reference_fn(self.phase_reference))
        self.pitch_reference = jnp.asarray(self.pitch_reference_fn(self.phase_reference))
        self.pitch_rate_reference = jnp.asarray(self.pitch_rate_reference_fn(self.phase_reference)) / self.flip_duration_s

        # Task Contacts:
        contact_geom_names = [
            # Lidar:
            'lidar_collision',
            # Front Legs:
            'front_right_foot_collision',
            'front_left_foot_collision',
            'front_right_calf_lower_collision',
            'front_left_calf_lower_collision',
            'front_right_hip_collision',
            'front_left_hip_collision',
            # Hind Legs:
            'hind_right_foot_collision',
            'hind_left_foot_collision',
            'hind_right_calf_lower_collision',
            'hind_left_calf_lower_collision',
            'hind_right_hip_collision',
            'hind_left_hip_collision',
        ]
        
        self.contact_geom_idx = jnp.array([
            self._mj_model.geom(name).id for name in contact_geom_names
        ])

        self.hip_contact_geom_idx = jnp.array([
            self._mj_model.geom('hind_right_hip_collision').id,
            self._mj_model.geom('hind_left_hip_collision').id,
        ])

        # Task Specific Observation Details:
        self.num_observations = 31 + self.nu + self.filter.observation_size
        self.num_privileged_observations = self.num_observations + 48 + self.nu + (2 * self.phase_step_lookahead) + len(self.contact_geom_idx)
        
        # State Observation Mask:
        state_mask = jnp.concatenate([
            jnp.ones(3, dtype=bool),
            jnp.ones(3, dtype=bool),
            jnp.ones(12, dtype=bool),
            jnp.ones(12, dtype=bool),
            jnp.ones(12, dtype=bool),
            jnp.zeros(1, dtype=bool),
            jnp.ones(self.filter.observation_size, dtype=bool),
        ])

        # Privileged Observation Mask:
        privileged_mask = jnp.concatenate([
            state_mask,
            jnp.ones(3, dtype=bool),
            jnp.ones(3, dtype=bool),
            jnp.ones(3, dtype=bool),
            jnp.ones(3, dtype=bool),
            jnp.ones(3, dtype=bool),
            jnp.ones(1, dtype=bool),
            jnp.ones(12, dtype=bool),
            jnp.ones(12, dtype=bool),
            jnp.ones(self.nu, dtype=bool),
            jnp.zeros(4, dtype=bool),
            jnp.ones(3, dtype=bool),
            jnp.ones(self.phase_step_lookahead, dtype=bool),
            jnp.ones(self.phase_step_lookahead, dtype=bool),
            jnp.zeros(len(self.contact_geom_idx), dtype=bool),
            jnp.zeros(1, dtype=bool),
        ])

        assert state_mask.shape[0] == self.num_observations, f"State mask length {state_mask.shape[0]} does not match number of observations {self.num_observations}."
        assert privileged_mask.shape[0] == self.num_privileged_observations, f"Privileged observation mask length {privileged_mask.shape[0]} does not match number of privileged observations {self.num_privileged_observations}."

        self.observation_mask = {
            'state': state_mask,
            'privileged_state': privileged_mask,
        }


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

        data = mjx.forward(self._mjx_model, data)

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

        # Command Sampling: (Time for Consecutive Flips)
        rng, command_frequency_key = jax.random.split(rng)
        seconds_until_next_command = jax.random.uniform(
            command_frequency_key,
            minval=self.command_config.command_frequency[0],
            maxval=self.command_config.command_frequency[1],
        )
        steps_until_next_command = jnp.round(
            seconds_until_next_command / self.dt
        ).astype(jnp.int32)

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
            'phase_step': jnp.int32(0),
            'flip_done': False,
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
        rng, cmd_frequency_key = jax.random.split(state.info['rng'], 2)

        # Disturbance: (Force based)
        if self.disturbance_config.magnitudes[1] > 0.0:
            state = self.maybe_apply_perturbation(state)

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
        self_collision_contacts = jnp.array([
            data.sensordata[self._mj_model.sensor_adr[sensor_id]] > 0
            for sensor_id in self.self_collision_contact_sensor
        ])
        termination_contacts = jnp.array([
            data.sensordata[self._mj_model.sensor_adr[sensor_id]] > 0
            for sensor_id in self.termination_contact_sensor
        ])

        # Body Velocity and Orientation:
        global_body_velocity = self.get_global_linear_velocity(data)
        global_angular_velocity = self.get_global_angular_velocity(data)
        local_body_velocity = self.get_local_linear_velocity(data)
        local_angular_velocity = self.get_gyro(data)
        projected_gravity = self.get_gravity(data)
        up_vector = self.get_upvector(data)

        # Observation data:
        observation = self.get_observation(
            data,
            state.info,
        )

        # Termination:
        unwanted_contacts = jnp.concatenate([unwanted_contacts, self_collision_contacts])
        done = self._get_termination(
            data,
            termination_contacts,
            unwanted_contacts,
            projected_gravity,
            imu_height,
            state.info['phase_step'],
            terminate_on_unwanted_contacts=self.environment_config.terminate_on_unwanted_contacts,
            terminate_on_extreme_landing_compression=self.environment_config.terminate_on_extreme_landing_compression,
        )

        # Rewards:
        rewards = {
            # Tracking Rewards:
            'tracking_height_reference': self._reward_tracking_height(
                imu_height, 
                state.info['phase_step'],
                self.height_sigma,
            ),
            'tracking_pitch_reference': self._reward_tracking_pitch(
                projected_gravity,
                state.info['phase_step'],
            ),
            'spin': self._reward_spin(
                local_angular_velocity,
                state.info['phase_step'],
                feet_contacts,
            ),
            'brake': self._reward_brake(
                local_angular_velocity,
                state.info['phase_step'],
                self.brake_sigma,
            ),
            # Regularization Costs:
            'unwanted_spin': self._cost_unwanted_spin(
                local_angular_velocity,
            ),
            'pose_regularization': self._cost_pose_regularization(joint_angles),
            'orientation_regularization': self._cost_orientation_regularization(
                up_vector,
                state.info['phase_step'],
            ),
            # Effort Costs:
            'torque': self._cost_torques(data.actuator_force),
            'action_rate': self._cost_action_rate(action, state.info['previous_action']),
            'acceleration': self._cost_acceleration(
                data.qacc,
            ),
            'mechanical_power': self._cost_mechanical_power(
                data,
                state.info['phase_step'],
            ),
            # Landing Regularization Costs:
            'dof_limit': self._cost_dof_limit(joint_angles),
            'base_clearance': self._cost_base_clearance(
                data,
                state.info['phase_step'],
            ),
            # Auxilary Costs:
            'stand_still': self._cost_stand_still(
                joint_angles,
                state.info['phase_step'],
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

        # Phase Logic:
        state.info['flip_done'] = (state.info['phase_step'] >= self.num_phase_steps)
        state.info['phase_step'] = jnp.where(
            done | (state.info['flip_done'] & (state.info['steps_until_next_command'] <= 0)),
            jnp.int32(0),
            jnp.minimum(state.info['phase_step'] + 1, self.num_phase_steps),
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
        termination_contacts: jax.Array,
        unwanted_contacts: jax.Array,
        projected_gravity: jax.Array,
        imu_height: jax.Array,
        phase_step: jax.Array,
        terminate_on_unwanted_contacts: bool = False,
        terminate_on_extreme_landing_compression: bool = False,
    ) -> jax.Array:
        # Termination Condition:
        done = jnp.any(termination_contacts)
        done |= terminate_on_unwanted_contacts * jnp.any(unwanted_contacts)

        phase = phase_step / self.num_phase_steps

        # Terminate Unwanted Contacts during Lift Off:
        is_liftoff_phase = phase <= self.phase_frames['liftoff']
        done |= is_liftoff_phase * jnp.any(unwanted_contacts)

        # Terminate Extreme Landing Compression:
        is_landing_phase = phase >= (self.phase_frames['apex'] + 0.2)
        right_hind_hip_height = data.geom_xpos[self.hip_contact_geom_idx[0]][2]
        left_hind_hip_height = data.geom_xpos[self.hip_contact_geom_idx[1]][2]
        right_hind_hip_clearance_violation = right_hind_hip_height < 0.11
        left_hind_hip_clearance_violation = left_hind_hip_height < 0.11
        done |= terminate_on_extreme_landing_compression & is_landing_phase & (
            right_hind_hip_clearance_violation | left_hind_hip_clearance_violation
        )

        # Tracking Failure Conditions:
        phase = phase_step / self.num_phase_steps
        is_flight_phase = (phase > self.phase_frames['liftoff']) & (phase < self.phase_frames['end'])
        # Height Tracking Failure:
        target_height = self.height_reference[phase_step]
        height_failure = is_flight_phase & ((target_height - imu_height) > 0.2)
        # Pitch Tracking Failure:
        pitch_reference = self.pitch_reference[phase_step]
        target_gravity = jnp.array([jnp.sin(pitch_reference), 0.0, -jnp.cos(pitch_reference)])
        dot_product = jnp.dot(projected_gravity, target_gravity)
        pitch_failure = is_flight_phase & (dot_product < 0.0)
        
        kinematic_failure = height_failure | pitch_failure
        done |= kinematic_failure

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
                phase,
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

        # Phase:
        phase = jnp.asarray([state_info['phase_step']], dtype=jnp.float32) / self.num_phase_steps

        observation = jnp.concatenate([
            noisy_angular_rate,                         # 3
            noisy_projected_gravity,                    # 3
            noisy_joint_positions - self.default_pose,  # 12
            noisy_joint_velocities,                     # 12
            state_info['previous_action'],              # 12 or 24
            phase,                                      # 1
            filter_observation,                         # Based on filter
        ])

        # Critic Observation:
        imu_height = jnp.asarray([data.site_xpos[self.imu_site_idx][2]])
        accelerometer = self.get_accelerometer(data)
        global_angular_velocity = self.get_global_angular_velocity(data)
        actuator_force = data.actuator_force

        # Reference Information:
        offsets = jnp.arange(self.phase_step_lookahead)
        reference_steps = state_info['phase_step'] + offsets
        reference_steps = jnp.minimum(reference_steps, self.num_phase_steps)
        reference_height_horizon = self.height_reference[reference_steps]
        reference_pitch_horizon = self.pitch_reference[reference_steps]

        # Contact Distances:
        contact_distances = data.geom_xpos[self.contact_geom_idx, 2]

        privileged_observation = jnp.concatenate([
            observation,                                                                                
            accelerometer,                                                                              # 3
            gyroscope,                                                                                  # 3
            projected_gravity,                                                                          # 3
            linear_velocity,                                                                            # 3
            global_angular_velocity,                                                                    # 3
            imu_height,                                                                                 # 1
            q - self.default_pose,                                                                      # 12
            qd,                                                                                         # 12
            actuator_force,                                                                             # 12 or 24
            state_info['previous_contact'],                                                             # 4
            data.xfrc_applied[self.base_idx, :3],                                                       # 3
            reference_height_horizon,                                                                   # phase_step_lookahead
            reference_pitch_horizon,                                                                    # phase_step_lookahead
            contact_distances,                                                                          # len(contact_geom_idx)
            jnp.asarray([
                state_info['steps_since_previous_disturbance'] >= state_info['steps_until_next_disturbance']
            ]),                                                                                         # 1
        ])

        return {
            'state': observation,
            'privileged_state': privileged_observation,
        }

    def _reward_tracking_height(
        self,
        imu_height: jax.Array,
        phase_step: jax.Array,
        kernel_sigma: float = 1.0,
    ) -> jax.Array:
        target_height = self.height_reference[phase_step]
        error = jnp.square(target_height - imu_height)
        return jnp.exp(-error / kernel_sigma)
    
    def _reward_tracking_pitch(
        self,
        projected_gravity: jax.Array,
        phase_step: jax.Array,
    ) -> jax.Array:
        pitch_reference = self.pitch_reference[phase_step]
        target_gravity = jnp.array([jnp.sin(pitch_reference), 0.0, -jnp.cos(pitch_reference)])
        dot_product = jnp.dot(projected_gravity, target_gravity)
        normalized = 0.5 * dot_product + 0.5
        tracking_condition = jnp.where((phase_step / self.num_phase_steps) < 1.0, 1.0, 0.0)
        return tracking_condition * jnp.square(normalized)
    
    def _reward_spin(
        self,
        local_angular_velocity: jax.Array,
        phase_step: jax.Array,
        feet_contacts: jax.Array,
    ) -> jax.Array:
        phase = phase_step / self.num_phase_steps
        
        # Phase Windows:
        brake_start = self.phase_frames['apex'] + 0.2
        ramp_up = (phase - self.phase_frames['liftoff']) * 1 / (self.phase_frames['apex'] - self.phase_frames['liftoff'])
        ramp_up = jnp.clip(ramp_up, 0.0, 1.0)
        ramp_down = 1.0 + (phase - brake_start) * -1 / (self.phase_frames['end'] - brake_start)
        ramp_down = jnp.clip(ramp_down, 0.0, 1.0)
        phase_multiplier = jnp.minimum(ramp_up, ramp_down)
        
        # Contact and Direction Gates:
        in_air = jnp.where(jnp.sum(feet_contacts) <= 0.0, 1.0, 0.0)
        is_backflip = jnp.where(local_angular_velocity[1] <= 0.0, 1.0, 0.0)

        # Spin Rate Reward:
        spin_rate = jnp.abs(local_angular_velocity[1])
        
        # Using the Derivative of the Pitch Reference as the Target Spin Rate:
        # target_spin = jnp.abs(self.pitch_rate_reference[phase_step]) + 1e-5
        
        flight_duration = self.flip_duration_s * (self.phase_frames['end'] - self.phase_frames['liftoff'])
        target_spin = ((2 * jnp.pi) / flight_duration)

        spin_reward = jnp.tanh(spin_rate / target_spin)
        
        return in_air * is_backflip * phase_multiplier * spin_reward

    def _reward_brake(
        self,
        local_angular_velocity: jax.Array,
        phase_step: jax.Array,
        kernel_sigma: float = 1.0,
    ) -> jax.Array:
        phase = phase_step / self.num_phase_steps
        
        # Phase Window:
        brake_start = self.phase_frames['apex'] + 0.2
        ramp_up = (phase - brake_start) * 1 / (self.phase_frames['end'] - brake_start)
        phase_multiplier = jnp.clip(ramp_up, 0.0, 1.0)
        
        # Reward:
        spin_rate = local_angular_velocity[1]
        target_spin = self.pitch_rate_reference[phase_step]
        error = jnp.square(spin_rate - target_spin)
        reward = jnp.exp(-error / kernel_sigma)
        return phase_multiplier * reward

    def _cost_unwanted_spin(
        self,
        local_angular_velocity: jax.Array,
    ) -> jax.Array:
        return jnp.square(local_angular_velocity[0]) + jnp.square(local_angular_velocity[2])

    def _cost_pose_regularization(
        self,
        qpos: jax.Array,
    ) -> jax.Array:
        # Pose Regularization:
        error = jnp.sum(jnp.square(qpos - self.default_pose))
        return error
    
    def _cost_orientation_regularization(
        self,
        base_z_axis: jax.Array,
        phase_step: jax.Array,
    ) -> jax.Array:
        # Penalize non flat base orientation
        no_command = (phase_step / self.num_phase_steps) >= 1.0
        return jnp.sum(jnp.square(base_z_axis[:2])) * no_command

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

    def _cost_mechanical_power(
        self,
        data: mjx.Data,
        phase_step: jax.Array,
    ) -> jax.Array:
        phase = phase_step / self.num_phase_steps
        is_landing = jnp.where(phase >= (self.phase_frames['apex'] + 0.2), 1.0, 0.0)
        
        torques = data.actuator_force
        velocities = data.qvel[6:]
        power = jnp.abs(torques * velocities)
        
        return is_landing * jnp.sum(jnp.square(power))

    def _cost_dof_limit(self, qpos: jax.Array) -> jax.Array:
        upper_margin = jnp.maximum(0.0, qpos - (self.joint_ub - 0.1))
        lower_margin = jnp.maximum(0.0, (self.joint_lb + 0.1) - qpos)
        return jnp.sum(upper_margin) + jnp.sum(lower_margin)

    def _cost_base_clearance(
        self, 
        data: mjx.Data,
        phase_step: jax.Array
    ) -> jax.Array:
        phase = phase_step / self.num_phase_steps
        is_landing = jnp.where(phase >= (self.phase_frames['apex'] + 0.2), 1.0, 0.0)
        imu_height = data.site_xpos[self.imu_site_idx][2]
        right_hind_hip_height = data.geom_xpos[self._mj_model.geom('hind_right_hip_collision').id][2]
        left_hind_hip_height = data.geom_xpos[self._mj_model.geom('hind_left_hip_collision').id][2]
        base_clearance_violation = jnp.maximum(0.0, 0.20 - imu_height)
        right_hind_hip_clearance_violation = jnp.maximum(0.0, 0.18 - right_hind_hip_height)
        left_hind_hip_clearance_violation = jnp.maximum(0.0, 0.18 - left_hind_hip_height)
        clearance_violation = jnp.square(base_clearance_violation) + jnp.square(right_hind_hip_clearance_violation) + jnp.square(left_hind_hip_clearance_violation)
        return is_landing * clearance_violation

    def _cost_stand_still(
        self,
        joint_angles: jax.Array,
        phase_step: jax.Array,
    ) -> jax.Array:
        # Penalize motion at zero commands
        no_command = (phase_step / self.num_phase_steps) >= 1.0
        return jnp.sum(jnp.abs(joint_angles - self.default_pose)) * (no_command)

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
