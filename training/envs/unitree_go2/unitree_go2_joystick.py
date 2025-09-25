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

from training.envs.utilities import collisions

from training.envs.unitree_go2.config import (
    RewardConfig,
    NoiseConfig,
    DisturbanceConfig,
    CommandConfig,
)

# Types:
PRNGKey: TypeAlias = jax.Array


class UnitreeGo2Env(PipelineEnv):
    """Environment for training the Unitree Go2 quadruped joystick policy in MJX."""

    def __init__(
        self,
        filename: str = 'scene_mjx.xml',
        reward_config: RewardConfig = RewardConfig(),
        noise_config: NoiseConfig = NoiseConfig(),
        disturbance_config: DisturbanceConfig = DisturbanceConfig(),
        command_config: CommandConfig = CommandConfig(),
        action_scale: float = 0.5,
        **kwargs,
    ):
        self.filename = f'mjcf/{filename}'
        self.filepath = os.path.join(
            os.path.dirname(__file__),
            self.filename,
        )
        sys = mjcf.load(self.filepath)

        self.step_dt = 0.02
        sys = sys.tree_replace({'opt.timestep': 0.004})

        n_frames = kwargs.pop('n_frames', int(self.step_dt / sys.opt.timestep))
        super().__init__(sys, backend='mjx', n_frames=n_frames)

        self.kernel_sigma = reward_config.kernel_sigma
        self.target_air_time = reward_config.target_air_time
        self.mode_time = reward_config.mode_time
        self.command_threshold = reward_config.command_threshold
        self.velocity_threshold = reward_config.velocity_threshold
        self.foot_height = reward_config.foot_height
        self.foot_clearance_velocity_scale = reward_config.foot_clearance_velocity_scale
        self.foot_clearance_sigma = reward_config.foot_clearance_sigma
        reward_config_dict = flax.serialization.to_state_dict(reward_config)
        del reward_config_dict['kernel_sigma']
        del reward_config_dict['target_air_time']
        del reward_config_dict['mode_time']
        del reward_config_dict['command_threshold']
        del reward_config_dict['velocity_threshold']
        del reward_config_dict['foot_height']
        del reward_config_dict['foot_clearance_velocity_scale']
        del reward_config_dict['foot_clearance_sigma']
        self.reward_config = reward_config_dict

        self.noise_config = noise_config
        self.disturbance_config = disturbance_config
        self.command_config = command_config

        self.floor_geom_idx = self.sys.mj_model.geom('floor').id
        self.base_idx = mujoco.mj_name2id(
            sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, 'base_link'
        )
        self.base_link_mass = self.sys.mj_model.body_subtreemass[self.base_idx]

        self.action_scale = action_scale
        self.init_q = jnp.array(sys.mj_model.keyframe('home').qpos)
        self.init_qd = jnp.zeros(sys.nv)
        self.default_pose = jnp.array(sys.mj_model.keyframe('home').qpos[7:])
        self.default_ctrl = jnp.array(sys.mj_model.keyframe('home').ctrl)
        self.joint_lb = jnp.array([
            -1.0472, -1.5708, -2.7227,
            -1.0472, -1.5708, -2.7227,
            -1.0472, -0.5236, -2.7227,
            -1.0472, -0.5236, -2.7227,
        ])
        self.joint_ub = jnp.array([
            1.0472, 3.4907, -0.83776,
            1.0472, 3.4907, -0.83776,
            1.0472, 4.5379, -0.83776,
            1.0472, 4.5379, -0.83776,
        ])

        # Sites and Bodies:
        feet_geom = [
            'front_right',
            'front_left',
            'hind_right',
            'hind_left',
        ]
        feet_geom_idx = [
            self.sys.mj_model.geom(name).id for name in feet_geom
        ]
        assert not any(id_ == -1 for id_ in feet_geom_idx), 'Site not found.'
        self.feet_geom_idx = np.array(feet_geom_idx)
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

        # Sensors:
        self.feet_position_sensor = [
            "fr_pos",
            "fl_pos",
            "hr_pos",
            "hl_pos",
        ]
        self.feet_linear_velocity_sensor = [
            "fr_global_linvel",
            "fl_global_linvel",
            "hr_global_linvel",
            "hl_global_linvel",
        ]

        # Observation Size:
        self.num_observations = 45
        self.num_privileged_observations = self.num_observations + 91

    def sample_command(
        self,
        rng: jax.Array,
    ) -> jax.Array:
        _, command_key, stand_still_key = jax.random.split(rng, 3)

        command = jax.random.uniform(
            command_key,
            shape=(3,),
            minval=-self.command_config.command_range,
            maxval=self.command_config.command_range,
        )
        stand_still_mask = jax.random.bernoulli(
            stand_still_key,
            p=self.command_config.command_mask_probability,
            shape=(1,),
        )

        command = stand_still_mask * command

        return command

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

        # Initialize State:
        pipeline_state = self.pipeline_init(qpos, qvel)

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

        state_info = {
            'rng': rng,
            'previous_action': jnp.zeros(12),
            'previous_velocity': jnp.zeros(12),
            'command': command,
            'steps_until_next_command': steps_until_next_command,
            'previous_contact': jnp.zeros(4, dtype=bool),
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
            pipeline_state, state_info,
        )

        reward, done = jnp.zeros(2)
        done = jnp.float64(done) if jax.config.x64_enabled else jnp.float32(done)

        metrics = {}
        for k in state_info['rewards']:
            metrics[k] = state_info['rewards'][k]
        metrics['total_distance'] = 0.0
        metrics['swing_peak'] = jnp.zeros(())

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
        rng, cmd_key, cmd_frequency_key = jax.random.split(state.info['rng'], 3)

        # Disturbance: (Force based)
        state = self.maybe_apply_perturbation(state)

        # Physics step:
        motor_targets = self.default_ctrl + action * self.action_scale
        pipeline_state = self.pipeline_step(
            state.pipeline_state, motor_targets,
        )

        imu_height = pipeline_state.site_xpos[self.imu_site_idx][2]
        joint_angles = pipeline_state.q[7:]
        joint_velocities = pipeline_state.qd[6:]

        # Foot contact data based on z-position:
        contact = jnp.array([
            collisions.geoms_colliding(pipeline_state, geom_id, self.floor_geom_idx)
            for geom_id in self.feet_geom_idx
        ])
        contact_filt = contact | state.info['previous_contact']
        first_contact = (state.info['feet_air_time'] > 0) * contact_filt

        # Feet Air and Contact Time:
        state.info['feet_contact_time'] += self.dt
        state.info['feet_air_time'] += self.dt

        state.info['previous_air_time'] = jnp.where(
            contact, state.info['feet_air_time'], state.info['previous_air_time'],
        )
        state.info['previous_contact_time'] = jnp.where(
            ~contact, state.info['feet_contact_time'], state.info['previous_contact_time'],
        )
        state.info['feet_air_time'] *= ~contact
        state.info['feet_contact_time'] *= contact

        # Foot Swing Peak Height:
        foot_position = pipeline_state.site_xpos[self.feet_site_idx]
        foot_position_z = foot_position[..., -1]
        state.info['swing_peak'] = jnp.maximum(
            state.info['swing_peak'], foot_position_z,
        )

        # Body Velocity:
        body_velocity = self.get_global_linvel(pipeline_state)

        # Observation data:
        observation = self.get_observation(
            pipeline_state,
            state.info,
        )

        # Done if joint limits are reached or robot is falling:
        done = self.get_upvector(pipeline_state)[-1] < 0.0
        done |= imu_height < 0.05
        # These can become rewards:
        done |= jnp.any(joint_angles < self.joint_lb)
        done |= jnp.any(joint_angles > self.joint_ub)

        # Rewards:
        rewards = {
            'tracking_linear_velocity': (
                self._reward_tracking_velocity(state.info['command'], body_velocity)
            ),
            'tracking_angular_velocity': (
                self._reward_tracking_yaw_rate(state.info['command'], self.get_gyro(pipeline_state))
            ),
            'linear_z_velocity': self._reward_vertical_velocity(
                body_velocity,
            ),
            'angular_xy_velocity': self._reward_angular_velocity(
                self.get_global_angvel(pipeline_state),
            ),
            'orientation_regularization': self._reward_orientation_regularization(
                self.get_upvector(pipeline_state),
            ),
            'torque': self._reward_torques(pipeline_state.actuator_force),
            'action_rate': self._reward_action_rate(action, state.info['previous_action']),
            'acceleration': self._reward_acceleration(
                pipeline_state.qacc,
            ),
            'stand_still': self._reward_stand_still(
                state.info['command'], joint_angles,
            ),
            'foot_slip': self._reward_foot_slip(
                pipeline_state, contact, state.info['command'],
            ),
            'air_time': self._reward_air_time(
                state.info['feet_air_time'],
                state.info['feet_contact_time'],
                state.info['command'],
                body_velocity,
                self.mode_time,
                self.command_threshold,
                self.velocity_threshold,
            ),
            'foot_clearance': self._reward_foot_clearance(
                pipeline_state,
                self.target_height,
                self.foot_clearance_velocity_scale,
                self.foot_clearance_sigma,
            ),
            'termination': jnp.float64(
                self._reward_termination(done)
            ) if jax.config.x64_enabled else jnp.float32(
                self._reward_termination(done)
            ),
        }
        rewards = {
            k: v * self.reward_config[k] for k, v in rewards.items()
        }
        reward = jnp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)

        # State management
        state.info['previous_action'] = action
        state.info['previous_velocity'] = joint_velocities
        state.info['previous_contact'] = contact
        state.info['swing_peak'] *= ~contact
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
        state.metrics['total_distance'] = math.normalize(
            pipeline_state.x.pos[self.base_idx - 1])[1]
        state.metrics['swing_peak'] = jnp.mean(
            state.info['swing_peak']
        )
        state.metrics.update(state.info['rewards'])

        done = jnp.float64(done) if jax.config.x64_enabled else jnp.float32(done)

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
                command,
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
            state_info['previous_action'],              # 12
            state_info['command'],                      # 3
        ])

        accelerometer = self.get_accelerometer(pipeline_state)
        linear_velocity = self.get_local_linvel(pipeline_state)
        global_angular_velocity = self.get_global_angvel(pipeline_state)
        actuator_force = pipeline_state.actuator_force
        feet_velocity = self.get_feet_velocity(pipeline_state).ravel()

        privileged_observation = jnp.concatenate([
            observation,                                                                                # 45
            accelerometer,                                                                              # 3
            gyroscope,                                                                                  # 3
            projected_gravity,                                                                          # 3
            linear_velocity,                                                                            # 3
            global_angular_velocity,                                                                    # 3
            q - self.default_pose,                                                                      # 12
            qd,                                                                                         # 12
            actuator_force,                                                                             # 12
            feet_velocity,                                                                              # 12
            state_info['previous_contact'],                                                             # 4
            state_info['feet_air_time'],                                                                # 4
            state_info['feet_contact_time'],                                                            # 4
            state_info['previous_air_time'],                                                            # 4
            state_info['previous_contact_time'],                                                        # 4
            state_info['swing_peak'],                                                                   # 4
            pipeline_state.xfrc_applied[self.base_idx, :3],                                             # 3
            jnp.asarray([
                state_info['steps_since_previous_disturbance'] >= state_info['steps_until_next_disturbance']
            ]),                                                                                         # 1
        ])
        # Size: 91

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

    def _reward_vertical_velocity(
        self, global_base_linvel: jax.Array
    ) -> jax.Array:
        # Penalize z axis base linear velocity
        return jnp.square(global_base_linvel[2])

    def _reward_angular_velocity(
        self, global_base_angvel: jax.Array,
    ) -> jax.Array:
        # Penalize xy axes base angular velocity
        return jnp.sum(jnp.square(global_base_angvel[:2]))

    def _reward_orientation_regularization(
        self, base_z_axis: jax.Array,
    ) -> jax.Array:
        # Penalize non flat base orientation
        return jnp.sum(jnp.square(base_z_axis[:2]))

    def _reward_pose_regularization(
        self, qpos: jax.Array,
    ) -> jax.Array:
        weight = jnp.array([1.0, 1.0, 0.1] * 4) / 12.0
        error = jnp.sum(jnp.square(qpos - self.default_pose) * weight)
        return jnp.exp(-error)

    def _reward_torques(self, torques: jax.Array) -> jax.Array:
        # Penalize torques
        return jnp.sqrt(jnp.sum(jnp.square(torques))) + jnp.sum(jnp.abs(torques))

    def _reward_action_rate(
        self, action: jax.Array, previous_action: jax.Array
    ) -> jax.Array:
        # Penalize changes in actions
        return jnp.sum(jnp.square(action - previous_action))

    def _reward_mechanical_power(
        self, qd: jax.Array, torques: jax.Array
    ) -> jax.Array:
        # Penalize mechanical power
        return jnp.sum(jnp.abs(torques) * jnp.abs(qd))

    def _reward_acceleration(
        self, qacc: jax.Array,
    ) -> jax.Array:
        # Penalize Motor/Joint Acceleration
        return jnp.sqrt(jnp.sum(jnp.square(qacc)))

    def _reward_stand_still(
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
        t_max = jnp.max([air_time, contact_time])
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

    def _air_time_variance_penalty(
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
        pipeline_state: base.State,
        target_height: float = 0.1,
        velocity_scale: float = 2.0,
        sigma: float = 0.05,
    ) -> jax.Array:
        foot_position = pipeline_state.site_xpos[self.feet_site_idx]
        foot_height = foot_position[..., -1]
        foot_error = jnp.square(foot_height - target_height)
        foot_velocity = self.get_feet_velocity(pipeline_state)[..., :2]
        foot_velocity_norm = jnp.linalg.norm(foot_velocity)
        foot_velocity_tanh = jnp.tanh(velocity_scale * foot_velocity_norm)
        error = jnp.sum(foot_error * foot_velocity_tanh)
        return jnp.exp(-error / sigma)

    def _reward_foot_slip(
        self,
        pipeline_state: base.State,
        contact: jax.Array,
        commands: jax.Array,
    ) -> jax.Array:
        # Penalize foot slip
        command_norm = jnp.linalg.norm(commands)
        foot_velocity = self.get_feet_velocity(pipeline_state)
        foot_velocity_xy = foot_velocity[..., :2]
        velocity_xy_sq = jnp.sum(jnp.square(foot_velocity_xy), axis=-1)
        return jnp.sum(velocity_xy_sq * contact) * (command_norm > 0.1)

    def _reward_termination(self, done: jax.Array) -> jax.Array:
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

    def get_global_linvel(self, pipeline_state: base.State) -> jax.Array:
        return self.get_sensor_data(
            self.sys.mj_model, pipeline_state, "global_linvel"
        )

    def get_global_angvel(self, pipeline_state: base.State) -> jax.Array:
        return self.get_sensor_data(
            self.sys.mj_model, pipeline_state, "global_angvel"
        )

    def get_local_linvel(self, pipeline_state: base.State) -> jax.Array:
        return self.get_sensor_data(
            self.sys.mj_model, pipeline_state, "local_linvel"
        )

    def get_accelerometer(self, pipeline_state: base.State) -> jax.Array:
        return self.get_sensor_data(
            self.sys.mj_model, pipeline_state, "imu_acceleration"
        )

    def get_gyro(self, pipeline_state: base.State) -> jax.Array:
        return self.get_sensor_data(self.sys.mj_model, pipeline_state, "imu_gyro")

    def get_feet_pos(self, pipeline_state: base.State) -> jax.Array:
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
        command: np.ndarray,
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
            q - self.default_ctrl,
            qd,
            previous_action,
            command,
        ])

        return {
            'state': observation,
            'privileged_state': np.zeros((self.num_privileged_observations,)),
        }

    def hardware_observation(
        self,
        imu_state: Any,
        motor_state: Any,
        command: np.ndarray,
        previous_action: np.ndarray,
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

        # Set to Correct Data Type:
        base_rotation = np.asarray(imu_state.quaternion, dtype=np.float32)
        gyroscope = np.asarray(imu_state.gyroscope, dtype=np.float32)
        joint_positions = np.asarray(motor_state.q, dtype=np.float32)
        joint_velocities = np.asarray(motor_state.qd, dtype=np.float32)

        # Calculate Projected Gravity:
        inverse_base_rotation = quat_inv(base_rotation)
        projected_gravity = rotate(
            np.array([0.0, 0.0, -1.0]),
            inverse_base_rotation,
        )

        observation = np.concatenate([
            gyroscope,
            projected_gravity,
            joint_positions - self.default_ctrl,
            joint_velocities,
            previous_action,
            command,
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
