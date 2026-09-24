"""
    Spot Environment: Joystick Locomotion Task
"""

from typing import Any, Dict, TypeAlias

import jax
import numpy as np
import jax.numpy as jnp

import flax
import flax.serialization

from mujoco import mjx
from mujoco.mjx._src import math as mjx_math
from mujoco_playground._src import mjx_env

from training.envs.spot import base
from training.envs.spot import diagnostics
from training.envs.spot import randomize
from training.envs.spot.config import (
    RewardConfig,
    NoiseConfig,
    DisturbanceConfig,
    CommandConfig,
    EnvironmentConfig,
)

# Types:
PRNGKey: TypeAlias = jax.Array


class SpotJoystickEnv(base.SpotEnv):
    """Environment for training the Spot quadruped joystick policy in MJX."""

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
            noise_config=noise_config,
            disturbance_config=disturbance_config,
            command_config=command_config,
            **kwargs,
        )

        # The policy only controls the 12 leg joints (self.num_leg_joints,
        # set in base.py); the arm is not policy-controlled and holds its
        # default pose -- see base.py's _simulation_step.

        # Task Specific Reward Implementation: weights and hyperparameters
        # are separate dataclasses (RewardConfig.weights / .hyperparameters),
        # so there's no manual key-deletion list to keep in sync --
        # self.reward_config is exactly the per-term weight dict `step()`
        # looks up, no more no less.
        hyperparameters = reward_config.hyperparameters
        self.kernel_sigma = hyperparameters.kernel_sigma
        self.target_air_time = hyperparameters.target_air_time
        self.mode_time = hyperparameters.mode_time
        self.command_threshold = hyperparameters.command_threshold
        self.velocity_threshold = hyperparameters.velocity_threshold
        self.ramp_at_vel = hyperparameters.ramp_at_vel
        self.ramp_rate = hyperparameters.ramp_rate
        self.target_foot_height = hyperparameters.target_foot_height
        self.foot_clearance_velocity_scale = hyperparameters.foot_clearance_velocity_scale
        self.foot_clearance_sigma = hyperparameters.foot_clearance_sigma
        self.stand_still_scale = hyperparameters.stand_still_scale
        self.window_steps = hyperparameters.window_steps
        self.reward_config = flax.serialization.to_state_dict(reward_config.weights)

        # Task Specific Observation Details:
        # State: linvel(3) + angvel(3) + gravity(3) + joint_pos(n) + joint_vel(n)
        #        + previous_action(num_leg_joints) + command(3) + filter.
        self.num_observations = (
            3 + 3 + 3 + self.num_joints + self.num_joints + self.num_leg_joints + 3
            + self.filter.observation_size
        )
        # Privileged: state + accelerometer(3) + raw angvel/gravity/linvel(3 each)
        #             + global_angvel(3) + raw joint_pos/vel(n each) + actuator_force(nu)
        #             + feet_velocity(12) + previous_contact/gait timers(4*6)
        #             + xfrc_applied(3) + disturbance_flag(1).
        self.num_privileged_observations = (
            self.num_observations
            + 3 + 3 + 3 + 3 + 3
            + self.num_joints + self.num_joints
            + self.nu
            + 12
            + 4 + 4 + 4 + 4 + 4 + 4
            + 3 + 1
        )

        # State Observation Mask: True marks a continuous signal that should
        # be normalized (e.g. via running mean/std); False marks scripted or
        # boolean signals that shouldn't be.
        state_mask = jnp.concatenate([
            jnp.ones(3, dtype=bool),                       # linear velocity
            jnp.ones(3, dtype=bool),                       # angular velocity
            jnp.ones(3, dtype=bool),                       # projected gravity
            jnp.ones(self.num_joints, dtype=bool),         # joint position - default
            jnp.ones(self.num_joints, dtype=bool),         # joint velocity
            jnp.ones(self.num_leg_joints, dtype=bool),     # previous action
            jnp.zeros(3, dtype=bool),                      # command (scripted)
            jnp.ones(self.filter.observation_size, dtype=bool),
        ])

        # Privileged Observation Mask:
        privileged_mask = jnp.concatenate([
            state_mask,
            jnp.ones(3, dtype=bool),                       # accelerometer
            jnp.ones(3, dtype=bool),                       # angular velocity (raw)
            jnp.ones(3, dtype=bool),                       # projected gravity (raw)
            jnp.ones(3, dtype=bool),                       # linear velocity (raw)
            jnp.ones(3, dtype=bool),                       # global angular velocity
            jnp.ones(self.num_joints, dtype=bool),         # joint position - default (raw)
            jnp.ones(self.num_joints, dtype=bool),         # joint velocity (raw)
            jnp.ones(self.nu, dtype=bool),                 # actuator force
            jnp.ones(12, dtype=bool),                      # feet velocity
            jnp.zeros(4, dtype=bool),                      # previous contact (boolean)
            jnp.ones(4, dtype=bool),                       # feet air time
            jnp.ones(4, dtype=bool),                       # feet contact time
            jnp.ones(4, dtype=bool),                       # previous air time
            jnp.ones(4, dtype=bool),                       # previous contact time
            jnp.ones(4, dtype=bool),                       # swing peak
            jnp.ones(3, dtype=bool),                       # xfrc applied
            jnp.zeros(1, dtype=bool),                      # disturbance active flag
        ])

        assert state_mask.shape[0] == self.num_observations, (
            f"State mask length {state_mask.shape[0]} does not match "
            f"number of observations {self.num_observations}."
        )
        assert privileged_mask.shape[0] == self.num_privileged_observations, (
            f"Privileged mask length {privileged_mask.shape[0]} does not "
            f"match number of privileged observations {self.num_privileged_observations}."
        )

        self.observation_mask = {
            'state': state_mask,
            'privileged_state': privileged_mask,
        }

    @property
    def action_size(self) -> int:
        # Overrides base.SpotEnv.action_size (= total actuator count): the
        # policy only controls the 12 leg joints, not the arm.
        return self.num_leg_joints

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

        # Small Joint Perturbation: leg joints only -- the arm starts exactly
        # at its default pose, since it isn't part of the task.
        rng, key = jax.random.split(rng)
        delta = jax.random.uniform(
            key,
            shape=(self.num_leg_joints,),
            minval=-0.1,
            maxval=0.1,
        )
        qpos = qpos.at[7:7 + self.num_leg_joints].set(
            qpos[7:7 + self.num_leg_joints] + delta
        )

        # Initialize State: every actuator is direct-torque now, so zero
        # ctrl means zero commanded torque everywhere (the arm's PD law
        # brings it to its default pose over the first few steps, rather
        # than a MuJoCo position servo holding it there from the start).
        ctrl = jnp.zeros(self.nu)

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

        # Actuation Delay: domain-randomized per channel, resampled each
        # reset like command/disturbance_magnitude above (see
        # randomize.sample_actuation_delay for why this isn't in
        # domain_randomize itself).
        rng, delay_key = jax.random.split(rng)
        actuation_delay = randomize.sample_actuation_delay(delay_key, self.nu)

        # Foot Contacts:
        feet_contacts = jnp.array([
            data.sensordata[self._mj_model.sensor_adr[sensor_id]] > 0
            for sensor_id in self.foot_to_floor_sensor
        ])

        state_info = {
            'rng': rng,
            'previous_action': jnp.zeros(self.num_leg_joints),
            'previous_joint_positions': jnp.zeros(self.num_joints),
            'previous_velocity': jnp.zeros(self.num_joints),
            'command': command,
            'steps_until_next_command': steps_until_next_command,
            'previous_contact': feet_contacts,
            'feet_air_time': jnp.zeros(4),
            'feet_contact_time': jnp.zeros(4),
            'previous_air_time': jnp.zeros(4),
            'previous_contact_time': jnp.zeros(4),
            'touchdown_window': jnp.zeros(4),
            'liftoff_window': jnp.zeros(4),
            'swing_peak': jnp.zeros(4),
            'rewards': {k: 0.0 for k in self.reward_config.keys()},
            'steps_until_next_disturbance': steps_until_next_disturbance,
            'disturbance_duration': disturbance_duration,
            'disturbance_duration_steps': disturbance_duration_steps,
            'steps_since_previous_disturbance': 0,
            'disturbance_step': 0,
            'disturbance_magnitude': disturbance_magnitude,
            'disturbance_direction': jnp.array([0.0, 0.0, 0.0]),
            'filter_state': filter_state,
            'delay_state': self.delay_line.init(),
            'actuation_delay': actuation_delay,
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
        metrics.update({name: 0.0 for name in diagnostics.DIAGNOSTIC_NAMES})
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

        # Apply Action Filter:
        filtered_action, filter_state = self.filter.apply(
            action, state.info['filter_state']
        )
        state.info['filter_state'] = filter_state

        # Physics step: `filtered_action` is the raw per-leg-joint action;
        # base.py's _simulation_step turns it into a qpos setpoint
        # (default_pose + action_scale * action) and runs the actuation
        # pipeline (delay -> PD law -> knee limit) to get torque.
        # `delay_state` must be threaded back out, same as `filter_state`
        # above; `actuation_delay` is this env's fixed-for-the-episode
        # domain-randomized sample from reset().
        data, delay_state = self._step(
            state.data,
            filtered_action,
            state.info['delay_state'],
            state.info['actuation_delay'],
        )
        state.info['delay_state'] = delay_state

        joint_angles = data.qpos[7:]
        joint_velocities = data.qvel[6:]

        # Sensor Contacts:
        feet_contacts = jnp.array([
            data.sensordata[self._mj_model.sensor_adr[sensor_id]] > 0
            for sensor_id in self.foot_to_floor_sensor
        ])
        unwanted_contacts = jnp.concatenate([
            jnp.array([
                data.sensordata[self._mj_model.sensor_adr[sensor_id]] > 0
                for sensor_id in self.thigh_to_floor_sensor
            ]),
            jnp.array([
                data.sensordata[self._mj_model.sensor_adr[sensor_id]] > 0
                for sensor_id in self.calf_to_floor_sensor
            ]),
            jnp.array([
                data.sensordata[self._mj_model.sensor_adr[sensor_id]] > 0
                for sensor_id in self.leg_self_collision_sensor
            ]),
            jnp.array([
                data.sensordata[self._mj_model.sensor_adr[sensor_id]] > 0
                for sensor_id in self.leg_to_leg_collision_sensor
            ]),
            jnp.array([
                data.sensordata[self._mj_model.sensor_adr[sensor_id]] > 0
                for sensor_id in self.arm_to_torso_collision_sensor
            ]),
        ])

        # Feet Air and Contact Time:
        # `previous_air_time`/`previous_contact_time` hold the duration of the
        # most recently *completed* swing/stance phase, updated only on the
        # touchdown/liftoff transition (not the whole subsequent phase) so
        # they don't collapse to 0 the very next step.
        touchdown = feet_contacts & ~state.info['previous_contact']
        liftoff = ~feet_contacts & state.info['previous_contact']
        state.info['previous_air_time'] = jnp.where(
            touchdown, state.info['feet_air_time'], state.info['previous_air_time'],
        )
        state.info['previous_contact_time'] = jnp.where(
            liftoff, state.info['feet_contact_time'], state.info['previous_contact_time'],
        )

        # Synchronized-contact detection window: hold each foot's
        # touchdown/liftoff "recently happened" flag for a few steps so
        # near-simultaneous (not just exactly simultaneous) transitions are
        # still caught as synchronized.
        state.info['touchdown_window'] = jnp.where(
            touchdown, self.window_steps, jnp.maximum(state.info['touchdown_window'] - 1, 0),
        )
        state.info['liftoff_window'] = jnp.where(
            liftoff, self.window_steps, jnp.maximum(state.info['liftoff_window'] - 1, 0),
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
            feet_contacts,
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
            # Leg joints only: the arm isn't policy-controlled, so its
            # gravity-hold torque/pose shouldn't be penalized.
            'torque': self._cost_torques(
                data.actuator_force[:self.num_leg_joints],
            ),
            'action_rate': self._cost_action_rate(action, state.info['previous_action']),
            'acceleration': self._cost_acceleration(
                data.qacc[6:6 + self.num_leg_joints],
            ),
            'stand_still': self._cost_stand_still(
                state.info['command'],
                local_body_velocity,
                joint_angles,
                self.stand_still_scale,
                self.velocity_threshold,
            ),
            'foot_slip': self._cost_foot_slip(
                data, feet_contacts,
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
            'gait_timing_variance': self._cost_gait_timing_variance(
                state.info['previous_air_time'],
                state.info['previous_contact_time'],
            ),
            'synchronized_contact': self._cost_synchronized_contact(
                state.info['touchdown_window'], state.info['liftoff_window'],
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

        # Weight-independent diagnostics (see diagnostics.py); must run
        # before swing_peak is cleared and previous_action is overwritten.
        diagnostic_values = self._diagnostics(
            command=state.info['command'],
            local_velocity=local_body_velocity,
            gyro=self.get_gyro(data),
            action=action,
            previous_action=state.info['previous_action'],
            feet_contacts=feet_contacts,
            touchdown=touchdown,
            touchdown_window=state.info['touchdown_window'],
            swing_peak=state.info['swing_peak'],
            foot_velocity=self.get_feet_velocity(data),
            unwanted_contacts=unwanted_contacts,
            data=data,
            joint_velocities=joint_velocities,
            upvector=self.get_upvector(data),
            done=done,
        )

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
        state.metrics.update(diagnostic_values)

        done = jnp.float64(done) if jax.config.x64_enabled else jnp.float32(done)

        state = state.replace(
            data=data,
            obs=observation,
            reward=reward,
            done=done,
        )
        return state

    def get_termination(self, data: mjx.Data) -> jax.Array:
        """Terminates when the torso, a thigh or an upper calf (knee) touches the floor."""
        termination_sensors = np.concatenate([
            self.torso_to_floor_sensor,
            self.thigh_to_floor_sensor,
            self.calf_upper_to_floor_sensor,
        ])
        termination_contacts = jnp.array([
            data.sensordata[self._mj_model.sensor_adr[sensor_id]] > 0
            for sensor_id in termination_sensors
        ])
        return jnp.any(termination_contacts)

    def get_observation(
        self,
        data: mjx.Data,
        contacts: jax.Array,
        state_info: dict[str, Any],
    ) -> Dict[str, jax.Array]:
        """
            Observation: [
                linear_velocity,
                angular_velocity,
                projected_gravity,
                relative_joint_positions,
                joint_velocities,
                previous_action,
                command,
                filter_observation,
            ]
        """
        q = data.qpos[7:]
        qd = data.qvel[6:]

        # Linear Velocity Noise:
        linear_velocity = self.get_local_linear_velocity(data)
        state_info['rng'], noise_key = jax.random.split(state_info['rng'])
        linear_velocity_noise = jax.random.uniform(
            noise_key,
            shape=linear_velocity.shape,
            minval=-self.noise_config.linear_velocity,
            maxval=self.noise_config.linear_velocity,
        )
        noisy_linear_velocity = linear_velocity + linear_velocity_noise

        # Angular Velocity Noise:
        angular_velocity = self.get_gyro(data)
        state_info['rng'], noise_key = jax.random.split(state_info['rng'])
        angular_velocity_noise = jax.random.uniform(
            noise_key,
            shape=angular_velocity.shape,
            minval=-self.noise_config.angular_velocity,
            maxval=self.noise_config.angular_velocity,
        )
        noisy_angular_velocity = angular_velocity + angular_velocity_noise

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
            noisy_linear_velocity,                        # 3
            noisy_angular_velocity,                       # 3
            noisy_projected_gravity,                      # 3
            noisy_joint_positions - self.default_pose,    # num_joints
            noisy_joint_velocities,                       # num_joints
            state_info['previous_action'],                # nu
            state_info['command'],                        # 3
            filter_observation,                           # Dynamic based on filter
        ])

        accelerometer = self.get_accelerometer(data)
        global_angular_velocity = self.get_global_angular_velocity(data)
        actuator_force = data.actuator_force
        feet_velocity = self.get_feet_velocity(data).ravel()

        privileged_observation = jnp.concatenate([
            observation,                                                                                # state size
            accelerometer,                                                                              # 3
            angular_velocity,                                                                           # 3
            projected_gravity,                                                                           # 3
            linear_velocity,                                                                             # 3
            global_angular_velocity,                                                                     # 3
            q - self.default_pose,                                                                       # num_joints
            qd,                                                                                          # num_joints
            actuator_force,                                                                              # nu
            feet_velocity,                                                                               # 12
            state_info['previous_contact'],                                                              # 4
            state_info['feet_air_time'],                                                                 # 4
            state_info['feet_contact_time'],                                                             # 4
            state_info['previous_air_time'],                                                             # 4
            state_info['previous_contact_time'],                                                         # 4
            state_info['swing_peak'],                                                                    # 4
            data.xfrc_applied[self.base_idx, :3],                                                        # 3
            jnp.asarray([
                state_info['steps_since_previous_disturbance'] >= state_info['steps_until_next_disturbance']
            ]),                                                                                          # 1
        ])

        return {
            'state': observation,
            'privileged_state': privileged_observation,
        }

    def _reward_tracking_velocity(
        self, commands: jax.Array, local_velocity: jax.Array
    ) -> jax.Array:
        # Tracking of linear velocity commands (xy axes)
        error = jnp.sum(jnp.square(commands[:2] - local_velocity[:2]))
        # Ramp: above ramp_at_vel commanded speed, scale the reward up by
        # ramp_rate per additional m/s, so tracking fast commands isn't
        # worth the same as tracking slow/stationary ones.
        command_magnitude = jnp.linalg.norm(commands[:2])
        ramp = jnp.maximum(
            1.0 + self.ramp_rate * (command_magnitude - self.ramp_at_vel), 1.0,
        )
        return jnp.exp(-error / self.kernel_sigma) * ramp

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
        # Penalize non flat base orientation. L2 norm (~sin(tilt), closer to
        # linear) rather than sum-of-squares (~sin^2(tilt)) -- matches
        # IsaacLab's Spot `base_orientation_penalty` shape. Note this is a
        # steeper penalty than the old quadratic form at small tilts, so the
        # weight will likely need retuning.
        return jnp.linalg.norm(base_z_axis[:2])

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
        commands: jax.Array,
        body_velocity: jax.Array,
        joint_angles: jax.Array,
        stand_still_scale: float = 2.0,
        velocity_threshold: float = 0.5,
    ) -> jax.Array:
        # Regularizes leg pose toward default only when told to stand still, so
        # the robot settles to a neutral stance without the penalty fighting
        # leg swing while walking. Gates on actual body velocity too (not just
        # command) so recovering from a push isn't mistaken for "at rest".
        command_norm = jnp.linalg.norm(commands)
        body_velocity_norm = jnp.linalg.norm(body_velocity)
        n = self.num_leg_joints
        deviation = jnp.sum(jnp.abs(joint_angles[:n] - self.default_pose[:n]))
        is_still = (command_norm < 1e-3) & (body_velocity_norm <= velocity_threshold)
        return jnp.where(is_still, stand_still_scale * deviation, 0.0)

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

    def _cost_gait_timing_variance(
        self,
        previous_air_time: jax.Array,
        previous_contact_time: jax.Array,
    ) -> jax.Array:
        # Penalize uneven leg usage: variance, across the 4 feet, of each
        # foot's most recently *completed* swing/stance duration. Does not
        # by itself distinguish gait patterns (a symmetric trot and a
        # synchronized pronk both drive this to ~0) -- see
        # `_cost_synchronized_contact` for that.
        air_time_variance = jnp.var(
            jnp.clip(previous_air_time, max=0.5),
        )
        contact_time_variance = jnp.var(
            jnp.clip(previous_contact_time, max=0.5),
        )
        return air_time_variance + contact_time_variance

    def _cost_synchronized_contact(
        self,
        touchdown_window: jax.Array,
        liftoff_window: jax.Array,
    ) -> jax.Array:
        # Penalize 3+ feet touching down/lifting off within a short window of
        # each other (e.g. pronking/bounding) without prescribing which
        # staggered gait pattern (trot/pace/walk) should be used instead --
        # exactly 2 feet together (a diagonal pair, as in a trot) is exempt.
        n_touchdown = jnp.sum(touchdown_window > 0)
        n_liftoff = jnp.sum(liftoff_window > 0)
        return jnp.square(jnp.maximum(n_touchdown - 2, 0)) + jnp.square(jnp.maximum(n_liftoff - 2, 0))

    def _reward_foot_clearance(
        self,
        data: mjx.Data,
        target_foot_height: float = 0.1,
        velocity_scale: float = 2.0,
        sigma: float = 0.05,
    ) -> jax.Array:
        # `minimum` intentionally only enforces a floor on clearance height;
        # overshoot is free here by design -- energy costs (torque,
        # acceleration, action_rate) are what discourage lifting higher
        # than necessary, not this term.
        foot_position = data.site_xpos[self.feet_site_idx]
        foot_height = jnp.minimum(foot_position[..., -1], target_foot_height)
        foot_error = jnp.square(foot_height - target_foot_height)
        foot_velocity = self.get_feet_velocity(data)[..., :2]
        foot_velocity_norm = jnp.linalg.norm(foot_velocity, axis=-1)
        foot_velocity_tanh = jnp.tanh(velocity_scale * foot_velocity_norm)
        error = jnp.sum(foot_error * foot_velocity_tanh)
        return jnp.exp(-error / sigma)

    def _diagnostics(
        self,
        command: jax.Array,
        local_velocity: jax.Array,
        gyro: jax.Array,
        action: jax.Array,
        previous_action: jax.Array,
        feet_contacts: jax.Array,
        touchdown: jax.Array,
        touchdown_window: jax.Array,
        swing_peak: jax.Array,
        foot_velocity: jax.Array,
        unwanted_contacts: jax.Array,
        data: mjx.Data,
        joint_velocities: jax.Array,
        upvector: jax.Array,
        done: jax.Array,
    ) -> dict[str, jax.Array]:
        """Raw per-step quantities for scorecard.compute_scorecard.

        Unlike `rewards`, none of these are scaled by reward weights, so
        they stay meaningful when a tuner changes (or zeroes) a weight.
        """
        n = self.num_leg_joints
        moving = (jnp.linalg.norm(command) >= 1e-3).astype(jnp.float32)
        still = 1.0 - moving
        contacts = feet_contacts.astype(jnp.float32)
        num_contacts = jnp.sum(contacts)
        action_delta_sq = jnp.mean(jnp.square(action - previous_action))
        body_speed = jnp.linalg.norm(local_velocity[:2])
        # Steps with exactly two feet down, classified by which pair (feet
        # order: front_left, front_right, rear_left, rear_right): a trot is
        # diagonal, a pace is same-side, a bound is the front or rear axle.
        two_contact = num_contacts == 2
        fl, fr, rl, rr = feet_contacts
        diagonal_pair = (fl & rr & ~fr & ~rl) | (fr & rl & ~fl & ~rr)
        same_side_pair = (fl & rl & ~fr & ~rr) | (fr & rr & ~fl & ~rl)
        axle_pair = (fl & fr & ~rl & ~rr) | (rl & rr & ~fl & ~fr)
        diagonal_agreement = (
            (feet_contacts[0] == feet_contacts[3])
            & (feet_contacts[1] == feet_contacts[2])
        )
        diagnostic_values = {
            'diag_moving_steps': moving,
            'diag_still_steps': still,
            'diag_linear_velocity_error_sq': moving * jnp.sum(
                jnp.square(command[:2] - local_velocity[:2]),
            ),
            'diag_yaw_rate_error_sq': moving * jnp.square(command[2] - gyro[2]),
            'diag_body_speed': moving * body_speed,
            'diag_foot_clearance_sum': moving * jnp.sum(swing_peak * touchdown),
            'diag_touchdown_count': moving * jnp.sum(touchdown),
            'diag_foot_slip_speed_sum': jnp.sum(
                jnp.linalg.norm(foot_velocity[..., :2], axis=-1) * contacts,
            ),
            'diag_foot_contact_count': num_contacts,
            'diag_flight_steps': moving * (num_contacts == 0),
            'diag_synchronized_touchdown_steps': moving * (
                jnp.sum(touchdown_window > 0) >= 3
            ),
            'diag_diagonal_agreement_steps': moving * diagonal_agreement,
            'diag_moving_contact_count': moving * num_contacts,
            'diag_two_contact_steps': moving * two_contact,
            'diag_two_contact_diagonal_steps': moving * diagonal_pair,
            'diag_two_contact_same_side_steps': moving * same_side_pair,
            'diag_two_contact_axle_steps': moving * axle_pair,
            'diag_moving_action_delta_sq': moving * action_delta_sq,
            'diag_still_action_delta_sq': still * action_delta_sq,
            'diag_still_joint_velocity_sq': still * jnp.mean(
                jnp.square(joint_velocities[:n]),
            ),
            'diag_still_body_speed': still * body_speed,
            'diag_mechanical_power': jnp.sum(
                jnp.abs(data.actuator_force[:n] * joint_velocities[:n]),
            ),
            # Fraction of leg joints commanding ~full range / ~full torque:
            # action saturation without torque saturation means the action
            # scale is too small; torque saturation means bang-bang or a
            # scale that reaches the limit.
            'diag_action_saturation': jnp.mean(jnp.abs(action) >= 0.95),
            'diag_torque_saturation': jnp.mean(
                jnp.abs(data.actuator_force[:n]) >= 0.95 * self.leg_torque_limit,
            ),
            'diag_unwanted_contacts': jnp.sum(unwanted_contacts),
            'diag_tilt': 1.0 - upvector[-1],
            'diag_terminated': done,
        }
        # Float32 throughout so the metrics keep a fixed dtype in scan carries.
        return {
            k: jnp.asarray(v, dtype=jnp.float32)
            for k, v in diagnostic_values.items()
        }

    def _cost_foot_slip(
        self,
        data: mjx.Data,
        contact: jax.Array,
    ) -> jax.Array:
        # Penalize foot slip. Always active (not gated on command) -- a
        # planted foot sliding is undesirable whether standing or walking.
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
