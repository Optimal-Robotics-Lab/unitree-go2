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
        **kwargs,
    ) -> None:
        super().__init__(
            environment_config=environment_config,
            reward_config=reward_config,
            noise_config=noise_config,
            disturbance_config=disturbance_config,
            **kwargs,
        )

    def reset(self, rng: PRNGKey) -> mjx_env.State:
        # Choose Initial Position and Velocity:
        initial_qpos = self.init_q
        initial_qvel = self.init_qd
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

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        _, rng = jax.random.split(state.info['rng'])

        # Disturbance: (Force based)
        if self.disturbance_config.magnitudes[1] > 0.0:
            state = self.maybe_apply_perturbation(state)

        # Physics step:
        motor_targets = state.data.ctrl + action * self.action_scale
        data = mjx_env.step(
            self._mjx_model, state.data, motor_targets, self._n_substeps,
        )

        imu_height = data.site_xpos[self.imu_site_idx][2]
        joint_angles = data.qpos[7:]
        joint_velocities = data.qvel[6:]
        base_velocity = data.qvel[:6]
        forward_vector = data.site_xmat[self.imu_site_idx] @ jnp.array([1.0, 0.0, 0.0])

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

        # Observation data:
        observation = self.get_observation(
            data,
            state.info,
        )

        # Termination Condition:
        done = self._get_termination(
            data,
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
            'torque': self._cost_torques(data.actuator_force),
            'action_rate': self._cost_action_rate(
                action, state.info['previous_action'],
            ),
            'acceleration': self._cost_acceleration(
                data.qacc,
            ),
            'base_velocity': self._cost_base_velocity(
                base_velocity,
            ),
            'stand_still': self._cost_stand_still(
                forward_vector, joint_angles,
            ),
            'feet_contact': self._cost_feet_contact(feet_contacts, alpha=1.0),
            'feet_slip': self._cost_feet_slip(
                data, feet_contacts
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
        state.info['previous_action'] = jnp.where(
            done, jnp.zeros_like(action), action,
        )
        state.info['previous_velocity'] = jnp.where(
            done, jnp.zeros_like(joint_velocities), joint_velocities,
        )
        state.info['previous_feet_contact'] = jnp.where(
            done, jnp.zeros_like(feet_contacts), feet_contacts,
        )
        state.info['previous_unwanted_contacts'] = jnp.where(
            done, jnp.zeros_like(unwanted_contacts), unwanted_contacts,
        )
        state.info['previous_motor_targets'] = jnp.where(
            done, self.default_ctrl, motor_targets,
        )
        state.info['time'] = jnp.where(done, 0.0, state.info['time'] + self.dt)

        state = state.replace(
            data=data,
            obs=observation,
            reward=reward,
            done=done,
        )
        return state

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
        ])

        accelerometer = self.get_accelerometer(data)
        linear_velocity = self.get_local_linear_velocity(data)
        global_angular_velocity = self.get_global_angular_velocity(data)
        actuator_force = data.actuator_force
        imu_height = jnp.asarray([data.site_xpos[self.imu_site_idx][2]])

        privileged_observation = jnp.concatenate([
            observation,                # 42 or 54
            accelerometer,              # 3
            gyroscope,                  # 3
            projected_gravity,          # 3
            linear_velocity,            # 3
            global_angular_velocity,    # 3
            q,                          # 12
            qd,                         # 12
            actuator_force,             # 12 or 24
            imu_height,                 # 1
        ])

        return {
            'state': observation,
            'privileged_state': privileged_observation,
        }

    def _get_termination(
        self,
        data: mjx.Data,
        termination_contacts: jax.Array,
        terminate_on_contact: bool = False,
    ) -> jax.Array:
        # Termination Condition:
        done = self.get_upvector(data)[-1] < -0.25
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
