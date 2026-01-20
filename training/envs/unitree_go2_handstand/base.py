import os

import jax
import jax.numpy as jnp
import numpy as np

import flax
import flax.serialization

from ml_collections import config_dict

import mujoco
from mujoco import mjx

from mujoco_playground._src import mjx_env

from training.envs.unitree_go2_handstand.config import (
    RewardConfig,
    NoiseConfig,
    DisturbanceConfig,
    EnvironmentConfig,
)


class UnitreeGo2Env(mjx_env.MjxEnv):
    """Base class for Unitree Go2 environments."""

    def __init__(
        self,
        environment_config: EnvironmentConfig = EnvironmentConfig(),
        reward_config: RewardConfig = RewardConfig(),
        noise_config: NoiseConfig = NoiseConfig(),
        disturbance_config: DisturbanceConfig = DisturbanceConfig(),
    ) -> None:
        config = config_dict.ConfigDict()
        config.ctrl_dt = environment_config.control_timestep
        config.sim_dt = environment_config.optimizer_timestep
        super().__init__(config)

        self.filename = f'mjcf/{environment_config.filename}'
        self.filepath = os.path.join(
            os.path.dirname(__file__),
            self.filename,
        )

        mj_model = mujoco.MjModel.from_xml_path(
            self.filepath,
        )
        mj_model.opt.timestep = environment_config.optimizer_timestep
        self._mj_model = mj_model
        self._mjx_model = mjx.put_model(self._mj_model, impl=environment_config.impl)

        # Increase offscreen framebuffer size to render at higher resolutions.
        self._mj_model.vis.global_.offwidth = 3840
        self._mj_model.vis.global_.offheight = 2160

        self.step_dt = environment_config.control_timestep
        self.time_step = self._mj_model.opt.timestep
        self._n_substeps = int(self.step_dt / self.time_step)
        self._mj_model.opt.ccd_iterations = 20

        # Parse Configs:
        self.base_sigma = reward_config.base_sigma
        self.orientation_sigma = reward_config.orientation_sigma
        self.pose_sigma = reward_config.pose_sigma
        reward_config_dict = flax.serialization.to_state_dict(reward_config)
        del reward_config_dict['base_sigma']
        del reward_config_dict['orientation_sigma']
        del reward_config_dict['pose_sigma']
        self.reward_config = reward_config_dict

        self.environment_config = environment_config
        self.noise_config = noise_config
        self.disturbance_config = disturbance_config

        # Constants Setup:
        self.floor_geom_idx = self._mj_model.geom('floor').id
        self.base_idx = mujoco.mj_name2id(
            self._mj_model, mujoco.mjtObj.mjOBJ_BODY.value, 'base_link'
        )
        self.base_link_mass = self._mj_model.body_subtreemass[self.base_idx]

        self.action_scale = environment_config.action_scale
        self.init_q = jnp.array(self._mj_model.keyframe('home').qpos)
        self.init_qd = jnp.zeros(self._mj_model.nv)
        self.default_pose = jnp.array(self._mj_model.keyframe('home').qpos[7:])
        self.default_ctrl = jnp.array(self._mj_model.keyframe('home').ctrl)
        self.joint_lb, self.joint_ub = self._mj_model.jnt_range[1:].T

        # Soft Joint Limits:
        c = (self.joint_lb + self.joint_ub) / 2.0
        r = self.joint_ub - self.joint_lb
        self.soft_joint_limit_factor = 0.9
        self.soft_lb = c - 0.5 * r * self.soft_joint_limit_factor
        self.soft_ub = c + 0.5 * r * self.soft_joint_limit_factor

        self.num_joints = self._mj_model.nv - 6
        self.nu = self._mj_model.nu

        # Task Pose:
        self.footstand_q = jnp.array(self._mj_model.keyframe('footstand').qpos)
        self.footstand_pose = jnp.array(self._mj_model.keyframe('footstand').qpos[7:])
        self.desired_height = 0.53
        self.tracking_vector = jnp.array([0.0, 0.0, 1.0])
        self.target_pose_threshold = 0.9

        # Sites and Bodies:
        feet_geom = [
            'front_right_foot_collision',
            'front_left_foot_collision',
            'hind_right_foot_collision',
            'hind_left_foot_collision',
        ]
        feet_geom_idx = [
            self._mj_model.geom(name).id for name in feet_geom
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
            mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_SITE.value, f)
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
            mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_BODY.value, c)
            for c in calf_body
        ]
        assert not any(id_ == -1 for id_ in calf_body_idx), 'Body not found.'
        self.calf_body_idx = np.array(calf_body_idx)
        imu_site_idx = mujoco.mj_name2id(
            self._mj_model, mujoco.mjtObj.mjOBJ_SITE.value, 'imu'
        )
        assert not any(id_ == -1 for id_ in [imu_site_idx]), 'IMU site not found.'
        self.imu_site_idx = np.array(imu_site_idx)

        # Sensors:
        self.feet_position_sensor = [
            "front_right_position",
            "front_left_position",
            "hind_right_position",
            "hind_left_position",
        ]
        self.feet_linear_velocity_sensor = [
            "front_right_global_linear_velocity",
            "front_left_global_linear_velocity",
            "hind_right_global_linear_velocity",
            "hind_left_global_linear_velocity",
        ]

        # Contact Sensors:
        feet_sensor_names = [
            "front_right_foot_to_floor",
            "front_left_foot_to_floor",
            "hind_right_foot_to_floor",
            "hind_left_foot_to_floor",
        ]
        self.feet_contact_sensor = [
            self._mj_model.sensor(f'{foot_sensor_name}').id
            for foot_sensor_name in feet_sensor_names
        ]

        unwanted_contact_sensor_names = [
            "front_right_calf_upper_to_floor",
            "front_right_calf_lower_to_floor",
            "front_left_calf_upper_to_floor",
            "front_left_calf_lower_to_floor",
            "hind_right_calf_upper_to_floor",
            "hind_right_calf_lower_to_floor",
            "hind_left_calf_upper_to_floor",
            "hind_left_calf_lower_to_floor",
        ]
        self.unwanted_contact_sensor = [
            self._mj_model.sensor(f'{sensor_name}').id
            for sensor_name in unwanted_contact_sensor_names
        ]

        termination_sensor_names = [
            "left_torso_to_floor",
            "right_torso_to_floor",
        ]
        termination_sensor_names.extend(unwanted_contact_sensor_names)
        self.termination_contact_sensor = [
            self._mj_model.sensor(f'{termination_sensor_name}').id
            for termination_sensor_name in termination_sensor_names
        ]
        self.terminate_on_contact = environment_config.terminate_on_contact

        # Observation Size:
        self.num_observations = 30 + self.nu
        self.num_privileged_observations = self.num_observations + 40 + self.nu

    # Sensor readings.
    @staticmethod
    def get_sensor_data(
        model: mujoco.MjModel, data: mjx.Data, sensor_name: str
    ) -> jax.Array:
        """Gets sensor data given sensor name."""
        sensor_id = model.sensor(sensor_name).id
        sensor_adr = model.sensor_adr[sensor_id]
        sensor_dim = model.sensor_dim[sensor_id]
        return data.sensordata[sensor_adr: sensor_adr + sensor_dim]

    def get_upvector(self, data: mjx.Data) -> jax.Array:
        return self.get_sensor_data(self._mj_model, data, "upvector")

    def get_gravity(self, data: mjx.Data) -> jax.Array:
        return data.site_xmat[self.imu_site_idx].T @ jnp.array([0, 0, -1])

    def get_global_linear_velocity(self, data: mjx.Data) -> jax.Array:
        return self.get_sensor_data(
            self._mj_model, data, "global_linear_velocity"
        )

    def get_global_angular_velocity(self, data: mjx.Data) -> jax.Array:
        return self.get_sensor_data(
            self._mj_model, data, "global_angular_velocity"
        )

    def get_local_linear_velocity(self, data: mjx.Data) -> jax.Array:
        return self.get_sensor_data(
            self._mj_model, data, "local_linear_velocity"
        )

    def get_accelerometer(self, data: mjx.Data) -> jax.Array:
        return self.get_sensor_data(
            self._mj_model, data, "imu_acceleration"
        )

    def get_gyro(self, data: mjx.Data) -> jax.Array:
        return self.get_sensor_data(self._mj_model, data, "imu_gyro")

    def get_feet_position(self, data: mjx.Data) -> jax.Array:
        return jnp.vstack([
            self.get_sensor_data(self._mj_model, data, sensor_name)
            for sensor_name in self.feet_position_sensor
        ])

    def get_feet_velocity(self, data: mjx.Data) -> jax.Array:
        return jnp.vstack([
            self.get_sensor_data(self._mj_model, data, sensor_name)
            for sensor_name in self.feet_linear_velocity_sensor
        ])

    # Accessors.

    @property
    def xml_path(self) -> str:
        return self.filepath

    @property
    def action_size(self) -> int:
        return self._mjx_model.nu

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model
