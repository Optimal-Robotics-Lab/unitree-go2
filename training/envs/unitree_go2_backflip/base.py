import functools
import os

import jax
import jax.numpy as jnp
import numpy as np

from ml_collections import config_dict

import mujoco
from mujoco import mjx

from mujoco_playground._src import mjx_env

from training.envs.unitree_go2_backflip.config import (
    NoiseConfig,
    DisturbanceConfig,
    CommandConfig,
    EnvironmentConfig,
    MotorConfig,
)

import training.envs.utilities.filter as filters


class UnitreeGo2Env(mjx_env.MjxEnv):
    """Base class for Unitree Go2 environments."""

    def __init__(
        self,
        environment_config: EnvironmentConfig = EnvironmentConfig(),
        noise_config: NoiseConfig = NoiseConfig(),
        disturbance_config: DisturbanceConfig = DisturbanceConfig(),
        command_config: CommandConfig = CommandConfig(),
        motor_config: MotorConfig | None = None,
        filter_impl: filters.Filter = filters.NoFilter(),
        model_parameters: dict | None = None,
        **kwargs,
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

        # Model Override:
        if model_parameters:
            params = model_parameters
            if params is not None:
                for k, v in params.items():
                    if 'actuator_dynprm' in k:
                        value = getattr(mj_model, k)
                        value[:, 0] = v
                        setattr(mj_model, k, value)
                    if 'dof_frictionloss' in k or 'dof_damping' in k or 'dof_armature' in k:
                        value = getattr(mj_model, k)
                        value[6:] = v
                        setattr(mj_model, k, value)
                    if 'qpos0' in k:
                        value = getattr(mj_model, k)
                        value[7:] = v
                        setattr(mj_model, k, value)

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

        # Wrap Step Function:
        self._step = functools.partial(self._simulation_step, n_substeps=self._n_substeps)

        # Parse Configs:
        self.environment_config = environment_config
        self.noise_config = noise_config
        self.disturbance_config = disturbance_config
        self.command_config = command_config
        self.motor_config = motor_config
        self.filter = filter_impl

        # Constants Setup:
        self.floor_geom_idx = self._mj_model.geom('floor').id
        self.base_idx = mujoco.mj_name2id(
            self._mj_model, mujoco.mjtObj.mjOBJ_BODY.value, 'base_link'
        )
        self.robot_mass = self._mj_model.body_subtreemass[self.base_idx]

        self.home_qpos = jnp.array(self._mj_model.keyframe('home').qpos)
        self.home_qvel = jnp.zeros(self._mj_model.nv)
        self.default_pose = jnp.array(self._mj_model.keyframe('home').qpos[7:])
        self.default_ctrl = jnp.array(self._mj_model.keyframe('home').ctrl)
        self.joint_lb, self.joint_ub = self._mj_model.jnt_range[1:].T

        self.action_scale = environment_config.action_scale
        if self.action_scale is None:
            dist_to_upper = self.joint_ub - self.default_ctrl
            dist_to_lower = self.default_ctrl - self.joint_lb
            self.action_scale = jnp.minimum(dist_to_upper, dist_to_lower)

        self.nu = self._mj_model.nu
        self.nv = self._mj_model.nv
        self.num_joints = self.nv - 6

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
            "hind_right_thigh_to_floor",
            "hind_left_thigh_to_floor",
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


    # Custom Step Method to Capture Acutator Pipeline:
    def _simulation_step(self, data: mjx.Data, action: jax.Array, n_substeps: int) -> mjx.Data:

        # Compute Target Joint Positions from Action:
        target_qpos = self.default_pose + action * self.action_scale
        target_qpos = jnp.clip(target_qpos, self.joint_lb, self.joint_ub)

        if self.motor_config is not None:
            def motor_model(mj_data: mjx.Data, target_qpos: jax.Array) -> jax.Array:
                # Extract Joint States:
                joint_positions = mj_data.qpos[7:]
                joint_velocities = mj_data.qvel[6:]

                # PD Control Law
                desired_torque = self.motor_config.kp * (target_qpos - joint_positions) \
                    - self.motor_config.kv * joint_velocities

                # Torque Speed Curve:
                available_torque = self.motor_config.tau_max - (self.motor_config.damping_slope * jnp.abs(joint_velocities))
                available_torque = jnp.maximum(available_torque, 0.0)

                # Apply Torque Limits:
                torque = jnp.clip(desired_torque, -available_torque, available_torque)

                return torque
        else:
            def motor_model(mj_data: mjx.Data, target_qpos: jax.Array) -> jax.Array:
                return target_qpos

        # Run Physics Substeps:
        def _substep(carry: mjx.Data, unused_t) -> tuple[mjx.Data, None]:
            ctrl = motor_model(carry, target_qpos)
            if self.environment_config.impl == 'warp':
                ctrl = ctrl.astype(jnp.float32)
            data = carry.replace(ctrl=ctrl)
            return mjx.step(self._mjx_model, data), None

        # Scan over substeps:
        data, _ = jax.lax.scan(_substep, data, None, length=n_substeps)

        return data

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

    def get_forwardvector(self, data: mjx.Data) -> jax.Array:
        return self.get_sensor_data(self._mj_model, data, "forwardvector")

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
