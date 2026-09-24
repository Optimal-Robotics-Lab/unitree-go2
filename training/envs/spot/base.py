"""
    Spot Environment Base Class:
"""
import functools

import jax
import jax.numpy as jnp
import numpy as np

from ml_collections import config_dict

import mujoco
from mujoco import mjx

from mujoco_playground._src import mjx_env

from training.envs.spot.config import (
    EnvironmentConfig,
    NoiseConfig,
    DisturbanceConfig,
    CommandConfig,
)
from training.envs.spot import motor
from training.envs.spot import transmission_constants as tc
from training.envs.spot.delay import DelayLine

import training.envs.utilities.filter as filters


def _lookup_id(model: mujoco.MjModel, kind: str, name: str) -> int:
    """Looks up a named model element's id, failing loudly if it's missing."""
    id_ = getattr(model, kind)(name).id
    if id_ == -1:
        raise ValueError(f"{kind} '{name}' not found in model.")
    return id_


# Actuator declaration order in mjcf/controllers/torque_control.xml; the
# leg/arm split is a robot-structure fact (which actuators exist), not a
# task decision (which ones the policy drives) -- `num_leg_joints` lives
# here so both base.py and any task subclass agree on it.
_LEG_ACTUATOR_NAMES = (
    'front_left_hip', 'front_left_thigh', 'front_left_calf',
    'front_right_hip', 'front_right_thigh', 'front_right_calf',
    'rear_left_hip', 'rear_left_thigh', 'rear_left_calf',
    'rear_right_hip', 'rear_right_thigh', 'rear_right_calf',
)
_ARM_ACTUATOR_NAMES = (
    'arm_sh0', 'arm_sh1', 'arm_el0', 'arm_el1', 'arm_wr0', 'arm_wr1', 'arm_f1x',
)
_NUM_LEGS = 4


def _tile_per_leg(values) -> jax.Array:
    """Repeats per-joint-type (abduction, thigh, calf) values for every leg."""
    return jnp.tile(jnp.asarray(values, dtype=jnp.float32), _NUM_LEGS)


# Torso capsule geoms (`<name>_collision`) in mjcf/models/spot_simplified.xml.
_TORSO_COLLISION_NAMES = (
    'torso_upper_left', 'torso_upper_right', 'torso_lower_left', 'torso_lower_right',
)
_KNEE_ACTUATOR_INDICES = tuple(
    i for i, name in enumerate(_LEG_ACTUATOR_NAMES) if name.endswith('_calf')
)


class SpotEnv(mjx_env.MjxEnv):
    """Base class for Spot environments."""

    def __init__(
        self,
        environment_config: EnvironmentConfig = EnvironmentConfig(),
        noise_config: NoiseConfig = NoiseConfig(),
        disturbance_config: DisturbanceConfig = DisturbanceConfig(),
        command_config: CommandConfig = CommandConfig(),
        filter_impl: filters.Filter = filters.NoFilter(),
        **kwargs,
    ) -> None:
        config = config_dict.ConfigDict()
        config.ctrl_dt = environment_config.control_timestep
        config.sim_dt = environment_config.optimizer_timestep
        super().__init__(config)

        self.environment_config = environment_config
        self.filepath = environment_config.mjcf_path

        mj_model = mujoco.MjModel.from_xml_path(str(self.filepath))
        mj_model.opt.timestep = environment_config.optimizer_timestep
        mj_model.opt.ccd_iterations = environment_config.ccd_iterations
        # Increase offscreen framebuffer size to render at higher resolutions.
        mj_model.vis.global_.offwidth = environment_config.render_width
        mj_model.vis.global_.offheight = environment_config.render_height

        self._mj_model = mj_model
        self._mjx_model = mjx.put_model(self._mj_model, impl=environment_config.impl)

        self.step_dt = environment_config.control_timestep
        self.time_step = self._mj_model.opt.timestep
        n_substeps = self.step_dt / self.time_step
        if not np.isclose(n_substeps, round(n_substeps)):
            raise ValueError(
                f'control_timestep ({self.step_dt}) must be an integer multiple '
                f'of optimizer_timestep ({self.time_step}).'
            )
        self._n_substeps = round(n_substeps)
        self._step = functools.partial(self._simulation_step, n_substeps=self._n_substeps)

        self.noise_config = noise_config
        self.disturbance_config = disturbance_config
        self.command_config = command_config
        self.filter = filter_impl

        # Constants Setup:
        self.base_idx = _lookup_id(self._mj_model, 'body', 'body')
        self.robot_mass = self._mj_model.body_subtreemass[self.base_idx]

        self.home_qpos = jnp.array(self._mj_model.keyframe('home').qpos)
        self.home_qvel = jnp.zeros(self._mj_model.nv)
        self.default_pose = jnp.array(self._mj_model.keyframe('home').qpos[7:])
        self.default_ctrl = jnp.array(self._mj_model.keyframe('home').ctrl)
        self.joint_lb, self.joint_ub = self._mj_model.jnt_range[1:].T

        self.nu = self._mj_model.nu
        self.nv = self._mj_model.nv
        self.num_joints = self.nv - 6

        self.num_leg_joints = len(_LEG_ACTUATOR_NAMES)

        # qpos_setpoint = default_pose + action_scale * action (legs only;
        # the arm isn't policy-controlled -- see SpotJoystickEnv).
        self.action_scale = environment_config.action_scale
        if isinstance(self.action_scale, (tuple, list)):
            self.action_scale = _tile_per_leg(self.action_scale)
        elif self.action_scale is None:
            dist_to_upper = (
                self.joint_ub[:self.num_leg_joints] - self.default_pose[:self.num_leg_joints]
            )
            dist_to_lower = (
                self.default_pose[:self.num_leg_joints] - self.joint_lb[:self.num_leg_joints]
            )
            self.action_scale = jnp.minimum(dist_to_upper, dist_to_lower)

        # Actuation pipeline: every actuator is direct-torque (`<motor>`, see
        # torque_control.xml), so a PD law is mandatory -- there's no MuJoCo
        # `<position>` actuator left to fall back on. Legs use prescribed,
        # soft gains from EnvironmentConfig (kp/kv); the arm uses Boston
        # Dynamics' own published gains (ARM_KP/ARM_KV) from
        # transmission_constants.py. `tau_max`/`omega_max` are left `None`
        # (no sourced torque-speed envelope for either) -- `forcerange` in
        # the MJCF still hard-clips torque as before, so `torque_speed_
        # saturate` is left out of the stage tuple rather than fed
        # placeholder numbers.
        actuator_names = list(_LEG_ACTUATOR_NAMES + _ARM_ACTUATOR_NAMES)
        qpos_ids, qvel_ids = motor.resolve_actuator_ids(self._mj_model, actuator_names)
        self.motor_model = motor.MotorModel(
            qpos_ids=qpos_ids,
            qvel_ids=qvel_ids,
            kp=jnp.concatenate([
                _tile_per_leg(environment_config.kp), tc.ARM_KP,
            ]),
            kv=jnp.concatenate([
                _tile_per_leg(environment_config.kv), tc.ARM_KV,
            ]),
        )
        # Torque limits [Nm] of the leg actuators, for saturation diagnostics.
        self.leg_torque_limit = jnp.asarray(
            self._mj_model.actuator_forcerange[:self.num_leg_joints, 1],
        )

        # Actuation delay: domain-randomized per channel in [0, 4ms], sampled
        # per env in SpotJoystickEnv.reset() (see randomize.py's
        # sample_actuation_delay -- not domain_randomize itself, since delay
        # isn't an mjx.Model field). use_interp is fixed True everywhere:
        # the fractional read subsumes the discrete one exactly when a
        # sampled delay lands on a substep boundary, so there's no
        # expressiveness lost by not also randomizing the read mode.
        self.delay_line = DelayLine(
            num_channels=self._mj_model.nu, dt=self.time_step, max_delay=0.005,
        )
        self.actuation_use_interp = jnp.ones(self._mj_model.nu, dtype=bool)

        self.actuation_fn = motor.build_actuation_function((
            motor.delay_stage(self.delay_line),
            motor.control_law,
            motor.knee_torque_limit_stage(jnp.array(_KNEE_ACTUATOR_INDICES)),
        ))

        # Sites:
        feet_names = ['front_left', 'front_right', 'rear_left', 'rear_right']
        self.feet_site_idx = np.array([
            _lookup_id(self._mj_model, 'site', f'{leg}_foot') for leg in feet_names
        ])
        self.imu_site_idx = _lookup_id(self._mj_model, 'site', 'imu')

        # Sensors used by the shared reading utilities below:
        self.feet_position_sensor = [f'{leg}_foot_position' for leg in feet_names]
        self.feet_linear_velocity_sensor = [
            f'{leg}_global_linear_velocity' for leg in feet_names
        ]

        # Contact sensors, exposed 1:1 with the MJCF sensor files. Grouping
        # these into "unwanted"/"terminal" categories is a task decision,
        # not made here.
        self.torso_to_floor_sensor = self._sensor_ids([
            f'{torso}_to_floor' for torso in _TORSO_COLLISION_NAMES
        ])
        self.thigh_to_floor_sensor = self._sensor_ids([
            f'{leg}_thigh_to_floor' for leg in feet_names
        ])
        self.calf_to_floor_sensor = self._sensor_ids([
            f'{leg}_calf_{part}_to_floor'
            for leg in feet_names for part in ('upper', 'lower')
        ])
        # Upper calf capsule only (the knee end): the lower one sits ~3 cm
        # above the floor in a normal stance, so it touches in ordinary strides.
        self.calf_upper_to_floor_sensor = self._sensor_ids([
            f'{leg}_calf_upper_to_floor' for leg in feet_names
        ])
        self.foot_to_floor_sensor = self._sensor_ids([
            f'{leg}_foot_to_floor' for leg in feet_names
        ])
        self.leg_self_collision_sensor = self._sensor_ids([
            f'{leg}_thigh_to_foot' for leg in feet_names
        ])
        leg_to_leg_names = []
        for side in ('left', 'right'):
            front, rear = f'front_{side}', f'rear_{side}'
            leg_to_leg_names.append(f'{front}_foot_to_{rear}_foot')
            for part in ('upper', 'lower'):
                leg_to_leg_names.append(f'{front}_foot_to_{rear}_calf_{part}')
                leg_to_leg_names.append(f'{rear}_foot_to_{front}_calf_{part}')
        self.leg_to_leg_collision_sensor = self._sensor_ids(leg_to_leg_names)
        self.arm_to_torso_collision_sensor = self._sensor_ids([
            f'{arm}_to_{torso}'
            for arm in ('arm_el1', 'arm_wr1', 'left_finger', 'right_finger')
            for torso in _TORSO_COLLISION_NAMES
        ])

    def _sensor_ids(self, names: list[str]) -> np.ndarray:
        return np.array([_lookup_id(self._mj_model, 'sensor', name) for name in names])

    # Physics step: `action` is the raw policy output for the 12 leg joints
    # only; the arm holds its default pose (see SpotJoystickEnv, which
    # isn't policy-controlled). `delay_state` and `delay` are threaded in
    # and back out rather than read off `self`, since both are per-env
    # (delay_state survives across env.step() calls; delay is this env's
    # domain-randomized sample from reset()) -- the caller carries
    # `delay_state` the same way it carries `state.info['filter_state']`.
    def _simulation_step(
        self,
        data: mjx.Data,
        action: jax.Array,
        delay_state: DelayLine.State,
        delay: jax.Array,
        n_substeps: int,
    ) -> tuple[mjx.Data, DelayLine.State]:
        leg_qpos_setpoint = jnp.clip(
            self.default_pose[:self.num_leg_joints] + action * self.action_scale,
            self.joint_lb[:self.num_leg_joints],
            self.joint_ub[:self.num_leg_joints],
        )
        qpos_setpoint = self.default_pose.at[:self.num_leg_joints].set(leg_qpos_setpoint)

        def _substep(
            carry: tuple[mjx.Data, DelayLine.State], unused_t,
        ) -> tuple[tuple[mjx.Data, DelayLine.State], None]:
            data, delay_state = carry
            pipeline_state = motor.ActuationPipelineState(
                data=data,
                motor_model=self.motor_model,
                qpos_setpoint=qpos_setpoint,
                qvel_setpoint=jnp.zeros(self.nu),
                feedforward_torque=jnp.zeros(self.nu),
                torque=jnp.zeros(self.nu),
                delay=delay,
                use_interp=self.actuation_use_interp,
                delay_state=delay_state,
            )
            pipeline_state = self.actuation_fn(pipeline_state)
            new_data = mjx.step(
                self._mjx_model, data.replace(ctrl=pipeline_state.torque),
            )
            return (new_data, pipeline_state.delay_state), None

        (data, delay_state), _ = jax.lax.scan(
            _substep, (data, delay_state), None, length=n_substeps,
        )
        return data, delay_state

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
        return self.get_sensor_data(self._mj_model, data, 'upvector')

    def get_forwardvector(self, data: mjx.Data) -> jax.Array:
        return self.get_sensor_data(self._mj_model, data, 'forwardvector')

    def get_gravity(self, data: mjx.Data) -> jax.Array:
        return data.site_xmat[self.imu_site_idx].T @ jnp.array([0, 0, -1])

    def get_global_linear_velocity(self, data: mjx.Data) -> jax.Array:
        return self.get_sensor_data(self._mj_model, data, 'global_linear_velocity')

    def get_global_angular_velocity(self, data: mjx.Data) -> jax.Array:
        return self.get_sensor_data(self._mj_model, data, 'global_angular_velocity')

    def get_local_linear_velocity(self, data: mjx.Data) -> jax.Array:
        return self.get_sensor_data(self._mj_model, data, 'local_linear_velocity')

    def get_accelerometer(self, data: mjx.Data) -> jax.Array:
        return self.get_sensor_data(self._mj_model, data, 'imu_acceleration')

    def get_gyro(self, data: mjx.Data) -> jax.Array:
        return self.get_sensor_data(self._mj_model, data, 'imu_gyro')

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
        return str(self.filepath)

    @property
    def action_size(self) -> int:
        return self._mjx_model.nu

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model
