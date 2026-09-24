"""
    Spot Environment Configuration:
"""
from pathlib import Path

import jax
import jax.numpy as jnp

import flax.struct


_DEFAULT_MJCF_PATH = Path(__file__).resolve().parent / 'mjcf' / 'scene_mjx_simplified.xml'


@flax.struct.dataclass
class EnvironmentConfig:
    mjcf_path: Path = _DEFAULT_MJCF_PATH
    impl: str = 'jax'
    control_timestep: float = 0.02
    optimizer_timestep: float = 0.001
    nconmax: int = 20 * 8192
    naccdmax: int = 0
    njmax: int = 50
    render_width: int = 3840
    render_height: int = 2160
    ccd_iterations: int = 20
    # Leg PD gains (abduction, thigh, calf), tiled across the 4 legs: a soft,
    # damped law (kv gives a damping ratio of ~0.7) the policy compensates
    # for. Check candidates with tuning/pd_design.py -- the thigh in particular
    # folds under load below kp ~60 (a bistable sag), so don't derive it from
    # a static holding torque.
    kp: tuple[float, float, float] = (60.0, 34.0, 152.0)
    kv: tuple[float, float, float] = (4.6, 3.5, 5.9)
    # qpos_setpoint = default_pose + action_scale * action (legs only; see
    # base.py) [rad]. A float applies to every leg joint, a 3-tuple is
    # (abduction, thigh, calf) tiled across the legs, and `None` auto-derives
    # per-joint from the smaller distance to either joint limit around the
    # default pose. The default is 0.75 * torque_limit / kp per joint type, so
    # a full-range action commands 75% of the torque limit: enough authority
    # without the bang-bang that a scale reaching the limit invites.
    action_scale: float | tuple[float, float, float] | None = (0.56, 0.99, 0.56)


@flax.struct.dataclass
class RewardWeights:
    # Rewards:
    tracking_linear_velocity: float = 1.5
    tracking_angular_velocity: float = 0.75
    # Orientation Regularization Terms:
    orientation_regularization: float = -2.5
    linear_z_velocity: float = -2.0
    angular_xy_velocity: float = -0.05
    # Energy Regularization Terms:
    torque: float = -2e-4
    action_rate: float = -0.2
    acceleration: float = -2.5e-7
    # Auxilary Terms:
    stand_still: float = -1.0
    termination: float = -1.0
    unwanted_contact: float = -0.5
    # Gait Reward Terms:
    foot_slip: float = -0.1
    air_time: float = 0.25
    foot_clearance: float = 0.5
    gait_timing_variance: float = -1.0
    synchronized_contact: float = -0.1


@flax.struct.dataclass
class RewardHyperparameters:
    # Gait:
    target_air_time: float = 0.5
    mode_time: float = 0.3
    command_threshold: float = 0.0
    velocity_threshold: float = 0.5
    # Linear Velocity Tracking Ramp: above `ramp_at_vel` [m/s] commanded
    # speed, the tracking reward is scaled up by `ramp_rate` per additional
    # m/s, so high-speed tracking isn't dominated by low-speed stationkeeping
    # (matches IsaacLab's `base_linear_velocity_reward`).
    ramp_at_vel: float = 1.0
    ramp_rate: float = 0.5
    # Foot Clearance:
    target_foot_height: float = 0.1
    foot_clearance_velocity_scale: float = 2.0
    foot_clearance_sigma: float = 0.01
    # Stand Still: scales the pose-deviation cost up at rest (deviation from
    # default pose is penalized at all times, more so when the robot isn't
    # commanded to move).
    stand_still_scale: float = 2.0
    # Synchronized Contact: number of steps a touchdown/liftoff event stays
    # "recent" for, so near-simultaneous (not just exactly simultaneous)
    # transitions still count as synchronized.
    window_steps: int = 3
    # Exponential kernel:
    kernel_sigma: float = 0.25


@flax.struct.dataclass
class RewardConfig:
    weights: RewardWeights = flax.struct.field(
        default_factory=RewardWeights,
    )
    hyperparameters: RewardHyperparameters = flax.struct.field(
        default_factory=RewardHyperparameters,
    )


@flax.struct.dataclass
class NoiseConfig:
    # Values match IsaacLab's official Spot flat-terrain policy observation
    # noise (`SpotObservationsCfg.PolicyCfg`, `Unoise` ranges).
    joint_position: float = 0.05
    joint_velocity: float = 0.5
    linear_velocity: float = 0.1
    angular_velocity: float = 0.1
    gravity_vector: float = 0.05


@flax.struct.dataclass
class DisturbanceConfig:
    wait_times: list[float] = flax.struct.field(
        default_factory=lambda: [1.0, 3.0],
    )
    durations: list[float] = flax.struct.field(
        default_factory=lambda: [0.05, 0.2],
    )
    magnitudes: list[float] = flax.struct.field(
        default_factory=lambda: [0.0, 3.0],
    )


@flax.struct.dataclass
class CommandConfig:
    command_range: jax.Array = flax.struct.field(
        default_factory=lambda: jnp.array([1.5, 1.0, 1.2]),
    )
    single_command_probability: float = 0.0
    command_mask_probability: float = 0.9
    command_frequency: list[float] = flax.struct.field(
        default_factory=lambda: [1.0, 5.0],
    )
