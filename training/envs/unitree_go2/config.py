"""
    Unitree Go2 Environment Configuration:
"""
import jax
import jax.numpy as jnp

import flax.struct


@flax.struct.dataclass
class RewardConfig:
    # Rewards:
    tracking_linear_velocity: float = 1.5
    tracking_angular_velocity: float = 0.75
    # Orientation Regularization Terms:
    orientation_regularization: float = -2.5
    linear_z_velocity: float = -2.0
    angular_xy_velocity: float = -0.05
    # Energy Regularization Terms:
    torque: float = -2e-4
    action_rate: float = -0.01
    acceleration: float = -2.5e-7
    # Auxilary Terms:
    stand_still: float = -1.0
    termination: float = -1.0
    unwanted_contact: float = -0.5
    # Gait Reward Terms:
    foot_slip: float = -0.1
    air_time: float = 0.25
    foot_clearance: float = 0.5
    gait_variance: float = -1.0
    # Gait Hyperparameters:
    target_air_time: float = 0.5
    mode_time: float = 0.3
    command_threshold: float = 0.0
    velocity_threshold: float = 0.5
    # Foot Clearance Reward Terms:
    target_foot_height: float = 0.1
    foot_clearance_velocity_scale: float = 2.0
    foot_clearance_sigma: float = 0.05
    # Hyperparameter for exponential kernel:
    kernel_sigma: float = 0.25


@flax.struct.dataclass
class NoiseConfig:
    joint_position: float = 0.05
    joint_velocity: float = 1.5
    gyroscope: float = 0.2
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


@flax.struct.dataclass
class EnvironmentConfig:
    filename: str = "scene_mjx.xml"
    impl: str = "jax"
    action_scale: float = 0.5
    control_timestep: float = 0.02
    optimizer_timestep: float = 0.004
    nconmax: int = 8 * 8192
    njmax: int = 12 + 48
