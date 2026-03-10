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
    # Cost of Transport Terms:
    # cost_of_transport_reward: float = 0.5
    # cost_of_transport_penalty: float = -0.001
    # Power Regularization Terms:
    electrical_power: float = -1.5e-3
    gravitational_power: float = -6e-3
    # Energy Regularization Terms:
    energy: float = -1e-5
    action_rate: float = -0.01
    acceleration: float = -2.5e-7
    # Auxilary Terms:
    termination: float = -1.0
    unwanted_contact: float = -0.5
    # Gait Reward Terms:
    impact: float = -0.5
    foot_slip: float = -0.1
    # Hyperparameter for exponential kernel:
    kernel_sigma: float = 0.25


@flax.struct.dataclass
class NoiseConfig:
    joint_position: float = 0.05
    joint_velocity: float = 1.5
    linear_velocity: float = 0.1
    gyroscope: float = 0.2
    gravity_vector: float = 0.05
    contact_dropout: float = 0.95


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


@flax.struct.dataclass
class MotorConfig:
    kp: float = 35.0
    kv: float = 0.5
    kt: float = 0.63895
    tau_base: float = 23.7
    omega_base: float = 30.0
    reduction_ratio: jax.Array = flax.struct.field(
        default_factory=lambda: jnp.array([1.0, 1.0, 45.43 / 23.7] * 4),
    )
    # Electrical Components:
    working_voltage: float = 24.0
    resistance: float = 0.5
    regen_efficiency: float = 0.3

    @property
    def tau_max(self) -> jax.Array:
        return self.tau_base * self.reduction_ratio

    @property
    def omega_max(self) -> jax.Array:
        return self.omega_base / self.reduction_ratio

    @property
    def damping_slope(self) -> jax.Array:
        return self.tau_max / self.omega_max
