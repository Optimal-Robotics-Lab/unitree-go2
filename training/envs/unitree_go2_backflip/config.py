"""
    Unitree Go2 Environment Configuration:
"""
import jax
import jax.numpy as jnp

import flax.struct

from training.envs.utilities.ecm import BaseBatteryConfig
from training.envs.utilities.motor_model import BaseMotorConfig


@flax.struct.dataclass
class RewardConfig:
    # Rewards:
    tracking_height_reference: float = 1.0
    tracking_pitch_reference: float = 1.0
    spin: float = 2.0
    brake: float = 0.5
    # Orientation Regularization Terms:
    unwanted_spin: float = -2.0
    pose_regularization: float = -0.01
    orientation_regularization: float = -0.5
    # Energy Regularization Terms:
    torque: float = -2e-4
    action_rate: float = -0.01
    acceleration: float = -2.5e-7
    mechanical_power: float = 0.0
    # Landing Regularization Terms:
    dof_limit: float = -0.5
    base_clearance: float = -1.0
    # Auxilary Terms:
    stand_still: float = -1.0
    foot_slip: float = -0.5
    termination: float = -1.0
    unwanted_contact: float = -0.5
    # Hyperparameter for exponential kernel:
    height_sigma: float = 0.05
    brake_sigma: float = 1.0


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
    command_frequency: list[float] = flax.struct.field(
        default_factory=lambda: [3.0, 5.0],
    )


@flax.struct.dataclass
class EnvironmentConfig:
    filename: str = "scene_mjx.xml"
    impl: str = "jax"
    action_scale: float | None = 0.5
    control_timestep: float = 0.02
    optimizer_timestep: float = 0.004
    terminate_on_unwanted_contacts: bool = False
    terminate_on_extreme_landing_compression: bool = False
    nconmax: int = 20 * 8192
    naccdmax: int = 0
    njmax: int = 50


@flax.struct.dataclass
class MotorConfig(BaseMotorConfig):
    kp: float = 35.0
    kv: float = 0.5
    tau_base: float = 23.7
    omega_base: float = 30.0
    reduction_ratio: jax.Array = flax.struct.field(
        default_factory=lambda: jnp.array([1.0, 1.0, 45.43 / 23.7] * 4),
    )

    # From the Simplexity Go2 Motor Analysis:
    internal_gear_ratio: float = 1 + (47.0 / 9.0)
    kt_q: float = 0.26
    ke_q: float = 0.26

    @property
    def kt(self) -> float:
        return self.kt_q * self.internal_gear_ratio

@flax.struct.dataclass
class BatteryConfig(BaseBatteryConfig):
    """Battery parameters for the Unitree Go2 BT2-05 pack."""
    # BMS Parameters:
    capacity_ah: float = 8.0            # Capacity in ampere-hours (Ah)
    r_s: float = 0.15                   # Series resistance (Ohms)
    r_p: float = 0.05                   # Polarization resistance (Ohms)
    c_p: float = 40.0                   # Polarization capacitance (Farads)
    i_continuous: float = 30.0          # Continuous safe current (Amps)
    i_peak_allowed: float = 120.0       # Safe transient peak limit (Amps)
    regen_current_limit: float = -3.5   # Maximum allowable charging current (Amps)
    thermal_threshold: float = 500.0    # I^2t trip threshold (A^2s)
    v_nominal: float = 29.6             # Nominal voltage (V)

    # FOC Parameters:
    internal_gear_ratio: float = 1 + (47.0 / 9.0)
    kt_q: float = 0.26
    ke_q: float = 0.26
    r_phase: float = 0.66

    @property
    def kt(self) -> float:
        return self.kt_q * self.internal_gear_ratio

    @property
    def ke(self) -> float:
        return self.ke_q * self.internal_gear_ratio
