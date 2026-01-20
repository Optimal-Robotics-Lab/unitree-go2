"""
    Unitree Go2 Environment Configuration:
"""
import flax.struct


@flax.struct.dataclass
class RewardConfig:
    # Rewards:
    tracking_base_pose: float = 1.0
    tracking_orientation: float = 1.0
    tracking_joint_pose: float = 1.0
    # Energy Regularization Terms:
    torque: float = -2e-4
    action_rate: float = -0.01
    acceleration: float = -2.5e-7
    # Penalty Terms:
    base_velocity: float = -1.0
    stand_still: float = -1.0
    feet_contact: float = 0.5
    feet_slip: float = -0.1
    unwanted_contact: float = -1.0
    termination: float = -1.0
    # MuJoCo Terms:
    pose: float = -0.1
    joint_limits: float = -0.5
    # Experimental Terms:
    symmetry: float = -0.1
    # Hyperparameter for exponential kernel:
    base_sigma: float = 0.25
    orientation_sigma: float = 0.25
    pose_sigma: float = 1.0


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
class EnvironmentConfig:
    filename: str = "scene_mjx.xml"
    impl: str = "jax"
    action_scale: float = 0.5
    control_timestep: float = 0.02
    optimizer_timestep: float = 0.004
    nconmax: int = 8 * 8192
    njmax: int = 12 + 48
    terminate_on_contact: bool = False
