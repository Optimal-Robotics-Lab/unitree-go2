from typing import Tuple, Any

import abc

import jax
import jax.numpy as jnp
import jax.typing as jtp


from flax import struct


@struct.dataclass
class BaseMotorConfig:
    # Controller Parameters:
    kp: jtp.ArrayLike
    kv: jtp.ArrayLike

    # Motor Parameters:
    kt: jtp.ArrayLike
    tau_base: jtp.ArrayLike
    omega_base: jtp.ArrayLike
    reduction_ratio: jtp.ArrayLike

    # Electrical Parameters:
    v_rated: jtp.ArrayLike
    v_nominal: jtp.ArrayLike
    r_series: jtp.ArrayLike
    r_phase: jtp.ArrayLike

    def __post_init__(self):
        assert bool(jnp.all(self.kp >= 0.0)), "Proportional gain must be >= 0."
        assert bool(jnp.all(self.kv >= 0.0)), "Derivative gain must be >= 0."
        assert bool(jnp.all(self.kt > 0.0)), "Torque constant must be > 0."
        assert bool(jnp.all(self.tau_base > 0.0)), "Base stall torque must be > 0."
        assert bool(jnp.all(self.omega_base > 0.0)), "Base no-load speed must be > 0."
        assert bool(jnp.all(self.reduction_ratio > 0.0)), "Reduction ratio must be > 0."
        assert bool(jnp.all(self.v_rated > 0.0)), "Rated voltage must be > 0."
        assert bool(jnp.all(self.v_nominal >= self.v_rated)), \
            "Nominal voltage must be >= rated voltage."
        assert bool(jnp.all(self.r_series >= 0.0)), "Series resistance must be >= 0."
        assert bool(jnp.all(self.r_phase >= 0.0)), "Phase resistance must be >= 0."


class MotorModel(abc.ABC):
    """Abstract base class for motor models as a PyTree."""
    motor_config: BaseMotorConfig

    @abc.abstractmethod
    def apply(self, *args: Any, **kwargs: Any) -> tuple[jax.Array, dict[str, jax.Array]]:
        """
        Applies motor constraints to the desired torque.
        
        Subclasses should override this method with their specific 
        required input arguments.
        """
        pass
    
    def joint_to_actuator_torque(self, joint_torque: jax.Array) -> jax.Array:
        """Divides torque by the external reduction ratio."""
        return joint_torque / self.motor_config.reduction_ratio

    def joint_to_actuator_velocity(self, joint_velocity: jax.Array) -> jax.Array:
        """Multiplies velocity by the external reduction ratio."""
        return joint_velocity * self.motor_config.reduction_ratio

    def actuator_to_joint_torque(self, actuator_torque: jax.Array) -> jax.Array:
        """Multiplies torque by the external reduction ratio."""
        return actuator_torque * self.motor_config.reduction_ratio

    def actuator_to_joint_velocity(self, actuator_velocity: jax.Array) -> jax.Array:
        """Divides velocity by the external reduction ratio."""
        return actuator_velocity / self.motor_config.reduction_ratio

@struct.dataclass
class PositionControl(MotorModel):
    motor_config: BaseMotorConfig

    def compute_desired_torque(self, target_joint_positions: jax.Array, joint_positions: jax.Array, joint_velocities: jax.Array) -> jax.Array:
        # PD Control Law
        desired_torque = self.motor_config.kp * (target_joint_positions - joint_positions) \
            - self.motor_config.kv * joint_velocities
        
        return jnp.asarray(desired_torque)

    def constrain_torque(self, actuator_torque: jax.Array, actuator_velocities: jax.Array) -> tuple[jax.Array, dict[str, jax.Array]]:
        """
            Applies motor constraints to the desired actuator torque.
            
            Input:
                actuator_torque: The desired actuator torque.
                actuator_velocities: The current actuator velocities.
            
            Returns:
                The constrained actuator torque and a dictionary of metrics.

        """

        # Motor Current Demands:
        clipped_actuator_torque = jnp.clip(actuator_torque, -self.motor_config.tau_base, self.motor_config.tau_base)
        i_motor = jnp.abs(clipped_actuator_torque) / self.motor_config.kt

        # Power Calculation:
        p_mechanical = jnp.maximum(clipped_actuator_torque * actuator_velocities, 0.0)
        p_loss = (i_motor**2) * self.motor_config.r_phase

        # Compute Bus Current and Voltage:
        i_bus = jnp.sum(p_mechanical + p_loss) / self.motor_config.v_nominal
        v_bus = self.motor_config.v_nominal - (i_bus * self.motor_config.r_series)
        v_bus = jnp.clip(v_bus, 0.0, self.motor_config.v_rated)

        # Scale Torque-Speed Curve for Voltage Sag:
        v_ratio = v_bus / self.motor_config.v_rated

        scaled_tau_base = self.motor_config.tau_base * v_ratio
        damping_slope = self.motor_config.tau_base / self.motor_config.omega_base

        # Calculate the available torque:
        available_torque = scaled_tau_base - (damping_slope * jnp.abs(actuator_velocities))
        available_torque = jnp.clip(available_torque, 0.0, scaled_tau_base)

        # Clip the actuator torque to the available range:
        torque = jnp.clip(actuator_torque, -available_torque, available_torque)

        is_voltage_sag = v_bus < self.motor_config.v_rated
        metrics = {
            "v_bus": v_bus,
            "i_bus": i_bus,
            "is_voltage_sag": is_voltage_sag,
        }

        return torque, metrics

    def apply(
        self, target_joint_positions: jax.Array, joint_positions: jax.Array, joint_velocities: jax.Array,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        # Calculate Desired Torque:
        desired_torque = self.compute_desired_torque(target_joint_positions, joint_positions, joint_velocities)
        
        # Convert from joint to actuator frame:
        actuator_torque = self.joint_to_actuator_torque(desired_torque)
        actuator_velocities = self.joint_to_actuator_velocity(joint_velocities)

        # Apply Motor Constraints:
        constrained_torque, metrics = self.constrain_torque(actuator_torque, actuator_velocities)

        # Convert the constrained torque back to the joint frame:
        joint_torque = self.actuator_to_joint_torque(constrained_torque)

        return joint_torque, metrics