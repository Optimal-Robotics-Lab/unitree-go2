from typing import Tuple, Any

import abc

import jax
import jax.numpy as jnp
import jax.typing as jtp


from flax import struct


@struct.dataclass
class BaseMotorConfig:
    kp: jtp.ArrayLike
    kv: jtp.ArrayLike
    tau_base: jtp.ArrayLike
    omega_base: jtp.ArrayLike
    reduction_ratio: jtp.ArrayLike

    @property
    def kt(self) -> jtp.ArrayLike:
        """Must be implemented by subclasses."""
        raise NotImplementedError

    @property
    def tau_max(self) -> jtp.ArrayLike:
        return self.tau_base * self.reduction_ratio

    @property
    def omega_max(self) -> jtp.ArrayLike:
        return self.omega_base / self.reduction_ratio

    @property
    def damping_slope(self) -> jtp.ArrayLike:
        return self.tau_max / self.omega_max


class MotorModel(abc.ABC):
    """Abstract base class for motor models as a PyTree."""
    motor_config: BaseMotorConfig

    @abc.abstractmethod
    def compute_desired_torque(self, *args: Any, **kwargs: Any) -> jax.Array:
        """
        Computes the applied motor torque.
        
        Subclasses should override this method with their specific 
        required input arguments.
        """
        pass

    @abc.abstractmethod
    def apply(self, *args: Any, **kwargs: Any) -> jax.Array:
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
    

@struct.dataclass
class PositionControl(MotorModel):
    motor_config: BaseMotorConfig

    def compute_desired_torque(self, target_joint_positions: jax.Array, joint_positions: jax.Array, joint_velocities: jax.Array) -> jax.Array:
        # PD Control Law
        desired_torque = self.motor_config.kp * (target_joint_positions - joint_positions) \
            - self.motor_config.kv * joint_velocities
        
        return jnp.asarray(desired_torque)
    
    def apply(self, desired_torque: jax.Array, joint_velocities: jax.Array) -> jax.Array:
        # Torque Speed Curve:
        available_torque = self.motor_config.tau_max - (self.motor_config.damping_slope * jnp.abs(joint_velocities))
        available_torque = jnp.maximum(available_torque, 0.0)

        # Apply Torque Limits:
        torque = jnp.clip(desired_torque, -available_torque, available_torque)

        return torque
