import jax
import jax.numpy as jnp
import jax.typing as jtp

from flax import struct


@struct.dataclass
class BaseBatteryConfig:
    """Battery Configuration Struct"""
    capacity_ah: jtp.ArrayLike          # Capacity in ampere-hours (Ah)
    r_s: jtp.ArrayLike                  # Pack series resistance (Ohms)
    r_p: jtp.ArrayLike                  # Polarization resistance (Ohms)
    c_p: jtp.ArrayLike                  # Polarization capacitance (Farads)
    i_continuous: jtp.ArrayLike         # Continuous safe current (Amps)
    i_peak_allowed: jtp.ArrayLike       # Safe transient peak limit (Amps)
    regen_current_limit: jtp.ArrayLike  # Maximum allowable charging current (Amps)
    thermal_threshold: jtp.ArrayLike    # I^2t trip threshold (A^2s)
    v_nominal: jtp.ArrayLike            # Nominal voltage (V)
    r_phase: jtp.ArrayLike              # Phase resistance (Ohms)

    @property
    def kt(self) -> jtp.ArrayLike:
        """Must be implemented by subclasses."""
        raise NotImplementedError

    @property
    def ke(self) -> jtp.ArrayLike:
        """Must be implemented by subclasses."""
        raise NotImplementedError


@struct.dataclass
class EquivalentCircuitModel:
    battery_config: BaseBatteryConfig
    dt: float

    @struct.dataclass
    class State:
        soc: jax.Array
        v_p: jax.Array
        s_thermal: jax.Array
        v_bus: jax.Array

    def init(self) -> State:
        initial_soc = 1.0
        initial_v_bus = 24.0 + (9.6 * initial_soc)
        return self.State(
            soc=jnp.array(initial_soc, dtype=jnp.float32),
            v_p=jnp.zeros((), dtype=jnp.float32),
            s_thermal=jnp.zeros((), dtype=jnp.float32),
            v_bus=jnp.array(initial_v_bus, dtype=jnp.float32),
        )

    def apply(self, actuator_torque: jax.Array, actuator_velocities: jax.Array, state: State) -> tuple[jax.Array, State]:
        # Extract q-axis phase current:
        i_q = jnp.abs(actuator_torque) / self.battery_config.kt
        
        # Estimate total power draw:
        p_mech = jnp.sum(jnp.maximum(actuator_torque * actuator_velocities, 0.0))
        p_loss = jnp.sum((i_q**2) * self.battery_config.r_phase)
        
        # Total pack current (I = P / V):
        safe_v_bus = jnp.maximum(state.v_bus, 18.0)
        i_pack = (p_mech + p_loss) / safe_v_bus

        # State of Charge (Coulomb Counting):
        soc_next = state.soc - (self.dt / (3600.0 * self.battery_config.capacity_ah)) * i_pack
        v_oc = 24.0 + 9.6 * soc_next  

        # RC Polarization Branch:
        decay = jnp.exp(-self.dt / (self.battery_config.r_p * self.battery_config.c_p))
        v_p_next = state.v_p * decay + self.battery_config.r_p * (1.0 - decay) * i_pack

        # Dynamic Bus Voltage (Voltage Sag):
        v_bus_next = v_oc - v_p_next - (i_pack * self.battery_config.r_s)

        # Update I^2t Thermal Accumulator:
        thermal_increase = ((i_pack**2) - (self.battery_config.i_continuous**2)) * self.dt
        s_thermal_next = jnp.maximum(0.0, state.s_thermal + thermal_increase)

        # BMS Throttling:
        thermal_limit_trigger = s_thermal_next >= self.battery_config.thermal_threshold
        i_limit = jnp.where(
            thermal_limit_trigger, 
            self.battery_config.i_continuous, 
            self.battery_config.i_peak_allowed,
        )

        # Actuator Dynamic Limits:
        i_max_local = i_limit / 12.0
        i_min_local = self.battery_config.regen_current_limit / 12.0
        
        max_driving_torque = i_max_local * self.battery_config.kt
        max_braking_torque = jnp.abs(i_min_local * self.battery_config.kt)

        # Calculate dynamic speed limit based on Back-EMF:
        safe_v_bus_clip = jnp.maximum(v_bus_next, 0.0) 
        omega_limit = (safe_v_bus_clip - (i_max_local * self.battery_config.r_phase)) / self.battery_config.ke

        # Actuator Clipping:
        clipped_actuator_torque = jnp.where(
            jnp.abs(actuator_velocities) > omega_limit,
            0.0,
            jnp.clip(actuator_torque, -max_braking_torque, max_driving_torque)
        )

        next_state = self.State(
            soc=soc_next,
            v_p=v_p_next,
            s_thermal=s_thermal_next,
            v_bus=v_bus_next,
        )

        return clipped_actuator_torque, next_state
