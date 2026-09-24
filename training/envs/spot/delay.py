"""
    Actuation Delay:

    Causal, per-channel setpoint delay for online rollouts.

    `DelayLine` is a fixed-length ring buffer of past setpoints, extended with
    a fractional-lag interpolated read so a delay isn't restricted to a whole multiple of
    the push period. `read_discrete` and `read_interp` both read the same
    buffer; `read` computes both and selects per channel.

    Call `push` once per physics substep (not once per control step) so a
    held setpoint's delay resolution matches the substep rate.
"""
import math

import jax
import jax.numpy as jnp

from flax import struct


@struct.dataclass
class DelayLine:
    """Fixed-length causal delay line for per-channel setpoint delay.

    Not batched over parallel envs internally -- vmap `State` the same way
    `mjx.Data` is vmapped elsewhere in this environment.

    Attributes:
        num_channels: Number of independently delayed channels (actuators).
        dt: Period between `push` calls [s] (the physics substep).
        max_delay: Largest delay this buffer must represent [s]; sizes the
            ring buffer and bounds any per-channel delay passed to `read`.
    """
    # Static configuration, not data: excluded from the pytree so it's never
    # traced or vmapped, only `State` is.
    num_channels: int = struct.field(pytree_node=False)
    dt: float = struct.field(pytree_node=False)
    max_delay: float = struct.field(pytree_node=False)

    @property
    def buffer_length(self) -> int:
        """Ring buffer length; one extra slot covers the interp read's ceil."""
        return math.ceil(self.max_delay / self.dt) + 2

    @struct.dataclass
    class State:
        buffer: jax.Array
        pointer: jax.Array
        initialized: jax.Array

    def init(self) -> State:
        """Returns an empty (zero-filled) buffer state."""
        return self.State(
            buffer=jnp.zeros((self.buffer_length, self.num_channels)),
            pointer=jnp.zeros((), dtype=jnp.int32),
            initialized=jnp.zeros((), dtype=jnp.bool_),
        )

    def push(self, state: State, value: jax.Array) -> State:
        """Writes `value` as the newest sample.

        On the first call after `init`, the whole buffer is filled with
        `value` so an immediate read isn't blending in stale zeros before
        the buffer has actually seen `buffer_length` pushes.

        Args:
            state: Current buffer state.
            value: New sample, shape `(num_channels,)`.
        """
        pointer = (state.pointer + 1) % self.buffer_length
        buffer = state.buffer.at[pointer].set(value)
        buffer = jnp.where(state.initialized, buffer, value[None, :])
        return self.State(
            buffer=buffer, pointer=pointer, initialized=jnp.asarray(True),
        )

    def _gather(self, state: State, lag_steps: jax.Array) -> jax.Array:
        index = (state.pointer - lag_steps) % self.buffer_length
        return state.buffer[index, jnp.arange(self.num_channels)]

    def read_discrete(self, state: State, delay: jax.Array) -> jax.Array:
        """Exact per-channel delay, snapped to the nearest substep.

        Args:
            state: Current buffer state.
            delay: Per-channel delay [s], shape `(num_channels,)`.
        """
        lag_steps = jnp.clip(
            jnp.round(delay / self.dt), 0, self.buffer_length - 1,
        ).astype(jnp.int32)
        return self._gather(state, lag_steps)

    def read_interp(self, state: State, delay: jax.Array) -> jax.Array:
        """Smooth per-channel delay, blending the two bracketing samples.

        Args:
            state: Current buffer state.
            delay: Per-channel delay [s], shape `(num_channels,)`.
        """
        lag = jnp.clip(delay / self.dt, 0.0, self.buffer_length - 1)
        lower = jnp.floor(lag).astype(jnp.int32)
        upper = jnp.clip(lower + 1, 0, self.buffer_length - 1)
        fraction = lag - lower.astype(jnp.float32)
        value_lower = self._gather(state, lower)
        value_upper = self._gather(state, upper)
        return value_lower + fraction * (value_upper - value_lower)

    def read(
        self, state: State, delay: jax.Array, use_interp: jax.Array,
    ) -> jax.Array:
        """Per-channel delay under each channel's own mode.

        Args:
            state: Current buffer state.
            delay: Per-channel delay [s], shape `(num_channels,)`.
            use_interp: Per-channel mode select, shape `(num_channels,)`;
                `True` reads `read_interp`, `False` reads `read_discrete`.
        """
        discrete_value = self.read_discrete(state, delay)
        interp_value = self.read_interp(state, delay)
        return jnp.where(use_interp, interp_value, discrete_value)

    def step(
        self,
        state: State,
        value: jax.Array,
        delay: jax.Array,
        use_interp: jax.Array,
    ) -> tuple[jax.Array, State]:
        """Pushes `value` and returns `(delayed_value, new_state)`.

        The single entry point a pipeline stage calls once per substep:
        record this substep's setpoint, then read back what the delay line
        says the actuator should see right now.
        """
        new_state = self.push(state, value)
        delayed_value = self.read(new_state, delay, use_interp)
        return delayed_value, new_state
