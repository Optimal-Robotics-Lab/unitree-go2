from typing import Tuple, Any

import abc

import jax
import jax.numpy as jnp

from flax import struct


class Filter(abc.ABC):
    """Abstract base class for action filters as a PyTree."""
    # We define action_dim as a field that all subclasses must have
    action_dim: int

    @abc.abstractmethod
    def init(self) -> Any:
        """Initializes and returns the filter state."""
        pass

    @abc.abstractmethod
    def apply(self, action: jax.Array, state: Any) -> Tuple[jax.Array, Any]:
        """Returns (filtered_action, new_filter_state)."""
        pass

    @abc.abstractmethod
    def get_observation(self, state: Any) -> jax.Array:
        """Returns the specific filter history required for the RL observation."""
        pass

    @property
    def observation_size(self) -> int:
        """Computes observation size dynamically based on a dummy init."""
        dummy_state = self.init()
        return self.get_observation(dummy_state).shape[-1]


@struct.dataclass
class NoFilter(Filter):
    action_dim: int = 0

    @struct.dataclass
    class State:
        empty: jax.Array

    def init(self) -> State:
        return self.State(empty=jnp.zeros((0,)))

    def apply(self, action: jax.Array, state: State) -> Tuple[jax.Array, State]:
        return action, state
        
    def get_observation(self, state: State) -> jax.Array:
        return state.empty


@struct.dataclass
class FirstOrderFilter(Filter):
    action_dim: int
    alpha: float

    @struct.dataclass
    class State:
        last_filtered_action: jax.Array

    def init(self) -> State:
        return self.State(last_filtered_action=jnp.zeros((self.action_dim,)))

    def apply(self, action: jax.Array, state: State) -> Tuple[jax.Array, State]:
        filtered_action = self.alpha * action + (1.0 - self.alpha) * state.last_filtered_action
        return filtered_action, self.State(last_filtered_action=filtered_action)

    def get_observation(self, state: State) -> jax.Array:
        return state.last_filtered_action


@struct.dataclass
class SecondOrderFilter(Filter):
    action_dim: int
    b0: float
    b1: float
    b2: float
    a1: float
    a2: float

    @struct.dataclass
    class State:
        x_t1: jax.Array
        x_t2: jax.Array
        y_t1: jax.Array
        y_t2: jax.Array

    def init(self) -> State:
        zeros = jnp.zeros((self.action_dim,))
        return self.State(x_t1=zeros, x_t2=zeros, y_t1=zeros, y_t2=zeros)

    def apply(self, action: jax.Array, state: State) -> Tuple[jax.Array, State]:
        # y[t] = b0*x[t] + b1*x[t-1] + b2*x[t-2] - a1*y[t-1] - a2*y[t-2]
        filtered_action = (
            self.b0 * action + self.b1 * state.x_t1 + self.b2 * state.x_t2 
            - self.a1 * state.y_t1 - self.a2 * state.y_t2
        )
        new_state = self.State(
            x_t1=action, 
            x_t2=state.x_t1,
            y_t1=filtered_action, 
            y_t2=state.y_t1
        )
        return filtered_action, new_state

    def get_observation(self, state: State) -> jax.Array:
        target_velocity = state.y_t1 - state.y_t2
        return jnp.concatenate([state.y_t1, target_velocity], axis=-1)
