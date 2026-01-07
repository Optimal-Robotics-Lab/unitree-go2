from typing import Any, Callable, Tuple, NamedTuple, Protocol, Mapping, TypeVar, Union

import jax.numpy as jnp
from mujoco_playground import mjx_env

Params = Any
PRNGKey = jnp.ndarray
NomralizationParams = Any
NetworkParams = Tuple[NomralizationParams, Params]
PolicyParams = Tuple[NomralizationParams, Params]
ActivationFn = Callable[[jnp.ndarray], jnp.ndarray]
Initializer = Callable[..., Any]

Observation = Union[jnp.ndarray, Mapping[str, jnp.ndarray]]
ObservationSize = Union[int, Mapping[str, Union[Tuple[int, ...], int]]]
Action = jnp.ndarray
PolicyData = Mapping[str, Any]
Metrics = dict[str, Any]

State = mjx_env.State
Env = mjx_env.MjxEnv

NetworkType = TypeVar('NetworkType')


class Transition(NamedTuple):
    observation: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    termination: jnp.ndarray
    next_observation: jnp.ndarray
    extras: Mapping[str, Any]


class Policy(Protocol):
    def __call__(
        self,
        x: jnp.ndarray,
        key: PRNGKey,
    ) -> Tuple[Action, PolicyData]:
        pass


class InputNormalizationFn(Protocol):
    def __call__(
        self,
        x: jnp.ndarray,
        normalization_params: NomralizationParams
    ) -> jnp.ndarray:
        pass


def identity_normalization_fn(
    x: jnp.ndarray,
    normalization_params: NomralizationParams
) -> jnp.ndarray:
    del normalization_params
    return x


class NetworkFactory(Protocol[NetworkType]):
    def __call__(
        self,
        observation_size: int,
        action_size: int,
        input_normalization_fn: InputNormalizationFn = identity_normalization_fn,
    ) -> NetworkType:
        pass
