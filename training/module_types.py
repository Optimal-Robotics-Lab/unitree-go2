# Original License Notice: https://github.com/google/brax/blob/main/brax/training/types.py

# Copyright 2025 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Any, Callable, Tuple, NamedTuple, Protocol, Mapping, TypeVar, Union

import jax
import flax.struct

import numpy as np
import jax.numpy as jnp
from brax import envs

Params = Any
PRNGKey = jnp.ndarray
NormalizationParams = Any
NetworkParams = Tuple[NormalizationParams, Params]
PolicyParams = Tuple[NormalizationParams, Params]
ActivationFn = Callable[[jnp.ndarray], jnp.ndarray]
Initializer = Callable[..., Any]

Observation = Union[jnp.ndarray, Mapping[str, jnp.ndarray]]
ObservationSize = Union[int, Mapping[str, Union[Tuple[int, ...], int]]]
Action = jnp.ndarray
Value = jnp.ndarray
PolicyData = Mapping[str, Any]
Metrics = dict[str, Any]

State = envs.State
Env = envs.Env

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
        normalization_params: NormalizationParams
    ) -> jnp.ndarray:
        pass


def identity_normalization_fn(
    x: jnp.ndarray,
    normalization_params: NormalizationParams
) -> jnp.ndarray:
    del normalization_params
    return x
