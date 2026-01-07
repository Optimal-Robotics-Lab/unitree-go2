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


from typing import Any, Callable, Tuple, NamedTuple, Protocol, Mapping, Union, List

import flax.nnx as nnx

import jax.numpy as jnp
import mujoco_playground

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

State = mujoco_playground.State
Env = mujoco_playground.MjxEnv

# Custom type for nested structures:
Leaf = Union[jnp.ndarray, float, int]
NestedArray = Union[
    Leaf,
    Mapping[Any, 'NestedArray'],
    List['NestedArray'],
    Tuple['NestedArray', ...],
    nnx.Dict,
    nnx.List,
]


class Transition(NamedTuple):
    observation: NestedArray
    action: NestedArray
    reward: NestedArray
    termination: NestedArray
    next_observation: NestedArray
    extras: Mapping[str, Any]


class Policy(Protocol):
    def __call__(
        self,
        x: NestedArray,
        key: PRNGKey,
    ) -> Tuple[Action, PolicyData]:
        pass
