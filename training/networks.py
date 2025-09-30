import dataclasses

from typing import Any, Callable, Sequence, Mapping

import jax
import jax.numpy as jnp
from flax import linen as nn

# Custom types:
import training.module_types as types
ActivationFn = Callable[[jnp.ndarray], jnp.ndarray]
Initializer = Callable[..., Any]


@dataclasses.dataclass
class FeedForwardNetwork:
    init: Callable[..., Any]
    apply: Callable[..., Any]


# Helper function for observation keys:
def _get_observation_size(
    observation_size: types.ObservationSize, observation_key: str
) -> int:
    observation_size = observation_size[observation_key] if isinstance(observation_size, Mapping) else observation_size
    return jax.tree.flatten(observation_size)[0][-1]


class Encoder(nn.Module):
    input_shape: Sequence[int]
    layer_sizes: Sequence[int]
    activation: ActivationFn = nn.tanh
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform()
    bias: bool = True
    layer_normalization: bool = False

    @nn.compact
    def __call__(self, x: jax.Array):
        for i, layer_size in enumerate(self.layer_sizes):
            x = nn.Conv(
                features=layer_size,
                kernel_init=self.kernel_init,
                use_bias=self.bias,
                name=f"conv_{i}",
            )(x)
            x = self.activation(x)
            if self.layer_normalization:
                x = nn.LayerNorm(name=f"layer_norm_{i}")(x)
        return x


class MLP(nn.Module):
    layer_sizes: Sequence[int]
    activation: ActivationFn = nn.tanh
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform()
    activate_final: bool = False
    bias: bool = True
    layer_normalization: bool = False

    @nn.compact
    def __call__(self, x: jax.Array):
        for i, layer_size in enumerate(self.layer_sizes):
            x = nn.Dense(
                features=layer_size,
                kernel_init=self.kernel_init,
                use_bias=self.bias,
                name=f"dense_{i}",
            )(x)
            if i != len(self.layer_sizes) - 1 or self.activate_final:
                x = self.activation(x)
                if self.layer_normalization:
                    x = nn.LayerNorm(name=f"layer_norm_{i}")(x)
        return x


def make_policy_network(
    input_size: types.ObservationSize,
    output_size: int,
    input_normalization_fn: types.InputNormalizationFn = types
    .identity_normalization_fn,
    layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = nn.tanh,
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform(),
    bias: bool = True,
    layer_normalization: bool = False,
    observation_key: str = "state",
) -> FeedForwardNetwork:
    """Intializes a policy network."""
    policy_network = MLP(
        layer_sizes=list(layer_sizes) + [output_size],
        activation=activation,
        kernel_init=kernel_init,
        bias=bias,
        layer_normalization=layer_normalization,
    )

    def apply(normalization_params, policy_params, x):
        x = input_normalization_fn(x, normalization_params)
        x = x if isinstance(x, jnp.ndarray) else x[observation_key]
        return policy_network.apply(policy_params, x)

    observation_size = _get_observation_size(
        input_size, observation_key
    )
    dummy_input = jnp.zeros((1, observation_size))
    return FeedForwardNetwork(
        init=lambda key: policy_network.init(key, dummy_input), apply=apply,
    )


def make_value_network(
    input_size: types.ObservationSize,
    input_normalization_fn: types.InputNormalizationFn = types
    .identity_normalization_fn,
    layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = nn.tanh,
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform(),
    bias: bool = True,
    layer_normalization: bool = False,
    observation_key: str = "state",
) -> FeedForwardNetwork:
    """Intializes a value network."""
    value_network = MLP(
        layer_sizes=list(layer_sizes) + [1],
        activation=activation,
        kernel_init=kernel_init,
        bias=bias,
        layer_normalization=layer_normalization,
    )

    def apply(normalization_params, value_params, x):
        x = input_normalization_fn(x, normalization_params)
        x = x if isinstance(x, jnp.ndarray) else x[observation_key]
        return jnp.squeeze(value_network.apply(value_params, x), axis=-1)

    observation_size = _get_observation_size(
        input_size, observation_key
    )
    dummy_input = jnp.zeros((1, observation_size))
    return FeedForwardNetwork(
        init=lambda key: value_network.init(key, dummy_input), apply=apply,
    )
