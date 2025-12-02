import dataclasses

from typing import Any, Callable, Sequence, Mapping

import jax
import jax.numpy as jnp
from flax import nnx
from jax import nn

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

    
class Block(nnx.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bias: bool,
        kernel_init: Initializer,
        activation: ActivationFn,
        apply_activation: bool,
        layer_normalization: bool,
        rngs: nnx.Rngs,
    ):
        self.linear_layer = nnx.Linear(
            in_features=in_features,
            out_features=out_features,
            use_bias=use_bias,
            kernel_init=kernel_init,
            rngs=rngs,
        )
        self.activation = activation if apply_activation else None
        apply_normalization = layer_normalization and apply_activation
        self.normalization_layer = nnx.LayerNorm(
            num_features=out_features,
            rngs=rngs,
        ) if apply_normalization else None

    def __call__(self, x: jax.Array):
        x = self.linear_layer(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.normalization_layer is not None:
            x = self.normalization_layer(x)
        return x

class MLP(nnx.Module):
    def __init__(
        self,
        features: Sequence[int],
        activation: ActivationFn = nn.tanh,
        kernel_init: Initializer = jax.nn.initializers.lecun_uniform(),
        activate_final: bool = False,
        bias: bool = True,
        layer_normalization: bool = False,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        in_features, out_features = features[:-1], features[1:]
        self.layers = nnx.List([
            Block(
                in_features=in_feature,
                out_features=out_feature,
                use_bias=bias,
                kernel_init=kernel_init,
                activation=activation,
                apply_activation=(i != len(out_features) - 1 or activate_final),
                layer_normalization=layer_normalization,
                rngs=rngs,
            )
            for i, (in_feature, out_feature) in enumerate(zip(in_features, out_features))
        ])

    def __call__(self, x: jax.Array):
        for layer in self.layers:
            x = layer(x)
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
    features = [input_size] + list(layer_sizes) + [output_size]
    policy_network = MLP(
        features=features,
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
    features = [input_size] + list(layer_sizes) + [1]
    value_network = MLP(
        features=features,
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
