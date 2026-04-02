from typing import Any, Callable, Sequence, Mapping, Optional, List, Tuple
import jax.typing as jtp

import math

import jax
import jax.numpy as jnp
from flax import nnx

from training import statistics

# Custom types:
import training.module_types as types
ActivationFn = Callable[[jtp.ArrayLike], jax.Array]
Initializer = Callable[..., Any]


# Helper function for observation keys:
def _get_observation_size(
    observation: types.Observation, observation_key: str
) -> int:
    observation_size = observation[observation_key] if isinstance(observation, Mapping) else observation
    return jax.tree.flatten(observation_size)[0][-1]


def _get_observation_shape(
    observation: types.Observation, observation_key: str
) -> Tuple[int, ...]:
    obs_shape = observation[observation_key] if isinstance(observation, Mapping) else observation
    return tuple(jax.tree.flatten(obs_shape)[0])


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
        if not callable(kernel_init):
            raise TypeError(f"Expected kernel_init to be a callable, got {type(kernel_init).__name__}.")

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
        activation: ActivationFn = jax.nn.tanh,
        kernel_init: Initializer | List[Initializer] = jax.nn.initializers.lecun_uniform(),
        activate_final: bool = False,
        bias: bool = True,
        layer_normalization: bool = False,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        if len(features) < 2:
            raise ValueError(
                f"The `features` sequence must contain at least an input and output size "
                f"(minimum length 2). Received features: {features}"
            )

        in_features, out_features = features[:-1], features[1:]
        num_layers = len(out_features)

        if isinstance(kernel_init, (list, tuple)):
            if len(kernel_init) != num_layers:
                raise ValueError(
                    f"Initializer length mismatch: The MLP expects {num_layers} layers "
                    f"based on the provided features {features}, but received a list of "
                    f"{len(kernel_init)} initializers."
                )
            kernel_inits = kernel_init
        else:
            kernel_inits = [kernel_init] * num_layers

        self.layers = nnx.List([
            Block(
                in_features=in_feature,
                out_features=out_feature,
                use_bias=bias,
                kernel_init=kernel_initialization,
                activation=activation,
                apply_activation=(i != len(out_features) - 1 or activate_final),
                layer_normalization=layer_normalization,
                rngs=rngs,
            )
            for i, (in_feature, out_feature, kernel_initialization) in enumerate(zip(in_features, out_features, kernel_inits))
        ])

    def __call__(self, x: jax.Array):
        for layer in self.layers:
            x = layer(x)
        return x


class ConvolutionalBlock(nnx.Module):
    def __init__(
        self,
        in_features: int | Tuple[int, ...],
        out_features: int | Tuple[int, ...],
        kernel_size: int | Sequence[int],
        strides: int | Sequence[int],
        padding: str | Sequence[Tuple[int, int]],
        use_bias: bool,
        kernel_init: Initializer,
        activation: ActivationFn,
        apply_activation: bool,
        layer_normalization: bool,
        rngs: nnx.Rngs,
    ):
        if not callable(kernel_init):
            raise TypeError(f"Expected kernel_init to be a callable, got {type(kernel_init).__name__}.")

        self.convolutional_layer = nnx.Conv(
            in_features=in_features,
            out_features=out_features,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
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
        x = self.convolutional_layer(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.normalization_layer is not None:
            x = self.normalization_layer(x)
        return x


class CNN(nnx.Module):
    def __init__(
        self,
        features: Sequence[int] | Sequence[Tuple[int, ...]],
        kernel_sizes: Sequence[int | Sequence[int]],
        strides: int | Sequence[int | Sequence[int]] = 1,
        paddings: str | Sequence[str | Sequence[Tuple[int, int]]] = 'SAME',
        activation: ActivationFn = jax.nn.tanh,
        kernel_init: Initializer | List[Initializer] = jax.nn.initializers.lecun_uniform(),
        activate_final: bool = True,
        bias: bool = True,
        layer_normalization: bool = False,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        if len(features) < 2:
            raise ValueError(
                f"The `features` sequence must contain at least an input and output size "
                f"(minimum length 2). Received features: {features}"
            )

        in_features, out_features = features[:-1], features[1:]
        num_layers = len(out_features)

        if not isinstance(strides, (list, tuple)):
            strides = [strides] * num_layers
            
        if isinstance(paddings, str) or not isinstance(paddings, (list, tuple)):
            paddings = [paddings] * num_layers

        if isinstance(kernel_init, (list, tuple)):
            if len(kernel_init) != num_layers:
                raise ValueError(
                    f"Initializer length mismatch: The MLP expects {num_layers} layers "
                    f"based on the provided features {features}, but received a list of "
                    f"{len(kernel_init)} initializers."
                )
            kernel_inits = kernel_init
        else:
            kernel_inits = [kernel_init] * num_layers

        self.layers = nnx.List([
            ConvolutionalBlock(
                in_features=in_feature,
                out_features=out_feature,
                kernel_size=kernel_size,
                strides=stride,
                padding=padding,
                use_bias=bias,
                kernel_init=kernel_initialization,
                activation=activation,
                apply_activation=(i != len(out_features) - 1 or activate_final),
                layer_normalization=layer_normalization,
                rngs=rngs,
            )
            for i, (in_feature, out_feature, kernel_size, stride, padding, kernel_initialization)
            in enumerate(zip(in_features, out_features, kernel_sizes, strides, paddings, kernel_inits))
        ])

    def __call__(self, x: jax.Array):
        for layer in self.layers:
            x = layer(x)
        return x

class Policy(nnx.Module):
    def __init__(
        self,
        input_size: types.ObservationSize,
        output_size: int,
        input_normalization: Optional[statistics.RunningStatistics] = None,
        layer_sizes: Sequence[int] = (256, 256),
        activation: ActivationFn = jax.nn.swish,
        kernel_init: Initializer | List[Initializer] = jax.nn.initializers.lecun_uniform(),
        bias: bool = True,
        layer_normalization: bool = False,
        observation_key: str = "state",
        *,
        rngs: nnx.Rngs,
    ):
        if output_size <= 0:
            raise ValueError(f"Policy output_size must be strictly positive. Received {output_size}.")

        self.input_normalization = input_normalization
        self.observation_key = observation_key
        self.squeeze_output = output_size == 1

        observation_size = _get_observation_size(
            input_size, observation_key
        )

        features = [observation_size] + list(layer_sizes) + [output_size]
        self.network = MLP(
            features=features,
            activation=activation,
            kernel_init=kernel_init,
            bias=bias,
            layer_normalization=layer_normalization,
            rngs=rngs,
        )

    def __call__(self, x: types.NestedArray):
        x = x if isinstance(x, jnp.ndarray) else x[self.observation_key]
        x = self.input_normalization(x) if self.input_normalization is not None else x
        out = self.network(x)
        if self.squeeze_output:
            out = jnp.squeeze(out, axis=-1)
        return out


class TemporalCNNPolicy(nnx.Module):
    def __init__(
        self,
        input_size: types.ObservationSize,
        output_size: int,
        input_normalization: Optional[statistics.RunningStatistics],
        cnn_features: Sequence[int] | Sequence[Tuple[int, ...]],
        cnn_kernel_sizes: Sequence[int | Sequence[int]],
        cnn_strides: int | Sequence[int | Sequence[int]] = 1,
        cnn_paddings: str | Sequence[Tuple[int, int]] = 'SAME',
        cnn_activation: ActivationFn = jax.nn.swish,
        cnn_kernel_init: Initializer | List[Initializer] = jax.nn.initializers.lecun_uniform(),
        cnn_bias: bool = True,
        cnn_layer_normalization: bool = False,
        mlp_layer_sizes: Sequence[int] = (256, 256),
        mlp_activation: ActivationFn = jax.nn.swish,
        mlp_kernel_init: Initializer | List[Initializer] = jax.nn.initializers.lecun_uniform(),
        mlp_bias: bool = True,
        mlp_layer_normalization: bool = False,
        observation_key: str = "state",
        *,
        rngs: nnx.Rngs,
    ):
        if output_size <= 0:
            raise ValueError(f"Policy output_size must be strictly positive. Received {output_size}.")

        self.input_normalization = input_normalization
        self.observation_key = observation_key
        self.squeeze_output = output_size == 1

        # Initialize CNN:
        input_shape = _get_observation_shape(input_size, observation_key)
        in_channels = input_shape[-1]
        full_cnn_features = [in_channels] + list(cnn_features)
        self.cnn = CNN(
            features=full_cnn_features,
            kernel_sizes=cnn_kernel_sizes,
            strides=cnn_strides,
            paddings=cnn_paddings,
            activation=cnn_activation,
            kernel_init=cnn_kernel_init,
            bias=cnn_bias,
            layer_normalization=cnn_layer_normalization,
            rngs=rngs,
        )

        dummy_input = jnp.zeros((1,) + input_shape)
        dummy_out = self.cnn(dummy_input)
        mlp_in_features = math.prod(dummy_out.shape[1:])

        # Initialize MLP:
        mlp_features = [mlp_in_features] + list(mlp_layer_sizes) + [output_size]
        self.mlp = MLP(
            features=mlp_features,
            activation=mlp_activation,
            kernel_init=mlp_kernel_init,
            bias=mlp_bias,
            layer_normalization=mlp_layer_normalization,
            rngs=rngs,
        )

    def __call__(self, x: types.NestedArray):
        x = x if isinstance(x, jnp.ndarray) else x[self.observation_key]
        x = self.input_normalization(x) if self.input_normalization is not None else x
        x = self.cnn(x)

        batch_dims = x.shape[:-2]
        flat_features = x.shape[-2] * x.shape[-1]
        x = jnp.reshape(x, (*batch_dims, flat_features))

        out = self.mlp(x)
        if self.squeeze_output:
            out = jnp.squeeze(out, axis=-1)
        return out
