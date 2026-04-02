from typing import Optional, Sequence, Tuple, List

import jax
import jax.numpy as jnp

import distrax

from flax import nnx

from training import networks
from training import statistics
import training.distribution_utilities as distribution_utilities
from training import module_types as types


class Agent(nnx.Module):
    def __init__(
        self,
        observation_size: types.ObservationSize,
        action_size: int,
        state_dependent_std: bool = True,
        policy_input_normalization: Optional[statistics.RunningStatistics] = None,
        value_input_normalization: Optional[statistics.RunningStatistics] = None,
        policy_layer_sizes: Sequence[int] = (256, 256),
        value_layer_sizes: Sequence[int] = (256, 256),
        activation: networks.ActivationFn = jax.nn.swish,
        policy_kernel_init: types.Initializer | List[types.Initializer] = jax.nn.initializers.lecun_uniform(),
        value_kernel_init: types.Initializer | List[types.Initializer] = jax.nn.initializers.lecun_uniform(),
        policy_observation_key: str = "state",
        value_observation_key: str = "state",
        action_distribution: distribution_utilities.ParametricDistribution = distribution_utilities
        .ParametricDistribution(distribution=distrax.Normal, bijector=distrax.Tanh()),
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        self.observation_size = observation_size
        self.action_size = action_size
        self.state_dependent_std = state_dependent_std

        # Validate Keys exist in Observation
        assert policy_observation_key in observation_size, (
            f"Policy observation key '{policy_observation_key}' not found in observation size."
        )
        assert value_observation_key in observation_size, (
            f"Value observation key '{value_observation_key}' not found in observation size."
        )

        """Creates the Policy and Value Networks for PPO."""
        policy_output_size = action_size * 2 if state_dependent_std else action_size
        self.policy = networks.Policy(
            input_size=observation_size,
            output_size=policy_output_size,
            input_normalization=policy_input_normalization,
            layer_sizes=policy_layer_sizes,
            activation=activation,
            kernel_init=policy_kernel_init,
            observation_key=policy_observation_key,
            rngs=rngs,
        )

        self.value = networks.Policy(
            input_size=observation_size,
            output_size=1,
            input_normalization=value_input_normalization,
            layer_sizes=value_layer_sizes,
            activation=activation,
            kernel_init=value_kernel_init,
            observation_key=value_observation_key,
            rngs=rngs,
        )

        self.action_log_std = nnx.Param(jnp.zeros(action_size))

        self.action_distribution = action_distribution

    def get_actions(
        self,
        x: types.Observation,
        key: types.PRNGKey,
        deterministic: bool = False,
    ) -> Tuple[types.Action, types.PolicyData]:
        """
        Forward pass for Policy Network.

        Args:
            x: The observation input to the policy network.
            key: A PRNGKey for sampling actions.
            deterministic: Whether to use deterministic actions (e.g., for evaluation).
        """

        logits = self.get_logits(x)

        if deterministic:
            actions = self.action_distribution.mode(logits)
            return actions, {}

        # Sample actions
        raw_actions = self.action_distribution.base_distribution_sample(logits, key)
        log_prob = self.action_distribution.log_prob(logits, raw_actions)
        actions = self.action_distribution.process_sample(raw_actions)

        return actions, {"log_prob": log_prob, "raw_action": raw_actions, "logits": logits}

    def get_logits(
        self,
        x: types.Observation,
    ) -> jnp.ndarray:
        """Forward pass to get policy logits without sampling."""
        policy_output = self.policy(x)

        if self.state_dependent_std:
            logits = policy_output
        else:
            mean = policy_output
            log_std = jnp.broadcast_to(self.action_log_std.value, mean.shape)
            logits = jnp.concatenate([mean, log_std], axis=-1)

        return logits

    def get_values(
        self,
        x: types.Observation,
    ) -> types.Value:
        """Forward pass for Value Network."""
        return self.value(x)

    def __call__(
        self,
        x: types.Observation,
        key: types.PRNGKey,
        deterministic: bool = False,
    ) -> Tuple[types.Action, types.Value, types.PolicyData]:
        """Unified forward pass for training or inference."""
        actions, info = self.get_actions(x, key, deterministic=deterministic)
        value = self.get_values(x)
        return actions, value, info


class TemporalCNNAgent(nnx.Module):
    def __init__(
        self,
        observation_size: types.ObservationSize,
        action_size: int,
        state_dependent_std: bool = True,
        policy_input_normalization: Optional[statistics.RunningStatistics] = None,
        value_input_normalization: Optional[statistics.RunningStatistics] = None,
        # Policy CNN Arguments:
        cnn_policy_features: Sequence[int] | Sequence[Tuple[int, ...]] = (32, 64),
        cnn_policy_kernel_sizes: int | Sequence[int | Sequence[int]] = 3,
        cnn_policy_strides: int | Sequence[int | Sequence[int]] = 1,
        cnn_policy_paddings: str | Sequence[str | Sequence[Tuple[int, int]]] = 'SAME',
        cnn_policy_activation: networks.ActivationFn = jax.nn.swish,
        cnn_policy_kernel_init: types.Initializer | List[types.Initializer] = jax.nn.initializers.lecun_uniform(),
        cnn_policy_bias: bool = True,
        cnn_policy_layer_normalization: bool = False,
        # Policy and Value MLP Arguments:
        mlp_policy_layer_sizes: Sequence[int] = (256, 256),
        mlp_value_layer_sizes: Sequence[int] = (256, 256),
        mlp_activation: networks.ActivationFn = jax.nn.swish,
        mlp_policy_kernel_init: types.Initializer | List[types.Initializer] = jax.nn.initializers.lecun_uniform(),
        mlp_value_kernel_init: types.Initializer | List[types.Initializer] = jax.nn.initializers.lecun_uniform(),
        mlp_policy_bias: bool = True,
        mlp_value_bias: bool = True,
        mlp_policy_layer_normalization: bool = False,
        mlp_value_layer_normalization: bool = False,
        policy_observation_key: str = "state",
        value_observation_key: str = "state",
        action_distribution: distribution_utilities.ParametricDistribution = distribution_utilities
        .ParametricDistribution(distribution=distrax.Normal, bijector=distrax.Tanh()),
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        self.observation_size = observation_size
        self.action_size = action_size
        self.state_dependent_std = state_dependent_std

        # Validate Keys exist in Observation
        assert policy_observation_key in observation_size, (
            f"Policy observation key '{policy_observation_key}' not found in observation size."
        )
        assert value_observation_key in observation_size, (
            f"Value observation key '{value_observation_key}' not found in observation size."
        )

        """Creates the Policy and Value Networks for PPO."""
        policy_output_size = action_size * 2 if state_dependent_std else action_size
        self.policy = networks.TemporalCNNPolicy(
            input_size=observation_size,
            output_size=policy_output_size,
            input_normalization=policy_input_normalization,
            cnn_features=cnn_policy_features,
            cnn_kernel_sizes=cnn_policy_kernel_sizes,
            cnn_strides=cnn_policy_strides,
            cnn_paddings=cnn_policy_paddings,
            cnn_activation=cnn_policy_activation,
            cnn_kernel_init=cnn_policy_kernel_init,
            cnn_bias=cnn_policy_bias,
            cnn_layer_normalization=cnn_policy_layer_normalization,
            mlp_layer_sizes=mlp_policy_layer_sizes,
            mlp_activation=mlp_activation,
            mlp_kernel_init=mlp_policy_kernel_init,
            mlp_bias=mlp_policy_bias,
            mlp_layer_normalization=mlp_policy_layer_normalization,
            observation_key=policy_observation_key,
            rngs=rngs,
        )

        self.value = networks.Policy(
            input_size=observation_size,
            output_size=1,
            input_normalization=value_input_normalization,
            layer_sizes=mlp_value_layer_sizes,
            activation=mlp_activation,
            kernel_init=mlp_value_kernel_init,
            bias=mlp_value_bias,
            layer_normalization=mlp_value_layer_normalization,
            observation_key=value_observation_key,
            rngs=rngs,
        )

        self.action_log_std = nnx.Param(jnp.zeros(action_size))

        self.action_distribution = action_distribution

    def get_actions(
        self,
        x: types.Observation,
        key: types.PRNGKey,
        deterministic: bool = False,
    ) -> Tuple[types.Action, types.PolicyData]:
        """
        Forward pass for Policy Network.

        Args:
            x: The observation input to the policy network.
            key: A PRNGKey for sampling actions.
            deterministic: Whether to use deterministic actions (e.g., for evaluation).
        """

        logits = self.get_logits(x)

        if deterministic:
            actions = self.action_distribution.mode(logits)
            return actions, {}

        # Sample actions
        raw_actions = self.action_distribution.base_distribution_sample(logits, key)
        log_prob = self.action_distribution.log_prob(logits, raw_actions)
        actions = self.action_distribution.process_sample(raw_actions)

        return actions, {"log_prob": log_prob, "raw_action": raw_actions, "logits": logits}

    def get_logits(
        self,
        x: types.Observation,
    ) -> jnp.ndarray:
        """Forward pass to get policy logits without sampling."""
        policy_output = self.policy(x)

        if self.state_dependent_std:
            logits = policy_output
        else:
            mean = policy_output
            log_std = jnp.broadcast_to(self.action_log_std.value, mean.shape)
            logits = jnp.concatenate([mean, log_std], axis=-1)

        return logits

    def get_values(
        self,
        x: types.Observation,
    ) -> types.Value:
        """Forward pass for Value Network."""
        return self.value(x)

    def __call__(
        self,
        x: types.Observation,
        key: types.PRNGKey,
        deterministic: bool = False,
    ) -> Tuple[types.Action, types.Value, types.PolicyData]:
        """Unified forward pass for training or inference."""
        actions, info = self.get_actions(x, key, deterministic=deterministic)
        value = self.get_values(x)
        return actions, value, info
