from typing import Optional, Sequence, Tuple

import jax
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
        policy_input_normalization: Optional[statistics.RunningStatistics] = None,
        value_input_normalization: Optional[statistics.RunningStatistics] = None,
        policy_layer_sizes: Sequence[int] = (256, 256),
        value_layer_sizes: Sequence[int] = (256, 256),
        activation: networks.ActivationFn = jax.nn.swish,
        policy_kernel_init: types.Initializer = jax.nn.initializers.lecun_uniform(),
        value_kernel_init: types.Initializer = jax.nn.initializers.lecun_uniform(),
        policy_observation_key: str = "state",
        value_observation_key: str = "state",
        action_distribution: distribution_utilities.ParametricDistribution = distribution_utilities
        .ParametricDistribution(distribution=distrax.Normal, bijector=distrax.Tanh()),
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        self.observation_size = observation_size
        self.action_size = action_size

        # Validate Keys exist in Observation
        assert policy_observation_key in observation_size, (
            f"Policy observation key '{policy_observation_key}' not found in observation size."
        )
        assert value_observation_key in observation_size, (
            f"Value observation key '{value_observation_key}' not found in observation size."
        )

        """Creates the Policy and Value Networks for PPO."""
        self.policy = networks.Policy(
            input_size=observation_size,
            output_size=2*action_size,
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
        logits = self.policy(x)

        if deterministic:
            actions = self.action_distribution.mode(logits)
            return actions, {}

        # Sample actions
        raw_actions = self.action_distribution.base_distribution_sample(logits, key)
        log_prob = self.action_distribution.log_prob(logits, raw_actions)
        actions = self.action_distribution.process_sample(raw_actions)

        return actions, {"log_prob": log_prob, "raw_action": raw_actions, "logits": logits}

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
