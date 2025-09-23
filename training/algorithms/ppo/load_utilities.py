from typing import Optional, Tuple, Any
import dataclasses
import os

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
import flax.linen as nn
import optax
import distrax

from brax.training.acme import running_statistics, specs
from brax.envs.base import Env

from training.algorithms.ppo import checkpoint_utilities
from training.algorithms.ppo import network_utilities as ppo_networks
from training.algorithms.ppo.network_utilities import PPONetworkParams
from training.algorithms.ppo.checkpoint_utilities import (
    RestoredCheckpoint, TrainState,
)
from training.distribution_utilities import ParametricDistribution


@dataclasses.dataclass
class Metadata:
    network_metadata: checkpoint_utilities.network_metadata
    loss_metadata: checkpoint_utilities.loss_metadata
    training_metadata: checkpoint_utilities.training_metadata


def load_network(checkpoint_name: str, environment: Env, restore_iteration: Optional[int] = None) -> Any:
    # Load Metadata:
    checkpoint_direrctory = os.path.join(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(__file__),
                ),
            ),
        ),
        f"checkpoints/{checkpoint_name}",
    )
    manager_options = checkpoint_utilities.default_checkpoint_options()

    registry = ocp.handlers.DefaultCheckpointHandlerRegistry()
    registry.add('train_state', ocp.args.PyTreeRestore, ocp.handlers.PyTreeCheckpointHandler)
    registry.add('network_metadata', ocp.args.PyTreeRestore, ocp.handlers.PyTreeCheckpointHandler)
    registry.add('loss_metadata', ocp.args.PyTreeRestore, ocp.handlers.PyTreeCheckpointHandler)
    registry.add('training_metadata', ocp.args.PyTreeRestore, ocp.handlers.PyTreeCheckpointHandler)

    metadata = checkpoint_utilities.load_checkpoint(
        directory=checkpoint_direrctory,
        options=manager_options,
        registry=registry,
        restore_iteration=restore_iteration,
    )
    
    network_metadata = metadata.network_metadata
    training_metadata = metadata.training_metadata

    env = environment
    key = jax.random.key(training_metadata['seed'])
    key, subkey = jax.random.split(key)
    env_state = jax.jit(env.reset)(subkey)

    # Restore Networks:
    if training_metadata['normalize_observations'] and not override_normalization_fn:
        normalization_fn = running_statistics.normalize
    else:
        normalization_fn = lambda x, y: x

    network_observation_shape = jax.tree.map(lambda x: x.shape[:], env_state.obs)
    network = ppo_networks.make_ppo_networks(
        observation_size=network_observation_shape,
        action_size=env.action_size,
        input_normalization_fn=normalization_fn,
        policy_layer_sizes=network_metadata['policy_layer_size'],
        value_layer_sizes=network_metadata['value_layer_size'],
        activation=eval(network_metadata['activation']),
        policy_kernel_init=eval(network_metadata['policy_kernel_init']),
        value_kernel_init=eval(network_metadata['value_kernel_init']),
        policy_observation_key=network_metadata['policy_observation_key'],
        value_observation_key=network_metadata['value_observation_key'],
        action_distribution=ParametricDistribution(
            distribution=distrax.Normal,
            bijector=distrax.Tanh(),
        ),
    )
    optimizer = eval(training_metadata['optimizer'])

    # Create Keys and Structures:
    key = jax.random.key(training_metadata['seed'])
    init_params = PPONetworkParams(
        policy_params=network.policy_network.init(key),
        value_params=network.value_network.init(key),
    )
    trainstate_observation_shape = jax.tree.map(
        lambda x: specs.Array(x.shape[-1:], jnp.dtype('float32')), env_state.obs
    )
    train_state = TrainState(
        opt_state=optimizer.init(init_params),
        params=init_params,
        normalization_params=running_statistics.init_state(
            trainstate_observation_shape,
        ),
        env_steps=0,
    )

    # Restore Train State:
    restored_train_state = checkpoint_utilities.load_train_state(
        directory=checkpoint_direrctory,
        options=manager_options,
        registry=registry,
        restore_iteration=restore_iteration,
        train_state=train_state,
    )
    train_state = restored_train_state.train_state

    params = (
        train_state.normalization_params, train_state.params.policy_params,
    )

    return network, params


def load_policy(checkpoint_name: str, environment: Env, restore_iteration: Optional[int] = None):
    # Load Metadata:
    checkpoint_direrctory = os.path.join(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(__file__),
                ),
            ),
        ),
        f"checkpoints/{checkpoint_name}",
    )
    manager_options = checkpoint_utilities.default_checkpoint_options()

    registry = ocp.handlers.DefaultCheckpointHandlerRegistry()
    registry.add('train_state', ocp.args.PyTreeRestore, ocp.handlers.PyTreeCheckpointHandler)
    registry.add('network_metadata', ocp.args.PyTreeRestore, ocp.handlers.PyTreeCheckpointHandler)
    registry.add('loss_metadata', ocp.args.PyTreeRestore, ocp.handlers.PyTreeCheckpointHandler)
    registry.add('training_metadata', ocp.args.PyTreeRestore, ocp.handlers.PyTreeCheckpointHandler)

    metadata = checkpoint_utilities.load_checkpoint(
        directory=checkpoint_direrctory,
        options=manager_options,
        registry=registry,
        restore_iteration=restore_iteration,
    )
    
    network_metadata = metadata.network_metadata
    loss_metadata = metadata.loss_metadata
    training_metadata = metadata.training_metadata

    env = environment
    key = jax.random.key(training_metadata['seed'])
    key, subkey = jax.random.split(key)
    env_state = jax.jit(env.reset)(subkey)

    # Restore Networks:
    if training_metadata['normalize_observations']:
        normalization_fn = running_statistics.normalize
    else:
        normalization_fn = lambda x, y: x

    network_observation_shape = jax.tree.map(lambda x: x.shape[:], env_state.obs)
    network = ppo_networks.make_ppo_networks(
        observation_size=network_observation_shape,
        action_size=env.action_size,
        input_normalization_fn=normalization_fn,
        policy_layer_sizes=network_metadata['policy_layer_size'],
        value_layer_sizes=network_metadata['value_layer_size'],
        activation=eval(network_metadata['activation']),
        policy_kernel_init=eval(network_metadata['policy_kernel_init']),
        value_kernel_init=eval(network_metadata['value_kernel_init']),
        policy_observation_key=network_metadata['policy_observation_key'],
        value_observation_key=network_metadata['value_observation_key'],
        action_distribution=ParametricDistribution(
            distribution=distrax.Normal,
            bijector=distrax.Tanh(),
        ),
    )
    optimizer = eval(training_metadata['optimizer'])

    # Create Keys and Structures:
    key = jax.random.key(training_metadata['seed'])
    init_params = PPONetworkParams(
        policy_params=network.policy_network.init(key),
        value_params=network.value_network.init(key),
    )
    trainstate_observation_shape = jax.tree.map(
        lambda x: specs.Array(x.shape[-1:], jnp.dtype('float32')), env_state.obs
    )
    train_state = TrainState(
        opt_state=optimizer.init(init_params),
        params=init_params,
        normalization_params=running_statistics.init_state(
            trainstate_observation_shape,
        ),
        env_steps=0,
    )

    # Restore Train State:
    restored_train_state = checkpoint_utilities.load_train_state(
        directory=checkpoint_direrctory,
        options=manager_options,
        registry=registry,
        restore_iteration=restore_iteration,
        train_state=train_state,
    )
    train_state = restored_train_state.train_state

    # Construct Policy:
    make_policy = ppo_networks.make_inference_fn(ppo_networks=network)
    params = (
        train_state.normalization_params, train_state.params.policy_params,
    )

    return make_policy, params, metadata


def load_checkpoint(
    checkpoint_name: str,
    environment: Env,
    restore_iteration: Optional[int] = None,
) -> Tuple[RestoredCheckpoint, Metadata]:
    # Load Metadata:
    checkpoint_direrctory = os.path.join(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(__file__),
                ),
            ),
        ),
        f"checkpoints/{checkpoint_name}",
    )

    manager_options = checkpoint_utilities.default_checkpoint_options()

    registry = ocp.handlers.DefaultCheckpointHandlerRegistry()
    registry.add('train_state', ocp.args.PyTreeRestore, ocp.handlers.PyTreeCheckpointHandler)
    registry.add('network_metadata', ocp.args.PyTreeRestore, ocp.handlers.PyTreeCheckpointHandler)
    registry.add('loss_metadata', ocp.args.PyTreeRestore, ocp.handlers.PyTreeCheckpointHandler)
    registry.add('training_metadata', ocp.args.PyTreeRestore, ocp.handlers.PyTreeCheckpointHandler)

    metadata = checkpoint_utilities.load_checkpoint(
        directory=checkpoint_direrctory,
        options=manager_options,
        registry=registry,
        restore_iteration=restore_iteration,
    )
    
    network_metadata = metadata.network_metadata
    loss_metadata = metadata.loss_metadata
    training_metadata = metadata.training_metadata

    env = environment
    key = jax.random.key(training_metadata['seed'])
    key, subkey = jax.random.split(key)
    env_state = jax.jit(env.reset)(subkey)

    # Restore Networks:
    if training_metadata['normalize_observations']:
        normalization_fn = running_statistics.normalize
    else:
        normalization_fn = lambda x, y: x

    network_observation_shape = jax.tree.map(lambda x: x.shape[:], env_state.obs)
    network = ppo_networks.make_ppo_networks(
        observation_size=network_observation_shape,
        action_size=env.action_size,
        input_normalization_fn=normalization_fn,
        policy_layer_sizes=network_metadata['policy_layer_size'],
        value_layer_sizes=network_metadata['value_layer_size'],
        activation=eval(network_metadata['activation']),
        policy_kernel_init=eval(network_metadata['policy_kernel_init']),
        value_kernel_init=eval(network_metadata['value_kernel_init']),
        policy_observation_key=network_metadata['policy_observation_key'],
        value_observation_key=network_metadata['value_observation_key'],
        action_distribution=ParametricDistribution(
            distribution=distrax.Normal,
            bijector=distrax.Tanh(),
        ),
    )
    optimizer = eval(training_metadata['optimizer'])

    # Create Keys and Structures:
    key = jax.random.key(training_metadata['seed'])
    init_params = PPONetworkParams(
        policy_params=network.policy_network.init(key),
        value_params=network.value_network.init(key),
    )
    trainstate_observation_shape = jax.tree.map(
        lambda x: specs.Array(x.shape[-1:], jnp.dtype('float32')), env_state.obs
    )
    train_state = TrainState(
        opt_state=optimizer.init(init_params),
        params=init_params,
        normalization_params=running_statistics.init_state(
            trainstate_observation_shape,
        ),
        env_steps=0,
    )

    # Restore Train State:
    restored_train_state = checkpoint_utilities.load_train_state(
        directory=checkpoint_direrctory,
        options=manager_options,
        registry=registry,
        restore_iteration=restore_iteration,
        train_state=train_state,
    )
    train_state = restored_train_state.train_state

    # Kind of redundant, but this gives us type hint instead of Any
    metadata = Metadata(
        network_metadata=network_metadata,
        loss_metadata=loss_metadata,
        training_metadata=training_metadata,
    )

    return (
        RestoredCheckpoint(network=network, train_state=train_state), metadata
    )
