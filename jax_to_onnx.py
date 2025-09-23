import os
from typing import Any, Optional

from absl import app, flags

import jax
import jax.numpy as jnp

import onnx
from jax2onnx import to_onnx

from training.envs.unitree_go2 import unitree_go2_joystick as unitree_go2

from flax import linen as nn
import optax
import distrax
from brax.envs.base import Env
from brax.training.acme import running_statistics, specs
import orbax.checkpoint as ocp
from training.algorithms.ppo import checkpoint_utilities
from training.algorithms.ppo import network_utilities as ppo_networks
from training.algorithms.ppo.network_utilities import PPONetworkParams
from training.algorithms.ppo.checkpoint_utilities import (
    TrainState,
)
from training.distribution_utilities import ParametricDistribution

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# jax2onnx turns this on and it messes things up:
# jax.config.update("jax_dynamic_shapes", False)

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'checkpoint_name', None, 'Desired checkpoint folder name to load.', short_name='c',
)
flags.DEFINE_integer(
    'checkpoint_iteration', None, 'Desired checkpoint iteration.', short_name='i',
)


def load_network(checkpoint_name: str, environment: Env, restore_iteration: Optional[int] = None) -> Any:
    # Load Metadata:
    checkpoint_directory = os.path.join(
        os.path.dirname(__file__),
        f"checkpoints/{checkpoint_name}",
    )
    manager_options = checkpoint_utilities.default_checkpoint_options()

    registry = ocp.handlers.DefaultCheckpointHandlerRegistry()
    registry.add('train_state', ocp.args.PyTreeRestore, ocp.handlers.PyTreeCheckpointHandler)
    registry.add('network_metadata', ocp.args.PyTreeRestore, ocp.handlers.PyTreeCheckpointHandler)
    registry.add('loss_metadata', ocp.args.PyTreeRestore, ocp.handlers.PyTreeCheckpointHandler)
    registry.add('training_metadata', ocp.args.PyTreeRestore, ocp.handlers.PyTreeCheckpointHandler)

    metadata = checkpoint_utilities.load_checkpoint(
        directory=checkpoint_directory,
        options=manager_options,
        registry=registry,
        restore_iteration=restore_iteration,
    )
    
    network_metadata = metadata.network_metadata
    training_metadata = metadata.training_metadata

    env = environment
    key = jax.random.key(training_metadata['seed'])
    key, subkey = jax.random.split(key)
    env_state = env.reset(subkey)

    # Restore Networks:
    '''
        jax2onnx cannot handle the normalization function using dicts.
        We will have to normalize the inputs before being passed to the network.
    '''
    normalization_fn = lambda x, y: x

    network_observation_shape = env.observation_size
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
        directory=checkpoint_directory,
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


def main(argv=None):
    # Load Policy:
    env = unitree_go2.UnitreeGo2Env(filename='scene_mjx.xml')
    ppo_networks, params = load_network(
        checkpoint_name=FLAGS.checkpoint_name,
        environment=env,
        restore_iteration=FLAGS.checkpoint_iteration,
    )
    
    policy_network = ppo_networks.policy_network
    action_distribution = ppo_networks.action_distribution
    normalization_params, policy_params = params
    mean, std = normalization_params.mean['state'], normalization_params.std['state']

    def inference_function(x: jax.Array) -> jax.Array:
        x = (x - mean) / std
        logits = policy_network.apply(normalization_params, policy_params, x)
        actions = action_distribution.mode(logits)
        return actions

    # Convert to ONNX
    onnx_model = to_onnx(
        inference_function,
        inputs=[(env.observation_size["state"][0],)],
        model_name=FLAGS.checkpoint_name,
        opset=11,
    )

    # Save the model
    output_path = f"onnx_models/{FLAGS.checkpoint_name}.onnx"
    onnx.save_model(onnx_model, output_path)


if __name__ == '__main__':
    app.run(main)
