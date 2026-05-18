from absl import app, flags, logging

import os
import functools
from pathlib import Path
import pickle

import jax

import jax.numpy as jnp
import flax.nnx as nnx

import distrax
import optax

import wandb

from training.envs.unitree_go2_handstand import unitree_go2_handstand
from training.envs.unitree_go2_handstand import config
from training.envs.unitree_go2_handstand import randomize
import training.envs.utilities.filter as filters

import training.statistics as statistics
import training.algorithms.ppo.agent as agent
from training.optimizer import OptimizerConfig, create_optimizer

from training.algorithms.ppo.loss_utilities import loss_function
from training.algorithms.ppo.train import train
from training import metrics_utilities
from training import checkpoint_utilities
from training import distribution_utilities

os.environ['XLA_FLAGS'] = (
    '--xla_gpu_enable_triton_softmax_fusion=true '
    '--xla_gpu_triton_gemm_any=True '
    '--xla_gpu_enable_async_collectives=true '
    '--xla_gpu_enable_latency_hiding_scheduler=true '
    '--xla_gpu_enable_highest_priority_async_stream=true '
)

logging.set_verbosity(logging.FATAL)


FLAGS = flags.FLAGS
flags.DEFINE_string(
    'tag', '', 'Tag for wandb run.', short_name='t',
)
flags.DEFINE_string(
    'parameter_checkpoint', None, 'Parameter checkpoint path to load.', short_name='p', required=True,
)
flags.DEFINE_string(
    'checkpoint', None, 'Checkpoint path to load.', short_name='c', required=True,
)


def main(argv=None):
    # Get FLAG.tag prefix:
    prefix, suffix = FLAGS.tag.split('-')
    if prefix not in ['vendor']:
        raise ValueError(f'Unknown FLAG.tag prefix: {prefix}')
    if suffix not in ['position', 'velocity', 'torque']:
        raise ValueError(f'Unknown FLAG.tag suffix: {suffix}')

    # Rehydrate Model from Parameter Checkpoint:
    model_params = None
    if FLAGS.parameter_checkpoint is not None:
        parameter_checkpoint_path = Path(FLAGS.parameter_checkpoint) / 'regressed_params.pkl'
        with open(parameter_checkpoint_path, 'rb') as f:
            params = pickle.load(f)

        # Get Regressed Parameters:
        model_params = {
            k: v
            for k, v in params.items()
            if not k.startswith('initial_')
        }

    # Configs:
    noise_config = config.NoiseConfig()
    disturbance_config = config.DisturbanceConfig()

    scene = f'scene_mjx_{prefix}_{suffix}.xml'

    # Setup Environments:
    motor_config = None
    if suffix == 'torque':
        motor_config = config.MotorConfig()

    # Setup Filter: (Currently Hardcodes action_dim)
    control_timestep = 0.02

    # First Order Filter:
    cutoff_frequency = 4.0
    tau = 1 / (2 * jnp.pi * cutoff_frequency)
    alpha = control_timestep / (tau + control_timestep)
    filter_impl = filters.FirstOrderFilter(
        action_dim=12,
        alpha=alpha,
    )

    environment_config = config.EnvironmentConfig(
        filename=scene,
        action_scale=None,
        control_timestep=control_timestep,
        optimizer_timestep=0.004,
        impl="warp",
    )

    env = unitree_go2_handstand.Handstand(
        environment_config=environment_config,
        noise_config=noise_config,
        disturbance_config=disturbance_config,
        motor_config=motor_config,
        model_params=model_params,
        filter_impl=filter_impl,
    )

    observation_size = env.observation_size
    action_size = env.action_size
    reference_observation = {
        key: jnp.zeros(value) for key, value in observation_size.items()
    }

    # Setup agent:
    policy_layer_size = [512, 256, 128,]
    value_layer_size = [512, 256, 128,]
    activation_fn = jax.nn.swish
    hidden_init = jax.nn.initializers.orthogonal(jnp.sqrt(2.0))
    output_init = jax.nn.initializers.orthogonal(0.01)
    policy_kernel_init = [hidden_init] * len(policy_layer_size) + [output_init]
    value_kernel_init = [hidden_init] * len(value_layer_size) + [output_init]
    policy_input_normalization = statistics.RunningStatistics(
        reference_input=reference_observation["state"],
    )
    value_input_normalization = statistics.RunningStatistics(
        reference_input=reference_observation["privileged_state"],
    )
    model = agent.Agent(
        observation_size=observation_size,
        action_size=action_size,
        state_dependent_std=False,
        policy_input_normalization=policy_input_normalization,
        value_input_normalization=value_input_normalization,
        policy_layer_sizes=policy_layer_size,
        value_layer_sizes=value_layer_size,
        activation=activation_fn,
        policy_kernel_init=policy_kernel_init,
        value_kernel_init=value_kernel_init,
        policy_observation_key="state",
        value_observation_key="privileged_state",
    )

    # Restore Checkpoint:
    restored_checkpoint = None
    if FLAGS.checkpoint is not None:
        restore_directory = os.path.join(
            os.path.dirname(__file__),
            f"checkpoints/{FLAGS.checkpoint}",
        )
        restore_manager = checkpoint_utilities.create_checkpoint_manager(
            checkpoint_directory=restore_directory,
        )
        restored_checkpoint, _ = checkpoint_utilities.restore_training_state(
            manager=restore_manager,
            agent=model,
        )
        optimizer = restored_checkpoint.optimizer
        nnx.update(model, restored_checkpoint.agent)

    # Create Evaluator: (Used for its utility)
    num_episode_steps = 500
    key = jax.random.PRNGKey(0)
    render_options = metrics_utilities.RenderOptions(
        filepath='policy_visualization',
    )
    evaluator = metrics_utilities.Evaluator(
        env=env,
        num_envs=2,
        episode_length=num_episode_steps,
        action_repeat=1,
        key=key,
        render_options=render_options,
    )

    # Visualize Policy:
    state = jax.jit(env.reset)(key)
    state_list = [state.data]

    inference_fn = functools.partial(
        model.get_actions,
        deterministic=True,
    )

    def loop(carry, unused):
        state, key = carry
        key, subkey = jax.random.split(key)
        state.info['command'] = jnp.array([0.0, 0.0, 0.0])
        action = inference_fn(state.obs, subkey)[0]
        state = env.step(state, action)
        return (state, key), (state.data.qpos, state.data.xpos, state.data.xquat)

    (final_state, _), states = jax.lax.scan(
        loop,
        (state, key),
        None,
        length=num_episode_steps,
    )
    
    # Render HTML:
    evaluator._render_html(
        states=states,
        iteration='visualization',
    )


if __name__ == '__main__':
    app.run(main)
