from absl import app, flags, logging
import os
import functools

import jax
import flax.linen as nn
import distrax
import optax

import wandb
import orbax.checkpoint as ocp

from training.envs.unitree_go2 import unitree_go2_joystick
from training.envs.unitree_go2 import config
from training.envs.unitree_go2 import randomize
from training.algorithms.ppo import network_utilities as ppo_networks
from training.algorithms.ppo.loss_utilities import loss_function
from training.distribution_utilities import ParametricDistribution
from training.algorithms.ppo.train import train
from training.algorithms.ppo import checkpoint_utilities
from training.algorithms.ppo.load_utilities import load_checkpoint
from training import metrics_utilities

os.environ['XLA_FLAGS'] = (
    '--xla_gpu_enable_triton_softmax_fusion=true '
    '--xla_gpu_triton_gemm_any=True '
    '--xla_gpu_enable_async_collectives=true '
    '--xla_gpu_enable_latency_hiding_scheduler=true '
    '--xla_gpu_enable_highest_priority_async_stream=true '
)

jax.config.update("jax_enable_x64", True)

logging.set_verbosity(logging.FATAL)


FLAGS = flags.FLAGS
flags.DEFINE_string(
    'checkpoint_name', None, 'Desired checkpoint folder name to load.', short_name='c',
)
flags.DEFINE_integer(
    'checkpoint_iteration', None, 'Desired checkpoint iteration.', short_name='i',
)
flags.DEFINE_string(
    'tag', '', 'Tag for wandb run.', short_name='t',
)


def main(argv=None):
    # Baseline Reward Config:
    reward_config = config.RewardConfig(
        # Rewards:
        tracking_linear_velocity=1.5,
        tracking_angular_velocity=0.75,
        # Orientation Regularization Terms:
        orientation_regularization=-5.0,
        linear_z_velocity=-2.0,
        angular_xy_velocity=-0.05,
        # Energy Regularization Terms:
        torque=-2e-4,
        action_rate=-0.1,
        acceleration=-2.5e-4,
        # Auxilary Terms:
        stand_still=-1.0,
        termination=-1.0,
        unwanted_contact=-1.0,
        # Gait Reward Terms:
        foot_slip=-0.5,
        air_time=0.75,
        foot_clearance=0.5,
        gait_variance=-1.0,
        # Gait Hyperparameters:
        target_air_time=0.25,
        mode_time=0.2,
        command_threshold=0.0,
        velocity_threshold=0.5,
        # Foot Clearance Reward Terms:
        target_foot_height=0.125,
        foot_clearance_velocity_scale=2.0,
        foot_clearance_sigma=0.05,
        # Hyperparameter for exponential kernel:
        kernel_sigma=0.25,
    )

    # Configs:
    noise_config = config.NoiseConfig()

    # Default Disturbance Config:
    disturbance_config = config.DisturbanceConfig()

    # Default Command Config:
    # command_config = config.CommandConfig()

    # Fast Command Tracking:
    command_config = config.CommandConfig(
        command_range=jax.numpy.array([1.5, 1.0, 3.14]),
        single_command_probability=0.0,
        command_mask_probability=0.9,
        command_frequency=[0.5, 2.0],
    )

    flat_terrain = 'scene_mjx.xml'

    environment_config = config.EnvironmentConfig(
        filename=flat_terrain,
        action_scale=0.5,
        control_timestep=0.02,
        optimizer_timestep=0.004,
        recover_from_footstand=True,
    )

    env = unitree_go2_joystick.UnitreeGo2Env(
        env_config=environment_config,
        reward_config=reward_config,
        noise_config=noise_config,
        disturbance_config=disturbance_config,
        command_config=command_config,
    )
    eval_env = unitree_go2_joystick.UnitreeGo2Env(
        env_config=environment_config,
        reward_config=reward_config,
        noise_config=noise_config,
        disturbance_config=disturbance_config,
        command_config=command_config,
    )

    # Metadata:
    policy_layer_size = [512, 256, 128,]
    value_layer_size = [512, 256, 128,]
    network_metadata = checkpoint_utilities.network_metadata(
        policy_layer_size=policy_layer_size,
        value_layer_size=value_layer_size,
        policy_depth=len(policy_layer_size),
        value_depth=len(value_layer_size),
        activation='nn.swish',
        policy_kernel_init='jax.nn.initializers.lecun_uniform()',
        value_kernel_init='jax.nn.initializers.variance_scaling(scale=0.01, mode="fan_in", distribution="uniform")',
        policy_observation_key='state',
        value_observation_key='privileged_state',
        action_distribution='ParametricDistribution(distribution=distrax.Normal, bijector=distrax.Tanh())',
    )
    loss_metadata = checkpoint_utilities.loss_metadata(
        clip_coef=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        gamma=0.99,
        gae_lambda=0.95,
        normalize_advantages=True,
    )
    training_metadata = checkpoint_utilities.training_metadata(
        num_epochs=20,
        num_training_steps=20,
        episode_length=1000,
        num_policy_steps=40,
        action_repeat=1,
        num_envs=8192,
        num_evaluation_envs=128,
        num_evaluations=1,
        deterministic_evaluation=True,
        reset_per_epoch=True,
        seed=42,
        batch_size=256,
        num_minibatches=32,
        num_ppo_iterations=4,
        normalize_observations=True,
        optimizer='optax.chain(optax.clip_by_global_norm(max_norm=1.0),optax.adam(3e-4))',
    )

    # Start Wandb and save metadata:
    run = wandb.init(
        project='Unitree-Go2',
        tags=[FLAGS.tag],
        config={
            'reward_config': reward_config,
            'network_metadata': network_metadata,
            'loss_metadata': loss_metadata,
            'training_metadata': training_metadata,
            'environment_config': environment_config,
        },
    )

    render_options = metrics_utilities.RenderOptions(
        filepath=run.name,
        num_envs=1,
        render_interval=5,
        spacing=1.0,
        duration=10.0,
        # Video Options:
        fps=10,
        video_format='html',
    )

    # Initialize Functions with Params:
    randomization_fn = randomize.domain_randomize
    make_networks_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_layer_sizes=network_metadata.policy_layer_size,
        value_layer_sizes=network_metadata.value_layer_size,
        activation=nn.swish,
        policy_kernel_init=jax.nn.initializers.lecun_uniform(),
        value_kernel_init=jax.nn.initializers.variance_scaling(scale=0.01, mode="fan_in", distribution="uniform"),
        policy_observation_key='state',
        value_observation_key='privileged_state',
        action_distribution=ParametricDistribution(
            distribution=distrax.Normal,
            bijector=distrax.Tanh(),
        ),
    )
    loss_fn = functools.partial(
        loss_function,
        clip_coef=loss_metadata.clip_coef,
        value_coef=loss_metadata.value_coef,
        entropy_coef=loss_metadata.entropy_coef,
        gamma=loss_metadata.gamma,
        gae_lambda=loss_metadata.gae_lambda,
        normalize_advantages=loss_metadata.normalize_advantages,
    )

    def progress_fn(iteration, num_steps, metrics):
        print(
            f'Iteration: {iteration} \t'
            f'Num Steps: {num_steps} \t'
            f'Episode Reward: {metrics["eval/episode_reward"]:.3f} \t'
        )
        if num_steps > 0:
            print(
                f'Training Loss: {metrics["training/loss"]:.3f} \t'
                f'Policy Loss: {metrics["training/policy_loss"]:.3f} \t'
                f'Value Loss: {metrics["training/value_loss"]:.3f} \t'
                f'Entropy Loss: {metrics["training/entropy_loss"]:.3f} \t'
                f'Training Wall Time: {metrics["training/walltime"]:.3f} \t'
            )
        print('\n')

    # Setup Checkpoint Manager:
    manager_options = ocp.CheckpointManagerOptions(
        save_interval_steps=5,
        create=True,
    )
    checkpoint_directory = os.path.join(
        os.path.dirname(__file__),
        f"checkpoints/{run.name}",
    )
    registry = ocp.handlers.DefaultCheckpointHandlerRegistry()
    registry.add('train_state', ocp.args.PyTreeSave, ocp.handlers.PyTreeCheckpointHandler)
    registry.add('network_metadata', ocp.args.PyTreeSave, ocp.handlers.PyTreeCheckpointHandler)
    registry.add('loss_metadata', ocp.args.PyTreeSave, ocp.handlers.PyTreeCheckpointHandler)
    registry.add('training_metadata', ocp.args.PyTreeSave, ocp.handlers.PyTreeCheckpointHandler)

    registry.add('train_state', ocp.args.PyTreeRestore, ocp.handlers.PyTreeCheckpointHandler)
    registry.add('network_metadata', ocp.args.PyTreeRestore, ocp.handlers.PyTreeCheckpointHandler)
    registry.add('loss_metadata', ocp.args.PyTreeRestore, ocp.handlers.PyTreeCheckpointHandler)
    registry.add('training_metadata', ocp.args.PyTreeRestore, ocp.handlers.PyTreeCheckpointHandler)

    restored_checkpoint = None
    if FLAGS.checkpoint_name is not None:
        restored_checkpoint, metadata = load_checkpoint(
            checkpoint_name=FLAGS.checkpoint_name,
            environment=env,
            restore_iteration=FLAGS.checkpoint_iteration,
        )

    checkpoint_fn = functools.partial(
        checkpoint_utilities.save_checkpoint,
        checkpoint_directory=checkpoint_directory,
        manager_options=manager_options,
        registry=registry,
        network_metadata=network_metadata,
        loss_metadata=loss_metadata,
        training_metadata=training_metadata,
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(max_norm=1.0),
        optax.adam(learning_rate=3e-4),
    )

    train_fn = functools.partial(
        train,
        num_epochs=training_metadata.num_epochs,
        num_training_steps=training_metadata.num_training_steps,
        episode_length=training_metadata.episode_length,
        num_policy_steps=training_metadata.num_policy_steps,
        action_repeat=training_metadata.action_repeat,
        num_envs=training_metadata.num_envs,
        num_evaluation_envs=training_metadata.num_evaluation_envs,
        num_evaluations=training_metadata.num_evaluations,
        deterministic_evaluation=training_metadata.deterministic_evaluation,
        reset_per_epoch=training_metadata.reset_per_epoch,
        seed=training_metadata.seed,
        batch_size=training_metadata.batch_size,
        num_minibatches=training_metadata.num_minibatches,
        num_ppo_iterations=training_metadata.num_ppo_iterations,
        normalize_observations=training_metadata.normalize_observations,
        network_factory=make_networks_factory,
        optimizer=optimizer,
        loss_function=loss_fn,
        progress_fn=progress_fn,
        randomization_fn=randomization_fn,
        checkpoint_fn=checkpoint_fn,
        restored_checkpoint=restored_checkpoint,
        wandb_run=run,
        render_options=render_options,
    )

    policy_generator, params, metrics = train_fn(
        environment=env,
        evaluation_environment=eval_env,
    )

    run.finish()


if __name__ == '__main__':
    app.run(main)
