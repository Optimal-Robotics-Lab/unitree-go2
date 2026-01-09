from absl import app, flags, logging
import os
import functools

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import flax.nnx as nnx

import distrax
import optax

import wandb

from training.envs.unitree_go2 import unitree_go2_joystick
from training.envs.unitree_go2 import config
from training.envs.unitree_go2 import randomize

import training.statistics as statistics
import training.algorithms.ppo.agent as agent
from training.optimizer import OptimizerConfig, create_optimizer

from training.algorithms.ppo.loss_utilities import loss_function
from training.algorithms.ppo.train import train
from training import metrics_utilities
from training import checkpoint_utilities

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
    'checkpoint_name', None, 'Desired checkpoint folder name to load.', short_name='c',
)
flags.DEFINE_integer(
    'checkpoint_iteration', None, 'Desired checkpoint iteration.', short_name='i',
)
flags.DEFINE_string(
    'tag', '', 'Tag for wandb run.', short_name='t',
)


def main(argv=None):
    # Get FLAG.tag prefix:
    prefix, suffix = FLAGS.tag.split('-')

    # Baseline Reward Config:
    if prefix == 'baseline':
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
            action_rate=-0.01,
            acceleration=-2.5e-5,
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
        command_config = config.CommandConfig()
    elif prefix == 'finetune':
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
        command_config = config.CommandConfig(
            command_range=jax.numpy.array([1.5, 1.0, 3.14]),
            command_mask_probability=0.9,
            command_frequency=[0.5, 2.0],
        )
    else:
        raise ValueError(f'Unknown FLAG.tag prefix: {prefix}')

    # Configs:
    noise_config = config.NoiseConfig()
    disturbance_config = config.DisturbanceConfig()

    if suffix == 'standard':
        scene = 'scene_mjx.xml'
    elif suffix == 'transparent':
        scene = 'scene_mjx_transparent.xml'
    else:
        raise ValueError(f'Unknown FLAG.tag suffix: {suffix}')

    # Setup Environments:
    environment_config = config.EnvironmentConfig(
        filename=scene,
        action_scale=0.5,
        control_timestep=0.02,
        optimizer_timestep=0.004,
    )

    env = unitree_go2_joystick.UnitreeGo2Env(
        environment_config=environment_config,
        reward_config=reward_config,
        noise_config=noise_config,
        disturbance_config=disturbance_config,
        command_config=command_config,
    )
    eval_env = unitree_go2_joystick.UnitreeGo2Env(
        environment_config=environment_config,
        reward_config=reward_config,
        noise_config=noise_config,
        disturbance_config=disturbance_config,
        command_config=command_config,
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
    policy_kernel_init = jax.nn.initializers.lecun_uniform()
    value_kernel_init = jax.nn.initializers.variance_scaling(
        scale=0.01, mode="fan_in", distribution="uniform",
    )
    policy_input_normalization = statistics.RunningStatistics(
        reference_input=reference_observation["state"],
    )
    value_input_normalization = statistics.RunningStatistics(
        reference_input=reference_observation["privileged_state"],
    )
    model = agent.Agent(
        observation_size=observation_size,
        action_size=action_size,
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

    # Setup Optimizer:
    optimizer_config = OptimizerConfig(
        learning_rate=3e-4,
        grad_clip_norm=1.0,
        desired_kl=None,
        min_learning_rate=1e-5,
        max_learning_rate=1e-2,
        kl_adjustment_factor=1.5,
    )
    optimizer = create_optimizer(optimizer_config)
    has_adaptive_kl_scheduler = True if optimizer_config.desired_kl is not None else False

    # Aggregate Metadata:
    agent_metadata = checkpoint_utilities.AgentMetadata(
        observation_size=env.observation_size,
        action_size=env.action_size,
        policy_layer_sizes=policy_layer_size,
        value_layer_sizes=value_layer_size,
        policy_input_normalization='statistics.RunningStatistics( \
            reference_input=reference_observation["state"], \
        )',
        value_input_normalization='statistics.RunningStatistics( \
            reference_input=reference_observation["privileged_state"], \
        )',
        activation='jax.nn.swish',
        policy_kernel_init='jax.nn.initializers.lecun_uniform()',
        value_kernel_init='value_kernel_init = jax.nn.initializers.variance_scaling( \
            scale=0.01, mode="fan_in", distribution="uniform", \
        )',
        policy_observation_key='state',
        value_observation_key='privileged_state',
        action_distribution='ParametricDistribution(distribution=distrax.Normal, bijector=distrax.Tanh())',
    )
    loss_metadata = checkpoint_utilities.LossMetadata(
        policy_clip_coef=0.2,
        value_clip_coef=0.4,
        value_coef=1.0,
        entropy_coef=0.01,
        gamma=0.99,
        gae_lambda=0.95,
        normalize_advantages=True,
    )
    training_metadata = checkpoint_utilities.TrainingMetadata(
        num_epochs=20,
        num_training_steps=20,
        episode_length=1000,
        num_policy_steps=40,
        action_repeat=1,
        num_envs=8192,
        num_evaluation_envs=128,
        deterministic_evaluation=True,
        reset_per_epoch=True,
        seed=42,
        batch_size=256,
        num_minibatches=32,
        num_ppo_iterations=4,
        normalize_observations=True,
    )

    # Start Wandb and save metadata:
    run = wandb.init(
        project='Test',
        tags=[FLAGS.tag],
        config={
            'reward_config': reward_config,
            'agent_metadata': agent_metadata,
            'loss_metadata': loss_metadata,
            'training_metadata': training_metadata,
            'optimizer_config': optimizer_config,
            'environment_config': environment_config,
            'noise_config': noise_config,
            'disturbance_config': disturbance_config,
            'command_config': command_config,
        },
    )

    render_options = metrics_utilities.RenderOptions(
        filepath=run.name,
        render_interval=5,
        duration=10.0,
    )

    # Initialize Functions with Params:
    randomization_fn = randomize.domain_randomize
    loss_fn = functools.partial(
        loss_function,
        policy_clip_coef=loss_metadata.policy_clip_coef,
        value_clip_coef=loss_metadata.value_clip_coef,
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

    # Restore Checkpoint:
    restored_checkpoint = None
    if FLAGS.checkpoint_name is not None:
        restore_directory = os.path.join(
            os.path.dirname(__file__),
            f"checkpoints/{FLAGS.checkpoint_name}",
        )
        restore_manager = checkpoint_utilities.create_checkpoint_manager(
            checkpoint_directory=restore_directory,
        )
        restored_checkpoint, _ = checkpoint_utilities.restore_training_state(
            manager=restore_manager,
            agent=model,
            iteration=FLAGS.checkpoint_iteration,
        )
        optimizer = restored_checkpoint.optimizer

    # Setup Checkpoint Manager:
    checkpoint_directory = os.path.join(
        os.path.dirname(__file__),
        f"checkpoints/{run.name}",
    )
    manager = checkpoint_utilities.create_checkpoint_manager(
        checkpoint_directory=checkpoint_directory,
        max_to_keep=5,
        save_interval_steps=1,
    )
    checkpoint_utilities.save_config(
        manager=manager,
        metadata=checkpoint_utilities.CheckpointMetadata(
            optimizer_config=optimizer_config,
            agent_metadata=agent_metadata,
            loss_metadata=loss_metadata,
            training_metadata=training_metadata,
        ),
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
        deterministic_evaluation=training_metadata.deterministic_evaluation,
        reset_per_epoch=training_metadata.reset_per_epoch,
        seed=training_metadata.seed,
        batch_size=training_metadata.batch_size,
        num_minibatches=training_metadata.num_minibatches,
        num_ppo_iterations=training_metadata.num_ppo_iterations,
        normalize_observations=training_metadata.normalize_observations,
        optimizer=optimizer,
        has_adaptive_kl_scheduler=has_adaptive_kl_scheduler,
        loss_function=loss_fn,
        progress_fn=progress_fn,
        checkpoint_manager=manager,
        restored_checkpoint=restored_checkpoint,
        randomization_fn=randomization_fn,
        wandb_run=run,
        render_options=render_options,
    )

    policy, metrics = train_fn(
        agent=model,
        environment=env,
        evaluation_environment=eval_env,
    )

    run.finish()


if __name__ == '__main__':
    app.run(main)
