from absl import app, flags, logging

import os

# Must be set before anything imports jax: importing the env modules below
# initializes the JAX backend (jnp constants at import time), after which
# these are silently ignored and JAX preallocates ~75% of GPU memory.
os.environ['XLA_FLAGS'] = (
    '--xla_gpu_triton_gemm_any=True '
    '--xla_gpu_enable_latency_hiding_scheduler=true '
    '--xla_gpu_enable_highest_priority_async_stream=true '
)
# Each run only claims memory as it's actually used instead of grabbing most
# of the device upfront, so runs can share one GPU.
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import dataclasses
import math
import json
import functools

import jax
import jax.numpy as jnp
import numpy as np

import wandb

from training.envs.spot import spot_joystick
from training.envs.spot import config
from training.envs.spot import randomize
from training.envs.spot import diagnostics
import training.envs.utilities.filter as filters

import training.statistics as statistics
import training.algorithms.ppo.agent as agent
from training.optimizer import OptimizerConfig, create_optimizer

from training.algorithms.ppo.loss_utilities import loss_function
from training.algorithms.ppo.train import train
from training import metrics_utilities
from training import checkpoint_utilities

logging.set_verbosity(logging.FATAL)


FLAGS = flags.FLAGS
flags.DEFINE_string(
    'tag', 'baseline', 'Tag for wandb run.', short_name='t',
)
flags.DEFINE_string(
    'reward_overrides', '{}',
    'JSON dict of RewardWeights/RewardHyperparameters field overrides, '
    'applied on top of the baseline reward config below -- e.g. '
    '\'{"tracking_linear_velocity": 2.0}\'. For launching several '
    'single-variable tuning trials concurrently without editing this file.',
)
flags.DEFINE_string(
    'reward_preset', '',
    'Name of a reward-override preset, tuning/presets/<name>.json, applied '
    'before --reward_overrides (which wins on conflicts).',
)
flags.DEFINE_boolean(
    'probe', False,
    'Short 5-epoch run (instead of 20) for screening hard failures; '
    'everything else matches a full run.',
)
flags.DEFINE_string(
    'kp', '',
    'Leg PD stiffness [Nm/rad]: one value or "abduction,thigh,calf". '
    'Default: EnvironmentConfig.kp.',
)
flags.DEFINE_string(
    'kv', '',
    'Leg PD damping [Nms/rad]: one value or "abduction,thigh,calf". '
    'Default: EnvironmentConfig.kv.',
)
flags.DEFINE_string(
    'action_scale', '',
    'Leg action scale [rad]: "none" (per-joint, from joint limits), one '
    'value, or "abduction,thigh,calf". Default: EnvironmentConfig.action_scale.',
)
flags.DEFINE_float(
    'filter_cutoff_hz', 0.0,
    'Cutoff [Hz] of a first-order low-pass filter on the leg actions '
    '(0 disables it).',
)
flags.DEFINE_string(
    'restore_run', '',
    'Name of a previous run under checkpoints/ to restore the agent and '
    'optimizer state from (finetuning). The observation layout must match '
    '(same filter setting).',
)
flags.DEFINE_integer(
    'restore_iteration', 0,
    'Checkpoint iteration to restore (0 = latest).',
)
flags.DEFINE_integer(
    'num_epochs', 0,
    'Overrides the epoch count when > 0 (default: 20, or 5 with --probe).',
)
flags.DEFINE_integer(
    'seed', 42, 'Training seed; vary it to measure run-to-run noise.',
)
flags.DEFINE_string(
    'metrics_file', '',
    'If set, append one JSON line of scalar metrics per iteration to this '
    'path (flushed immediately) so an external process can monitor the run '
    'and abort it early.',
)


def _parse_per_type(text: str) -> float | tuple[float, float, float]:
    """Parses "a" or "a,b,c" into a float or an (abduction, thigh, calf) tuple."""
    values = tuple(float(x) for x in text.split(','))
    if len(values) not in (1, 3):
        raise ValueError(f'Expected 1 or 3 comma-separated values, got {text!r}.')
    return values[0] if len(values) == 1 else values


def _control_overrides() -> dict:
    """EnvironmentConfig overrides from --kp/--kv/--action_scale."""
    overrides = {}
    for name in ('kp', 'kv'):
        text = getattr(FLAGS, name)
        if text:
            value = _parse_per_type(text)
            overrides[name] = (value,) * 3 if isinstance(value, float) else value
    if FLAGS.action_scale:
        overrides['action_scale'] = (
            None if FLAGS.action_scale.lower() == 'none'
            else _parse_per_type(FLAGS.action_scale)
        )
    return overrides


def main(argv=None):
    # Reward Config: Spot's own config.py defaults, written out explicitly
    # here (rather than a bare RewardConfig()) so every number is visible
    # and directly editable per run. These are our educated initial guess --
    # most were ported from IsaacLab's official SpotFlatEnvCfg earlier in
    # this project (see config.py's field comments for provenance); they
    # haven't been tuned against actual Spot training yet.
    #
    # --reward_overrides applies on top of this baseline via .replace(), so a
    # tuning loop can launch single-variable trials (e.g.
    # --reward_overrides='{"tracking_linear_velocity": 2.0}') without
    # editing this file.
    reward_overrides = {}
    if FLAGS.reward_preset:
        preset_path = os.path.join(
            os.path.dirname(__file__), 'tuning', 'presets',
            f'{FLAGS.reward_preset}.json',
        )
        with open(preset_path) as f:
            reward_overrides.update(json.load(f))
    reward_overrides.update(json.loads(FLAGS.reward_overrides))
    weight_fields = {f.name for f in dataclasses.fields(config.RewardWeights)}
    hyperparameter_fields = {
        f.name for f in dataclasses.fields(config.RewardHyperparameters)
    }
    unknown_fields = set(reward_overrides) - weight_fields - hyperparameter_fields
    if unknown_fields:
        raise ValueError(
            f'--reward_overrides has unknown field(s): {unknown_fields}. '
            f'Valid fields: {sorted(weight_fields | hyperparameter_fields)}'
        )
    weight_overrides = {
        k: v for k, v in reward_overrides.items() if k in weight_fields
    }
    hyperparameter_overrides = {
        k: v for k, v in reward_overrides.items() if k in hyperparameter_fields
    }

    reward_config = config.RewardConfig(
        weights=config.RewardWeights(
            # Rewards:
            tracking_linear_velocity=1.5,
            tracking_angular_velocity=0.75,
            # Orientation Regularization Terms:
            orientation_regularization=-2.5,
            linear_z_velocity=-2.0,
            angular_xy_velocity=-0.05,
            # Energy Regularization Terms:
            torque=-2e-4,
            action_rate=-0.2,
            acceleration=-2.5e-7,
            # Auxilary Terms:
            stand_still=-1.0,
            termination=-1.0,
            unwanted_contact=-0.5,
            # Gait Reward Terms:
            foot_slip=-0.1,
            air_time=0.25,
            foot_clearance=0.5,
            gait_timing_variance=-1.0,
            synchronized_contact=-0.1,
        ).replace(**weight_overrides),
        hyperparameters=config.RewardHyperparameters(
            target_air_time=0.5,
            mode_time=0.3,
            command_threshold=0.0,
            velocity_threshold=0.5,
            ramp_at_vel=1.0,
            ramp_rate=0.5,
            target_foot_height=0.1,
            foot_clearance_velocity_scale=2.0,
            foot_clearance_sigma=0.01,
            stand_still_scale=2.0,
            window_steps=3,
            kernel_sigma=0.25,
        ).replace(**hyperparameter_overrides),
    )
    command_config = config.CommandConfig()
    noise_config = config.NoiseConfig()
    disturbance_config = config.DisturbanceConfig()

    # num_envs is decided here (not just in training_metadata below) because
    # nconmax must scale with it: MJX-Warp's nconmax is a global contact
    # budget "across all worlds" (mjx.make_data's docstring), not per-env,
    # so reusing the full-scale nconmax at a smaller num_envs over-allocates
    # a contact buffer sized for envs that aren't running.
    num_envs = 8192
    # Control law (kp, kv, action_scale) defaults live in EnvironmentConfig;
    # --kp/--kv/--action_scale override them. action_scale=None (per-joint,
    # full range) is meant to be paired with an action filter, which this
    # script omits.
    environment_config = config.EnvironmentConfig(
        impl='warp', nconmax=20 * num_envs, **_control_overrides(),
    )

    # No action filter yet -- revisit once a baseline policy is training.
    filter_impl = filters.NoFilter()
    if FLAGS.filter_cutoff_hz > 0:
        # Same discretization as train.py's Go2 filter. action_dim is the 12
        # leg joints; the arm isn't policy-controlled.
        control_timestep = environment_config.control_timestep
        tau = 1.0 / (2.0 * math.pi * FLAGS.filter_cutoff_hz)
        filter_impl = filters.FirstOrderFilter(
            action_dim=12, alpha=control_timestep / (tau + control_timestep),
        )

    env = spot_joystick.SpotJoystickEnv(
        environment_config=environment_config,
        reward_config=reward_config,
        noise_config=noise_config,
        disturbance_config=disturbance_config,
        command_config=command_config,
        filter_impl=filter_impl,
    )
    eval_env = spot_joystick.SpotJoystickEnv(
        environment_config=environment_config,
        reward_config=reward_config,
        noise_config=noise_config,
        disturbance_config=disturbance_config,
        command_config=command_config,
        filter_impl=filter_impl,
    )

    observation_size = env.observation_size
    action_size = env.action_size
    reference_observation = {
        key: jnp.zeros(value) for key, value in observation_size.items()
    }

    # Setup agent: same actor-critic architecture as train.py's Go2 setup.
    policy_layer_size = [512, 256, 128]
    value_layer_size = [512, 256, 128]
    activation_fn = jax.nn.swish
    hidden_init = jax.nn.initializers.orthogonal(jnp.sqrt(2.0))
    output_init = jax.nn.initializers.orthogonal(0.01)
    policy_kernel_init = [hidden_init] * len(policy_layer_size) + [output_init]
    value_kernel_init = [hidden_init] * len(value_layer_size) + [output_init]
    # clip bounds normalized inputs: features that are constant in the first
    # batch (e.g. disturbance force) get std ~0.01 and would otherwise reach
    # the critic as ~1e5 once they first change, spiking the value loss.
    policy_input_normalization = statistics.RunningStatistics(
        reference_input=reference_observation["state"], clip=10.0,
    )
    value_input_normalization = statistics.RunningStatistics(
        reference_input=reference_observation["privileged_state"], clip=10.0,
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

    # Aggregate Metadata:
    agent_metadata = checkpoint_utilities.AgentMetadata(
        observation_size=env.observation_size,
        action_size=env.action_size,
        policy_layer_sizes=policy_layer_size,
        value_layer_sizes=value_layer_size,
        policy_input_normalization='statistics.RunningStatistics( \
            reference_input=reference_observation["state"], clip=10.0, \
        )',
        value_input_normalization='statistics.RunningStatistics( \
            reference_input=reference_observation["privileged_state"], clip=10.0, \
        )',
        activation='jax.nn.swish',
        policy_kernel_init='jax.nn.initializers.orthogonal(jnp.sqrt(2.0))',
        value_kernel_init='jax.nn.initializers.orthogonal(jnp.sqrt(2.0))',
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
    # --probe only shortens training (5 epochs); env count and episode length
    # match a full run, since 8192 envs is also the faster configuration.
    training_metadata = checkpoint_utilities.TrainingMetadata(
        num_epochs=FLAGS.num_epochs or (5 if FLAGS.probe else 20),
        num_training_steps=20,
        episode_length=1000,
        num_policy_steps=40,
        action_repeat=1,
        num_envs=num_envs,
        num_evaluation_envs=128,
        deterministic_evaluation=True,
        reset_per_epoch=False,
        seed=FLAGS.seed,
        batch_size=256,
        num_minibatches=32,
        num_ppo_iterations=4,
        normalize_observations=True,
    )

    # Setup Optimizer:
    optimizer_config = OptimizerConfig(
        optimizer_type="adam",
        scheduler_type="constant_schedule",
        optimizer_params={},
        scheduler_params={
            "value": 3e-4
        },
        grad_clip_norm=1.0,
    )
    optimizer = create_optimizer(optimizer_config)
    has_adaptive_kl_scheduler = (optimizer_config.scheduler_type == "adaptive_kl_schedule")

    # Sanitize Optimizer Config for Logging and Checkpointing:
    sanitized_optimizer_config = checkpoint_utilities.sanitize_config(optimizer_config)

    # Start Wandb and save metadata:
    run = wandb.init(
        project='Spot',
        tags=[FLAGS.tag],
        config={
            'reward_config': reward_config,
            'agent_metadata': agent_metadata,
            'loss_metadata': loss_metadata,
            'training_metadata': training_metadata,
            'environment_config': environment_config,
            'noise_config': noise_config,
            'disturbance_config': disturbance_config,
            'command_config': command_config,
            'optimizer_config': sanitized_optimizer_config,
        },
    )

    render_options = metrics_utilities.RenderOptions(
        filepath=run.name,
        render_interval=1,
        duration=10.0,
    )

    # Curriculum Functions:
    curriculum_fn = None

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
        scorecard = diagnostics.compute_scorecard(
            metrics, robot_mass=float(env.robot_mass), control_dt=env.dt,
        )
        run.summary.update(scorecard)
        if FLAGS.metrics_file:
            record = {'iteration': iteration, 'num_steps': num_steps}
            record.update({k: float(v) for k, v in metrics.items() if np.ndim(v) == 0})
            record.update(scorecard)
            with open(FLAGS.metrics_file, 'a') as f:
                f.write(json.dumps(record) + '\n')
        print(
            f'Iteration: {iteration} \t'
            f'Num Steps: {num_steps} \t'
            f'Episode Reward: {metrics["eval/episode_reward"]:.3f} \t',
            flush=True,
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
            optimizer_config=sanitized_optimizer_config,
            agent_metadata=agent_metadata,
            loss_metadata=loss_metadata,
            training_metadata=training_metadata,
        ),
    )

    restored_checkpoint = None
    if FLAGS.restore_run:
        source_manager = checkpoint_utilities.create_checkpoint_manager(
            checkpoint_directory=os.path.join(
                os.path.dirname(__file__), 'checkpoints', FLAGS.restore_run,
            ),
            max_to_keep=None,  # read-only: never prune the source run.
        )
        restored_checkpoint, _ = checkpoint_utilities.restore_training_state(
            source_manager, model, iteration=FLAGS.restore_iteration or None,
        )

    # The checkpoint's config.json has no environment settings; save the ones
    # deployment needs (see training/envs/spot/to_onnx.py).
    with open(os.path.join(checkpoint_directory, 'environment.json'), 'w') as f:
        json.dump({
            'kp': list(environment_config.kp),
            'kv': list(environment_config.kv),
            'action_scale': (
                list(environment_config.action_scale)
                if isinstance(environment_config.action_scale, tuple)
                else environment_config.action_scale
            ),
            'control_timestep': environment_config.control_timestep,
            'filter_cutoff_hz': FLAGS.filter_cutoff_hz,
            # Joint default pose (12 legs then 7 arm) the policy's offsets and targets are relative to.
            'default_pose': [float(x) for x in env.default_pose],
            # |vx|, |vy|, |yaw rate| the velocity command was sampled from.
            'command_range': [float(x) for x in command_config.command_range],
        }, f, indent=2)

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
        curriculum_fn=curriculum_fn,
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
