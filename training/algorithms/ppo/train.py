import functools
import time
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec

import flax
import flax.nnx as nnx

import optax

from mujoco import mjx

from mujoco_playground import wrapper
import training.module_types as types

from training.algorithms.ppo.agent import Agent

import training.algorithms.ppo.loss_utilities as loss_utilities
import training.training_utilities as training_utilities
import training.metrics_utilities as metrics_utilities
import training.checkpoint_utilities as checkpoint_utilities

import orbax.checkpoint as ocp

try:
    import wandb

except ImportError:
    class MockWandb:
        def init(self, *args, **kwargs):
            pass

        def log(self, *args, **kwargs):
            pass

        def finish(self, *args, **kwargs):
            pass

        def Video(self, *args, **kwargs):
            pass

        def Html(self, *args, **kwargs):
            pass

    wandb = MockWandb()


def train(
    agent: Agent,
    environment: types.Env,
    evaluation_environment: types.Env,
    num_epochs: int,
    num_training_steps: int,
    episode_length: int,
    eval_episode_length: int = 1000,
    num_policy_steps: int = 10,
    action_repeat: int = 1,
    num_envs: int = 1,
    num_evaluation_envs: int = 128,
    deterministic_evaluation: bool = False,
    reset_per_epoch: bool = False,
    seed: int = 0,
    batch_size: int = 32,
    num_minibatches: int = 16,
    num_ppo_iterations: int = 4,
    normalize_observations: bool = True,
    optimizer: optax.GradientTransformation = optax.adam(1e-4),
    has_adaptive_kl_scheduler: bool = False,
    loss_function: Callable[..., Tuple[jnp.ndarray, types.Metrics]] =
    loss_utilities.loss_function,
    progress_fn: Callable[[int, int, types.Metrics], None] = lambda *args: None,
    checkpoint_manager: Optional[ocp.CheckpointManager] = None,
    restored_checkpoint: Optional[checkpoint_utilities.RestoredCheckpoint] = None,
    randomization_fn: Optional[
        Callable[[mjx.Model, types.PRNGKey], Tuple[mjx.Model, mjx.Model]]
    ] = None,
    wandb_run: Optional[Any] = None,
    render_options: Optional[metrics_utilities.RenderOptions] = None,
):
    assert batch_size * num_minibatches % num_envs == 0

    # JAX Device management:
    devices = jax.devices()
    process_id = jax.process_index()

    mesh = Mesh(devices, axis_names=('batch',))

    # Sharding Specs
    # REPLICATED: Params, Optimizer State, Agent Stats (One copy on every GPU)
    s_replicated = NamedSharding(mesh, PartitionSpec())
    # SHARDED: Environment State, Observations, Random Keys (Split across GPUs)
    s_data = NamedSharding(mesh, PartitionSpec('batch'))

    assert num_envs % len(devices) == 0

    # Step increment counters:
    num_steps_per_train_step = (
        batch_size * num_minibatches * num_policy_steps * action_repeat
    )
    num_steps_per_epoch = (
        num_steps_per_train_step * num_training_steps
    )

    # Generate Random Key:
    key = jax.random.PRNGKey(seed)  # Brax wrappers rely on the old PRNGKey
    global_key, local_key = jax.random.split(key)
    del key
    local_key = jax.random.fold_in(local_key, process_id)
    local_key, env_key, eval_key = jax.random.split(local_key, 3)
    del global_key

    # Initialize Environment:
    _randomization_fn = None
    if randomization_fn is not None:
        randomization_key = jax.random.split(env_key, num_envs)
        _randomization_fn = functools.partial(
            randomization_fn, rng=randomization_key,  # type: ignore
        )

    env = wrapper.wrap_for_brax_training(
        env=environment,
        episode_length=episode_length,
        action_repeat=action_repeat,
        randomization_fn=_randomization_fn,
    )

    # Reset Function
    reset_fn = jax.jit(env.reset)

    # Initialize Env State
    envs_key = jax.random.split(env_key, num_envs)
    envs_key = jax.device_put(envs_key, s_data)

    env_state = reset_fn(envs_key)

    # Initialize Agent and Optimizer:
    params = nnx.state(agent, nnx.Param)
    opt_state = optimizer.init(params)
    current_step = 0

    if restored_checkpoint is not None:
        nnx.update(agent, restored_checkpoint.agent)
        opt_state = restored_checkpoint.opt_state
    
    agent = jax.device_put(agent, s_replicated)
    opt_state = jax.device_put(opt_state, s_replicated)

    def minibatch_step(carry, data: types.Transition,):
        agent, opt_state, key = carry
        key, subkey = jax.random.split(key)

        grad_fn = nnx.value_and_grad(
            loss_function, argnums=nnx.DiffState(0, nnx.Param), has_aux=True,
        )
        (loss, metrics), grads = grad_fn(agent, data, subkey)

        params = nnx.state(agent, nnx.Param)

        if has_adaptive_kl_scheduler:
            updates, opt_state = optimizer.update(
                grads,
                opt_state,
                params,
                kl_mean=metrics['kl_mean'],
            )
        else:
            updates, opt_state = optimizer.update(grads, opt_state, params)

        new_params = optax.apply_updates(params, updates)

        nnx.update(agent, new_params)

        return (agent, opt_state, key), metrics

    def sgd_step(
        carry,
        unused_t,
        data: types.Transition,
    ):
        agent, opt_state, key = carry
        key, permutation_key, grad_key = jax.random.split(key, 3)

        # Shuffle
        def permute(x):
            x = jax.random.permutation(permutation_key, x)
            x = jnp.reshape(x, (num_minibatches, -1) + x.shape[1:])
            return x

        shuffled_data = jax.tree.map(permute, data)

        (agent, opt_state, _), metrics = jax.lax.scan(
            minibatch_step,
            (agent, opt_state, grad_key),
            shuffled_data,
            length=num_minibatches,
        )
        return (agent, opt_state, key), metrics

    def train_step(
        carry: Tuple[Agent, optax.OptState, types.State, types.PRNGKey],
        unused_t,
    ) -> Tuple[Tuple[Agent, optax.OptState, types.State, types.PRNGKey], types.Metrics]:
        agent, opt_state, state, key = carry
        next_key, sgd_key, rollout_key = jax.random.split(key, 3)

        # Turn off training for data collection:
        def policy_fn(x: jnp.ndarray, key: types.PRNGKey):
            actions, value, info = agent(x, key)
            return actions, {**info, 'value': value}

        # Generate Episode Data:
        def f(carry, unused_t):
            current_state, key = carry
            key, subkey = jax.random.split(key)
            next_state, data = training_utilities.unroll_policy_steps(
                env=env,
                state=current_state,
                policy=policy_fn,
                key=key,
                num_steps=num_policy_steps,
                extra_fields=('truncation',),
            )
            return (next_state, subkey), data

        (state, _), data = jax.lax.scan(
            f,
            (state, rollout_key),
            (),
            length=batch_size * num_minibatches // num_envs,
        )

        # Swap leading dimensions: (T, B, ...) -> (B, T, ...)
        data = jax.tree.map(lambda x: jnp.swapaxes(x, 1, 2), data)
        data = jax.tree.map(
            lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), data,
        )

        def update_normalization_statistics(
            agent_to_update: Agent, batch_data: types.Transition,
        ) -> Agent:
            # Update Policy Stats (Actor)
            if agent_to_update.policy.input_normalization is not None:
                flat_observation = batch_data.observation[agent_to_update.policy.observation_key].reshape(
                    -1, *agent_to_update.observation_size[agent_to_update.policy.observation_key],  # type: ignore
                )
                agent_to_update.policy.input_normalization.update(
                    flat_observation,
                )
            # Update Value Stats (Critic)
            if agent_to_update.value.input_normalization is not None:
                flat_observation = batch_data.observation[agent_to_update.value.observation_key].reshape(
                    -1, *agent_to_update.observation_size[agent_to_update.value.observation_key],  # type: ignore
                )
                agent_to_update.value.input_normalization.update(
                    flat_observation,
                )
            return agent_to_update

        if normalize_observations and not has_adaptive_kl_scheduler:
            agent = update_normalization_statistics(agent, data)

        (agent, opt_state, _), metrics = jax.lax.scan(
            functools.partial(sgd_step, data=data),
            (agent, opt_state, sgd_key),
            (),
            length=num_ppo_iterations,
        )

        if normalize_observations and has_adaptive_kl_scheduler:
            agent = update_normalization_statistics(agent, data)

        return (agent, opt_state, state, next_key), metrics

    @nnx.jit(donate_argnames=('agent', 'opt_state', 'state'))
    def training_epoch(
        agent: Agent,
        opt_state: optax.OptState,
        state: types.State,
        key: types.PRNGKey,
    ) -> Tuple[Agent, optax.OptState, types.State, types.Metrics]:
        (agent, opt_state, state, _), loss_metrics = jax.lax.scan(
            train_step,
            (agent, opt_state, state, key),
            (),
            length=num_training_steps,
        )
        loss_metrics = jax.tree.map(jnp.mean, loss_metrics)
        return agent, opt_state, state, loss_metrics

    # Setup Evaluation Environment:
    eval_randomization_fn = None
    if randomization_fn is not None:
        eval_randomization_key = jax.random.split(
            eval_key, num_evaluation_envs,
        )
        eval_randomization_fn = functools.partial(
            randomization_fn, rng=eval_randomization_key,  # type: ignore
        )

    eval_env = wrapper.wrap_for_brax_training(
        env=evaluation_environment,
        episode_length=eval_episode_length,
        action_repeat=action_repeat,
        randomization_fn=eval_randomization_fn,
    )
    evaluator = metrics_utilities.Evaluator(
        env=eval_env,
        num_envs=num_evaluation_envs,
        episode_length=eval_episode_length,
        action_repeat=action_repeat,
        key=eval_key,
        render_options=render_options,
    )

    def apply_inference(agent, obs, key):
        return agent.get_actions(
            obs, key, deterministic=deterministic_evaluation
        )

    def run_evaluation(training_metrics, current_step, iteration):
        policy_fn = jax.tree_util.Partial(apply_inference, agent)

        metrics = evaluator.evaluate(
            policy_fn=policy_fn, training_metrics=training_metrics, iteration=iteration,
        )

        if wandb_run is not None:
            log_data = dict(metrics)
            if evaluator.render and evaluator.render_flag:
                log_data['Visualizer'] = wandb.Html(evaluator.current_filepath, inject=False)
            wandb_run.log(log_data)

        progress_fn(iteration, current_step, metrics)

        return metrics

    # Training Loop:
    try:
        with mesh:

            if process_id == 0:
                metrics = run_evaluation(
                    training_metrics={},
                    current_step=current_step,
                    iteration=0,
                )

            if checkpoint_manager is not None:
                checkpoint_utilities.save_checkpoint(
                    manager=checkpoint_manager,
                    iteration=0,
                    agent=agent,
                    opt_state=opt_state,
                )

            training_walltime = 0.0

            for epoch_iteration in range(num_epochs):
                start_time = time.time()

                local_key, epoch_key = jax.random.split(local_key)

                agent, opt_state, env_state, training_metrics = training_epoch(
                    agent, opt_state, env_state, epoch_key
                )

                # Compute Timing
                jax.tree.map(lambda x: x.block_until_ready(), training_metrics)
                epoch_duration = time.time() - start_time
                training_walltime += epoch_duration

                # Metrics:
                training_metrics = jax.device_get(training_metrics)
                steps_per_second = num_steps_per_epoch / epoch_duration
                metrics = {
                    'training/steps_per_second': steps_per_second,
                    'training/walltime': training_walltime,
                    **{f'training/{name}': value for name, value in training_metrics.items()},
                }

                # Update Step Counts:
                current_step += num_steps_per_epoch

                # If reset per epoch else Auto Reset:
                if reset_per_epoch:
                    envs_key = jax.random.split(local_key, num_envs)
                    envs_key = jax.device_put(envs_key, s_data)
                    env_state = reset_fn(envs_key)

                if process_id == 0:
                    metrics = run_evaluation(
                        training_metrics=metrics,
                        current_step=current_step,
                        iteration=epoch_iteration + 1,
                    )

                if checkpoint_manager is not None:
                    checkpoint_utilities.save_checkpoint(
                        manager=checkpoint_manager,
                        iteration=epoch_iteration + 1,
                        agent=agent,
                        opt_state=opt_state,
                    )
    finally:
        if process_id == 0 and checkpoint_manager is not None:
            checkpoint_manager.wait_until_finished()

    return jax.device_get(agent), metrics
