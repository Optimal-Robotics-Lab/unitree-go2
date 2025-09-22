from typing import Any, Optional, Union

import jax.numpy as jnp

import orbax.checkpoint as ocp
from orbax.checkpoint import CheckpointManager, CheckpointManagerOptions
import optax
import flax.struct
from brax.training.acme import running_statistics
import training.module_types as types
import training.distribution_utilities as distribution
import training.algorithms.ppo.network_utilities as ppo_networks


@flax.struct.dataclass
class TrainState:
    opt_state: optax.OptState
    params: ppo_networks.PPONetworkParams
    normalization_params: running_statistics.RunningStatisticsState
    env_steps: jnp.ndarray


@flax.struct.dataclass
class RestoredCheckpoint:
    network: ppo_networks.PPONetworks
    train_state: TrainState


@flax.struct.dataclass
class network_metadata:
    policy_layer_size: list[int]
    value_layer_size: list[int]
    policy_depth: int
    value_depth: int
    activation: Union[types.ActivationFn, str]
    policy_kernel_init: Union[types.Initializer, str]
    value_kernel_init: Union[types.Initializer, str]
    policy_observation_key: str
    value_observation_key: str
    action_distribution: Union[distribution.ParametricDistribution, str]


@flax.struct.dataclass
class loss_metadata:
    clip_coef: float
    value_coef: float
    entropy_coef: float
    gamma: float
    gae_lambda: float
    normalize_advantages: bool


@flax.struct.dataclass
class training_metadata:
    num_epochs: int
    num_training_steps: int
    episode_length: int
    num_policy_steps: int
    action_repeat: int
    num_envs: int
    num_evaluation_envs: int
    num_evaluations: int
    deterministic_evaluation: bool
    reset_per_epoch: bool
    seed: int
    batch_size: int
    num_minibatches: int
    num_ppo_iterations: int
    normalize_observations: bool
    optimizer: Union[optax.GradientTransformation, str]


def empty_network_metadata() -> network_metadata:
    return network_metadata(
        policy_layer_size=[],
        value_layer_size=[],
        policy_depth=0,
        value_depth=0,
        activation='',
        policy_kernel_init='',
        value_kernel_init='',
        policy_observation_key='',
        value_observation_key='',
        action_distribution='',
    )


def empty_loss_metadata() -> loss_metadata:
    return loss_metadata(
        clip_coef=0.0,
        value_coef=0.0,
        entropy_coef=0.0,
        gamma=0.0,
        gae_lambda=0.0,
        normalize_advantages=False,
    )


def empty_training_metadata() -> training_metadata:
    return training_metadata(
        num_epochs=0,
        num_training_steps=0,
        episode_length=0,
        num_policy_steps=0,
        action_repeat=0,
        num_envs=0,
        num_evaluation_envs=0,
        num_evaluations=0,
        deterministic_evaluation=False,
        reset_per_epoch=False,
        seed=0,
        batch_size=0,
        num_minibatches=0,
        num_ppo_iterations=0,
        normalize_observations=False,
        optimizer='',
    )


def default_checkpoint_options() -> CheckpointManagerOptions:
    options = CheckpointManagerOptions(
        max_to_keep=10,
        save_interval_steps=1,
        create=True,
    )
    return options


def default_checkpoint_metadata() -> dict:
    return {'iteration': 0}


def save_checkpoint(
    checkpoint_directory: str,
    manager_options: CheckpointManagerOptions,
    registry: ocp.handlers.CheckpointHandlerRegistry,
    iteration: int,
    train_state: TrainState,
    **metadata: Union[dict[str, Any], flax.struct.PyTreeNode],
) -> None:
    with ocp.CheckpointManager(
        directory=checkpoint_directory,
        options=manager_options,
        handler_registry=registry,
    ) as manager:
        # Save Checkpoint:
        args = {
            'train_state': ocp.args.PyTreeSave(train_state),  # type: ignore
        }
        metadata = {
            key: ocp.args.PyTreeSave(value) for key, value in metadata.items()  # type: ignore
        }
        args.update(metadata)  # type: ignore
        manager.save(
            iteration,
            args=ocp.args.Composite(
                **args,
            ),
        )


def load_train_state(
    directory: str,
    options: CheckpointManagerOptions,
    registry: ocp.handlers.CheckpointHandlerRegistry,
    restore_iteration: Optional[int] = None,
    **data: Union[dict[str, Any], flax.struct.PyTreeNode],
) -> Any:
    with ocp.CheckpointManager(
        directory=directory,
        options=options,
        handler_registry=registry,
    ) as manager:
        # Create abstract states:
        args = {
            key: ocp.args.PyTreeRestore(value) for key, value in data.items()
        }

        # Load Checkpoint:
        if restore_iteration is None:
            restore_iteration = manager.latest_step()

        restored = manager.restore(
            restore_iteration,
            args=ocp.args.Composite(
                **args,
            ),
        )

    return restored


def load_checkpoint(
    directory: str,
    options: CheckpointManagerOptions,
    registry: ocp.handlers.CheckpointHandlerRegistry,
    restore_iteration: Optional[int] = None,
) -> Any:
    with ocp.CheckpointManager(
        directory=directory,
        options=options,
        handler_registry=registry,
    ) as manager:
        # Load Checkpoint:
        if restore_iteration is None:
            restore_iteration = manager.latest_step()

        restored = manager.restore(
            restore_iteration,
        )

    return restored
