import os
from typing import Any, Dict, List
import dataclasses
import json

import jax
import flax.struct
import flax.nnx as nnx
import optax
import orbax.checkpoint as ocp

from training.optimizer import OptimizerConfig, create_optimizer


@dataclasses.dataclass
class AgentMetadata:
    observation_size: Dict[str, Any]
    action_size: int
    policy_layer_sizes: List[int]
    value_layer_sizes: List[int]
    policy_input_normalization: str
    value_input_normalization: str
    activation: str
    policy_kernel_init: str
    value_kernel_init: str
    policy_observation_key: str
    value_observation_key: str
    action_distribution: str


@dataclasses.dataclass
class LossMetadata:
    policy_clip_coef: float
    value_clip_coef: float
    value_coef: float
    entropy_coef: float
    gamma: float
    gae_lambda: float
    normalize_advantages: bool


@dataclasses.dataclass
class TrainingMetadata:
    num_epochs: int
    num_training_steps: int
    episode_length: int
    num_policy_steps: int
    action_repeat: int
    num_envs: int
    num_evaluation_envs: int
    deterministic_evaluation: bool
    reset_per_epoch: bool
    seed: int
    batch_size: int
    num_minibatches: int
    num_ppo_iterations: int
    normalize_observations: bool


@dataclasses.dataclass
class RestoredCheckpoint:
    agent: nnx.Module
    optimizer: optax.GradientTransformation
    opt_state: optax.OptState | None


@dataclasses.dataclass
class CheckpointMetadata:
    optimizer_config: OptimizerConfig
    agent_metadata: AgentMetadata
    loss_metadata: LossMetadata
    training_metadata: TrainingMetadata


def create_checkpoint_manager(
    checkpoint_directory: str,
    max_to_keep: int = 5,
    save_interval_steps: int = 1,
) -> ocp.CheckpointManager:
    """
        Creates a Orbax CheckpointManager with StandardCheckpointer.
    """
    options = ocp.CheckpointManagerOptions(
        max_to_keep=max_to_keep,
        save_interval_steps=save_interval_steps,
        create=True,
    )

    return ocp.CheckpointManager(
        os.path.abspath(checkpoint_directory),
        options=options,
    )


def save_config(manager: ocp.CheckpointManager, metadata: CheckpointMetadata):
    if not os.path.exists(manager.directory):
        os.makedirs(manager.directory)

    config_path = os.path.join(manager.directory, 'config.json')

    if not os.path.exists(config_path):
        metadata_dict = dataclasses.asdict(metadata)

        with open(config_path, 'w') as f:
            json.dump(metadata_dict, f, indent=2)


def save_checkpoint(
    manager: ocp.CheckpointManager,
    iteration: int,
    agent: nnx.Module,
    opt_state: optax.OptState,
) -> None:
    _, agent_state = nnx.split(agent)

    manager.save(
        iteration,
        args=ocp.args.Composite(
            agent=ocp.args.StandardSave(agent_state),
            opt_state=ocp.args.StandardSave(opt_state),
        ),
    )


def restore_training_state(manager, agent: nnx.Module, iteration=None):
    if iteration is None:
        iteration = manager.latest_step()

    config_path = os.path.join(manager.directory, 'config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with open(config_path, 'r') as f:
        metadata_dict = json.load(f)

    opt_config = OptimizerConfig(**metadata_dict.pop('optimizer_config'))
    metadata = CheckpointMetadata(optimizer_config=opt_config, **metadata_dict)

    optimizer = create_optimizer(opt_config)
    
    abstract_agent_state = nnx.state(agent)
    abstract_params = nnx.state(agent, nnx.Param)
    abstract_opt_state = jax.eval_shape(
        lambda: optimizer.init(abstract_params)
    )

    restore_args = ocp.args.Composite(
        agent=ocp.args.StandardRestore(abstract_agent_state),
        opt_state=ocp.args.StandardRestore(abstract_opt_state),
    )

    restored_state = manager.restore(iteration, args=restore_args)

    restored_checkpoint = RestoredCheckpoint(
        agent=restored_state.agent,
        optimizer=optimizer,
        opt_state=restored_state.opt_state,
    )

    return restored_checkpoint, metadata
