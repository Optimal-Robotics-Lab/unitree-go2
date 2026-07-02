from __future__ import annotations

import dataclasses
from typing import Any, Callable, Dict, Protocol, Tuple

import jax
import optax


# Optimizers that run their own line search over the full-batch objective.
_LINE_SEARCH_TYPES = frozenset({"lbfgs"})


@dataclasses.dataclass
class OptimizerConfig:
    """Config-constructible optimizer specification.

    ``optimizer_type`` / ``scheduler_type`` are optax attribute names resolved via
    ``getattr`` so any optax optimizer/scheduler is usable without code changes.
    """

    optimizer_type: str = "adamw"
    scheduler_type: str = "warmup_cosine_decay_schedule"
    optimizer_params: Dict[str, Any] = dataclasses.field(default_factory=dict)
    scheduler_params: Dict[str, Any] = dataclasses.field(default_factory=dict)
    grad_clip_norm: float = 1.0

    @property
    def is_line_search(self) -> bool:
        return self.optimizer_type in _LINE_SEARCH_TYPES


def _get_optax(name: str):
    try:
        return getattr(optax, name)
    except AttributeError:
        raise ValueError(
            f"Unknown optax attribute '{name}'. Check the optax documentation "
            f"for valid optimizer/scheduler names."
        ) from None


def _build_scheduler(cfg: OptimizerConfig) -> optax.Schedule:
    scheduler_cls = _get_optax(cfg.scheduler_type)
    params = dict(cfg.scheduler_params)
    return scheduler_cls(**params)


def build_optimizer(cfg: OptimizerConfig) -> optax.GradientTransformation:
    """Construct an optax ``GradientTransformation`` from ``cfg``."""
    components = []

    # Only clip for first-order optimizers.
    if cfg.grad_clip_norm and cfg.grad_clip_norm > 0 and not cfg.is_line_search:
        components.append(optax.clip_by_global_norm(cfg.grad_clip_norm))

    optimizer_cls = _get_optax(cfg.optimizer_type)

    if cfg.is_line_search:
        components.append(optimizer_cls(**cfg.optimizer_params))
    else:
        scheduler = _build_scheduler(cfg)
        opt_params = {k: v for k, v in cfg.optimizer_params.items() if k != "learning_rate"}
        components.append(optimizer_cls(learning_rate=scheduler, **opt_params))

    return optax.chain(*components)


StepMetrics = Dict[str, Any]


class Solver(Protocol):

    def init(self, params) -> optax.OptState: ...

    def step(
        self, params, opt_state, batch,
    ) -> Tuple[Any, optax.OptState, StepMetrics]: ...


@dataclasses.dataclass
class StochasticSolver:
    """First-order minibatch solver: update(grads, state, params)."""

    optimizer: optax.GradientTransformation
    value_and_grad_fn: Callable[[Any, Any], Tuple[jax.Array, Any]]

    def init(self, params: optax.Params) -> optax.OptState:
        return self.optimizer.init(params)

    def step(
        self, params: optax.Params, opt_state: optax.OptState, batch: Any,
    ) -> Tuple[optax.Params, optax.OptState, StepMetrics]:
        loss, grads = self.value_and_grad_fn(params, batch)
        updates, opt_state = self.optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, {"loss": loss, "grads": grads}


@dataclasses.dataclass
class LineSearchSolver:
    """Full-batch line-search solver (L-BFGS): update needs value/grad/value_fn.

    ``value_and_grad_from_state`` reuses the value/grad already computed inside the
    previous step's line search, so a step costs one extra objective evaluation at
    most. Requires a deterministic (full-batch) objective.
    """

    optimizer: optax.GradientTransformation
    value_and_grad_fn: Callable[..., Tuple[jax.Array, Any]]
    loss_fn: Callable[[Any, Any], jax.Array]

    def init(self, params: optax.Params) -> optax.OptState:
        return self.optimizer.init(params)

    def step(
        self, params: optax.Params, opt_state: optax.OptState, batch: Any,
    ) -> Tuple[optax.Params, optax.OptState, StepMetrics]:
        loss, grads = self.value_and_grad_fn(params, state=opt_state)
        updates, opt_state = self.optimizer.update(
            grads, opt_state, params, value=loss, grad=grads, value_fn=self.loss_fn,
        )
        params = optax.apply_updates(params, updates)
        return params, opt_state, {"loss": loss, "grads": grads}


def make_solver(
    cfg: OptimizerConfig,
    optimizer: optax.GradientTransformation,
    value_and_grad_fn: Callable[..., Tuple[jax.Array, Any]],
    loss_fn: Callable[[Any, Any], jax.Array] | None = None,
) -> Solver:
    """Select the solver matching ``cfg``."""
    if cfg.is_line_search:
        if loss_fn is None:
            raise ValueError("Line-search solvers require a loss_fn.")

        return LineSearchSolver(
            optimizer=optimizer, value_and_grad_fn=value_and_grad_fn, loss_fn=loss_fn,
        )
    return StochasticSolver(optimizer=optimizer, value_and_grad_fn=value_and_grad_fn)
