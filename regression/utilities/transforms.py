"""Parameter reparameterizations for system identification.

Maps unconstrained optimizer variables ``theta`` (centered so ``theta = 0`` is
the nominal value) to physical parameters. Keeping the decision variables
dimensionless and O(1) means physical units do not distort the optimization
landscape. Each transform is registered by name; a regression spec selects one
per parameter group via its ``transform`` key, so the reparameterization is
decoupled from both the loss and the optimizer.

Transforms
----------
affine : physical = nominal + theta * scale
    Unbounded additive offset. Preferred for the log-Cholesky base parameters:
    every theta in R^10 already maps to a physically-consistent inertia, so a
    hard bound is unnecessary and would only re-introduce the constraint the
    parameterization was designed to remove. ``scale`` acts as a preconditioner
    / prior width (1-sigma), not a limit -- pair with an L2 term for the soft
    prior toward nominal instead of a box.

affine_tanh : physical = nominal + tanh(theta) * scale
    Bounded additive offset, deviation limited to +/- ``scale``. Use only when
    a hard box is genuinely wanted; note the gradient dies as theta saturates,
    which can silently stall the optimizer at a bound.

log_exp : physical = nominal * exp(theta)
    Positive, multiplicative, unbounded. For strictly-positive scale parameters
    (joint damping, armature, static friction): a unit step in ``theta`` is a
    fixed *fractional* change, so the coordinate is scale-invariant and
    ``mean(theta**2)`` is a log-normal prior centered at nominal. Cannot go
    negative, unlike a nominal-centered additive transform.

log_exp_tanh : physical = nominal * exp(tanh(theta) * scale)
    Positive and multiplicative but range-limited to
    [nominal * exp(-scale), nominal * exp(scale)].
"""

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import jax
import jax.numpy as jnp

Array = jax.Array
TransformFn = Callable[[Array, Array, Optional[Array]], Array]

_ATANH_EPS = 1e-6
_LOG_EXP_CLIP = 12.0  # NaN guard on the exponent (exp(12) ~ 1.6e5); not a constraint


@dataclass(frozen=True)
class ParameterTransform:
    """A bijection between optimizer space (theta) and physical space.

    Both callables take ``(x, nominal, scale)``; ``scale`` may be ``None`` for
    transforms that do not use it (e.g. ``log_exp``).
    """

    forward: TransformFn
    inverse: TransformFn
    uses_scale: bool
    positive_anchor: bool = False


def _affine_forward(theta, nominal, scale):
    return nominal + theta * scale


def _affine_inverse(value, nominal, scale):
    return (value - nominal) / scale


def _affine_tanh_forward(theta, nominal, scale):
    return nominal + jnp.tanh(theta) * scale


def _affine_tanh_inverse(value, nominal, scale):
    ratio = jnp.clip((value - nominal) / scale, -1.0 + _ATANH_EPS, 1.0 - _ATANH_EPS)
    return jnp.arctanh(ratio)


def _log_exp_forward(theta, nominal, scale):
    safe_theta = jnp.clip(theta, min=-_LOG_EXP_CLIP, max=_LOG_EXP_CLIP)
    return nominal * jnp.exp(safe_theta)


def _log_exp_inverse(value, nominal, scale):
    return jnp.log(value / nominal)


def _log_exp_tanh_forward(theta, nominal, scale):
    safe_theta_scale = jnp.clip(jnp.tanh(theta) * scale, min=-_LOG_EXP_CLIP, max=_LOG_EXP_CLIP)
    return nominal * jnp.exp(safe_theta_scale)


def _log_exp_tanh_inverse(value, nominal, scale):
    ratio = jnp.clip(
        jnp.log(value / nominal) / scale, -1.0 + _ATANH_EPS, 1.0 - _ATANH_EPS
    )
    return jnp.arctanh(ratio)


TRANSFORMS: Dict[str, ParameterTransform] = {
    "affine": ParameterTransform(
        _affine_forward, _affine_inverse, uses_scale=True,
    ),
    "affine_tanh": ParameterTransform(
        _affine_tanh_forward, _affine_tanh_inverse, uses_scale=True,
    ),
    "log_exp": ParameterTransform(
        _log_exp_forward, _log_exp_inverse,
        uses_scale=False, positive_anchor=True,
    ),
    "log_exp_tanh": ParameterTransform(
        _log_exp_tanh_forward, _log_exp_tanh_inverse,
        uses_scale=True, positive_anchor=True,
    ),
}

DEFAULT_TRANSFORM = "affine"


def get_transform(name: str) -> ParameterTransform:
    try:
        return TRANSFORMS[name]
    except KeyError:
        raise KeyError(
            f"Unknown parameter transform '{name}'. "
            f"Available: {sorted(TRANSFORMS)}"
        ) from None


def validate_transform_spec(name: str, spec: dict) -> ParameterTransform:
    """Fail fast if a regression spec entry violates its transform's contract.

    Checks that ``field`` and a valid ``transform`` are present, and that ``scale`` is supplied iff the
    transform uses one. Returns the resolved transform for convenience.
    """
    if not isinstance(spec, dict):
        raise ValueError(f"Regression parameter '{name}' spec must be a dict, got {type(spec)}.")
    if "field" not in spec:
        raise ValueError(f"Regression parameter '{name}' is missing required key 'field'.")

    transform_name = spec.get("transform", DEFAULT_TRANSFORM)
    transform = get_transform(transform_name)

    has_scale = spec.get("scale") is not None
    if transform.uses_scale and not has_scale:
        raise ValueError(
            f"Parameter '{name}': transform '{transform_name}' requires a 'scale', "
            f"but none was provided."
        )
    if not transform.uses_scale and has_scale:
        raise ValueError(
            f"Parameter '{name}': transform '{transform_name}' is scale-free, "
            f"but a 'scale' was provided (likely stale config)."
        )
    return transform


def transform_parameters(
    opt_params: Dict[str, Array],
    nominal_parameters: Dict[str, Array],
    *,
    regression_spec: Dict[str, dict],
    parameter_scale: Dict[str, Array],
) -> Dict[str, Array]:
    """Map optimizer variables to physical parameters, per-group.

    Args:
        opt_params: Unconstrained decision variables, keyed by parameter name.
        nominal_parameters: Nominal (theta = 0) physical values, same keys.
        regression_spec: Per-parameter spec; ``spec['transform']`` selects the
            reparameterization (defaults to ``affine_tanh``).
        parameter_scale: Per-parameter scale passed to scale-using transforms.
    """
    physical = {}
    for name, theta in opt_params.items():
        transform = get_transform(regression_spec[name].get("transform", DEFAULT_TRANSFORM))
        physical[name] = transform.forward(
            theta, nominal_parameters[name], parameter_scale.get(name)
        )
    return physical


def untransform_parameters(
    physical_parameters: Dict[str, Array],
    nominal_parameters: Dict[str, Array],
    *,
    regression_spec: Dict[str, dict],
    parameter_scale: Dict[str, Array],
) -> Dict[str, Array]:
    """Inverse of :func:`transform_parameters` (physical -> optimizer space).

    Useful for initializing the optimizer from a non-nominal guess and for
    checking round-trip consistency.
    """
    theta = {}
    for name, value in physical_parameters.items():
        transform = get_transform(regression_spec[name].get("transform", DEFAULT_TRANSFORM))
        theta[name] = transform.inverse(
            value, nominal_parameters[name], parameter_scale.get(name)
        )
    return theta
