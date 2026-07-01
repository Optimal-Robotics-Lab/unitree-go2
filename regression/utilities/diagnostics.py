"""JIT-compatible diagnostics for the log-Cholesky inertia gradient path.

The map ``theta -> pi -> I_com`` is smooth (Rucker & Wensing), but MuJoCo stores
rotational inertia as principal moments (``body_inertia``) plus a frame
(``body_iquat``). Recovering that representation requires an eigendecomposition
of ``I_com``, whose *eigenvector* VJP scales like ``1 / (lambda_i - lambda_j)``.
Near-degenerate principal moments (rod-like links: Go2 thighs/calves) therefore
produce ill-conditioned or non-finite gradients even though the underlying
physics is smooth. These helpers let you watch for that during training without
breaking ``jit`` / ``vmap`` / ``grad``.
"""

from functools import partial

import jax
import jax.numpy as jnp


def principal_moment_gaps(inertia: jax.Array) -> jax.Array:
    """Minimum pairwise gap between the principal moments of an inertia tensor.

    This is the single scalar that predicts ``eigh`` gradient conditioning: the
    eigenvector VJP magnitude grows like ``1 / gap``. Batched over any leading
    dimensions of a ``(..., 3, 3)`` symmetric inertia. The input is detached, so
    this is purely an observable and never contributes to gradients.

    Args:
        inertia: ``(..., 3, 3)`` symmetric inertia tensor(s).

    Returns:
        ``(...)`` minimum ``|lambda_i - lambda_j|`` per tensor.
    """
    w = jnp.linalg.eigvalsh(jax.lax.stop_gradient(inertia))  # ascending, (..., 3)
    d01 = jnp.abs(w[..., 1] - w[..., 0])
    d12 = jnp.abs(w[..., 2] - w[..., 1])
    d02 = jnp.abs(w[..., 2] - w[..., 0])
    return jnp.minimum(jnp.minimum(d01, d12), d02)


@partial(jax.custom_vjp, nondiff_argnums=(0,))
def grad_probe(tag: str, x: jax.Array) -> jax.Array:
    """Identity on the forward pass; reports the cotangent on the backward pass.

    Insert this immediately before an operation whose gradient you want to watch
    (e.g. the ``eigh`` input) as ``x = grad_probe("eigh_in", x)``. During reverse
    -mode differentiation it prints whether the incoming cotangent is finite and
    its norm, which is exactly where the ``1 / gap`` blow-up first appears. The
    forward value and the propagated gradient are unchanged, so it is safe to
    leave in place; ``jax.debug.print`` keeps it ``jit`` / ``vmap`` compatible.

    Args:
        tag: Static label included in the printed line.
        x: Value flowing forward; its cotangent is inspected.

    Returns:
        ``x`` unchanged.
    """
    return x


def _grad_probe_fwd(tag: str, x: jax.Array):
    return x, None


def _grad_probe_bwd(tag: str, _residual, g: jax.Array):
    g_flat = jnp.ravel(g)
    finite_mask = jnp.isfinite(g_flat)
    g_safe = jnp.where(finite_mask, g_flat, 0.0)  # keep the print itself finite
    jax.debug.print(
        "[grad_probe:{t}] all_finite={f} |g|={n:.3e} max|g|={m:.3e}",
        t=tag,
        f=jnp.all(finite_mask),
        n=jnp.linalg.norm(g_safe),
        m=jnp.max(jnp.abs(g_safe)),
    )
    return (g,)  # pass the true cotangent through untouched


grad_probe.defvjp(_grad_probe_fwd, _grad_probe_bwd)
