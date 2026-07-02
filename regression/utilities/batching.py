from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp


def _leading_dim(tree: Any) -> int:
    return jax.tree.leaves(tree)[0].shape[0]


def chunk_dataset(dataset: Any, chunk_size: int) -> Any:
    """Reshape every leaf ``(N, ...) -> (n_chunks, chunk_size, ...)`` for lax.scan.

    Windows beyond the last full chunk are dropped (with a warning): a
    deterministic objective needs a fixed set of windows, and equal chunks are
    what make the chunked mean exact.
    """
    n = _leading_dim(dataset)
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}.")
    n_chunks = n // chunk_size
    if n_chunks == 0:
        raise ValueError(
            f"chunk_size ({chunk_size}) exceeds the number of samples ({n})."
        )
    dropped = n - n_chunks * chunk_size
    if dropped:
        print(
            f"WARNING: chunk_dataset dropping {dropped} of {n} samples "
            f"(chunk_size={chunk_size} does not divide N)."
        )
    return jax.tree.map(
        lambda x: x[: n_chunks * chunk_size].reshape(
            n_chunks, chunk_size, *x.shape[1:]
        ),
        dataset,
    )


def chunked_value(loss_fn: Callable[[Any, Any], jax.Array]) -> Callable[[Any, Any], jax.Array]:
    """Mean of ``loss_fn(params, chunk)`` over the leading chunk axis.

    Pure evaluation (no AD state is kept across chunks) -- suitable as the
    line-search ``value_fn``. For a loss of the form ``data_mean + reg(params)``
    the chunk-mean reproduces the unchunked loss exactly: the data term is a
    mean of equal-sized chunk means and the reg term is invariant across chunks.
    """

    def value(params: Any, chunked_batch: Any) -> jax.Array:
        n_chunks = _leading_dim(chunked_batch)

        def body(total, chunk):
            return total + loss_fn(params, chunk), None

        total, _ = jax.lax.scan(body, jnp.zeros(()), chunked_batch)
        return total / n_chunks

    return value


def chunked_value_and_grad(
    value_and_grad_fn: Callable[[Any, Any], Tuple[jax.Array, Any]],
) -> Callable[[Any, Any], Tuple[jax.Array, Any]]:
    """Accumulate ``(value, grad)`` over chunks; means match the unchunked result.

    ``value_and_grad_fn`` is applied per chunk (e.g. ``forward_mode_value_and_grad``
    of the per-chunk loss), so only one chunk's linearization and tangents are
    alive at a time. Works with any inner AD mode (forward or reverse).
    """

    def value_and_grad(params: Any, chunked_batch: Any) -> Tuple[jax.Array, Any]:
        n_chunks = _leading_dim(chunked_batch)

        def body(carry, chunk):
            total, grads = carry
            value, grad = value_and_grad_fn(params, chunk)
            return (total + value, jax.tree.map(jnp.add, grads, grad)), None

        init = (jnp.zeros(()), jax.tree.map(jnp.zeros_like, params))
        (total, grads), _ = jax.lax.scan(body, init, chunked_batch)
        return total / n_chunks, jax.tree.map(lambda g: g / n_chunks, grads)

    return value_and_grad
