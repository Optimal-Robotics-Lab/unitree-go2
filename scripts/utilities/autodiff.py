import jax
import jax.numpy as jnp
import jax.flatten_util

def forward_mode_value_and_grad(loss_fn: callable):
    """
    Returns a function that computes (value, grad) using forward-mode AD.
    Uses jax.linearize to run the primal simulation ONCE, caching linearization points.
    """
    def value_and_grad_fwd(params, *args):
        # Linearize: 
        loss_val, jvp_fn = jax.linearize(lambda p: loss_fn(p, *args), params)
        
        # Create Basis:
        flat_params, unflatten = jax.flatten_util.ravel_pytree(params)
        basis = jnp.eye(len(flat_params))
        
        # Vectorize the Linear Pass:
        flat_grads = jax.vmap(lambda v: jvp_fn(unflatten(v)))(basis)
        
        # Unflatten back to dict
        grads = unflatten(flat_grads)
        
        return loss_val, grads

    return value_and_grad_fwd
