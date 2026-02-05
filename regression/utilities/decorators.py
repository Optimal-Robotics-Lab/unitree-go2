import functools
import inspect

import jax
from jax import core


def force_static_args(*arg_names: str):
    """
        Decorator that mandates specific arguments are concrete (non-Tracers).
        This forces the caller to use functools.partial to bind these arguments
        before passing the function to jax.jit.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Bind arguments to the function signature
            sig = inspect.signature(func)
            try:
                bound_args = sig.bind(*args, **kwargs)
            except TypeError as e:
                # Add context to the error
                raise TypeError(f"Error binding arguments in {func.__name__}: {e}")
            
            bound_args.apply_defaults()
            
            for name in arg_names:
                if name in bound_args.arguments:
                    val = bound_args.arguments[name]
                    
                    # Efficiently check leaves for Tracers
                    leaves = jax.tree_util.tree_leaves(val)
                    if any(isinstance(leaf, core.Tracer) for leaf in leaves):
                        raise TypeError(
                            f"CRITICAL JAX ERROR: Argument '{name}' in '{func.__name__}' is a Tracer.\n"
                            f"This argument is marked as STATIC. You must bind it using "
                            f"functools.partial(func, {name}=...) before JIT compilation.\n"
                            f"Passing it dynamically forces JAX to re-trace or fails on non-array types."
                        )
            
            return func(*args, **kwargs)
        return wrapper
    return decorator
