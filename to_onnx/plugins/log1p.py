from typing import TYPE_CHECKING, Any
import jax
import jax.numpy as jnp
import numpy as np
from jax.core import ShapedArray
from jax.extend import core as jcore_ext
from jax.lax import log1p_p

from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, PLUGIN_REGISTRY
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG

if TYPE_CHECKING:
    from jax2onnx.converter import IRContext
    from jax.core import JaxprEqn
else:
    IRContext = Any
    JaxprEqn = Any


class Log1pPlugin(PrimitiveLeafPlugin):
    """ 
        Plugin for converting jax.lax.log1p to ONNX.
        ONNX ops: Log(Add(Constant(1.0), x)).
    """

    def lower(self, ctx: IRContext, eqn: JaxprEqn):
        builder = ctx.builder
        
        x_var = eqn.invars[0]
        out_var = eqn.outvars[0]

        x_val = ctx.get_value_for_var(
            x_var, name_hint=ctx.fresh_name("log1p_in")
        )
        out_spec = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("log1p_out")
        )

        dtype = x_var.aval.dtype
        val = np.array(1.0, dtype=dtype)
        aval = ShapedArray(shape=(), dtype=dtype)
        one_literal = jcore_ext.Literal(val, aval)
        
        one_const_val = ctx.get_value_for_var(
            one_literal, name_hint=ctx.fresh_name("one")
        )

        add_val = builder.add(one_const_val, x_val)
        log_val = builder.log(add_val, _outputs=[out_spec.name])

        log_val.type = out_spec.type
        log_val.shape = out_spec.shape

        ctx.bind_value_for_var(out_var, log_val)


try:
    PLUGIN_REGISTRY[log1p_p] = Log1pPlugin()
    PLUGIN_REGISTRY["log1p"] = Log1pPlugin()
    print("Successfully registered plugin for jax.lax.log1p primitive.")
except Exception as e:
    print(f"Warning: Could not register plugin for jax.lax.log1p primitive: {e}")
