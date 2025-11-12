import os
import jax
import jax.numpy as jnp
import numpy as np
from absl import app, flags
import functools  # We'll need this

# -----------------------------------------------------------------
# START: LOCAL PLUGIN FOR 'log1p'
# (This part is correct and necessary)
# -----------------------------------------------------------------
import jax2onnx
from typing import TYPE_CHECKING, Any
from jax.lax import log1p_p
from jax.core import ShapedArray
from jax.extend import core as jcore_ext
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, PLUGIN_REGISTRY

if TYPE_CHECKING:
    from jax2onnx.converter import IRContext
    from jax.core import JaxprEqn
else:
    IRContext = Any
    JaxprEqn = Any

os.environ['XLA_FLAGS'] = (
    '--xla_gpu_enable_triton_softmax_fusion=true '
    '--xla_gpu_triton_gemm_any=True '
    '--xla_gpu_enable_async_collectives=true '
    '--xla_gpu_enable_latency_hiding_scheduler=true '
    '--xla_gpu_enable_highest_priority_async_stream=true '
)

class Log1pPlugin(PrimitiveLeafPlugin):
    primitive_name = "log1p"
    def lower(self, ctx: IRContext, eqn: JaxprEqn):
        builder = ctx.builder
        x_var = eqn.invars[0]; out_var = eqn.outvars[0]
        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("log1p_in"))
        out_spec = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("log1p_out"))
        dtype = x_var.aval.dtype
        val = np.array(1.0, dtype=dtype); aval = ShapedArray(shape=(), dtype=dtype)
        one_literal = jcore_ext.Literal(val, aval)
        one_const_val = ctx.get_value_for_var(one_literal, name_hint=ctx.fresh_name("one"))
        add_val = builder.add(one_const_val, x_val)
        log_val = builder.log(add_val, _outputs=[out_spec.name])
        log_val.type = out_spec.type; log_val.shape = out_spec.shape
        ctx.bind_value_for_var(out_var, log_val)

try:
    PLUGIN_REGISTRY[log1p_p] = Log1pPlugin()
    PLUGIN_REGISTRY["log1p"] = Log1pPlugin()
    print("Successfully registered custom local plugin for 'log1p' primitive.")
except Exception as e:
    print(f"Warning: Could not register custom 'log1p' plugin: {e}")
# --------------------------------------------------


# --- IMPORTS (with sys.path fix) ---
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from training.envs.unitree_go2.unitree_go2_joystick import UnitreeGo2Env
    from training.algorithms.ppo.load_utilities import load_policy
except ImportError:
    print("Error: Could not import 'training' modules.")
    exit(1)

# --- MODEL DEFINITIONS (Copied from your prompt) ---
from flax import linen as nn
from typing import Sequence, Callable, Any

ActivationFn = Callable[[jnp.ndarray], jnp.ndarray]
Initializer = Callable[..., Any]

class MLP(nn.Module):
    layer_sizes: Sequence[int]
    activation: ActivationFn = nn.tanh
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform()
    activate_final: bool = False
    bias: bool = True
    layer_normalization: bool = False

    @nn.compact
    def __call__(self, x: jax.Array):
        for i, layer_size in enumerate(self.layer_sizes):
            x = nn.Dense(
                features=layer_size,
                kernel_init=self.kernel_init,
                use_bias=self.bias,
                name=f"dense_{i}",
            )(x)
            if i != len(self.layer_sizes) - 1 or self.activate_final:
                x = self.activation(x)
                if self.layer_normalization:
                    x = nn.LayerNorm(name=f"layer_norm_{i}")(x)
        return x
# --------------------------------------------------

os.environ["MUJOCO_GL"] = "egl"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

FLAGS = flags.FLAGS
flags.DEFINE_string('checkpoint_name', None, 'Desired checkpoint folder name to load.', short_name='c')

def main(argv=None):
    if not FLAGS.checkpoint_name:
        print("Error: Please provide a checkpoint name using -c or --checkpoint_name")
        return

    print(f"Loading JAX policy from checkpoint: {FLAGS.checkpoint_name}...")
    env = UnitreeGo2Env()
    # 'params' is a tuple: (normalization_params, policy_params)
    _, params, metadata = load_policy(
        checkpoint_name=FLAGS.checkpoint_name,
        environment=env,
    )
    
    obs_size = env.observation_size
    act_size = env.action_size
    try:
        state_input_shape = (1,) + obs_size["state"]
        print(f"State input shape: {state_input_shape}")
    except Exception as e:
        print(f"Error determining shapes: {e}")
        return

    example_input_array = jnp.zeros(state_input_shape, dtype=jnp.float32)

    # --- 1. Extract all parameters (as JAX arrays) ---
    normalization_params = params[0]
    policy_params = params[1]
    mean_state_jax = normalization_params.mean['state']
    std_state_jax = normalization_params.std['state']
    
    policy_layers = list(metadata.network_metadata['policy_layer_size'])
    final_layer_size = act_size * 2 # From your network_utilities.py

    # --- 2. Convert all JAX arrays to NumPy arrays ---
    # This is the key. JAX's tracer will treat these as
    # compile-time constants, not dynamic inputs.
    print("Converting JAX params to NumPy constants...")
    np_policy_params = jax.tree.map(np.array, policy_params)
    np_mean_state = np.array(mean_state_jax)
    np_std_state = np.array(std_state_jax)
    print("Conversion complete.")


    # --- 3. Define the "pure" function to be traced ---
    # This function closes over the NumPy arrays, which JAX
    # will treat as baked-in constants.
    def exportable_policy_fn(state_obs):
        
        # 1. Define the model architecture *inside* the function
        policy_network_def = MLP(
           layer_sizes=policy_layers + [final_layer_size],
           activation=nn.swish, # From your network_utilities.py
           activate_final=False
        )

        # 2. Manually normalize the state input
        #    This is the (data - mean) / std part.
        #    'np_mean_state' and 'np_std_state' are NumPy constants.
        norm_obs = (state_obs - np_mean_state) / np_std_state
        
        # 3. Apply the MLP
        #    'np_policy_params' is a pytree of NumPy constants.
        logits = policy_network_def.apply(np_policy_params, norm_obs) 
        
        # 4. Apply action distribution logic
        loc, _ = jnp.split(logits, 2, axis=-1)
        return jnp.tanh(loc)

    # --- 4. JIT the final, 1-argument function ---
    jitted_wrapper = jax.jit(exportable_policy_fn)

    print(f"Testing JAX wrapper with input shape: {example_input_array.shape}")
    jax_output = jitted_wrapper(example_input_array)
    print(f"JAX wrapper output shape: {jax_output.shape}. (Expected: (1, {act_size}))")

    output_dir = "onnx_models"
    output_path = f"{output_dir}/{FLAGS.checkpoint_name}_jax.onnx"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nConverting JAX model to ONNX at {output_path}...")
    jax2onnx.to_onnx(
        fn=jitted_wrapper,             # Pass the JIT-compiled 1-arg function
        inputs=[example_input_array],  # Pass only the single dynamic input
        opset=11,
        return_mode="file",
        output_path=output_path,
        model_name=f"{FLAGS.checkpoint_name}_jax",
        enable_double_precision=False
    )
    
    print("-----------------------------------------")
    print(f"✅ Successfully converted JAX model to ONNX!")
    print(f"Model saved to: {output_path}")
    print("-----------------------------------------")

if __name__ == '__main__':
    app.run(main)