import jax
from mujoco import mjx


def init_function(
    model: mjx.Model,
    qpos: jax.Array,
    qvel: jax.Array,
    ctrl: jax.Array,
) -> mjx.Data:
    data = mjx.make_data(model)
    data = data.replace(qpos=qpos, qvel=qvel, ctrl=ctrl)
    data = mjx.forward(model, data)
    return data

def step_function(
    model: mjx.Model,
    data: mjx.Data,
    ctrl: jax.Array,
    n_substeps: int,
) -> mjx.Data:
    data = data.replace(ctrl=ctrl)

    def loop(carry: mjx.Data, unused_t):
        return mjx.step(model, carry), None

    data, _ = jax.lax.scan(loop, data, None, length=n_substeps)
    return data
