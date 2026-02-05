from typing import Callable, Dict, Tuple
import jax
import flax.struct
import optax

TrainState = Tuple[Dict[str, jax.Array], optax.OptState]
ObjectiveFunction = Callable[[jax.Array, jax.Array], jax.Array]

@flax.struct.dataclass
class Dataset:
    qpos: jax.Array
    qvel: jax.Array
    actuator_force: jax.Array
    ctrl: jax.Array
