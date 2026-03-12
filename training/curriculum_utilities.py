from typing import Callable, Any

import jax

import jax.numpy as jnp
from mujoco_playground import wrapper

CurriculumFn = Callable[[jax.Array], Any]


def get_exponential_schedule(
    transition_begin: int = 0, 
    transition_steps: int = 2440, 
    init_value: float = 1e-3, 
    target_value: float = 1.0
) -> CurriculumFn:
    """Returns a function that computes an exponential curriculum factor."""
    growth_rate = target_value / init_value
    
    def schedule(step: jax.Array) -> jax.Array:
        rate_factor = jnp.clip(
            (step - transition_begin) / transition_steps, 0.0, 1.0
        )
        return init_value * (growth_rate ** rate_factor)
        
    return schedule


def get_constant_schedule(value: float = 1.0) -> CurriculumFn:
    """Returns a function that always outputs a constant factor."""
    def schedule(step: jax.Array) -> jax.Array:
        return jnp.full_like(step, value, dtype=jnp.float32)
        
    return schedule


class CurriculumWrapper(wrapper.Wrapper):
    def __init__(self, env, curriculum_fn: CurriculumFn):
        super().__init__(env)
        self.curriculum_fn = curriculum_fn

    def reset(self, rng):
        state = self.env.reset(rng)
        new_info = state.info.copy()

        global_step = jnp.zeros_like(state.reward, dtype=jnp.int32)
        new_info['global_step'] = global_step
        new_info['curriculum_fn_result'] = self.curriculum_fn(global_step)
        
        return state.replace(info=new_info)

    def step(self, state, action):
        current_step = state.info['global_step']
        state = self.env.step(state, action)
        
        new_step = current_step + 1        
        new_info = state.info.copy()
        new_info['global_step'] = new_step
        new_info['curriculum_fn_result'] = self.curriculum_fn(new_step)
        
        return state.replace(info=new_info)
