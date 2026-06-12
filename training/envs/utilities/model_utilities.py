import jax
import jax.numpy as jnp

import numpy as np

import mujoco
from mujoco import mjx

from training.envs.utilities.math import matrix_to_quaternion


def log_cholesky_inertia_matrix(theta: jax.Array) -> jax.Array:
    """
    Convert a vector of 10 parameters into a log-cholesky inertia matrix representation.
    """
    scale = jnp.exp(theta[0])
    
    U11 = jnp.exp(theta[1])
    U22 = jnp.exp(theta[2])
    U33 = jnp.exp(theta[3])
    
    U12 = theta[4]
    U23 = theta[5]
    U13 = theta[6]
    
    U14 = theta[7]
    U24 = theta[8]
    U34 = theta[9]
    U44 = 1.0

    U = scale * jnp.array([
        [U11, U12, U13, U14],
        [0.0, U22, U23, U24],
        [0.0, 0.0, U33, U34],
        [0.0, 0.0, 0.0, U44]
    ])

    return U @ U.T


def log_cholesky_to_mujoco(theta: dict[str, float]) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    Convert a vector of 10 parameters into a log-cholesky inertia matrix representation.

    Input:
        theta: [alpha, d1, d2, d3, s12, s23, s13, t1, t2, t3]

    Output:
        intertial matrix
    
    """

    inertia = log_cholesky_inertia_matrix(theta)

    # Extract mass and center of mass:
    body_mass = inertia[3, 3]
    h = inertia[0:3, 3]
    body_ipos = h / body_mass

    # Extract Sigma and compute inertia com:
    sigma = inertia[0:3, 0:3]
    inertia_origin =  jnp.linalg.trace(sigma) * jnp.eye(3) - sigma
    inertia_com = inertia_origin - body_mass * (jnp.dot(body_ipos, body_ipos) * jnp.eye(3) - jnp.outer(body_ipos, body_ipos))
    
    # Compute body inertia and orientation:
    body_inertia, rotation_matrix = jnp.linalg.eigh(inertia_com + 1e-6 * jnp.eye(3))
    body_iquat = matrix_to_quaternion(rotation_matrix)

    return body_mass, body_ipos, body_inertia, body_iquat


def rehydrate_model(
    model: mujoco.MjModel | mjx.Model,
    parameters: dict[str, jax.Array | np.ndarray],
    regression_spec: dict[str, dict]
) -> mujoco.MjModel | mjx.Model:
    """
        Applies regressed parameters to MjModel.
    """
    is_mjx = isinstance(model, mjx.Model)
    
    replace_kwargs = {}

    joint_start = model.body_jntadr[1]
    joint_type = model.jnt_type[joint_start]
    
    nq_offset = 7 if joint_type == mujoco.mjtJoint.mjJNT_FREE else 0
    nv_offset = 6 if joint_type == mujoco.mjtJoint.mjJNT_FREE else 0

    for k, v in parameters.items():
        if k not in regression_spec:
            continue

        spec = regression_spec[k]
        field = spec['field']

        if field == 'log_cholesky_inertia':
            body_ids = spec['body_ids']
            b_mass, b_ipos, b_inertia, b_iquat = jax.vmap(log_cholesky_to_mujoco)(jnp.asarray(v))

            if is_mjx:
                replace_kwargs['body_mass'] = model.body_mass.at[body_ids].set(b_mass)
                replace_kwargs['body_ipos'] = model.body_ipos.at[body_ids, :].set(b_ipos)
                replace_kwargs['body_inertia'] = model.body_inertia.at[body_ids, :].set(b_inertia)
                replace_kwargs['body_iquat'] = model.body_iquat.at[body_ids, :].set(b_iquat)
            else:
                b_mass, b_ipos, b_inertia, b_iquat = jax.device_get(
                    (b_mass, b_ipos, b_inertia, b_iquat)
                )
                model.body_mass[body_ids] = b_mass
                model.body_ipos[body_ids] = b_ipos
                model.body_inertia[body_ids] = b_inertia
                model.body_iquat[body_ids] = b_iquat
            
            continue

        if is_mjx:
            original_array = replace_kwargs.get(field, getattr(model, field))
            v_jax = jnp.asarray(v)
            
            if 'column' in spec:
                col_idx = spec['column']
                new_array = original_array.at[:, col_idx].set(v_jax)
            elif 'row' in spec:
                row_idx = spec['row']
                new_array = original_array.at[row_idx, :].set(v_jax)
            else:
                if any(x in k for x in ['dof_frictionloss', 'dof_damping', 'dof_armature']):
                    new_array = original_array.at[nv_offset:].set(v_jax)
                elif 'qpos0' in k:
                    new_array = original_array.at[nq_offset:].set(v_jax)
                else:
                    new_array = v_jax
            
            replace_kwargs[field] = new_array

        else:
            original_array = getattr(model, field)
            v_np = np.asarray(jax.device_get(v))
            if 'column' in spec:
                col_idx = spec['column']
                original_array[:, col_idx] = v_np
            elif 'row' in spec:
                row_idx = spec['row']
                original_array[row_idx, :] = v_np
            else:
                if any(x in k for x in ['dof_frictionloss', 'dof_damping', 'dof_armature']):
                    original_array[nv_offset:] = v_np
                elif 'qpos0' in k:
                    original_array[nq_offset:] = v_np
                else:
                    np.copyto(original_array, v_np)

    if is_mjx:
        return model.replace(**replace_kwargs)
    else:
        return model
