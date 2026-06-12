from typing import Any, Dict

import jax
import jax.numpy as jnp

import numpy as np

import mujoco
from mujoco import mjx

from regression.utilities.math import matrix_to_quaternion


def _check_base_type(model: mujoco.MjModel | mjx.Model) -> tuple[int, int]:
    joint_start = model.body_jntadr[1]
    joint_type = model.jnt_type[joint_start]

    if joint_type == mujoco.mjtJoint.mjJNT_FREE:
        return (6, 7)
    else:
        return (0, 0)


def hydrate_model(
    params: dict,
    mj_model: mujoco.MjModel,
    regression_spec: Dict[str, Dict[str, Any]],
) -> mujoco.MjModel:
    nq_offset, nv_offset = _check_base_type(mj_model)

    for k, v in params.items():
        spec = regression_spec[k]
        field = spec['field']
        
        if field == 'log_cholesky_inertia':
            body_ids = spec['body_ids']
            b_mass, b_ipos, b_inertia, b_iquat = jax.device_get(
                jax.vmap(log_cholesky_to_mujoco)(v)
            )
            mj_model.body_mass[body_ids] = b_mass
            mj_model.body_ipos[body_ids] = b_ipos
            mj_model.body_inertia[body_ids] = b_inertia
            mj_model.body_iquat[body_ids] = b_iquat
            continue

        value = getattr(mj_model, field, None)
        if value is None:
            continue

        if 'column' in spec:
            col_idx = spec['column']
            value[:, col_idx] = v
        else:
            value = v

        match k:
            case 'actuator_dynprm':
                if 'column' in spec:
                    col_idx = spec['column']
                    value[:, col_idx] = v
                elif 'row' in spec:
                    row_idx = spec['row']
                    value[row_idx, :] = v
                else:
                    value = v
            case k if any(x in k for x in ['dof_frictionloss', 'dof_damping', 'dof_armature']):
                value[nv_offset:] = v
            case 'qpos0':
                value[nq_offset:] = v
            case _:
                raise ValueError(f"Unknown parameter key: {k}")

        setattr(mj_model, k, value)

    return mj_model


def get_nominal_inertia_parameters(mj_model: mujoco.MjModel, body_id: int) -> np.ndarray:
    """Extracts the 10-D nominal Log-Cholesky base parameters from the default MuJoCo model."""
    mass = mj_model.body_mass[body_id]
    ipos = mj_model.body_ipos[body_id]
    inertia = mj_model.body_inertia[body_id]
    iquat = mj_model.body_iquat[body_id]

    xmat = np.empty(9)
    mujoco.mju_quat2Mat(xmat, iquat)
    R = xmat.reshape(3, 3)
    I_com = R @ np.diag(inertia) @ R.T

    skew_ipos = np.array([
        [0, -ipos[2], ipos[1]], 
        [ipos[2], 0, -ipos[0]], 
        [-ipos[1], ipos[0], 0]
    ])
    I_origin = I_com - (mass * skew_ipos @ skew_ipos)

    # Construct the 4x4 Pseudo-Inertia Matrix
    Sigma = 0.5 * np.trace(I_origin) * np.eye(3) - I_origin
    h = mass * ipos
    
    J = np.zeros((4, 4))
    J[:3, :3] = Sigma
    J[:3, 3] = h
    J[3, :3] = h
    J[3, 3] = mass

    indices = np.arange(3, -1, -1)
    J_reversed = J[indices][:, indices]
    L_prime = np.linalg.cholesky(J_reversed)
    U = L_prime[indices][:, indices]

    exp_alpha = U[3, 3]
    alpha = np.log(exp_alpha)
    d1 = np.log(U[0, 0] / exp_alpha)
    d2 = np.log(U[1, 1] / exp_alpha)
    d3 = np.log(U[2, 2] / exp_alpha)
    
    s12 = U[0, 1] / exp_alpha
    s13 = U[0, 2] / exp_alpha
    s23 = U[1, 2] / exp_alpha
    
    t1 = U[0, 3] / exp_alpha
    t2 = U[1, 3] / exp_alpha
    t3 = U[2, 3] / exp_alpha

    return np.array([alpha, d1, d2, d3, s12, s23, s13, t1, t2, t3])


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
