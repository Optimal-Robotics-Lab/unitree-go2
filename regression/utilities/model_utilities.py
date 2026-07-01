from typing import Any, Dict

import jax
import jax.numpy as jnp

import numpy as np

import mujoco
from mujoco import mjx

from regression.utilities.math import matrix_to_quaternion
from regression.utilities.diagnostics import grad_probe, principal_moment_gaps


def _check_base_type(model: mujoco.MjModel | mjx.Model) -> tuple[int, int]:
    joint_start = model.body_jntadr[1]
    joint_type = model.jnt_type[joint_start]

    if joint_type == mujoco.mjtJoint.mjJNT_FREE:
        return (6, 7)
    else:
        return (0, 0)


def rehydrate_model(
    params: dict,
    nominal_parameters: Dict[str, jax.Array],
    mj_model: mjx.Model,
    regression_spec: Dict[str, Dict[str, Any]],
) -> mujoco.MjModel:
    # Rehydrate the model with new parameters:
    replace_kwargs = {}
    for name, value in params.items():
        spec = regression_spec[name]
        field = spec['field']
        if field == 'log_cholesky_inertia':
            body_ids = spec['body_ids']
            b_mass, b_ipos, b_inertia, b_iquat = jax.vmap(
                log_cholesky_to_mujoco
            )(value, nominal_parameters['log_cholesky_inertia'])
            replace_kwargs['body_mass'] = mj_model.body_mass.at[body_ids].set(b_mass)
            replace_kwargs['body_ipos'] = mj_model.body_ipos.at[body_ids, :].set(b_ipos)
            replace_kwargs['body_inertia'] = mj_model.body_inertia.at[body_ids, :].set(b_inertia)
            replace_kwargs['body_iquat'] = mj_model.body_iquat.at[body_ids, :].set(b_iquat)

        elif 'column' in spec:
            col_idx = spec['column']
            original_array = getattr(mj_model, field)
            new_array = original_array.at[:, col_idx].set(value)
            replace_kwargs[field] = new_array

        else:
            replace_kwargs[field] = value

    hydrated_model = mj_model.replace(**replace_kwargs)

    return hydrated_model


# TODO: Create unified hydrate function. Should not be needed...
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


def get_nominal_theta(mj_model: mujoco.MjModel, body_id: int) -> np.ndarray:
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


def theta_to_pi(theta: jax.Array) -> jax.Array:
    """
    Convert the 10-D log-Cholesky base parameters theta into the 10-D inertial
    parameter vector pi.

    Input:
        theta: [alpha, d1, d2, d3, s12, s23, s13, t1, t2, t3]

    Output:
        pi: [m, hx, hy, hz, Ixx, Iyy, Izz, Ixy, Iyz, Ixz]

    """
    alpha = theta[0]
    d1 = theta[1]
    d2 = theta[2]
    d3 = theta[3]
    s12 = theta[4]
    s23 = theta[5]
    s13 = theta[6]
    t1 = theta[7]
    t2 = theta[8]
    t3 = theta[9]

    U = jnp.exp(alpha) * jnp.array([
        [jnp.exp(d1), s12,         s13,         t1],
        [0.0,         jnp.exp(d2), s23,         t2],
        [0.0,         0.0,         jnp.exp(d3), t3],
        [0.0,         0.0,         0.0,         1.0],
    ])

    # Pseudo-inertia J = [[Sigma, h], [h.T, m]], Sigma = 0.5*tr(Ibar)*I - Ibar.
    J = U @ U.T
    sigma = J[:3, :3]
    inertia_bar = jnp.trace(sigma) * jnp.eye(3) - sigma
    h = J[:3, 3]
    m = J[3, 3]

    pi = jnp.array([
        m,
        h[0], h[1], h[2],
        inertia_bar[0, 0], inertia_bar[1, 1], inertia_bar[2, 2],
        inertia_bar[0, 1], inertia_bar[1, 2], inertia_bar[0, 2],
    ])

    return pi


def get_icom_from_pi(pi: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
        Extract mass, center of mass (ipos), and the inertia tensor at the center of mass.
    """

    # Extract Mass and Center of Mass
    body_mass = jnp.maximum(pi[0], 1e-6)
    h = pi[1:4]
    body_ipos = h / body_mass

    # Inertia at Origin
    Ixx, Iyy, Izz = pi[4], pi[5], pi[6]
    Ixy, Iyz, Ixz = pi[7], pi[8], pi[9]

    I_origin = jnp.array([
        [Ixx, Ixy, Ixz],
        [Ixy, Iyy, Iyz],
        [Ixz, Iyz, Izz]
    ])

    # Shift to Center of Mass:
    cx, cy, cz = body_ipos[0], body_ipos[1], body_ipos[2]

    # (c.c)*I - outer(c, c)
    c_cross_square = jnp.array([
        [cy**2 + cz**2, -cx*cy, -cx*cz],
        [-cx*cy, cx**2 + cz**2, -cy*cz],
        [-cx*cz, -cy*cz, cx**2 + cy**2]
    ])

    I_com = I_origin - body_mass * c_cross_square

    return body_mass, body_ipos, I_com


def log_cholesky_to_mujoco(
    theta: jax.Array,
    theta_nominal: jax.Array | None = None,
    *,
    debug: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    Convert a vector of 10 parameters into a log-cholesky inertia matrix representation.

    Input:
        theta: [alpha, d1, d2, d3, s12, s23, s13, t1, t2, t3]
        theta_nominal: Nominal parameters for initialization
        debug: If True, insert JIT-compatible diagnostics around the eigh: print
            the minimum principal-moment gap (forward) and the finiteness/norm of
            the cotangent flowing into the eigh (backward). Off by default so the
            production path is unchanged; enable per-call via
            `functools.partial(log_cholesky_to_mujoco, debug=True)`.

    Output:
        body_mass, body_ipos, body_inertia, body_iquat

    """

    # pi: [m, hx, hy, hz, Ixx, Iyy, Izz, Ixy, Iyz, Ixz]
    pi = theta_to_pi(theta)

    # Extract mass and center of mass:
    body_mass, body_ipos, inertia_com = get_icom_from_pi(pi)

    # Nominal Regularization:
    if theta_nominal is not None:
        theta_nominal_frozen = jax.lax.stop_gradient(theta_nominal)
        pi_nominal = theta_to_pi(theta_nominal_frozen)
        _, _, inertia_com_nominal = get_icom_from_pi(pi_nominal)
        regularization = 1e-5 * inertia_com_nominal
    else:
        regularization = jnp.diag(jnp.array([1e-6, 2e-6, 3e-6]))

    eps_safeguard = jnp.diag(jnp.array([1e-9, 2e-9, 3e-9]))

    inertia_matrix = inertia_com + regularization + eps_safeguard

    # Debug:
    if debug:
        jax.debug.print(
            "[eigh] min_principal_gap={g:.3e}",
            g=principal_moment_gaps(inertia_matrix),
        )
        inertia_matrix = grad_probe("eigh_in", inertia_matrix)

    # Compute body inertia and orientation:
    body_inertia, rotation_matrix = jnp.linalg.eigh(inertia_matrix)

    det = jnp.linalg.det(rotation_matrix)
    parity = jax.lax.stop_gradient(jnp.where(det < 0.0, -1.0, 1.0))
    rotation_matrix = rotation_matrix.at[:, 2].multiply(parity)

    body_iquat = matrix_to_quaternion(rotation_matrix)

    return body_mass, body_ipos, body_inertia, body_iquat


def log_cholesky_conditioning(
    theta: jax.Array,
    theta_nominal: jax.Array | None = None,
) -> dict[str, jax.Array]:
    """Forward-only conditioning metrics for a batch of log-Cholesky parameters.

    Intended to be threaded out of a jitted training step as auxiliary data and
    logged (e.g. to wandb) — this is the clean, no-print monitoring path. The
    returned `min_principal_gap` is the per-body distance to inertial degeneracy;
    watch for it shrinking alongside a rising inertia gradient norm.

    Args:
        theta: `(n_bodies, 10)` batch of base parameters.
        theta_nominal: Optional `(n_bodies, 10)` nominal batch, so the reported
            gaps include the same regularization used by `log_cholesky_to_mujoco`.

    Returns:
        Dict with `min_principal_gap` `(n_bodies,)` and its `worst` scalar.
    """
    def _inertia(th, th_nom):
        _, _, inertia_com = get_icom_from_pi(theta_to_pi(th))
        if th_nom is not None:
            _, _, inertia_com_nominal = get_icom_from_pi(
                theta_to_pi(jax.lax.stop_gradient(th_nom))
            )
            inertia_com = inertia_com + 1e-5 * inertia_com_nominal
        return inertia_com + jnp.diag(jnp.array([1e-9, 2e-9, 3e-9]))

    if theta_nominal is None:
        inertia = jax.vmap(lambda th: _inertia(th, None))(theta)
    else:
        inertia = jax.vmap(_inertia)(theta, theta_nominal)

    gaps = principal_moment_gaps(inertia)
    return {"min_principal_gap": gaps, "worst": jnp.min(gaps)}
