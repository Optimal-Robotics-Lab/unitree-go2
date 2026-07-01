"""Human-readable, physical-space logging for regressed parameters.

The optimizer works in theta space (log-Cholesky base parameters as unbounded
affine offsets). For inspection we decode back to physical quantities -- mass,
center of mass, and the body-frame inertia about the CoM -- so logs are directly
comparable to CAD / datasheet values instead of opaque theta coordinates.
"""

from typing import Dict

import jax
import numpy as np

from regression.utilities.model_utilities import (
    theta_to_pi,
    get_icom_from_pi,
    log_cholesky_conditioning,
)

# Unique entries of the symmetric CoM-frame inertia, in body-frame axes.
_INERTIA_KEYS = ("Ixx", "Iyy", "Izz", "Ixy", "Ixz", "Iyz")


def decode_inertia(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode log-Cholesky theta to physical inertia.

    Args:
        theta: ``(n_bodies, 10)`` log-Cholesky base parameters.

    Returns:
        ``(mass (n,), com (n, 3), inertia_com (n, 3, 3))`` -- mass, center of
        mass, and the body-frame inertia tensor about the CoM. Uses the smooth
        ``theta -> pi -> I_com`` path (no eigendecomposition), so the entries are
        frame-consistent and directly comparable across epochs.
    """
    mass, com, inertia = jax.vmap(lambda th: get_icom_from_pi(theta_to_pi(th)))(theta)
    return np.asarray(mass), np.asarray(com), np.asarray(inertia)


def build_param_log(
    physical_params: Dict[str, np.ndarray],
    regression_spec: dict,
    joint_names: list[str],
) -> Dict[str, float]:
    """Flat, physical-space log dict for wandb (keys grouped by ``/``).

    Inertia parameters are decoded to mass / CoM / inertia; dof parameters are
    already physical and logged per joint. The eigh conditioning metric
    (``min_gap``) is included per body.
    """
    log: Dict[str, float] = {}
    for name, value in physical_params.items():
        spec = regression_spec[name]

        if spec["field"] == "log_cholesky_inertia":
            body_names = spec["body_names"]
            mass, com, inertia = decode_inertia(value)
            gaps = np.asarray(
                log_cholesky_conditioning(value)["min_principal_gap"]
            )
            for i, body in enumerate(body_names):
                base = f"params/inertia/{body}"
                log[f"{base}/mass"] = float(mass[i])
                log[f"{base}/com_x"] = float(com[i, 0])
                log[f"{base}/com_y"] = float(com[i, 1])
                log[f"{base}/com_z"] = float(com[i, 2])
                m = inertia[i]
                entries = (m[0, 0], m[1, 1], m[2, 2], m[0, 1], m[0, 2], m[1, 2])
                for key, val in zip(_INERTIA_KEYS, entries):
                    log[f"{base}/{key}"] = float(val)
                log[f"diagnostics/inertia/{body}/min_gap"] = float(gaps[i])
        else:
            v = np.asarray(value)
            for i, joint in enumerate(joint_names):
                if i < v.shape[0]:
                    log[f"params/{name}/{joint}"] = float(v[i])

    return log


def format_param_summary(
    physical_params: Dict[str, np.ndarray],
    nominal_params: Dict[str, np.ndarray],
    regression_spec: dict,
    joint_names: list[str],
) -> str:
    """Compact console table of physical values vs. their initial (nominal)."""
    lines: list[str] = []

    # Inertia: one row per body (mass, CoM in mm, diagonal inertia).
    for name, value in physical_params.items():
        if regression_spec[name]["field"] != "log_cholesky_inertia":
            continue
        body_names = regression_spec[name]["body_names"]
        mass, com, inertia = decode_inertia(value)
        m0, _, _ = decode_inertia(np.asarray(nominal_params[name]))
        lines.append("Inertia [mass kg (init) | CoM mm | Ixx Iyy Izz kg·m²]:")
        for i, body in enumerate(body_names):
            diag = (inertia[i, 0, 0], inertia[i, 1, 1], inertia[i, 2, 2])
            lines.append(
                f"  {body:<18} {mass[i]:6.3f} ({m0[i]:5.3f})  "
                f"({com[i,0]*1e3:6.1f},{com[i,1]*1e3:6.1f},{com[i,2]*1e3:6.1f})  "
                f"({diag[0]:.2e} {diag[1]:.2e} {diag[2]:.2e})"
            )

    # DoF: one row per joint, all dof params (physical | init) side by side.
    dof_fields = {
        name: name.replace("dof_", "")
        for name in physical_params
        if regression_spec[name]["field"].startswith("dof_")
    }
    if dof_fields:
        header = "  " + f"{'joint':<14}" + "".join(f"{s:>22}" for s in dof_fields.values())
        lines.append("DoF [physical (init)]:")
        lines.append(header)
        for i, joint in enumerate(joint_names):
            cells = []
            for name in dof_fields:
                cur = float(np.asarray(physical_params[name])[i])
                init = float(np.asarray(nominal_params[name])[i])
                cells.append(f"{cur:.4g} ({init:.3g})")
            lines.append("  " + f"{joint:<14}" + "".join(f"{c:>22}" for c in cells))

    return "\n".join(lines)
