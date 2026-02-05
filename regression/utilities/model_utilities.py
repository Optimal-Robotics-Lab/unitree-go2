from typing import Dict

import mujoco
from mujoco import mjx


def _check_base_type(model: mujoco.MjModel | mjx.Model) -> tuple[int, int]:
    joint_start = model.body_jntadr[1]
    joint_type = model.jnt_type[joint_start]
    
    if joint_type == mujoco.mjtJoint.mjJNT_FREE:
        return (6, 7)
    else:
        return (0, 0)


def hydrate_model(
    params: dict,
    mj_model: mujoco.MjModel | mjx.Model,
    regression_spec: Dict[str, Dict[str, any]],
) -> mujoco.MjModel | mjx.Model:
    nq_offset, nv_offset = _check_base_type(mj_model)

    for k, v in params.items():
        spec = regression_spec[k]
        field = spec['field']

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
