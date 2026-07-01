from typing import Tuple

from ml_collections import ConfigDict

import jax.numpy as jnp

import mujoco

from regression.utilities import model_utilities
from regression.utilities import transforms


def log_cholesky_affine_scale(
    mass_cv: float = 0.2,
    inertia_cv: float = 0.3,
    shear_std: float = 0.1,
    com_std: float = 0.01,
) -> tuple:
    """Principled ``affine`` scale for the log-Cholesky theta, from physical priors.

    Under the ``affine`` transform ``physical = nominal + theta * scale`` with an
    L2 term on ``theta``, ``scale`` is the per-component 1-sigma prior width. Each
    entry is derived from a stated physical uncertainty via the correct functional
    relationship rather than guessed, and returned in the theta layout
    ``[alpha, d1, d2, d3, s12, s23, s13, t1, t2, t3]``:

      alpha : mass ~ exp(2*alpha)          -> scale = mass_cv / 2
      d1-d3 : principal 2nd moment ~ exp(2*d) -> scale = inertia_cv / 2
      s     : dimensionless shear (linear)  -> scale = shear_std (direct)
      t     : center of mass, meters        -> scale = com_std (absolute length)

    Args:
        mass_cv: relative (1-sigma) mass uncertainty, e.g. 0.2 for +/-20%.
        inertia_cv: relative (1-sigma) uncertainty in the principal moments.
        shear_std: prior std on the dimensionless shear terms (products of inertia
            relative to the principal moments).
        com_std: center-of-mass position std in meters.

    Note: because ``affine`` is unbounded, these are soft prior widths, not limits
    -- the optimizer can still reach a value many sigma from nominal if the data
    demands it.
    """
    scale_alpha = mass_cv / 2.0
    scale_d = inertia_cv / 2.0
    return (
        scale_alpha,
        scale_d, scale_d, scale_d,
        shear_std, shear_std, shear_std,
        com_std, com_std, com_std,
    )


def get_default_config():
    config = ConfigDict()

    # File Paths
    config.scene_file = 'regression/mjcf/scene_mjx_transparent.xml'
    config.datasets = ('regression/data/chirp-trajectories',)
    config.evaluation_dataset = 'regression/data/chirp-trajectories'

    # Physics Settings
    config.physics = ConfigDict()
    config.physics.timestep = 0.004
    config.physics.control_rate = 0.02
    config.physics.solver = 'newton'    # 'newton', 'cg', 'pgs'
    config.physics.iterations = 5
    config.physics.ls_iterations = 5
    config.physics.use_reverse_mode = False

    # Training Hyperparameters
    config.training = ConfigDict()
    config.training.seed = 0
    config.training.num_epochs = 20
    config.training.batches_per_epoch = 256
    config.training.minibatch_size = 25
    config.training.window_length = 25

    # Optimizer Settings
    config.optimizer = ConfigDict()
    config.optimizer.lr_init = 1e-5
    config.optimizer.lr_peak = 1e-2
    config.optimizer.lr_end = 1e-6
    config.optimizer.warmup_pct = 0.1
    config.optimizer.weight_decay = 1e-4
    config.optimizer.clip_norm = 1.0

    # Loss Function Weights
    config.loss = ConfigDict()
    config.loss.weights = ConfigDict({
        'position': 1.0,
        'velocity': 1.0,
        'actuator_force': 1.0,
    })
    config.loss.regularization_weights = ConfigDict({
        # 'dof_frictionloss': 0.0,
        # 'dof_damping': 0.0,
        # 'dof_armature': 0.0,
        'log_cholesky_inertia': 1e-3,
    })

    # Loss Function Type: rmse, mse, mae, huber
    config.loss.type = 'mse'

    # Parameters to Regress:
    config.regression = ConfigDict()

    config.regression.dof_frictionloss = ConfigDict({
        'field': 'dof_frictionloss', 'transform': 'log_exp', 'reference': 0.1,
    })
    config.regression.dof_damping = ConfigDict({
        'field': 'dof_damping', 'transform': 'log_exp', 'reference': 0.1,
    })
    config.regression.dof_armature = ConfigDict({
        'field': 'dof_armature', 'transform': 'log_exp', 'reference': 0.01,
    })
    # Unbounded affine keeps the log-Cholesky search unconstrained; 'scale' is a
    # physical-prior width (see log_cholesky_affine_scale), not a box.
    config.regression.log_cholesky_inertia = ConfigDict({
        'field': 'log_cholesky_inertia',
        'transform': 'affine',
        'body_names': [
            'front_right_hip', 'front_right_thigh', 'front_right_calf',
            'front_left_hip', 'front_left_thigh', 'front_left_calf',
            'hind_right_hip', 'hind_right_thigh', 'hind_right_calf',
            'hind_left_hip', 'hind_left_thigh', 'hind_left_calf',
        ],
        'scale': log_cholesky_affine_scale(
            mass_cv=0.2, inertia_cv=0.3, shear_std=0.1, com_std=0.01,
        ),
    })

    # WandB
    config.wandb = ConfigDict()
    config.wandb.project = "Parameter-Regression-Unitree-Go2"
    config.wandb.group = None

    return config


def build_parameter_scale(regression_spec: dict) -> dict:
    """Per-parameter transform scale in theta space.

    Returns the ``scale`` each scale-using transform consumes (e.g. the
    affine_tanh deviation half-width), or ``None`` for scale-free transforms
    (e.g. log_exp). Scales are absolute in theta space and independent of the
    nominal, so no nominal is needed here.
    """
    scales: dict[str, jnp.ndarray | None] = {}
    for name, spec in regression_spec.items():
        transforms.validate_transform_spec(name, spec)
        scale = spec.get('scale')
        if scale is None:
            scales[name] = None
            continue
        scale = jnp.asarray(scale, dtype=jnp.float32)
        if not bool(jnp.all(jnp.isfinite(scale) & (scale > 0.0))):
            raise ValueError(
                f"Parameter '{name}': 'scale' must be finite and strictly positive, "
                f"got {spec['scale']}."
            )
        scales[name] = scale
    return scales


def process_regression_spec(mj_model: mujoco.MjModel, regression_spec: ConfigDict | dict) -> Tuple[dict, dict]:
    """
        Process the regression spec to get the initial parameter values from the Mujoco model.
    """
    params: dict[str, jnp.ndarray] = {}
    regression_dict = regression_spec.to_dict() if isinstance(regression_spec, ConfigDict) else regression_spec

    for name, spec in regression_dict.items():
        transform = transforms.validate_transform_spec(name, spec)

        if spec['field'] == 'log_cholesky_inertia':
            body_ids: list[int] = []
            thetas: list[jnp.ndarray] = []
            for b_name in spec['body_names']:
                b_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, b_name)
                if b_id == -1:
                    raise ValueError(f"Body '{b_name}' not found in model.")

                body_ids.append(b_id)
                thetas.append(model_utilities.get_nominal_theta(mj_model, b_id))

            params[name] = jnp.array(thetas)
            spec['body_ids'] = jnp.array(body_ids, dtype=jnp.int32)
        else:
            model_val: jnp.ndarray = getattr(mj_model, spec['field'])
            if 'column' in spec:
                model_val = model_val[:, spec['column']]
            if 'reference' in spec:
                val = jnp.full(model_val.shape, spec['reference'], dtype=jnp.float32)
            else:
                val = jnp.asarray(model_val)
            params[name] = val

        # Multiplicative transforms need a strictly-positive anchor
        if transform.positive_anchor and not bool(jnp.all(params[name] > 0.0)):
            raise ValueError(
                f"Parameter '{name}' uses transform '{spec.get('transform')}', which "
                f"requires a strictly-positive anchor, but its nominal/reference has "
                f"non-positive entries. Provide a positive 'reference'."
            )

    return params, regression_dict
