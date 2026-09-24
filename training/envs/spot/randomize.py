"""
    Spot Domain Randomization:
"""
import jax
from mujoco import mjx

from training.module_types import PRNGKey


def domain_randomize(
    mjx_model: mjx.Model, rng: PRNGKey,
) -> tuple[mjx.Model, mjx.Model]:
    @jax.vmap
    def randomize_parameters(rng):
        # IDs:
        FLOOR_GEOM_ID = 0
        TORSO_BODY_ID = 1

        # Number of joints (excluding floating base):
        num_joints = (mjx_model.nv - 6)

        # Floor Friction:
        rng, key = jax.random.split(rng)
        geom_friction = jax.random.uniform(key, minval=0.3, maxval=1.0)
        friction = mjx_model.geom_friction.at[FLOOR_GEOM_ID, 0].set(
            geom_friction,
        )

        # Joint Friction:
        rng, key = jax.random.split(rng)
        frictionloss = mjx_model.dof_frictionloss[6:] * jax.random.uniform(
            key, shape=(num_joints,), minval=0.9, maxval=1.1,
        )
        dof_frictionloss = mjx_model.dof_frictionloss.at[6:].set(frictionloss)

        # Damping:
        rng, key = jax.random.split(rng)
        damping = mjx_model.dof_damping[6:] * jax.random.uniform(
            key, shape=(num_joints,), minval=0.9, maxval=1.1,
        )
        dof_damping = mjx_model.dof_damping.at[6:].set(damping)

        # Armature:
        rng, key = jax.random.split(rng)
        armature = mjx_model.dof_armature[6:] * jax.random.uniform(
            key, shape=(num_joints,), minval=0.9, maxval=1.1,
        )
        dof_armature = mjx_model.dof_armature.at[6:].set(armature)

        # Center of Mass offset:
        rng, key = jax.random.split(rng)
        inertia_offset = jax.random.uniform(
            key, (3,), minval=-0.05, maxval=0.05,
        )
        body_ipos = mjx_model.body_ipos.at[TORSO_BODY_ID].set(
            mjx_model.body_ipos[TORSO_BODY_ID] + inertia_offset,
        )

        # Link mass randomization:
        rng, key = jax.random.split(rng)
        delta = jax.random.uniform(
            key, (mjx_model.nbody,), minval=0.9, maxval=1.1,
        )
        body_mass = mjx_model.body_mass.at[:].set(mjx_model.body_mass * delta)

        # Torso mass randomization:
        rng, key = jax.random.split(rng)
        delta = jax.random.uniform(
            key, minval=-2.5, maxval=2.5,
        )
        body_mass = body_mass.at[TORSO_BODY_ID].set(
            body_mass[TORSO_BODY_ID] + delta
        )

        # Joint reference randomization:
        rng, key = jax.random.split(rng)
        qpos0 = mjx_model.qpos0
        delta = jax.random.uniform(
            key, shape=(num_joints,), minval=-0.05, maxval=0.05,
        )
        qpos0 = qpos0.at[7:].set(qpos0[7:] + delta)

        return (
            friction,
            dof_frictionloss,
            dof_damping,
            dof_armature,
            body_ipos,
            body_mass,
            qpos0,
        )

    (
        friction,
        dof_frictionloss,
        dof_damping,
        dof_armature,
        body_ipos,
        body_mass,
        qpos0,
    ) = randomize_parameters(rng)

    in_axes = jax.tree.map(lambda x: None, mjx_model)
    in_axes = in_axes.tree_replace({
        'geom_friction': 0,
        'dof_frictionloss': 0,
        'dof_damping': 0,
        'dof_armature': 0,
        'body_ipos': 0,
        'body_mass': 0,
        'qpos0': 0,
    })

    mjx_model = mjx_model.tree_replace({
        'geom_friction': friction,
        'dof_frictionloss': dof_frictionloss,
        'dof_damping': dof_damping,
        'dof_armature': dof_armature,
        'body_ipos': body_ipos,
        'body_mass': body_mass,
        'qpos0': qpos0,
    })  # type: ignore

    return mjx_model, in_axes


def sample_actuation_delay(rng: PRNGKey, num_channels: int) -> jax.Array:
    """Samples independent per-channel actuation delay [s].

    Deliberately not part of `domain_randomize`: delay isn't an `mjx.Model`
    field, and `domain_randomize` must return exactly `(mjx.Model,
    mjx.Model)` to work as `training.algorithms.ppo.train`'s
    `randomization_fn` (see its call sites, e.g. train.py). Called from
    `SpotJoystickEnv.reset()` instead, using the same per-env rng that
    already randomizes `command`/`disturbance_magnitude` there.

    Args:
        rng: PRNGKey for this env's sample.
        num_channels: Number of independently delayed actuators.

    Returns:
        Delay [s], shape `(num_channels,)`, uniform in [0, 4ms].
    """
    return jax.random.uniform(rng, shape=(num_channels,), minval=0.0, maxval=0.004)
