"""
    Unitree Go2 Domain Randomization:
"""
import jax
from mujoco import mjx

from training.module_types import PRNGKey


def domain_randomize(
    mjx_model: mjx.Model, rng: PRNGKey,
) -> tuple[mjx.Model, mjx.Model]:
    @jax.vmap
    def randomize_parameters(rng):
        # Body IDs:
        FLOOR_BODY_ID = 0
        TORSO_BODY_ID = 1

        # Floor Friction:
        rng, key = jax.random.split(rng)
        geom_friction = jax.random.uniform(key, minval=0.6, maxval=1.2)
        friction = mjx_model.geom_friction.at[FLOOR_BODY_ID, 0].set(
            geom_friction,
        )

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
            key, minval=-1.0, maxval=1.0,
        )
        body_mass = mjx_model.body_mass.at[TORSO_BODY_ID].set(
            mjx_model.body_mass[TORSO_BODY_ID] + delta
        )

        return (
            friction,
            body_ipos,
            body_mass,
        )

    (
        friction,
        body_ipos,
        body_mass,
    ) = randomize_parameters(rng)

    in_axes = jax.tree.map(lambda x: None, mjx_model)
    in_axes = in_axes.tree_replace({
        'geom_friction': 0,
        'body_ipos': 0,
        'body_mass': 0,
    })

    mjx_model = mjx_model.tree_replace({
        'geom_friction': friction,
        'body_ipos': body_ipos,
        'body_mass': body_mass,
    })  # type: ignore

    return mjx_model, in_axes
