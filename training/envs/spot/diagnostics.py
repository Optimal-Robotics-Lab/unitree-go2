"""Weight-independent evaluation scorecard for Spot.

`SpotJoystickEnv` emits raw per-step `diag_*` metrics. The evaluator sums
them over each episode, so `compute_scorecard` divides by the matching step
counts to get physical quantities. Nothing here depends on reward weights,
which is what lets a reward tuner be judged by something it cannot game by
zeroing a penalty.
"""

import math

_GRAVITY = 9.81

DIAGNOSTIC_NAMES = (
    'diag_moving_steps',
    'diag_still_steps',
    'diag_linear_velocity_error_sq',
    'diag_yaw_rate_error_sq',
    'diag_body_speed',
    'diag_foot_clearance_sum',
    'diag_touchdown_count',
    'diag_foot_slip_speed_sum',
    'diag_foot_contact_count',
    'diag_flight_steps',
    'diag_synchronized_touchdown_steps',
    'diag_diagonal_agreement_steps',
    'diag_moving_contact_count',
    'diag_two_contact_steps',
    'diag_two_contact_diagonal_steps',
    'diag_two_contact_same_side_steps',
    'diag_two_contact_axle_steps',
    'diag_moving_action_delta_sq',
    'diag_still_action_delta_sq',
    'diag_still_joint_velocity_sq',
    'diag_still_body_speed',
    'diag_mechanical_power',
    'diag_action_saturation',
    'diag_torque_saturation',
    'diag_unwanted_contacts',
    'diag_tilt',
    'diag_terminated',
)


def _ratio(numerator: float, denominator: float) -> float | None:
    """Returns numerator / denominator, or None if the denominator is ~0."""
    if denominator < 1e-9:
        return None
    return numerator / denominator


def _root(value: float | None) -> float | None:
    return None if value is None else math.sqrt(value)


def compute_scorecard(
    metrics: dict[str, float], robot_mass: float, control_dt: float,
) -> dict[str, float | None]:
    """Converts evaluator metrics into physical, weight-independent scores.

    Args:
        metrics: Evaluator output containing `eval/episode_diag_*` (episode
            sums, averaged over evaluation envs) and `eval/avg_episode_length`.
        robot_mass: Total robot mass [kg], for cost of transport.
        control_dt: Control period [s], for step frequency.

    Returns:
        Scorecard values keyed by name; None where the quantity is undefined
        (e.g. no stand-still steps occurred). Units are in each key's suffix.
    """
    def total(name: str) -> float:
        return float(metrics[f'eval/episode_{name}'])

    steps = float(metrics['eval/avg_episode_length'])
    moving = total('diag_moving_steps')
    still = total('diag_still_steps')
    mean_speed = _ratio(total('diag_body_speed'), moving)
    mean_power = _ratio(total('diag_mechanical_power'), steps)

    # Touchdowns per foot per second while moving. Contact chatter counts as
    # extra touchdowns, so this is an upper bound on the true step rate.
    step_frequency = _ratio(
        total('diag_touchdown_count'), 4.0 * moving * control_dt,
    )
    stride_length = None
    if mean_speed is not None and step_frequency:
        stride_length = mean_speed / step_frequency
    two_contact = total('diag_two_contact_steps')

    cost_of_transport = None
    if mean_speed is not None and mean_power is not None:
        cost_of_transport = _ratio(
            mean_power, robot_mass * _GRAVITY * mean_speed,
        )

    return {
        'score/episode_length_steps': steps,
        'score/linear_velocity_rmse_mps': _root(
            _ratio(total('diag_linear_velocity_error_sq'), moving),
        ),
        'score/yaw_rate_rmse_radps': _root(
            _ratio(total('diag_yaw_rate_error_sq'), moving),
        ),
        'score/foot_clearance_m': _ratio(
            total('diag_foot_clearance_sum'), total('diag_touchdown_count'),
        ),
        'score/foot_slip_mps': _ratio(
            total('diag_foot_slip_speed_sum'),
            total('diag_foot_contact_count'),
        ),
        'score/flight_fraction': _ratio(total('diag_flight_steps'), moving),
        'score/synchronized_touchdown_fraction': _ratio(
            total('diag_synchronized_touchdown_steps'), moving,
        ),
        'score/diagonal_agreement_fraction': _ratio(
            total('diag_diagonal_agreement_steps'), moving,
        ),
        'score/mean_body_speed_mps': mean_speed,
        'score/step_frequency_hz': step_frequency,
        'score/stride_length_m': stride_length,
        'score/two_contact_fraction': _ratio(two_contact, moving),
        'score/two_contact_diagonal_fraction': _ratio(
            total('diag_two_contact_diagonal_steps'), two_contact,
        ),
        'score/two_contact_same_side_fraction': _ratio(
            total('diag_two_contact_same_side_steps'), two_contact,
        ),
        'score/two_contact_axle_fraction': _ratio(
            total('diag_two_contact_axle_steps'), two_contact,
        ),
        'score/mean_feet_in_contact': _ratio(
            total('diag_moving_contact_count'), moving,
        ),
        'score/moving_action_delta_rms': _root(
            _ratio(total('diag_moving_action_delta_sq'), moving),
        ),
        'score/still_action_delta_rms': _root(
            _ratio(total('diag_still_action_delta_sq'), still),
        ),
        'score/still_joint_velocity_rms_radps': _root(
            _ratio(total('diag_still_joint_velocity_sq'), still),
        ),
        'score/still_body_drift_mps': _ratio(
            total('diag_still_body_speed'), still,
        ),
        'score/cost_of_transport': cost_of_transport,
        'score/action_saturation_fraction': _ratio(
            total('diag_action_saturation'), steps,
        ),
        'score/torque_saturation_fraction': _ratio(
            total('diag_torque_saturation'), steps,
        ),
        'score/unwanted_contacts_per_step': _ratio(
            total('diag_unwanted_contacts'), steps,
        ),
        'score/tilt': _ratio(total('diag_tilt'), steps),
        'score/termination_rate_per_step': _ratio(
            total('diag_terminated'), steps,
        ),
    }
