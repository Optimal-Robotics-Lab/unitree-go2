import pathlib
import yaml

from absl import app

import numpy as np
import scipy


# Verify this calculation:
def diebold_mariano_test(error_x, error_y, h=1, metric='MSE', harvey_correction=True):
    """
    Robust Diebold-Mariano test with Harvey Correction and Bartlett weights.

    Parameters:
        real (array): The actual values.
        pred1 (array): Predictions from model 1.
        pred2 (array): Predictions from model 2.
        h (int): Forecast horizon (h=1 for standard regression).
        metric (str): 'MSE' or 'MAE'.
        harvey_correction (bool): If True, adjusts for small sample sizes.

    Returns:
        dm_stat (float): The test statistic.
        p_value (float): The p-value (Two-sided).
    """

    # Compute the errors and the loss differential (d)
    e1 = error_x
    e2 = error_y

    if metric == 'MSE':
        d = e1**2 - e2**2
    elif metric == 'MAE':
        d = np.abs(e1) - np.abs(e2)

    T = len(d)
    d_mean = np.mean(d)

    # Compute the long-run variance (using Bartlett kernel / Newey-West)
    gamma0 = np.var(d, ddof=0)

    sum_gamma = 0
    for k in range(1, h):
        gamma_k = np.cov(d[k:], d[:-k])[0, 1] if k < T else 0
        weight = 1 - (k / h)
        sum_gamma += weight * gamma_k

    # Long-run variance estimate
    V_d = gamma0 + 2 * sum_gamma

    # Avoid division by zero
    if V_d <= 1e-12:
        return 0.0, 1.0

    # Compute DM statistic (standard)
    dm_stat = d_mean / np.sqrt(V_d / T)

    # Apply Harvey Correction:
    if harvey_correction:
        harvey_adj = np.sqrt((T + 1 - 2*h + h*(h-1)/T) / T)
        dm_stat = harvey_adj * dm_stat

    # Compute P-Value using Student's t-distribution
    # Degrees of freedom = T - 1
    p_value = 2 * (1 - scipy.stats.t.cdf(np.abs(dm_stat), df=T-1))

    return dm_stat, p_value


def main(argv=None):
    # Load yaml:
    data_directory = pathlib.Path(__file__).parent.parent
    directories = [
        data_directory / f"processed/standard-replay-position",
        data_directory / f"processed/transparent-replay-position",
    ]

    stats = {}
    for directory in directories:
        prefix = directory.name.split('-')[0]
        with open(directory / "stats.yaml", 'r') as f:
            stats[prefix] = yaml.safe_load(f)

    name_to_index = {
        'front_right_hip': 0,
        'front_right_thigh': 1,
        'front_right_calf': 2,
        'front_left_hip': 3,
        'front_left_thigh': 4,
        'front_left_calf': 5,
        'hind_right_hip': 6,
        'hind_right_thigh': 7,
        'hind_right_calf': 8,
        'hind_left_hip': 9,
        'hind_left_thigh': 10,
        'hind_left_calf': 11,
    }

    stats_results = {}
    for name in name_to_index.keys():
        stats_results[name] = {}
        for error_type in ['position_error', 'velocity_error', 'torque_error']:
            if error_type == 'position_error':
                loss_differentials = np.array(stats['standard'][error_type])[:, 7:] - np.array(stats['transparent'][error_type])[:, 7:]
            elif error_type == 'velocity_error':
                loss_differentials = np.array(stats['standard'][error_type])[:, 6:] - np.array(stats['transparent'][error_type])[:, 6:]
            else:
                loss_differentials = np.array(stats['standard'][error_type]) - np.array(stats['transparent'][error_type])

            loss_differentials = loss_differentials[:, name_to_index[name]]

            standard_error = np.array(stats['standard'][error_type])[:, name_to_index[name]]
            transparent_error = np.array(stats['transparent'][error_type])[:, name_to_index[name]]
            standard_error_abs = np.abs(standard_error)
            transparent_error_abs = np.abs(transparent_error)

            # T-Test:
            # t_stat, p_value = scipy.stats.ttest_1samp(loss_differentials, popmean=0.0)
            t_stat, p_value = scipy.stats.ttest_ind(
                standard_error_abs,
                transparent_error_abs,
                equal_var=False,
                alternative='less'
            )

            # Diebold-Mariano test:
            dm_stat, dm_p_value = diebold_mariano_test(
                standard_error,
                transparent_error,
                h=1,
                metric='MSE',
            )

            # Mann-Whitney U Test:
            u_stat, u_p_value = scipy.stats.mannwhitneyu(
                standard_error_abs,
                transparent_error_abs,
                alternative='less',
            )

            print(f"Joint: {name}")
            print(f"Error Type: {error_type}")
            print(f"  T-Test: statistic={t_stat}, p-value={p_value}")
            print(f"  Diebold-Mariano Test: statistic={dm_stat}, p-value={dm_p_value}")
            print(f"  Mann-Whitney U Test: statistic={u_stat}, p-value={u_p_value}")
            print("")

            stats_results[name][error_type] = {
                'ttest_statistic': float(t_stat),
                'ttest_p_value': float(p_value),
                'diebold_mariano_statistic': float(dm_stat),
                'diebold_mariano_p_value': float(dm_p_value),
                'mannwhitneyu_statistic': float(u_stat),
                'mannwhitneyu_p_value': float(u_p_value),
            }

    with open(data_directory / "processed" / "comparison_stats.yaml", 'w') as f:
        yaml.dump(stats_results, f)


if __name__ == '__main__':
    app.run(main)
