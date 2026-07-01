from typing import Dict, Any, List
from collections import defaultdict
from ml_collections import ConfigDict

import jax
import jax.numpy as jnp
import numpy as np

from mujoco import mjx

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import wandb

from regression.utilities.constants import JOINT_NAMES
from regression.utilities.factories import get_objective_fn
from regression.utilities.mjx_utilities import init_function, step_function
from regression.utilities.model_utilities import log_cholesky_to_mujoco
from regression.utilities import resampling
from regression.utilities.config import validate_rates


def _group_joints_by_leg(joint_names: List[str]) -> Dict[str, List[int]]:
    """
        Helper to group joints into legs based on naming prefixes (e.g., FL_hip -> FL).
    """
    groups = defaultdict(list)
    for i, name in enumerate(joint_names):
        prefix = name.split('_')[0] 
        groups[prefix].append(i)
    return dict(groups)


def create_interactive_plot(
    t: np.ndarray,
    joint_names: List[str],
    targets: Dict[str, np.ndarray],
    baseline: Dict[str, np.ndarray],
    optimized: Dict[str, np.ndarray],
    trial_idx: int
) -> go.Figure:
    """
        Generates a Plotly figure with Tabs for each leg.
        
        Layout Structure:
        - Columns: Joints (Hip, Thigh, Calf)
        - Rows: Metrics (Position, Velocity, Torque)
    """
    leg_groups = _group_joints_by_leg(joint_names)
    leg_names = list(leg_groups.keys())
    
    # Grid: 3 Rows (Metrics) x 3 Cols (Joints)
    rows = 3
    cols = 3
    
    # Titles for the Columns (First Row Only)
    column_titles = ["Hip", "Thigh", "Calf"]
    
    fig = make_subplots(
        rows=rows, 
        cols=cols, 
        shared_xaxes=True,
        subplot_titles=column_titles,
        vertical_spacing=0.08,
        horizontal_spacing=0.05
    )

    # Metric Configurations
    # Row 0 -> Position, Row 1 -> Velocity, Row 2 -> Torque
    metric_keys = ['qpos', 'qvel', 'actuator_force']
    y_axis_labels = ["Position (rad)", "Velocity (rad/s)", "Torque (Nm)"]
    hover_units = ["(rad)", "(rad/s)", "(Nm)"]
    
    colors = {'Target': 'black', 'Baseline': 'red', 'Optimized': 'blue'}
    line_styles = {'Target': 'solid', 'Baseline': 'dash', 'Optimized': 'solid'}
    opacities = {'Target': 0.4, 'Baseline': 1.0, 'Optimized': 1.0}

    # -- Create Traces --
    trace_indices_per_leg = defaultdict(list)
    current_trace_idx = 0

    for leg_name in leg_names:
        joint_indices = leg_groups[leg_name]
        
        for col_idx, joint_global_idx in enumerate(joint_indices[:3]): 
            
            joint_name = joint_names[joint_global_idx]
            
            for row_idx, metric in enumerate(metric_keys):
                
                unit = hover_units[row_idx]

                data_map = {
                    'Target': targets[metric][:, joint_global_idx],
                    'Baseline': baseline[metric][:, joint_global_idx],
                    'Optimized': optimized[metric][:, joint_global_idx]
                }

                for label, data_array in data_map.items():
                    is_visible = (leg_name == leg_names[0])
                    
                    trace = go.Scatter(
                        x=t,
                        y=data_array,
                        mode='lines',
                        name=f"{label}",
                        legendgroup=label,
                        showlegend=(col_idx == 0 and row_idx == 0),
                        line=dict(
                            color=colors[label], 
                            dash=line_styles[label],
                            width=2 if label == 'Optimized' else 1.5
                        ),
                        opacity=opacities[label],
                        visible=is_visible,
                        hovertemplate=f"<b>{label}</b><br>Joint: {joint_name}<br>Time: %{{x:.2f}}s<br>Val: %{{y:.3f}} {unit}<extra></extra>"
                    )
                    
                    fig.add_trace(trace, row=row_idx+1, col=col_idx+1)
                    
                    trace_indices_per_leg[leg_name].append(current_trace_idx)
                    current_trace_idx += 1
    
    # Y-Axes: Only label the first column
    for r in range(1, 4):
        fig.update_yaxes(title_text=y_axis_labels[r-1], row=r, col=1)

    # X-Axes: Only label the last row
    for c in range(1, 4):
        fig.update_xaxes(title_text="Time (s)", row=3, col=c)

    buttons = []
    for leg_name in leg_names:
        visibility = [False] * current_trace_idx
        for idx in trace_indices_per_leg[leg_name]:
            visibility[idx] = True
            
        button = dict(
            label=leg_name,
            method="update",
            args=[
                {"visible": visibility},
                {"title.text": f"Comparison: Leg {leg_name} (Trial {trial_idx})"}
            ]
        )
        buttons.append(button)

    # -- Layout Updates --
    fig.update_layout(
        title=dict(
            text=f"Comparison: Leg {leg_names[0]} (Trial {trial_idx})",
            x=0.5,
            y=0.98
        ),
        margin=dict(t=140),
        updatemenus=[dict(
            type="buttons",
            direction="left",
            buttons=buttons,
            pad={"r": 10, "t": 10},
            showactive=True,
            x=0.0,
            xanchor="left",
            y=1.12,
            yanchor="top"
        )],
        height=900,
        hovermode="x unified",
        template="plotly_white"
    )

    return fig


def evaluate(
    key: jax.Array,
    mjx_model_static: mjx.Model,
    initial_params: Dict[str, jax.Array],
    optimized_params: Dict[str, jax.Array],
    dataset: Dict[str, jax.Array],
    config: ConfigDict,
    regression_spec: Dict[str, Any],
    wandb_run: Any,
):  
    # Get Config Settings:
    n_sim_per_obs, _n_obs_per_ctrl = validate_rates(
        sim_dt=config.physics.timestep,
        observation_dt=config.data.observation_rate,
        control_dt=config.physics.control_rate,
        state_dt=config.data.state_rate,
    )
    objective_metric = get_objective_fn(config.loss.type)
    objective_weights = config.loss.weights.to_dict()

    # Resample the eval trajectories onto the observation grid (matching training).
    state_dt = config.data.state_rate
    control_dt = config.physics.control_rate
    obs_dt = config.data.observation_rate
    n_obs = resampling.num_resampled_steps(dataset['qpos'].shape[1], state_dt, obs_dt)
    qpos_all = resampling.resample_linear(dataset['qpos'], state_dt, obs_dt, n_obs)
    qvel_all = resampling.resample_linear(dataset['qvel'], state_dt, obs_dt, n_obs)
    force_all = resampling.resample_linear(dataset['actuator_force'], state_dt, obs_dt, n_obs)
    ctrl_all = resampling.resample_zoh(dataset['ctrl'], control_dt, obs_dt, n_obs)

    # Sample a Random Trajectory (concrete index for numpy slicing).
    n_trials = dataset['ctrl'].shape[0]
    trial_idx = int(jax.random.randint(key, (), 0, n_trials))

    # Set Targets and Initial State:
    qpos = jnp.asarray(qpos_all[trial_idx])
    qvel = jnp.asarray(qvel_all[trial_idx])
    actuator_force = jnp.asarray(force_all[trial_idx])
    ctrl_setpoints = jnp.asarray(ctrl_all[trial_idx])[:-1]

    qpos_init, qvel_init = qpos[0], qvel[0]
    qpos_target = qpos[1:]
    qvel_target = qvel[1:]
    actuator_force_target = actuator_force[1:]

    def run_trajectory(params_dict):
        # Rehydrate model
        replace_kwargs = {}
        for name, value in params_dict.items():
            spec = regression_spec[name]
            field = spec['field']
            if field == 'log_cholesky_inertia':
                body_ids = spec['body_ids']
                b_mass, b_ipos, b_inertia, b_iquat = jax.vmap(log_cholesky_to_mujoco)(value)
                replace_kwargs['body_mass'] = mjx_model_static.body_mass.at[body_ids].set(b_mass)
                replace_kwargs['body_ipos'] = mjx_model_static.body_ipos.at[body_ids, :].set(b_ipos)
                replace_kwargs['body_inertia'] = mjx_model_static.body_inertia.at[body_ids, :].set(b_inertia)
                replace_kwargs['body_iquat'] = mjx_model_static.body_iquat.at[body_ids, :].set(b_iquat)
            elif 'column' in spec:
                col_idx = spec['column']
                original_array = getattr(mjx_model_static, field)
                new_array = original_array.at[:, col_idx].set(value)
                replace_kwargs[field] = new_array
            else:
                replace_kwargs[field] = value
            
        model_dynamic = mjx_model_static.replace(**replace_kwargs)

        def rollout(setpoints, qpos_init, qvel_init):
            d = init_function(
                model_dynamic, qpos_init, qvel_init, setpoints[0],
            )

            def step(carry, xs):
                d = step_function(model_dynamic, carry, xs, n_sim_per_obs)
                return d, (d.qpos, d.qvel, d.actuator_force)

            _, (qpos, qvel, actuator_force) = jax.lax.scan(step, d, setpoints)
            return qpos, qvel, actuator_force

        return rollout(ctrl_setpoints, qpos_init, qvel_init)

    rollout_fn = jax.jit(run_trajectory)
    
    # Rollout Baseline vs Optimized Model:
    base_qpos, base_qvel, base_actuator_force = rollout_fn(initial_params)
    opt_qpos, opt_qvel, opt_actuator_force = rollout_fn(optimized_params)
    
    # Calculate Metrics:
    def compute_weighted_loss(qpos_pred, qvel_pred, force_pred):
        losses = {
            'position': objective_metric(qpos_pred, qpos_target),
            'velocity': objective_metric(qvel_pred, qvel_target),
            'actuator_force': objective_metric(force_pred, actuator_force_target),
        }
        losses = {k: v * objective_weights[k] for k, v in losses.items()}
        return losses, sum(losses.values())

    base_losses, base_loss = compute_weighted_loss(base_qpos, base_qvel, base_actuator_force)
    opt_losses, opt_loss = compute_weighted_loss(opt_qpos, opt_qvel, opt_actuator_force)
    
    metrics = {
        "eval/baseline_pos_loss": base_losses['position'],
        "eval/baseline_vel_loss": base_losses['velocity'],
        "eval/baseline_force_loss": base_losses['actuator_force'],
        "eval/baseline_loss": base_loss,
        "eval/optimized_pos_loss": opt_losses['position'],
        "eval/optimized_vel_loss": opt_losses['velocity'],
        "eval/optimized_force_loss": opt_losses['actuator_force'],
        "eval/optimized_loss": opt_loss,
    }
    
    t = np.arange(1, n_obs) * config.data.observation_rate
    
    def to_np(x): return np.array(x)

    targets_dict = {
        'qpos': to_np(qpos_target),
        'qvel': to_np(qvel_target),
        'actuator_force': to_np(actuator_force_target)
    }
    baseline_dict = {
        'qpos': to_np(base_qpos),
        'qvel': to_np(base_qvel),
        'actuator_force': to_np(base_actuator_force)
    }
    optimized_dict = {
        'qpos': to_np(opt_qpos),
        'qvel': to_np(opt_qvel),
        'actuator_force': to_np(opt_actuator_force)
    }

    fig = create_interactive_plot(
        t, 
        JOINT_NAMES, 
        targets_dict, 
        baseline_dict, 
        optimized_dict,
        int(trial_idx)
    )
    
    metrics["eval/trajectory_evaluation"] = wandb.Plotly(fig)
    
    wandb_run.log(metrics)
