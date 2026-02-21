from absl import app, flags

import sys
from pathlib import Path
import yaml

import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go


FLAGS = flags.FLAGS
flags.DEFINE_string(
    'yaml_path', None, 'Path to the reward_comparison.yaml file.', short_name='y', required=True
)


def load_and_format_data(yaml_path):
    """Loads YAML and formats it into a comprehensive Pandas DataFrame."""
    path = Path(yaml_path)
    if not path.exists():
        print(f"Error: Could not find {path}", file=sys.stderr)
        sys.exit(1)

    with path.open('r') as f:
        data = yaml.safe_load(f)

    # --- NEW: Dictionary mapping raw directory names to publication names ---
    policy_name_map = {
        "regressed-position-forward": "Regressed Parameter",
        "regressed-position-dr-forward": "Regressed Parameter w/ Domain Randomization",
        "transparent-position-forward": "Transparent Parameter",
        "transparent-position-dr-forward": "Transparent Parameter w/ Domain Randomization",
        "uniform-position-dr-forward": "Uniform Parameter Domain Randomization",
        "vendor-position-forward": "Vendor Baseline",
        "vendor-position-dr-forward": "Vendor Baseline w/ Domain Randomization",
    }
    # ------------------------------------------------------------------------

    records = []
    for run_name, run_data in data['runs'].items():
        raw_system_name = run_name.rsplit('-', 1)[0]

        clean_system_name = policy_name_map.get(raw_system_name, raw_system_name)

        record = {
            "System": clean_system_name,
            "Run": run_name,
            "Total_Reward": run_data['reward'],
        }

        config = run_data.get('original_config', {})
        for metric_name, value in config.items():
            if metric_name != 'sum':
                record[metric_name] = value

        records.append(record)

    return pd.DataFrame(records)


def plot_aggregate_performance(df, output_dir):
    """Generates the Box + Swarm plot for total reward distributions."""
    order = df.groupby('System')['Total_Reward'].mean().sort_values(ascending=False).index

    fig = px.box(
        df,
        x='Total_Reward',
        y='System',
        color='System',
        points='all',
        category_orders={"System": order.tolist()}
    )

    fig.update_layout(
        title=dict(text="Policy Performance Distribution Across Hardware Deployments", font=dict(size=18)),
        xaxis_title="Total Reward Sum",
        yaxis_title="Policy Configuration",
        showlegend=False,
        template="simple_white",
        width=1000,
        height=600
    )

    fig.write_html(output_dir / "aggregate_performance.html")
    fig.write_image(output_dir / "aggregate_performance.pdf")
    print("Saved: aggregate_performance (HTML/PDF)")


def plot_radar_chart(df, output_dir):
    """Generates a Radar Chart comparing the mean normalized sub-metrics."""
    metrics_to_plot = [
        # Rewards:
        'tracking_linear_velocity',
        'tracking_angular_velocity',
        # Orientation Regularization:
        'orientation_regularization',
        'linear_z_velocity',
        'angular_xy_velocity',
        # Energy Regularization:
        'torque',
        'action_rate',
        # Gait Shaping:
        'foot_slip',
        'air_time',
        'foot_clearance',
        'gait_variance',
    ]

    metrics_to_plot = [m for m in metrics_to_plot if m in df.columns]
    mean_df = df.groupby('System')[metrics_to_plot].mean()

    # Normalize data between 0 and 1 so they can share a polar axis
    normalized_df = (mean_df - mean_df.min()) / (mean_df.max() - mean_df.min())
    normalized_df = normalized_df.fillna(0.5)

    # Plot the top 3 systems by total reward
    top_systems = df.groupby('System')['Total_Reward'].mean().nlargest(4).index
    plot_df = normalized_df.loc[top_systems]

    categories = list(plot_df.columns)

    fig = go.Figure()
    colors = px.colors.qualitative.Plotly

    for idx, (system_name, row) in enumerate(plot_df.iterrows()):
        # Duplicate the first value to close the loop on the radar chart
        r_vals = row.values.tolist() + [row.values[0]]
        theta_vals = categories + [categories[0]]

        fig.add_trace(go.Scatterpolar(
            r=r_vals,
            theta=theta_vals,
            fill='toself',
            name=system_name,
            line_color=colors[idx % len(colors)]
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1.1], tickfont=dict(color="grey", size=10))
        ),
        title=dict(text="Normalized Multi-Objective Trade-offs (Top 3 Policies)", font=dict(size=18)),
        showlegend=True,
        template="plotly_white",
        width=800,
        height=800
    )

    fig.write_html(output_dir / "radar_tradeoffs.html")
    fig.write_image(output_dir / "radar_tradeoffs.pdf")
    print("Saved: radar_tradeoffs (HTML/PDF)")


def plot_stacked_bar(df, output_dir):
    """Generates a Stacked Bar Chart showing positive rewards vs negative penalties."""
    metric_cols = [c for c in df.columns if c not in ['System', 'Run', 'Total_Reward', 'action_rate', 'stand_still']]
    mean_df = df.groupby('System')[metric_cols].mean().reset_index()

    order = df.groupby('System')['Total_Reward'].mean().sort_values(ascending=False).index

    # Melt dataframe so Plotly Express can color by metric
    melted_df = mean_df.melt(
        id_vars='System',
        value_vars=metric_cols,
        var_name='Metric',
        value_name='Reward Contribution'
    )

    # barmode='relative' automatically stacks negative values downward
    fig = px.bar(
        melted_df,
        x='System',
        y='Reward Contribution',
        color='Metric',
        barmode='relative',
        category_orders={"System": order.tolist()}
    )

    fig.update_layout(
        title=dict(text="Reward Composition (Positive Tracking vs. Negative Penalties)", font=dict(size=18)),
        xaxis_title="Policy Configuration",
        template="simple_white",
        width=1200,
        height=700
    )

    # Draw a distinct zero line
    fig.add_hline(y=0, line_width=1.5, line_color="black")

    fig.write_html(output_dir / "reward_composition.html")
    fig.write_image(output_dir / "reward_composition.pdf")
    print("Saved: reward_composition (HTML/PDF)")


def main(argv=None):
    yaml_path = Path(FLAGS.yaml_path).resolve()
    output_dir = yaml_path.parent

    print(f"Loading data from: {yaml_path.name}")
    df = load_and_format_data(yaml_path)

    print("Generating figures...")
    plot_aggregate_performance(df, output_dir)
    plot_radar_chart(df, output_dir)
    plot_stacked_bar(df, output_dir)

    print(f"\nAll plots saved successfully to: {output_dir}")


if __name__ == '__main__':
    app.run(main)
