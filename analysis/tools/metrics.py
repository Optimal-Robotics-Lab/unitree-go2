from absl import app, flags

import sys
from pathlib import Path
import matplotlib
import yaml

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, PathPatch, Rectangle

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

    # policy_name_map = {
    #     "regressed-position-forward": "Regressed Parameter",
    #     "regressed-position-dr-forward": "Regressed Parameter w/ Domain Randomization",
    #     "transparent-position-forward": "Transparent Parameter",
    #     "transparent-position-dr-forward": "Transparent Parameter w/ Domain Randomization",
    #     "uniform-position-dr-forward": "Uniform Parameter Domain Randomization",
    #     "vendor-position-forward": "Vendor",
    #     "vendor-position-dr-forward": "Vendor w/ Domain Randomization",
    # }

    policy_name_map = {
        "fresh-armadillo-6": "Regressed Parameter",
        "rose-fog-12": "Regressed Parameter w/ Domain Randomization",
        "treasured-sky-23": "Transparent Parameter",
        "resilient-oath-10": "Transparent Parameter w/ Domain Randomization",
        "major-bush-14": "Uniform Parameter Domain Randomization",
        "astral-bee-25": "Vendor",
        "lilac-resonance-8": "Vendor w/ Domain Randomization",
    }

    records = []
    for run_name, run_data in data['runs'].items():
        raw_system_name = run_name.rsplit('-', 1)[0]

        clean_system_name = policy_name_map.get(raw_system_name, raw_system_name)

        record = {
            "System": clean_system_name,
            "Run": run_name,
            "Total_Reward": run_data['reward'],
            "Episodic_Reward": run_data.get('episodic_reward', np.nan),
        }

        config = run_data.get('original_config', {})
        for metric_name, value in config.items():
            if metric_name != 'sum':
                record[metric_name] = value

        records.append(record)

    return pd.DataFrame(records)


# def plot_aggregate_performance(df, output_dir):
#     """Generates the Box + Swarm plot for total reward distributions."""
#     colors = px.colors.qualitative.Plotly
#     color_map = {
#         "Regressed Parameter": colors[0],
#         "Regressed Parameter w/ Domain Randomization": colors[0],    
#         "Transparent Parameter": colors[1],
#         "Transparent Parameter w/ Domain Randomization": colors[1],
#         "Vendor": colors[2],
#         "Vendor w/ Domain Randomization": colors[2],
#         "Uniform Parameter Domain Randomization": colors[3],
#     }

#     order = df.groupby('System')['Total_Reward'].mean().sort_values(ascending=False).index

#     fig = px.box(
#         df,
#         x='Total_Reward',
#         y='System',
#         color='System',
#         points='all',
#         category_orders={"System": order.tolist()},
#         color_discrete_map=color_map,
#     )

#     for trace in fig.data:
#         if "Domain Randomization" in trace.name:
#             # Hatches: '\\', 'x', '-', '|', '+', or '.'
#             import pdb; pdb.set_trace()
#             trace.fillpattern = dict(shape="/", fillmode="overlay")

#     fig.update_layout(
#         title=dict(text="Policy Performance Distribution Across Hardware Deployments", font=dict(size=18)),
#         xaxis_title="Total Reward Sum",
#         yaxis_title="Policy Configuration",
#         showlegend=False,
#         template="simple_white",
#         width=1000,
#         height=600
#     )

#     fig.write_html(output_dir / "aggregate_performance.html")
#     fig.write_image(output_dir / "aggregate_performance.pdf")
#     print("Saved: aggregate_performance (HTML/PDF)")

def plot_aggregate_performance(df, output_dir):
    """Generates a Seaborn Box + Strip plot with hatches, offset matched scatter points, and full whiskers."""
    df = df.copy()
    rename_map = {
        "Regressed Parameter": "Regressed",
        "Regressed Parameter w/ Domain Randomization": "Regressed (DR)",
        "Transparent Parameter": "Transparent",
        "Transparent Parameter w/ Domain Randomization": "Transparent (DR)",
        "Vendor": "Vendor",
        "Vendor w/ Domain Randomization": "Vendor (DR)",
        "Uniform Parameter Domain Randomization": "Uniform (DR)",
    }
    df['System'] = df['System'].replace(rename_map)

    order = df.groupby('System')['Total_Reward'].mean().sort_values(ascending=False).index

    base_palette = sns.color_palette()
    color_map = {
        "Regressed": base_palette[0],
        "Regressed (DR)": base_palette[0],
        "Transparent": base_palette[1],
        "Transparent (DR)": base_palette[1],
        "Vendor": base_palette[2],
        "Vendor (DR)": base_palette[2],
        "Uniform (DR)": base_palette[3],
    }

    fig, ax = plt.subplots(figsize=(3.5, 4.0))
    # plt.figure(figsize=(10, 6))

    ax = sns.boxplot(
        data=df,
        x='Total_Reward',
        y='System',
        order=order,
        palette=color_map,
        showfliers=False,
        width=0.4,
        zorder=3,
        whis=(0, 100),
    )

    for patch in ax.patches:
        patch.set_zorder(3)

    lines_per_box = len(ax.lines) // len(ax.patches)

    for i, box in enumerate(ax.patches):
        system_name = order[i]

        base_color = box.get_facecolor()
        rgb = base_color[:3]

        edge_color = (*rgb, 1.0)
        face_color = (*rgb, 0.3)

        box.set_facecolor(face_color)
        box.set_edgecolor(edge_color)
        box.set_linewidth(1.0)

        if "(DR)" in system_name:
            hatch_overlay = PathPatch(
                box.get_path(),
                transform=box.get_transform(),
                facecolor='none',
                edgecolor=edge_color,
                hatch='///',
                hatch_linewidth=1.0,
                linewidth=1.0,
                zorder=box.get_zorder() + 0.1
            )
            ax.add_patch(hatch_overlay)

        # 3. Whisker, Caps, and Median Lines
        start_idx = i * lines_per_box
        end_idx = start_idx + lines_per_box

        for line in ax.lines[start_idx:end_idx]:
            line.set_color(edge_color)
            line.set_linewidth(1.0)

    n_collections = len(ax.collections)

    sns.stripplot(
        data=df,
        x='Total_Reward',
        y='System',
        order=order,
        hue='System',
        palette=color_map,
        alpha=0.6,
        size=4,
        jitter=0.05,
        zorder=1,
        legend=False,
    )

    for col in ax.collections[n_collections:]:
        offsets = col.get_offsets()
        if len(offsets) > 0:
            offsets[:, 1] += 0.4
            col.set_offsets(offsets)

    # plt.xlabel("Total Reward Sum", fontsize=12, fontweight='bold')
    # plt.ylabel("", fontsize=12)

    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)

    # sns.despine()

    # 6. Formatting (Scaled down for 3.5 inch width)
    ax.set_xlabel("Total Reward Sum", fontsize=9, fontweight="bold")
    ax.set_ylabel("")
    ax.tick_params(axis='both', labelsize=8)
    sns.despine(ax=ax)

    legend_elements = [
        Patch(facecolor='gray', alpha=0.3, edgecolor='black', label='Baseline Policy'),
        Patch(facecolor='gray', alpha=0.3, edgecolor='black', hatch='///', label='Domain Randomization (DR)')
    ]

    plt.legend(handles=legend_elements, loc='best', frameon=True, fontsize=6)

    plt.tight_layout()
    output_pdf = output_dir / "aggregate_performance.pdf"
    plt.savefig(output_pdf, format='pdf', bbox_inches='tight')
    print(f"Saved: {output_pdf}")
    plt.close()


def plot_grouped_performance(df, output_dir):
    """Generates a grouped Box + Strip plot with matching colors, hatches, and offset scatter points."""
    df = df.copy()
    output_dir = Path(output_dir)

    rename_map = {
        "Regressed Parameter": "Regressed",
        "Regressed Parameter w/ Domain Randomization": "Regressed (DR)",
        "Transparent Parameter": "Transparent",
        "Transparent Parameter w/ Domain Randomization": "Transparent (DR)",
        "Vendor": "Vendor",
        "Vendor w/ Domain Randomization": "Vendor (DR)",
        "Uniform Parameter Domain Randomization": "Uniform (DR)",
    }
    df['System'] = df['System'].replace(rename_map)

    # Extract Base and Variant using fast, vectorized operations
    df['Base_Policy'] = df['System'].str.replace(r' \(DR\)', '', regex=True)
    df['Variant'] = np.where(df['System'].str.contains('(DR)', regex=False), 'DR', 'Baseline')

    # Determine the y-axis order dynamically based on mean Total_Reward
    order = (
        df.groupby('Base_Policy')['Total_Reward']
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )

    # Inject dummy rows safely (Bulk creation is 100x faster than looping pd.concat)
    existing_pairs = set(zip(df['Base_Policy'], df['Variant']))
    missing_rows = [
        {'System': f"{base} ({variant})", 'Base_Policy': base, 'Variant': variant, 'Total_Reward': np.nan}
        for base in order for variant in ['Baseline', 'DR']
        if (base, variant) not in existing_pairs
    ]
    if missing_rows:
        df = pd.concat([df, pd.DataFrame(missing_rows)], ignore_index=True)

    # 2. Assign strictly matching colors safely
    base_palette = sns.color_palette()
    color_map = {
        "Regressed": base_palette[0],
        "Transparent": base_palette[1],
        "Vendor": base_palette[2],
        "Uniform": base_palette[3],
    }

    matplotlib.rcParams['hatch.linewidth'] = 1.0
    fig, ax = plt.subplots(figsize=(3.5, 4.0))

    # 3. Create Grouped Boxplot
    sns.boxplot(
        data=df,
        x='Total_Reward',
        y='Base_Policy',
        hue='Variant',
        hue_order=['Baseline', 'DR'],
        order=order,
        showfliers=False,
        width=0.5,
        zorder=3,
        whis=(0, 100),
        legend=False,
        ax=ax
    )

    # Box Labels:
    labels = [
        (variant, base)
        for variant in ['Baseline', 'DR']
        for base in order
    ]

    # Remove the Uniform Baseline
    labels.remove(('Baseline', 'Uniform'))

    # 4. Clean Coordinate Mapping for Colors, Hatches, and Gaps
    dodge_offset = 0.1
    scale = 0.75
    grouped_lines = np.split(np.asarray(ax.lines), len(ax.patches))
    boxes = [p for p in ax.patches if isinstance(p, (PathPatch, Rectangle))]

    for box, lines, (variant, base) in zip(boxes, grouped_lines, labels):
        if not box.get_visible():
            continue

        # Set Colors:
        rgb = color_map.get(base, base_palette[0])[:3]
        face_color = (*rgb, 0.3)
        edge_color = (*rgb, 1.0)
        box.set_facecolor(face_color)
        box.set_edgecolor(edge_color)
        box.set_linewidth(1.0)

        # Set Hatch:
        if variant == 'DR':
            hatch_overlay = PathPatch(
                box.get_path(),
                transform=box.get_transform(),
                facecolor='none',
                edgecolor=edge_color,
                hatch='///',
                hatch_linewidth=1.0,
                linewidth=1.0,
                zorder=box.get_zorder() + 0.1
            )
            ax.add_patch(hatch_overlay)

        # Box Spacing:
        y_idx = order.index(base)
        if base == 'Uniform':
            target_y_center = y_idx
        else:
            target_y_center = y_idx + (dodge_offset if variant == 'DR' else -dodge_offset)

        # Scale Box Height:
        path = box.get_path()
        vertices = path.vertices
        y_center = (vertices[:, 1].max() + vertices[:, 1].min()) / 2.0
        shift = target_y_center - y_center
        vertices[:, 1] = target_y_center + (vertices[:, 1] - y_center) * scale

        for line in lines:
            line.set_ydata(line.get_ydata() + shift)
            line.set_color(edge_color)
            line.set_linewidth(1.0)

    # 5. Scatter Points
    sns.stripplot(
        data=df,
        x='Total_Reward',
        y='Base_Policy',
        hue='Variant',
        hue_order=['Baseline', 'DR'],
        order=order,
        dodge=True,
        alpha=0.6,
        size=3,
        jitter=0.05,
        zorder=1,
        legend=False,
        ax=ax
    )

    scatter_labels = [
        (variant, base)
        for base in order
        for variant in ['Baseline', 'DR']
    ]

    for (variant, base), collection in zip(scatter_labels, ax.collections):
        offsets = collection.get_offsets()
        if base != 'Uniform':
            if variant != 'DR':
                offsets[:, 1] -= dodge_offset
            else:
                offsets[:, 1] += dodge_offset

        rgb = color_map.get(base, base_palette[0])[:3]
        color = (*rgb, 1.0)
        collection.set_facecolors(color)
        collection.set_edgecolors(color)
        collection.set_offsets(offsets)

    # Set x lim:
    ax.set_xlim(6, 12)

    # 7. Column Formatting
    ax.set_xlabel("Total Reward Sum", fontsize=8, fontweight="bold")
    ax.set_ylabel("")
    ax.tick_params(axis='both', labelsize=8)
    sns.despine(ax=ax)

    # Custom legend representing the variants
    legend_elements = [
        Patch(facecolor='gray', alpha=0.3, edgecolor='black', label='Baseline Policy'),
        Patch(facecolor='gray', alpha=0.3, edgecolor='black', hatch='////', label='Domain Randomization (DR)')
    ]
    ax.legend(handles=legend_elements, loc='best', frameon=True, fontsize=6)

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_pdf = output_dir / "reward_comparison.pdf"

    plt.savefig(output_pdf, format='pdf', bbox_inches='tight')
    print(f"Saved: {output_pdf}")
    plt.close(fig)


def plot_episodic_grouped_performance(df, output_dir):
    """Generates a grouped Box + Strip plot for the Episodic Reward."""
    df = df.copy()
    output_dir = Path(output_dir)

    rename_map = {
        "Regressed Parameter": "Regressed",
        "Regressed Parameter w/ Domain Randomization": "Regressed (DR)",
        "Transparent Parameter": "Transparent",
        "Transparent Parameter w/ Domain Randomization": "Transparent (DR)",
        "Vendor": "Vendor",
        "Vendor w/ Domain Randomization": "Vendor (DR)",
        "Uniform Parameter Domain Randomization": "Uniform (DR)",
    }
    df['System'] = df['System'].replace(rename_map)

    # Extract Base and Variant using fast, vectorized operations
    df['Base_Policy'] = df['System'].str.replace(r' \(DR\)', '', regex=True)
    df['Variant'] = np.where(df['System'].str.contains('(DR)', regex=False), 'DR', 'Baseline')

    # Determine the y-axis order dynamically based on mean Episodic_Reward
    order = (
        df.groupby('Base_Policy')['Episodic_Reward']
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )

    # Inject dummy rows safely 
    existing_pairs = set(zip(df['Base_Policy'], df['Variant']))
    missing_rows = [
        {'System': f"{base} ({variant})", 'Base_Policy': base, 'Variant': variant, 'Total_Reward': np.nan, 'Episodic_Reward': np.nan}
        for base in order for variant in ['Baseline', 'DR']
        if (base, variant) not in existing_pairs
    ]
    if missing_rows:
        df = pd.concat([df, pd.DataFrame(missing_rows)], ignore_index=True)

    # Assign strictly matching colors safely
    base_palette = sns.color_palette()
    color_map = {
        "Regressed": base_palette[0],
        "Transparent": base_palette[1],
        "Vendor": base_palette[2],
        "Uniform": base_palette[3],
    }

    matplotlib.rcParams['hatch.linewidth'] = 1.0
    fig, ax = plt.subplots(figsize=(3.5, 4.0))

    # Create Grouped Boxplot
    sns.boxplot(
        data=df,
        x='Episodic_Reward',
        y='Base_Policy',
        hue='Variant',
        hue_order=['Baseline', 'DR'],
        order=order,
        showfliers=False,
        width=0.5,
        zorder=3,
        whis=(0, 100),
        legend=False,
        ax=ax
    )

    # Box Labels
    labels = [
        (variant, base)
        for variant in ['Baseline', 'DR']
        for base in order
    ]
    if ('Baseline', 'Uniform') in labels:
        labels.remove(('Baseline', 'Uniform'))

    # Clean Coordinate Mapping for Colors, Hatches, and Gaps
    dodge_offset = 0.1
    scale = 0.75
    grouped_lines = np.split(np.asarray(ax.lines), len(ax.patches))
    boxes = [p for p in ax.patches if isinstance(p, (PathPatch, Rectangle))]

    for box, lines, (variant, base) in zip(boxes, grouped_lines, labels):
        if not box.get_visible():
            continue

        rgb = color_map.get(base, base_palette[0])[:3]
        face_color = (*rgb, 0.3)
        edge_color = (*rgb, 1.0)
        box.set_facecolor(face_color)
        box.set_edgecolor(edge_color)
        box.set_linewidth(1.0)

        if variant == 'DR':
            hatch_overlay = PathPatch(
                box.get_path(),
                transform=box.get_transform(),
                facecolor='none',
                edgecolor=edge_color,
                hatch='///',
                hatch_linewidth=1.0,
                linewidth=1.0,
                zorder=box.get_zorder() + 0.1
            )
            ax.add_patch(hatch_overlay)

        y_idx = order.index(base)
        target_y_center = y_idx if base == 'Uniform' else y_idx + (dodge_offset if variant == 'DR' else -dodge_offset)

        path = box.get_path()
        vertices = path.vertices
        y_center = (vertices[:, 1].max() + vertices[:, 1].min()) / 2.0
        shift = target_y_center - y_center
        vertices[:, 1] = target_y_center + (vertices[:, 1] - y_center) * scale

        for line in lines:
            line.set_ydata(line.get_ydata() + shift)
            line.set_color(edge_color)
            line.set_linewidth(1.0)

    # Scatter Points
    sns.stripplot(
        data=df,
        x='Episodic_Reward',  # <-- UPDATED
        y='Base_Policy',
        hue='Variant',
        hue_order=['Baseline', 'DR'],
        order=order,
        dodge=True,
        alpha=0.6,
        size=3,
        jitter=0.05,
        zorder=1,
        legend=False,
        ax=ax
    )

    scatter_labels = [
        (variant, base)
        for base in order
        for variant in ['Baseline', 'DR']
    ]

    for (variant, base), collection in zip(scatter_labels, ax.collections):
        offsets = collection.get_offsets()
        if len(offsets) > 0:
            if base != 'Uniform':
                offsets[:, 1] += dodge_offset if variant == 'DR' else -dodge_offset

            rgb = color_map.get(base, base_palette[0])[:3]
            color = (*rgb, 1.0)
            collection.set_facecolors(color)
            collection.set_edgecolors(color)
            collection.set_offsets(offsets)

    # Set x lim:
    ax.set_xlim(6, 12)

    # Column Formatting
    ax.set_xlabel("Episodic Reward", fontsize=8, fontweight="bold")
    ax.set_ylabel("")
    ax.tick_params(axis='both', labelsize=8)
    sns.despine(ax=ax)

    # Custom legend
    legend_elements = [
        Patch(facecolor='gray', alpha=0.3, edgecolor='black', label='Baseline Policy'),
        Patch(facecolor='gray', alpha=0.3, edgecolor='black', hatch='////', label='Domain Randomization (DR)')
    ]
    ax.legend(handles=legend_elements, loc='best', frameon=True, fontsize=6)

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)

    output_pdf = output_dir / "episodic_reward_comparison.pdf" 

    plt.savefig(output_pdf, format='pdf', bbox_inches='tight')
    print(f"Saved: {output_pdf}")
    plt.close(fig)


def plot_average_episodic_reward_bar(df, output_dir):
    """Generates a simple bar plot of the average episodic reward across policies."""
    df = df.copy()
    output_dir = Path(output_dir)

    # Drop NaNs just in case some runs lack episodic reward data
    df = df.dropna(subset=['Episodic_Reward'])

    # Sort the y-axis dynamically based on the average episodic reward
    order = (
        df.groupby('System')['Episodic_Reward']
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )

    # Use the same color palette mappings
    base_palette = sns.color_palette()
    color_map = {
        "Regressed Parameter": base_palette[0],
        "Regressed Parameter w/ Domain Randomization": base_palette[0],
        "Transparent Parameter": base_palette[1],
        "Transparent Parameter w/ Domain Randomization": base_palette[1],
        "Vendor": base_palette[2],
        "Vendor w/ Domain Randomization": base_palette[2],
        "Uniform Parameter Domain Randomization": base_palette[3],
    }

    matplotlib.rcParams['hatch.linewidth'] = 1.0 
    fig, ax = plt.subplots(figsize=(5.5, 4.0))

    # Seaborn's barplot automatically averages the values and plots error bars
    sns.barplot(
        data=df,
        x='Episodic_Reward',
        y='System',
        order=order,
        palette=color_map,
        errorbar='sd',       # Show standard deviation as error bars
        capsize=0.15,        # Add caps to error bars
        edgecolor='black',
        linewidth=1.0,
        ax=ax
    )

    # Apply hatching to the Domain Randomization variants
    for i, bar in enumerate(ax.patches):
        if i < len(order):
            system_name = order[i]
            # Make bars slightly transparent
            bar.set_alpha(0.8)
            # Add hatching if it's a DR variant
            if "Domain Randomization" in system_name:
                bar.set_hatch('////')

    # Formatting
    ax.set_xlabel("Average Episodic Reward", fontsize=10, fontweight="bold")
    ax.set_ylabel("") 
    ax.tick_params(axis='both', labelsize=8)
    sns.despine(ax=ax)

    plt.tight_layout()
    output_pdf = output_dir / "episodic_reward_bar.pdf"
    plt.savefig(output_pdf, format='pdf', bbox_inches='tight')
    print(f"Saved: {output_pdf}")
    plt.close(fig)


def plot_simple_bar(output_dir):
    """Generates a simple bar plot for three specific metrics."""
    output_dir = Path(output_dir)

    data = np.array([0.22, 0.31, 0.35, 0.59, 1.01, 0.68, 0.71, 1.55])

    fig, ax = plt.subplots(figsize=(4.0, 4.0))

    # Create the bar plot
    sns.barplot(
        data,
        ax=ax,
        palette="Blues_d",
        edgecolor="black",
        linewidth=1.0
    )

    # Apply hatching to the Domain Randomization variants
    # for i, bar in enumerate(ax.patches):
    #     if i < len(labels):
    #         system_name = labels[i]
    #         # Make bars slightly transparent
    #         bar.set_alpha(0.8)
    #         # Add hatching if it's a DR variant
    #         if "(DR)" in system_name:
    #             bar.set_hatch('////')

    # Optional: Add the exact value above each bar
    # for i, val in enumerate(data):
    #     ax.text(i, val + 0.02, f'{val:.2f}', ha='center', va='bottom', fontsize=9)

    # Formatting
    ax.set_ylabel("Value", fontsize=10, fontweight="bold")
    ax.tick_params(axis='both', labelsize=9)

    # Make the y-axis range slightly taller than the max value so the text fits
    ax.set_ylim(0, max(data) * 1.15)

    sns.despine(ax=ax)

    plt.tight_layout()
    output_pdf = output_dir / "simple_metrics_bar.pdf"
    plt.savefig(output_pdf, format='pdf', bbox_inches='tight')
    print(f"Saved: {output_pdf}")
    plt.close(fig)


def plot_radar_chart(df, output_dir):
    df = df.copy()
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
        'acceleration',
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
        'acceleration',
        # Gait Shaping:
        'foot_slip',
        'air_time',
        'foot_clearance'
    ]
    
    # Filter the list to only include columns actually present in the dataframe
    metric_cols = [c for c in metrics_to_plot if c in df.columns]
    
    mean_df = df.groupby('System')[metric_cols].mean().reset_index()

    # Determine order based on total reward
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
    # df = load_and_format_data(yaml_path)

    print("Generating figures...")
    # plot_aggregate_performance(df, output_dir)
    # plot_grouped_performance(df, output_dir)
    # plot_episodic_grouped_performance(df, output_dir)
    # plot_radar_chart(df, output_dir)
    # plot_stacked_bar(df, output_dir)
    # plot_average_episodic_reward_bar(df, output_dir)
    plot_simple_bar(output_dir)

    print(f"\nAll plots saved successfully to: {output_dir}")


if __name__ == '__main__':
    app.run(main)
