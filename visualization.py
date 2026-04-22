import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


def plot_time_series(df, parameter, stations=None, title=None, figsize=(14, 5)):
    """
    Plot a time series of one electrical parameter for one or more stations.

    Parameters:
    df (pandas.DataFrame): Power system data.
    parameter (str): Column name to plot.
    stations (list, optional): Station IDs to include. None = all.
    title (str, optional): Plot title override.
    figsize (tuple): Figure dimensions.

    Returns:
    matplotlib.figure.Figure
    """
    if stations is None:
        stations = df['station_id'].unique()

    fig, ax = plt.subplots(figsize=figsize)
    colors = ['steelblue', 'coral', 'mediumseagreen']

    for idx, station in enumerate(stations):
        station_data = df[df['station_id'] == station].sort_values('timestamp')
        ax.plot(
            station_data['timestamp'],
            station_data[parameter],
            label=station,
            color=colors[idx % len(colors)],
            linewidth=0.7,
            alpha=0.85,
        )

    label = parameter.replace('_', ' ').title()
    ax.set_xlabel('Time')
    ax.set_ylabel(label)
    ax.set_title(title or f'{label} Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=30)
    plt.tight_layout()
    return fig


def plot_power_triangle(real_power, reactive_power, station_id=''):
    """
    Draw the power triangle for given real and reactive power values.

    Parameters:
    real_power (float): Real power in MW.
    reactive_power (float): Reactive power in MVAR.
    station_id (str): Label for the plot title.

    Returns:
    matplotlib.figure.Figure
    """
    apparent_power = np.sqrt(real_power**2 + reactive_power**2)
    power_factor   = real_power / apparent_power if apparent_power > 0 else 0
    angle_rad      = np.arctan2(reactive_power, real_power)
    angle_deg      = np.degrees(angle_rad)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Draw the three sides as arrows
    arrow_props = dict(arrowstyle='->', lw=2.5)
    ax.annotate('', xy=(real_power, reactive_power), xytext=(0, 0),
                arrowprops=dict(**arrow_props, color='royalblue'))
    ax.annotate('', xy=(real_power, 0), xytext=(0, 0),
                arrowprops=dict(**arrow_props, color='seagreen'))
    ax.annotate('', xy=(real_power, reactive_power), xytext=(real_power, 0),
                arrowprops=dict(**arrow_props, color='tomato'))

    # Side labels
    offset = apparent_power * 0.06
    ax.text(real_power / 2, -offset * 1.8,
            f'P = {real_power:.1f} MW', ha='center',
            color='seagreen', fontsize=11, fontweight='bold')
    ax.text(real_power + offset, reactive_power / 2,
            f'Q = {reactive_power:.1f} MVAR', ha='left',
            color='tomato', fontsize=11, fontweight='bold')
    ax.text(real_power / 2 - offset * 2, reactive_power / 2 + offset,
            f'S = {apparent_power:.1f} MVA', ha='right',
            color='royalblue', fontsize=11, fontweight='bold')

    # Angle arc
    theta = np.linspace(0, angle_rad, 60)
    r = apparent_power * 0.18
    ax.plot(r * np.cos(theta), r * np.sin(theta), 'k-', lw=1.5)
    ax.text(r * 1.3 * np.cos(angle_rad / 2),
            r * 1.3 * np.sin(angle_rad / 2),
            f'φ = {angle_deg:.1f}°\nPF = {power_factor:.3f}',
            fontsize=10, ha='left')

    ax.set_xlim(-apparent_power * 0.12, apparent_power * 1.25)
    ax.set_ylim(-apparent_power * 0.25, apparent_power * 1.1)
    ax.set_xlabel('Real Power (MW)')
    ax.set_ylabel('Reactive Power (MVAR)')
    ax.set_title(f'Power Triangle{" — " + station_id if station_id else ""}')
    ax.axhline(0, color='k', lw=0.5)
    ax.axvline(0, color='k', lw=0.5)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_load_patterns(patterns, title='System-Wide Load Patterns'):
    """
    Plot average daily (by hour) and seasonal (by month) load patterns as bar charts.

    Parameters:
    patterns (dict): Output of analysis.identify_load_patterns().
    title (str): Suptitle for the figure.

    Returns:
    matplotlib.figure.Figure
    """
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(patterns['daily'].index, patterns['daily'].values,
                color='steelblue', alpha=0.85, edgecolor='white', linewidth=0.5)
    axes[0].set_xlabel('Hour of Day')
    axes[0].set_ylabel('Average Real Power (MW)')
    axes[0].set_title('Average Daily Load Pattern')
    axes[0].set_xticks(range(0, 24, 2))
    axes[0].grid(True, alpha=0.3, axis='y')

    months_present = patterns['seasonal'].index.tolist()
    axes[1].bar(
        range(len(months_present)),
        patterns['seasonal'].values,
        color='coral', alpha=0.85, edgecolor='white', linewidth=0.5,
    )
    axes[1].set_xticks(range(len(months_present)))
    axes[1].set_xticklabels([month_names[m - 1] for m in months_present])
    axes[1].set_xlabel('Month')
    axes[1].set_ylabel('Average Real Power (MW)')
    axes[1].set_title('Average Seasonal Load Pattern')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_load_heatmap(df):
    """
    Plot a heatmap of average real power by hour of day and month.

    Parameters:
    df (pandas.DataFrame): Power system data.

    Returns:
    matplotlib.figure.Figure
    """
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    df = df.copy()
    df['hour']  = df['timestamp'].dt.hour
    df['month'] = df['timestamp'].dt.month

    pivot = df.groupby(['month', 'hour'])['real_power_mw'].mean().unstack()

    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(pivot.values, aspect='auto', cmap='YlOrRd', origin='lower')
    plt.colorbar(im, ax=ax, label='Avg Real Power (MW)')

    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Month')
    ax.set_title('Load Heatmap — Average Real Power by Hour and Month')
    ax.set_xticks(range(0, 24, 2))
    ax.set_xticklabels(range(0, 24, 2))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([month_names[m - 1] for m in pivot.index])
    plt.tight_layout()
    return fig


def plot_fault_timeline(faults_df, df):
    """
    Plot voltage time series for all stations with fault events highlighted in red.

    Parameters:
    faults_df (pandas.DataFrame): Output of analysis.detect_faults().
    df (pandas.DataFrame): Full power system data.

    Returns:
    matplotlib.figure.Figure or None if no faults.
    """
    if faults_df.empty:
        print("No faults detected — skipping fault timeline plot.")
        return None

    stations = df['station_id'].unique()
    colors   = ['steelblue', 'coral', 'mediumseagreen']

    fig, axes = plt.subplots(len(stations), 1, figsize=(14, 4 * len(stations)), sharex=True)
    if len(stations) == 1:
        axes = [axes]

    for i, station in enumerate(stations):
        s = df[df['station_id'] == station].sort_values('timestamp')
        axes[i].plot(s['timestamp'], s['voltage_pu'],
                     color=colors[i % len(colors)], linewidth=0.6, label=station)

        sf = faults_df[faults_df['station_id'] == station]
        if not sf.empty:
            axes[i].scatter(sf['timestamp'], sf['voltage_pu'],
                            color='red', zorder=5, s=25, label='Fault detected')

        axes[i].axhline(0.95, color='orange', linestyle='--', lw=1.2, alpha=0.8, label='Limits (0.95 / 1.05 pu)')
        axes[i].axhline(1.05, color='orange', linestyle='--', lw=1.2, alpha=0.8)
        axes[i].set_ylabel('Voltage (pu)')
        axes[i].set_title(station)
        axes[i].legend(loc='upper right', fontsize=8)
        axes[i].grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time')
    plt.suptitle('Voltage Time Series with Fault Detection', fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_grid_health_scores(health_scores):
    """
    Plot a color-coded bar chart of the composite grid health score per station.

    Green ≥ 90, Orange ≥ 75, Red < 75.

    Parameters:
    health_scores (dict): Output of analysis.calculate_grid_health_score().

    Returns:
    matplotlib.figure.Figure
    """
    stations = list(health_scores.keys())
    scores   = list(health_scores.values())
    colors   = ['seagreen' if s >= 90 else 'darkorange' if s >= 75 else 'crimson'
                for s in scores]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(stations, scores, color=colors, alpha=0.88, edgecolor='white', linewidth=1.5)

    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{score:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=13)

    ax.set_ylim(0, 108)
    ax.set_ylabel('Health Score (0–100)')
    ax.set_title('Composite Grid Health Score by Substation')
    ax.axhline(90, color='seagreen',   linestyle='--', lw=1.5, label='Good (≥ 90)')
    ax.axhline(75, color='darkorange', linestyle='--', lw=1.5, label='Fair (≥ 75)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    return fig
