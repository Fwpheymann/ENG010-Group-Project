import pandas as pd
import numpy as np

# Grid standard constants (IEEE 1159 / ANSI C84.1)
VOLTAGE_LOWER_LIMIT = 0.95   # per unit
VOLTAGE_UPPER_LIMIT = 1.05   # per unit
VOLTAGE_SAG_THRESHOLD = 0.90 # per unit — below this is a fault-level sag
MIN_POWER_FACTOR = 0.90
NOMINAL_VOLTAGE_PU = 1.0


def load_data(filepath):
    """
    Load and validate power system data from a CSV file.

    Parameters:
    filepath (str): Path to the CSV file.

    Returns:
    pandas.DataFrame or None: Loaded DataFrame, or None if loading fails.
    """
    required_cols = [
        'timestamp', 'station_id', 'voltage_pu', 'current_pu',
        'real_power_mw', 'reactive_power_mvar', 'power_factor'
    ]
    try:
        df = pd.read_csv(filepath, parse_dates=['timestamp'])
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        print(f"Loaded {len(df):,} records from '{filepath}'.")
        return df
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def calculate_statistics(df, station_id=None):
    """
    Calculate mean, median, and standard deviation for each measurement.

    Parameters:
    df (pandas.DataFrame): Power system data.
    station_id (str, optional): Filter to a single station. None = all stations.

    Returns:
    dict: Nested dict of {column: {mean, median, std, min, max}}.
    """
    data = df[df['station_id'] == station_id] if station_id else df
    numeric_cols = [
        'voltage_pu', 'current_pu', 'real_power_mw',
        'reactive_power_mvar', 'power_factor'
    ]
    stats = {}
    for col in numeric_cols:
        stats[col] = {
            'mean':   round(data[col].mean(), 4),
            'median': round(data[col].median(), 4),
            'std':    round(data[col].std(), 4),
            'min':    round(data[col].min(), 4),
            'max':    round(data[col].max(), 4),
        }
    return stats


def identify_load_patterns(df):
    """
    Identify average daily and seasonal load patterns across all stations.

    Parameters:
    df (pandas.DataFrame): Power system data.

    Returns:
    dict: {'daily': Series by hour, 'seasonal': Series by month}
    """
    df = df.copy()
    df['hour'] = df['timestamp'].dt.hour
    df['month'] = df['timestamp'].dt.month

    daily_pattern    = df.groupby('hour')['real_power_mw'].mean()
    seasonal_pattern = df.groupby('month')['real_power_mw'].mean()

    return {'daily': daily_pattern, 'seasonal': seasonal_pattern}


def check_grid_standards(df):
    """
    Compare measurements against IEEE grid standards and return violations.

    Parameters:
    df (pandas.DataFrame): Power system data.

    Returns:
    pandas.DataFrame: Rows describing each violation, or empty DataFrame.
    """
    violations = []
    stations = df['station_id'].unique().tolist()

    # while loop — iterate through each station
    i = 0
    while i < len(stations):
        station = stations[i]
        station_data = df[df['station_id'] == station]

        for _, row in station_data.iterrows():
            if row['voltage_pu'] < VOLTAGE_SAG_THRESHOLD:
                violations.append({
                    'timestamp':  row['timestamp'],
                    'station_id': station,
                    'type':       'Voltage Sag (Critical)',
                    'value':      round(row['voltage_pu'], 4),
                    'limit':      VOLTAGE_SAG_THRESHOLD,
                    'severity':   'CRITICAL',
                })
            elif row['voltage_pu'] < VOLTAGE_LOWER_LIMIT:
                violations.append({
                    'timestamp':  row['timestamp'],
                    'station_id': station,
                    'type':       'Under-Voltage',
                    'value':      round(row['voltage_pu'], 4),
                    'limit':      VOLTAGE_LOWER_LIMIT,
                    'severity':   'WARNING',
                })
            elif row['voltage_pu'] > VOLTAGE_UPPER_LIMIT:
                violations.append({
                    'timestamp':  row['timestamp'],
                    'station_id': station,
                    'type':       'Over-Voltage',
                    'value':      round(row['voltage_pu'], 4),
                    'limit':      VOLTAGE_UPPER_LIMIT,
                    'severity':   'WARNING',
                })

            if row['power_factor'] < MIN_POWER_FACTOR:
                violations.append({
                    'timestamp':  row['timestamp'],
                    'station_id': station,
                    'type':       'Low Power Factor',
                    'value':      round(row['power_factor'], 4),
                    'limit':      MIN_POWER_FACTOR,
                    'severity':   'WARNING',
                })
        i += 1

    return pd.DataFrame(violations) if violations else pd.DataFrame()


def calculate_power_quality_indices(df):
    """
    Calculate power quality indices (voltage compliance, PF compliance, load factor).

    Parameters:
    df (pandas.DataFrame): Power system data.

    Returns:
    dict: {station_id: {index_name: value, ...}}
    """
    results = {}
    for station_id in df['station_id'].unique():
        s = df[df['station_id'] == station_id]

        v_compliant = (
            (s['voltage_pu'] >= VOLTAGE_LOWER_LIMIT) &
            (s['voltage_pu'] <= VOLTAGE_UPPER_LIMIT)
        ).mean() * 100

        pf_compliant = (s['power_factor'] >= MIN_POWER_FACTOR).mean() * 100

        avg_load  = s['real_power_mw'].mean()
        peak_load = s['real_power_mw'].max()
        load_factor = avg_load / peak_load if peak_load > 0 else 0

        mean_v_dev = np.abs(s['voltage_pu'] - NOMINAL_VOLTAGE_PU).mean()

        results[station_id] = {
            'voltage_compliance_pct':    round(v_compliant, 2),
            'pf_compliance_pct':         round(pf_compliant, 2),
            'load_factor':               round(load_factor, 4),
            'mean_voltage_deviation_pu': round(mean_v_dev, 4),
        }
    return results


def detect_faults(df):
    """
    Detect voltage anomalies using a 24-hour rolling z-score (>3σ = fault).

    Parameters:
    df (pandas.DataFrame): Power system data.

    Returns:
    pandas.DataFrame: Detected fault events, or empty DataFrame.
    """
    faults = []
    for station_id in df['station_id'].unique():
        s = df[df['station_id'] == station_id].sort_values('timestamp').copy()

        roll_mean = s['voltage_pu'].rolling(window=24, center=True, min_periods=1).mean()
        roll_std  = s['voltage_pu'].rolling(window=24, center=True, min_periods=1).std()

        z_scores = np.abs((s['voltage_pu'] - roll_mean) / roll_std.replace(0, np.nan))
        anomalies = s[z_scores > 3]

        for _, row in anomalies.iterrows():
            if row['voltage_pu'] < VOLTAGE_SAG_THRESHOLD:
                fault_type = 'Voltage Sag'
            elif row['voltage_pu'] > VOLTAGE_UPPER_LIMIT:
                fault_type = 'Voltage Swell'
            else:
                fault_type = 'Voltage Anomaly'

            faults.append({
                'timestamp':     row['timestamp'],
                'station_id':    station_id,
                'fault_type':    fault_type,
                'voltage_pu':    round(row['voltage_pu'], 4),
                'real_power_mw': round(row['real_power_mw'], 2),
            })

    return pd.DataFrame(faults) if faults else pd.DataFrame()


def calculate_grid_health_score(df):
    """
    Compute a composite grid health score (0–100) per station.

    Weights: 40% voltage compliance, 40% PF compliance, 20% load factor.

    Parameters:
    df (pandas.DataFrame): Power system data.

    Returns:
    dict: {station_id: health_score}
    """
    pqi = calculate_power_quality_indices(df)
    scores = {}
    for station_id, idx in pqi.items():
        score = (
            0.40 * idx['voltage_compliance_pct'] +
            0.40 * idx['pf_compliance_pct'] +
            0.20 * min(idx['load_factor'] * 100, 100)
        )
        scores[station_id] = round(score, 2)
    return scores
