"""
data_pipeline.py
Core data processing and analytics module for CNC Machine Monitoring & Anomaly Detection.
Reproduces and consolidates the analytics logic from code1.ipynb, code2.ipynb, and code3.ipynb.
"""

import os
import glob
import numpy as np
import pandas as pd


def get_existing_file(candidates):
    for f in candidates:
        if os.path.exists(f):
            return f
    # Also check case-insensitive match or wildcard
    for f in candidates:
        matches = glob.glob(f.replace('(', '*').replace(')', '*'))
        if matches:
            return matches[0]
    return candidates[0]


def load_raw_data():
    path = get_existing_file([
        "CNC_Task4_Complete_Data (1).csv",
        "CNC_Task4_Complete_Data.csv",
        "CNC_Task3_Processed_Data (1).csv",
        "initial data.csv"
    ])
    df = pd.read_csv(path)
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values(["Machine_ID", "Timestamp"]).reset_index(drop=True)
    return df


def compute_kpi_summary(df):
    """
    Computes all Task 4 KPIs across all machines:
    - Status counts & distribution
    - Utilization %, Running Hours
    - Total Production, Production Rate (parts/hr)
    - Total Rejected Parts, Rejection Rate (%)
    - Total Downtime (hours)
    - Average Temperature, Average Vibration, Maximum Motor Current, Average Cycle Time
    - Operational Variability (CV %)
    """
    total_obs = df.groupby('Machine_ID').size()
    running_obs = df[df['Machine_Status'] == 'Running'].groupby('Machine_ID').size().reindex(total_obs.index, fill_value=0)
    
    utilization = (running_obs / total_obs * 100).round(2)
    running_hours = (running_obs * 5 / 60).round(2) # 5 min sampling
    
    total_prod = df.groupby('Machine_ID')['Production_Count'].sum()
    prod_rate = (total_prod / running_hours).round(2)
    
    total_rejected = df.groupby('Machine_ID')['Rejected_Parts'].sum()
    rejection_rate = ((total_rejected / total_prod) * 100).round(2)
    
    downtime_statuses = ['Stopped', 'Fault', 'Maintenance']
    downtime_obs = df[df['Machine_Status'].isin(downtime_statuses)].groupby('Machine_ID').size()
    downtime_hours = (downtime_obs * 5 / 60).reindex(total_obs.index, fill_value=0).round(2)
    
    avg_temp = df.groupby('Machine_ID')['Temperature'].mean().round(2)
    avg_vib = df.groupby('Machine_ID')['Vibration'].mean().round(2)
    max_curr = df.groupby('Machine_ID')['Motor_Current'].max().round(2)
    avg_cycle = df.groupby('Machine_ID')['Cycle_Time'].mean().round(2)
    
    # Variability (CV %) across 6 parameters
    var_cols = ['Temperature', 'Vibration', 'Motor_Current', 'Spindle_RPM', 'Pressure', 'Cycle_Time']
    valid_var_cols = [c for c in var_cols if c in df.columns]
    cv_table = df.groupby('Machine_ID')[valid_var_cols].agg(lambda x: (x.std() / x.mean()) * 100 if x.mean() != 0 else 0)
    overall_variability = cv_table.mean(axis=1).round(2)
    
    # Injected abnormal (Task 4 statistical 95th percentile rule)
    if 'Injected_Abnormal' in df.columns:
        abnormal_obs = df.groupby('Machine_ID')['Injected_Abnormal'].sum()
    else:
        limits = {col: df[col].quantile(0.95) for col in valid_var_cols}
        injected = sum(df[col] > limits[col] for col in ['Temperature', 'Vibration', 'Motor_Current'] if col in limits) >= 2
        abnormal_obs = injected.groupby(df['Machine_ID']).sum()
        
    summary_df = pd.DataFrame({
        'Average_Temperature': avg_temp,
        'Average_Vibration': avg_vib,
        'Maximum_Motor_Current': max_curr,
        'Average_Cycle_Time': avg_cycle,
        'Utilization_%': utilization,
        'Running_Hours': running_hours,
        'Total_Production': total_prod,
        'Total_Rejected': total_rejected,
        'Production_Rate': prod_rate,
        'Rejection_Rate_%': rejection_rate,
        'Downtime_Hours': downtime_hours,
        'Abnormal_Observations': abnormal_obs,
        'Operational_Variability_%': overall_variability
    })
    
    return summary_df


def compute_task5_alarms(df, k_rule=2, z_limit=2, ma_window=5, ma_k=2, persist=3):
    """
    Task 5 multi-algorithm anomaly and alarm detection across all machines:
    - Rule-based thresholding (mean + k*std)
    - Z-score thresholding (> z_limit)
    - Moving average deviation (> ma_k)
    - Consensus voting (at least 2 agree)
    - Persistence filter (at least persist consecutive samples)
    """
    sensors = ["Temperature", "Vibration", "Motor_Current"]
    valid_sensors = [s for s in sensors if s in df.columns]
    
    running = df[df["Machine_Status"] == "Running"]
    baseline = running.groupby("Machine_ID")[valid_sensors].agg(["mean", "std"])
    
    results = []
    for m, g in df.groupby("Machine_ID", sort=False):
        g = g.copy()
        is_run = g["Machine_Status"] == "Running"
        
        for s in valid_sensors:
            mean = baseline.loc[m, (s, "mean")]
            std = baseline.loc[m, (s, "std")]
            if pd.isna(std) or std == 0:
                std = np.nan
                
            g["thr_" + s] = mean + k_rule * std
            g["z_" + s] = (g[s] - mean) / std
            g["ma_" + s] = g[s].where(is_run).rolling(ma_window, min_periods=3).mean()
            g["ma_dev_" + s] = (g["ma_" + s] - mean) / std
            
        # 1. Rule Flag: All available sensors exceed threshold
        rule_conditions = [(g[s] > g["thr_" + s]) for s in valid_sensors]
        if rule_conditions:
            rule_eval = rule_conditions[0]
            for c in rule_conditions[1:]:
                rule_eval = rule_eval & c
            g["Rule_Flag"] = is_run & rule_eval
        else:
            g["Rule_Flag"] = False
            
        # 2. Z Flag: At least 2 sensors exceed Z limit
        z_cols = ["z_" + s for s in valid_sensors]
        n_z = (g[z_cols] > z_limit).sum(axis=1)
        g["Z_Flag"] = is_run & (n_z >= 2)
        
        # 3. MA Flag: At least 2 sensors exceed MA dev threshold
        ma_dev_cols = ["ma_dev_" + s for s in valid_sensors]
        n_ma = (g[ma_dev_cols] > ma_k).sum(axis=1)
        g["MA_Flag"] = is_run & (n_ma >= 2)
        
        # Consensus: at least 2 methods agree
        g["Combined_Flag"] = (g[["Rule_Flag", "Z_Flag", "MA_Flag"]].sum(axis=1) >= 2)
        
        # Persistence: consecutive samples
        g["Alarm"] = (
            g["Combined_Flag"].astype(int)
            .rolling(persist, min_periods=persist).sum() >= persist
        )
        
        results.append(g)
        
    full_df = pd.concat(results).sort_index()
    
    # Group consecutive alarms into distinct events
    events = []
    for m, g in full_df.groupby("Machine_ID"):
        g = g.reset_index(drop=True)
        if not g["Alarm"].any():
            continue
        group_id = (g["Alarm"] != g["Alarm"].shift()).cumsum()
        for _, block in g[g["Alarm"]].groupby(group_id[g["Alarm"]]):
            events.append({
                "Machine_ID": m,
                "Start": block["Timestamp"].iloc[0],
                "End": block["Timestamp"].iloc[-1],
                "Duration_Minutes": (block["Timestamp"].iloc[-1] - block["Timestamp"].iloc[0]).total_seconds() / 60 + 5,
                "Samples": len(block),
                "Max_Vibration": block["Vibration"].max() if "Vibration" in block.columns else np.nan,
                "Max_Temperature": block["Temperature"].max() if "Temperature" in block.columns else np.nan,
                "Max_Current": block["Motor_Current"].max() if "Motor_Current" in block.columns else np.nan
            })
            
    events_df = pd.DataFrame(events)
    if events_df.empty:
        events_df = pd.DataFrame(columns=[
            "Machine_ID", "Start", "End", "Duration_Minutes", "Samples",
            "Max_Vibration", "Max_Temperature", "Max_Current"
        ])
        
    return full_df, baseline, events_df


def generate_simulated_reading(machine_id, baseline, prev_reading=None, inject_anomaly=False):
    """
    Generates a realistic synthetic sensor reading for real-time simulation.
    """
    s_mean = baseline.xs("mean", level=1, axis=1).loc[machine_id]
    s_std = baseline.xs("std", level=1, axis=1).loc[machine_id]
    
    # Base nominal values
    temp_base = s_mean.get("Temperature", 48.0)
    temp_std = s_std.get("Temperature", 2.6)
    
    vib_base = s_mean.get("Vibration", 1.4)
    vib_std = s_std.get("Vibration", 0.15)
    
    curr_base = s_mean.get("Motor_Current", 11.0)
    curr_std = s_std.get("Motor_Current", 0.55)
    
    if inject_anomaly:
        temp = np.random.normal(temp_base + 3.0 * temp_std, temp_std * 0.8)
        vib = np.random.normal(vib_base + 5.0 * vib_std, vib_std * 1.5)
        curr = np.random.normal(curr_base + 3.5 * curr_std, curr_std * 0.8)
        status = "Running"
        cycle_time = np.random.normal(55.0, 3.0)
    else:
        # 85% Running, 10% Idle, 5% Maintenance
        rand_status = np.random.random()
        if rand_status < 0.82:
            status = "Running"
            temp = np.random.normal(temp_base, temp_std * 0.6)
            vib = np.random.normal(vib_base, vib_std * 0.6)
            curr = np.random.normal(curr_base, curr_std * 0.6)
            cycle_time = max(20.0, np.random.normal(42.0, 4.0))
        elif rand_status < 0.94:
            status = "Idle"
            temp = max(25.0, temp_base - 10.0 + np.random.normal(0, 1.0))
            vib = max(0.1, np.random.normal(0.2, 0.05))
            curr = max(0.5, np.random.normal(2.0, 0.3))
            cycle_time = np.nan
        else:
            status = "Maintenance"
            temp = max(22.0, np.random.normal(25.0, 1.0))
            vib = max(0.05, np.random.normal(0.1, 0.02))
            curr = 0.0
            cycle_time = np.nan
            
    reading = {
        "Machine_ID": machine_id,
        "Machine_Status": status,
        "Temperature": round(float(temp), 2),
        "Vibration": round(float(vib), 3),
        "Motor_Current": round(float(curr), 2),
        "Spindle_RPM": round(float(np.random.normal(2800, 150)) if status == "Running" else 0.0, 1),
        "Pressure": round(float(np.random.normal(6.0, 0.4)), 2),
        "Cycle_Time": round(float(cycle_time), 2) if not pd.isna(cycle_time) else np.nan,
        "Production_Count": int(np.random.choice([0, 1, 2, 3], p=[0.1, 0.7, 0.15, 0.05])) if status == "Running" else 0,
        "Rejected_Parts": int(np.random.choice([0, 1], p=[0.97, 0.03])) if status == "Running" else 0
    }
    return reading
