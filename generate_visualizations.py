"""
generate_visualizations.py
Generates publication-quality charts and an interactive HTML report
for all 8 deliverables:
1. Machine Operating Status
2. Temperature and Vibration Trends
3. Production Counts & Production Rates
4. Machine Comparison (Multi-metric & Radar)
5. Utilization Comparison & Downtime Breakdown
6. Rejection-Rate Comparison
7. Detected Abnormal Periods
8. Alarm Summary
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import data_pipeline as dp

# Set aesthetic style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 14

OUT_DIR = "visualizations"
os.makedirs(OUT_DIR, exist_ok=True)


def generate_all():
    print("Loading data and computing analytics...")
    df = dp.load_raw_data()
    kpi_summary = dp.compute_kpi_summary(df)
    full_task5, baseline, events_df = dp.compute_task5_alarms(df)
    
    # -------------------------------------------------------------------------
    # 1. Machine Operating Status
    # -------------------------------------------------------------------------
    print("1. Generating Machine Operating Status visualization...")
    status_pct = pd.crosstab(df['Machine_ID'], df['Machine_Status'], normalize='index') * 100
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    palette = sns.color_palette("tab10", n_colors=status_pct.shape[1])
    status_pct.plot(kind='bar', stacked=True, ax=ax, color=palette, edgecolor='black', linewidth=0.5)
    ax.set_title("Machine Operating Status Distribution (% of Time)")
    ax.set_xlabel("CNC Machine ID")
    ax.set_ylabel("Percentage of Total Time (%)")
    ax.set_ylim(0, 100)
    plt.xticks(rotation=0)
    ax.legend(title="Operating Status", bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    fig_path1 = os.path.join(OUT_DIR, "1_machine_operating_status.png")
    fig.savefig(fig_path1)
    plt.close()
    
    # -------------------------------------------------------------------------
    # 2. Temperature and Vibration Trends
    # -------------------------------------------------------------------------
    print("2. Generating Temperature & Vibration Trends...")
    # Plot CNC03 (the machine with the detected anomaly)
    cnc03 = full_task5[full_task5['Machine_ID'] == 'CNC03'].copy()
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True, dpi=150)
    # Temperature
    axes[0].plot(cnc03['Timestamp'], cnc03['Temperature'], color='#1f77b4', alpha=0.6, label='Raw Temperature (°C)')
    axes[0].plot(cnc03['Timestamp'], cnc03['ma_Temperature'], color='#ff7f0e', linewidth=1.8, label='5-Sample Moving Avg')
    axes[0].axhline(cnc03['thr_Temperature'].iloc[0], color='red', linestyle='--', linewidth=1.5, label='Threshold (Mean + 2σ)')
    axes[0].fill_between(cnc03['Timestamp'], cnc03['Temperature'].min(), cnc03['Temperature'].max(),
                         where=cnc03['Alarm'].to_numpy(), color='red', alpha=0.3, label='Active Alarm Region')
    axes[0].set_ylabel("Temperature (°C)")
    axes[0].set_title("CNC03: Temperature Trend & Multi-Sensor Alarm Window")
    axes[0].legend(loc='upper right', frameon=True)
    
    # Vibration
    axes[1].plot(cnc03['Timestamp'], cnc03['Vibration'], color='#2ca02c', alpha=0.6, label='Raw Vibration')
    axes[1].plot(cnc03['Timestamp'], cnc03['ma_Vibration'], color='#9467bd', linewidth=1.8, label='5-Sample Moving Avg')
    axes[1].axhline(cnc03['thr_Vibration'].iloc[0], color='red', linestyle='--', linewidth=1.5, label='Threshold (Mean + 2σ)')
    axes[1].fill_between(cnc03['Timestamp'], cnc03['Vibration'].min(), cnc03['Vibration'].max(),
                         where=cnc03['Alarm'].to_numpy(), color='red', alpha=0.3, label='Active Alarm Region')
    axes[1].set_ylabel("Vibration")
    axes[1].set_xlabel("Timestamp")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    axes[1].set_title("CNC03: Vibration Trend & Multi-Sensor Alarm Window")
    axes[1].legend(loc='upper right', frameon=True)
    
    plt.xticks(rotation=20)
    plt.tight_layout()
    fig_path2 = os.path.join(OUT_DIR, "2_temperature_vibration_trends.png")
    fig.savefig(fig_path2)
    plt.close()

    # -------------------------------------------------------------------------
    # 3. Production Counts & Production Rates
    # -------------------------------------------------------------------------
    print("3. Generating Production Counts & Rates...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    
    machines = kpi_summary.index
    total_prod = kpi_summary['Total_Production']
    prod_rate = kpi_summary['Production_Rate']
    good_parts = total_prod - kpi_summary['Total_Rejected']
    rejected_parts = kpi_summary['Total_Rejected']
    
    # Stacked good vs rejected parts
    ax1.bar(machines, good_parts, label='Good Parts', color='#2ca02c', edgecolor='black', linewidth=0.5)
    ax1.bar(machines, rejected_parts, bottom=good_parts, label='Rejected Parts', color='#d62728', edgecolor='black', linewidth=0.5)
    ax1.set_title("Total Production Output (Good vs Rejected)")
    ax1.set_ylabel("Total Parts")
    ax1.set_xlabel("Machine ID")
    for i, (m, tot) in enumerate(zip(machines, total_prod)):
        ax1.text(i, tot + 150, f"{tot:,}", ha='center', va='bottom', fontweight='bold', fontsize=9)
    ax1.legend()
    
    # Production rate
    bars = ax2.bar(machines, prod_rate, color='#3470a3', edgecolor='black', linewidth=0.5)
    ax2.set_title("Production Rate (Parts per Running Hour)")
    ax2.set_ylabel("Parts / Hour")
    ax2.set_xlabel("Machine ID")
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1, f"{height:.1f}", ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    fig_path3 = os.path.join(OUT_DIR, "3_production_counts_and_rate.png")
    fig.savefig(fig_path3)
    plt.close()

    # -------------------------------------------------------------------------
    # 4. Machine Comparison (Radar & Normalized Profile)
    # -------------------------------------------------------------------------
    print("4. Generating Machine Comparison Matrix & Radar...")
    metrics_to_compare = [
        'Average_Temperature', 'Average_Vibration', 'Maximum_Motor_Current',
        'Average_Cycle_Time', 'Utilization_%', 'Production_Rate', 'Operational_Variability_%'
    ]
    
    # Normalized 0-100% scale for fair comparison
    comp_df = kpi_summary[metrics_to_compare].copy()
    norm_comp = (comp_df - comp_df.min()) / (comp_df.max() - comp_df.min() + 1e-6) * 100
    
    fig = plt.figure(figsize=(10, 8), dpi=150)
    ax = fig.add_subplot(111, polar=True)
    
    categories = [
        'Avg Temp', 'Avg Vib', 'Max Current',
        'Cycle Time', 'Utilization', 'Prod Rate', 'Variability'
    ]
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for idx, (m_id, row) in enumerate(norm_comp.iterrows()):
        values = row.tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=m_id, color=colors[idx])
        ax.fill(angles, values, color=colors[idx], alpha=0.1)
        
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontweight='bold', fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_title("Machine Benchmark Comparison Profile (Normalized % Scale)", pad=20, fontweight='bold')
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1))
    plt.tight_layout()
    fig_path4 = os.path.join(OUT_DIR, "4_machine_comparison_radar.png")
    fig.savefig(fig_path4)
    plt.close()

    # -------------------------------------------------------------------------
    # 5. Utilization Comparison & Downtime Breakdown
    # -------------------------------------------------------------------------
    print("5. Generating Utilization & Downtime Comparison...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    
    # Utilization %
    bars = ax1.bar(machines, kpi_summary['Utilization_%'], color='#2ca02c', edgecolor='black', linewidth=0.5)
    ax1.axhline(80, color='red', linestyle='--', linewidth=1.5, label='Target Benchmark (80%)')
    ax1.set_ylim(0, 100)
    ax1.set_title("Machine Utilization (%)")
    ax1.set_ylabel("Utilization (%)")
    ax1.set_xlabel("Machine ID")
    for bar in bars:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., h + 1, f"{h:.2f}%", ha='center', va='bottom', fontweight='bold', fontsize=9)
    ax1.legend(loc='lower right')
    
    # Running vs Downtime Hours
    w = 0.35
    x = np.arange(len(machines))
    ax2.bar(x - w/2, kpi_summary['Running_Hours'], w, label='Running Hours', color='#1f77b4', edgecolor='black', linewidth=0.5)
    ax2.bar(x + w/2, kpi_summary['Downtime_Hours'], w, label='Downtime Hours', color='#e377c2', edgecolor='black', linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(machines)
    ax2.set_title("Operating Hours vs Downtime Hours")
    ax2.set_ylabel("Hours")
    ax2.set_xlabel("Machine ID")
    ax2.legend()
    
    plt.tight_layout()
    fig_path5 = os.path.join(OUT_DIR, "5_utilization_downtime_comparison.png")
    fig.savefig(fig_path5)
    plt.close()

    # -------------------------------------------------------------------------
    # 6. Rejection-Rate Comparison
    # -------------------------------------------------------------------------
    print("6. Generating Rejection-Rate Comparison...")
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    rej_rates = kpi_summary['Rejection_Rate_%']
    bar_colors = ['#d62728' if r == rej_rates.max() else '#3470a3' for r in rej_rates]
    bars = ax.bar(machines, rej_rates, color=bar_colors, edgecolor='black', linewidth=0.5)
    ax.axhline(rej_rates.mean(), color='orange', linestyle='--', linewidth=1.5, label=f'Fleet Mean ({rej_rates.mean():.2f}%)')
    ax.set_title("Rejection Rate by CNC Machine (%)")
    ax.set_ylabel("Rejection Rate (%)")
    ax.set_xlabel("Machine ID")
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.05, f"{h:.2f}%", ha='center', va='bottom', fontweight='bold', fontsize=9)
    ax.legend()
    plt.tight_layout()
    fig_path6 = os.path.join(OUT_DIR, "6_rejection_rate_comparison.png")
    fig.savefig(fig_path6)
    plt.close()

    # -------------------------------------------------------------------------
    # 7. Detected Abnormal Periods (Task 5 CNC03 Zoom & Injected Comparison)
    # -------------------------------------------------------------------------
    print("7. Generating Detected Abnormal Periods visualization...")
    alarm_times = cnc03.loc[cnc03['Alarm'], 'Timestamp']
    t1 = alarm_times.min() - pd.Timedelta(hours=3)
    t2 = alarm_times.max() + pd.Timedelta(hours=3)
    zoom_cnc03 = cnc03[(cnc03['Timestamp'] >= t1) & (cnc03['Timestamp'] <= t2)]
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 9), sharex=True, dpi=150)
    
    # Vibration
    ax1.plot(zoom_cnc03['Timestamp'], zoom_cnc03['Vibration'], label='Vibration', color='#2ca02c')
    ax1.plot(zoom_cnc03['Timestamp'], zoom_cnc03['ma_Vibration'], 'g--', label='Moving Avg (w=5)')
    ax1.plot(zoom_cnc03['Timestamp'], zoom_cnc03['thr_Vibration'], 'k:', label='Threshold (μ+2σ)')
    ax1.fill_between(zoom_cnc03['Timestamp'], 0, 1, where=zoom_cnc03['Alarm'].to_numpy(),
                     transform=ax1.get_xaxis_transform(), color='red', alpha=0.3, label='Persisted Alarm Window')
    ax1.set_ylabel("Vibration")
    ax1.set_title("Detected Abnormal Period Detail: CNC03 (2026-01-05 13:45 to 14:25)")
    ax1.legend(loc='upper left', fontsize=8)
    
    # Temperature
    ax2.plot(zoom_cnc03['Timestamp'], zoom_cnc03['Temperature'], label='Temperature (°C)', color='#ff7f0e')
    ax2.plot(zoom_cnc03['Timestamp'], zoom_cnc03['ma_Temperature'], color='brown', linestyle='--', label='Moving Avg')
    ax2.plot(zoom_cnc03['Timestamp'], zoom_cnc03['thr_Temperature'], 'k:', label='Threshold (μ+2σ)')
    ax2.fill_between(zoom_cnc03['Timestamp'], 0, 1, where=zoom_cnc03['Alarm'].to_numpy(),
                     transform=ax2.get_xaxis_transform(), color='red', alpha=0.3)
    ax2.set_ylabel("Temperature (°C)")
    ax2.legend(loc='upper left', fontsize=8)
    
    # Motor Current
    ax3.plot(zoom_cnc03['Timestamp'], zoom_cnc03['Motor_Current'], label='Motor Current (A)', color='#1f77b4')
    ax3.plot(zoom_cnc03['Timestamp'], zoom_cnc03['ma_Motor_Current'], color='blue', linestyle='--', label='Moving Avg')
    ax3.plot(zoom_cnc03['Timestamp'], zoom_cnc03['thr_Motor_Current'], 'k:', label='Threshold (μ+2σ)')
    ax3.fill_between(zoom_cnc03['Timestamp'], 0, 1, where=zoom_cnc03['Alarm'].to_numpy(),
                     transform=ax3.get_xaxis_transform(), color='red', alpha=0.3)
    ax3.set_ylabel("Current (A)")
    ax3.set_xlabel("Timestamp")
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax3.legend(loc='upper left', fontsize=8)
    
    plt.tight_layout()
    fig_path7 = os.path.join(OUT_DIR, "7_detected_abnormal_period_detail.png")
    fig.savefig(fig_path7)
    plt.close()

    # -------------------------------------------------------------------------
    # 8. Alarm Summary
    # -------------------------------------------------------------------------
    print("8. Generating Alarm Summary visualization...")
    alarm_summary_data = {
        'Machine_ID': machines,
        'Rule_Flag_Counts': full_task5.groupby('Machine_ID')['Rule_Flag'].sum(),
        'Z_Flag_Counts': full_task5.groupby('Machine_ID')['Z_Flag'].sum(),
        'MA_Flag_Counts': full_task5.groupby('Machine_ID')['MA_Flag'].sum(),
        'Consensus_Flags': full_task5.groupby('Machine_ID')['Combined_Flag'].sum(),
        'Persisted_Alarm_Events': [len(events_df[events_df['Machine_ID'] == m]) for m in machines]
    }
    alarm_table = pd.DataFrame(alarm_summary_data).set_index('Machine_ID')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    
    # Diagnostic flags breakdown
    w = 0.2
    x = np.arange(len(machines))
    ax1.bar(x - 1.5*w, alarm_table['Rule_Flag_Counts'], w, label='Rule Flags (All 3 Sensors)', color='#9467bd')
    ax1.bar(x - 0.5*w, alarm_table['Z_Flag_Counts'], w, label='Z-Score Flags (>=2 Sensors)', color='#ff7f0e')
    ax1.bar(x + 0.5*w, alarm_table['MA_Flag_Counts'], w, label='Moving Avg Flags (>=2 Sensors)', color='#1f77b4')
    ax1.bar(x + 1.5*w, alarm_table['Consensus_Flags'], w, label='Consensus Combined Flags', color='#d62728')
    ax1.set_xticks(x)
    ax1.set_xticklabels(machines)
    ax1.set_title("Anomaly Detection Diagnostic Trigger Counts")
    ax1.set_ylabel("Observation Count")
    ax1.set_xlabel("Machine ID")
    ax1.legend(fontsize=8)
    
    # Persisted alarm events per machine
    bars = ax2.bar(machines, alarm_table['Persisted_Alarm_Events'], color='tomato', edgecolor='black', linewidth=0.5)
    ax2.set_title("Persisted Alarm Events per Machine (Task 5)")
    ax2.set_ylabel("Number of Detected Events")
    ax2.set_xlabel("Machine ID")
    ax2.set_yticks(range(0, max(alarm_table['Persisted_Alarm_Events']) + 3))
    for bar in bars:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., h + 0.05, str(int(h)), ha='center', va='bottom', fontweight='bold', fontsize=10)
        
    plt.tight_layout()
    fig_path8 = os.path.join(OUT_DIR, "8_alarm_summary_diagnostics.png")
    fig.savefig(fig_path8)
    plt.close()

    # -------------------------------------------------------------------------
    # Generate Interactive HTML Report
    # -------------------------------------------------------------------------
    print("Generating Interactive HTML Report...")
    generate_html_report(df, kpi_summary, full_task5, events_df, alarm_table)
    
    print("All visualizations successfully created in:", OUT_DIR)


def generate_html_report(df, kpi, full_task5, events, alarm_table):
    # Plotly interactive overview
    fig_status = px.bar(
        pd.crosstab(df['Machine_ID'], df['Machine_Status'], normalize='index').reset_index().melt(id_vars='Machine_ID', value_name='Percentage', var_name='Status'),
        x='Machine_ID', y='Percentage', color='Status', title="Machine Operating Status Distribution",
        barmode='stack', height=400
    )
    
    fig_trend = px.line(
        full_task5[full_task5['Machine_ID'] == 'CNC03'],
        x='Timestamp', y=['Temperature', 'ma_Temperature', 'thr_Temperature'],
        title="CNC03 Temperature & Threshold Trend",
        labels={'value': 'Temperature (°C)', 'variable': 'Signal'},
        height=400
    )
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>CNC Fleet Analytics & Diagnostics Report</title>
        <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background: #f8fafc; color: #1e293b; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            h1 {{ color: #0f172a; border-bottom: 2px solid #cbd5e1; padding-bottom: 10px; }}
            h2 {{ color: #1e3a8a; margin-top: 30px; }}
            .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
            .card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); border-left: 4px solid #3b82f6; }}
            .card h3 {{ margin: 0; font-size: 13px; color: #64748b; text-transform: uppercase; }}
            .card .val {{ font-size: 26px; font-weight: bold; margin-top: 5px; color: #0f172a; }}
            table {{ width: 100%; border-collapse: collapse; background: white; margin: 20px 0; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
            th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #e2e8f0; font-size: 14px; }}
            th {{ background: #f1f5f9; font-weight: 600; color: #475569; }}
            tr:hover {{ background: #f8fafc; }}
            .badge-alert {{ background: #fee2e2; color: #991b1b; padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 12px; }}
            .badge-ok {{ background: #dcfce7; color: #166534; padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 12px; }}
            .img-container {{ text-align: center; margin: 25px 0; }}
            img {{ max-width: 100%; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.08); }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🏭 CNC Fleet Monitoring & Anomaly Detection Report</h1>
            <p>Comprehensive Diagnostic Analysis across CNC01, CNC02, CNC03, CNC04, and CNC05</p>
            
            <div class="kpi-grid">
                <div class="card"><h3>Total Machines</h3><div class="val">5 Active</div></div>
                <div class="card"><h3>Fleet Avg Utilization</h3><div class="val">{kpi['Utilization_%'].mean():.2f}%</div></div>
                <div class="card"><h3>Total Production</h3><div class="val">{kpi['Total_Production'].sum():,}</div></div>
                <div class="card"><h3>Fleet Rejection Rate</h3><div class="val">{(kpi['Total_Rejected'].sum() / kpi['Total_Production'].sum() * 100):.2f}%</div></div>
                <div class="card" style="border-left-color: #ef4444;"><h3>Critical Alarm Events</h3><div class="val">{len(events)} Event</div></div>
            </div>

            <h2>1. Machine Operating Status & Fleet Comparison</h2>
            {kpi[['Utilization_%', 'Running_Hours', 'Downtime_Hours', 'Total_Production', 'Production_Rate', 'Rejection_Rate_%', 'Average_Temperature', 'Average_Vibration', 'Operational_Variability_%']].to_html(classes="table")}

            <div class="img-container">
                <img src="1_machine_operating_status.png" alt="Operating Status Distribution">
            </div>

            <h2>2. Temperature & Vibration Trends</h2>
            <div class="img-container">
                <img src="2_temperature_vibration_trends.png" alt="Temperature & Vibration Trends">
            </div>

            <h2>3. Production Counts & Production Rates</h2>
            <div class="img-container">
                <img src="3_production_counts_and_rate.png" alt="Production Output and Rates">
            </div>

            <h2>4. Machine Comparison Profile (Radar Benchmark)</h2>
            <div class="img-container">
                <img src="4_machine_comparison_radar.png" alt="Machine Comparison Profile">
            </div>

            <h2>5. Machine Utilization & Downtime Breakdown</h2>
            <div class="img-container">
                <img src="5_utilization_downtime_comparison.png" alt="Utilization & Downtime Comparison">
            </div>

            <h2>6. Rejection-Rate & Quality Comparison</h2>
            <div class="img-container">
                <img src="6_rejection_rate_comparison.png" alt="Rejection-Rate Comparison">
            </div>

            <h2>7. Detected Abnormal Periods</h2>
            <p>Task 5 Multi-Sensor Consensus Alarm Detection Log:</p>
            {events.to_html(classes="table") if not events.empty else "<p>No persistent alarms detected.</p>"}
            <div class="img-container">
                <img src="7_detected_abnormal_period_detail.png" alt="Detected Abnormal Period Detail">
            </div>

            <h2>8. Alarm Summary & Diagnostics Trigger Breakdown</h2>
            {alarm_table.to_html(classes="table")}
            <div class="img-container">
                <img src="8_alarm_summary_diagnostics.png" alt="Alarm Summary Diagnostics">
            </div>
        </div>
    </body>
    </html>
    """
    with open(os.path.join(OUT_DIR, "dashboard_report.html"), "w", encoding="utf-8") as f:
        f.write(html_content)


if __name__ == "__main__":
    generate_all()
