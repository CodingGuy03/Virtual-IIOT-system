"""
app.py
Interactive CNC Fleet Monitoring & Anomaly Detection Dashboard.
Provides all 8 deliverables + dynamic real-time sensor telemetry simulation.
"""

import time
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import data_pipeline as dp

# Page Configuration
st.set_page_config(
    page_title="CNC Smart Fleet Diagnostics & Anomaly Dashboard",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern Industrial Dashboard
st.markdown("""
<style>
    /* Metric Card Styling */
    .metric-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 18px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.08), 0 2px 4px -1px rgba(0, 0, 0, 0.04);
        border-left: 5px solid #2563eb;
        margin-bottom: 12px;
    }
    .metric-card-danger {
        background-color: #fef2f2;
        border-radius: 10px;
        padding: 18px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.08);
        border-left: 5px solid #ef4444;
        margin-bottom: 12px;
    }
    .metric-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0f172a;
        margin-top: 4px;
    }
    .metric-subtitle {
        font-size: 0.8rem;
        color: #94a3b8;
        margin-top: 2px;
    }
    /* Status Badges */
    .badge-running {
        background-color: #dcfce7;
        color: #15803d;
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .badge-idle {
        background-color: #fef9c3;
        color: #854d0e;
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .badge-maint {
        background-color: #ffedd5;
        color: #c2410c;
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .badge-alarm {
        background-color: #fee2e2;
        color: #b91c1c;
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.8rem;
        animation: pulse 1.5s infinite;
    }
</style>
""", unsafe_allow_html=True)


# Data Caching
@st.cache_data
def load_all_data():
    df = dp.load_raw_data()
    kpi_summary = dp.compute_kpi_summary(df)
    full_task5, baseline, events_df = dp.compute_task5_alarms(df)
    return df, kpi_summary, full_task5, baseline, events_df


df, kpi_summary, full_task5, baseline, events_df = load_all_data()
all_machines = sorted(df['Machine_ID'].unique().tolist())

# Sidebar Navigation & Settings
st.sidebar.image("https://img.icons8.com/color/96/cnc-machine.png", width=64)
st.sidebar.title("Fleet Diagnostics")
st.sidebar.caption("Condition Monitoring & Anomaly Detection")

menu = st.sidebar.radio(
    "Navigation Menu",
    [
        "🎛️ Fleet Overview",
        "⚙️ Operating Status",
        "📈 Temp & Vibration Trends",
        "📦 Production Counts & Rate",
        "⚖️ Machine Comparison",
        "⏱️ Utilization & Downtime",
        "🎯 Rejection-Rate Comparison",
        "⚠️ Detected Abnormal Periods",
        "🚨 Alarm Summary & Tuning",
        "🔄 Real-Time Sensor Simulation"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔍 Global Machine Filter")
selected_machine = st.sidebar.selectbox("Filter Specific Machine", ["All Machines"] + all_machines)

# Session State for Real-Time Simulation
if "sim_history" not in st.session_state:
    st.session_state.sim_history = []
if "sim_active" not in st.session_state:
    st.session_state.sim_active = False
if "sim_machine" not in st.session_state:
    st.session_state.sim_machine = "CNC03"
if "sim_speed" not in st.session_state:
    st.session_state.sim_speed = 3
if "sim_force_anomaly" not in st.session_state:
    st.session_state.sim_force_anomaly = False


# =============================================================================
# 1. FLEET OVERVIEW
# =============================================================================
if menu == "🎛️ Fleet Overview":
    st.title("🏭 CNC Fleet Health & Performance Overview")
    st.markdown("Real-time telemetry aggregation and performance indicators across all CNC units.")

    # Top KPI Cards
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Monitored Fleet</div>
            <div class="metric-value">{len(all_machines)} Units</div>
            <div class="metric-subtitle">CNC01 to CNC05 Active</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        avg_util = kpi_summary['Utilization_%'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Avg Utilization</div>
            <div class="metric-value">{avg_util:.1f}%</div>
            <div class="metric-subtitle">Target: 80.0%</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        total_p = kpi_summary['Total_Production'].sum()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Production</div>
            <div class="metric-value">{total_p:,}</div>
            <div class="metric-subtitle">Total Parts Manufactured</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        fleet_rej = (kpi_summary['Total_Rejected'].sum() / total_p) * 100
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Fleet Rejection Rate</div>
            <div class="metric-value">{fleet_rej:.2f}%</div>
            <div class="metric-subtitle">Benchmark &lt; 2.5%</div>
        </div>
        """, unsafe_allow_html=True)
    with col5:
        critical_alarms = len(events_df)
        st.markdown(f"""
        <div class="metric-card-danger">
            <div class="metric-title">Detected Alarms</div>
            <div class="metric-value">{critical_alarms} Event</div>
            <div class="metric-subtitle">CNC03 Sustained Anomaly</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 📊 Fleet Executive Matrix")
    styled_kpi = kpi_summary[[
        'Utilization_%', 'Running_Hours', 'Downtime_Hours', 'Total_Production',
        'Production_Rate', 'Rejection_Rate_%', 'Average_Temperature',
        'Average_Vibration', 'Operational_Variability_%'
    ]].copy()
    
    st.dataframe(
        styled_kpi.style.highlight_max(axis=0, subset=['Utilization_%', 'Production_Rate'], color='#bbf7d0')
                        .highlight_max(axis=0, subset=['Rejection_Rate_%', 'Average_Temperature', 'Operational_Variability_%'], color='#fecaca')
                        .format("{:.2f}"),
        use_container_width=True
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Machine Status Breakdown (% Time)")
        status_pct = pd.crosstab(df['Machine_ID'], df['Machine_Status'], normalize='index') * 100
        fig_stat = px.bar(
            status_pct.reset_index().melt(id_vars='Machine_ID', var_name='Status', value_name='Percentage'),
            x='Machine_ID', y='Percentage', color='Status', barmode='stack',
            color_discrete_sequence=px.colors.qualitative.Safe,
            labels={'Percentage': 'Time (%)'}
        )
        fig_stat.update_layout(margin=dict(l=20, r=20, t=30, b=20), height=350)
        st.plotly_chart(fig_stat, use_container_width=True)

    with col_b:
        st.subheader("Production Output vs Quality Loss")
        fig_prod = go.Figure()
        fig_prod.add_trace(go.Bar(
            x=kpi_summary.index, y=kpi_summary['Total_Production'] - kpi_summary['Total_Rejected'],
            name='Good Parts', marker_color='#22c55e'
        ))
        fig_prod.add_trace(go.Bar(
            x=kpi_summary.index, y=kpi_summary['Total_Rejected'],
            name='Scrap Parts', marker_color='#ef4444'
        ))
        fig_prod.update_layout(barmode='stack', margin=dict(l=20, r=20, t=30, b=20), height=350)
        st.plotly_chart(fig_prod, use_container_width=True)


# =============================================================================
# 2. OPERATING STATUS
# =============================================================================
elif menu == "⚙️ Operating Status":
    st.title("⚙️ Machine Operating Status Analysis")
    st.markdown("Tracks time distribution across operational states (Running, Idle, Maintenance, Fault, Stopped, Starting, Warmup).")

    m_filter = all_machines if selected_machine == "All Machines" else [selected_machine]
    filtered_df = df[df['Machine_ID'].isin(m_filter)]

    # Crosstab counts and percentages
    status_counts = pd.crosstab(filtered_df['Machine_ID'], filtered_df['Machine_Status'])
    status_pct = pd.crosstab(filtered_df['Machine_ID'], filtered_df['Machine_Status'], normalize='index') * 100

    col1, col2 = st.columns([3, 2])
    with col1:
        st.subheader("Operational Status Distribution (% of Total Duration)")
        melted_pct = status_pct.reset_index().melt(id_vars='Machine_ID', var_name='Status', value_name='Percentage')
        fig = px.bar(
            melted_pct, x='Machine_ID', y='Percentage', color='Status',
            barmode='stack', text_auto='.1f',
            color_discrete_sequence=px.colors.qualitative.Prism
        )
        fig.update_layout(yaxis=dict(range=[0, 100], title="Percentage (%)"), height=420)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Exact Observation Counts")
        st.dataframe(status_counts, use_container_width=True)
        st.markdown("""
        **Operational Insights:**
        - **Running:** Primary active cutting/machining state.
        - **Idle:** Waiting for material/loading between cycles.
        - **Downtime Category:** Includes `Stopped`, `Fault`, and `Maintenance` states.
        """)

    st.subheader("📅 Operational State Timeline (Chronological State Sequence)")
    target_mach = st.selectbox("Select Machine for Timeline Inspection", m_filter)
    sample_timeline = filtered_df[filtered_df['Machine_ID'] == target_mach].tail(300)
    
    fig_time = px.scatter(
        sample_timeline, x='Timestamp', y='Machine_Status', color='Machine_Status',
        title=f"Recent 300 Telemetry Intervals State Transitions for {target_mach}",
        height=320, color_discrete_sequence=px.colors.qualitative.Dark24
    )
    fig_time.update_traces(marker=dict(size=8, symbol='square'))
    st.plotly_chart(fig_time, use_container_width=True)


# =============================================================================
# 3. TEMPERATURE & VIBRATION TRENDS
# =============================================================================
elif menu == "📈 Temp & Vibration Trends":
    st.title("📈 Temperature & Vibration Condition Trends")
    st.markdown("Continuous sensor time series with rolling averages, statistical upper thresholds, and detected anomaly highlights.")

    target_mach = selected_machine if selected_machine != "All Machines" else st.selectbox("Select Target Machine", all_machines, index=2)
    mach_data = full_task5[full_task5['Machine_ID'] == target_mach].copy()

    # Time Window Filter
    min_date = mach_data['Timestamp'].min().to_pydatetime()
    max_date = mach_data['Timestamp'].max().to_pydatetime()
    time_range = st.slider("Select Telemetry Time Window", min_value=min_date, max_value=max_date, value=(min_date, max_date))
    mach_data = mach_data[(mach_data['Timestamp'] >= time_range[0]) & (mach_data['Timestamp'] <= time_range[1])]

    # Metric Badges
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Avg Temperature", f"{mach_data['Temperature'].mean():.2f} °C", f"Max: {mach_data['Temperature'].max():.2f} °C")
    with c2:
        st.metric("Avg Vibration", f"{mach_data['Vibration'].mean():.2f}", f"Max: {mach_data['Vibration'].max():.2f}")
    with c3:
        st.metric("Avg Motor Current", f"{mach_data['Motor_Current'].mean():.2f} A", f"Max: {mach_data['Motor_Current'].max():.2f} A")
    with c4:
        active_alarms = mach_data['Alarm'].sum()
        st.metric("Alarm Sample Count", f"{active_alarms}", "Samples in Alarm State" if active_alarms > 0 else "Normal")

    # Interactive Subplots for Temp and Vibration
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        subplot_titles=(f"{target_mach} - Temperature Trend (°C)", f"{target_mach} - Vibration Trend"),
        vertical_spacing=0.1
    )

    # Temperature Trace
    fig.add_trace(go.Scatter(x=mach_data['Timestamp'], y=mach_data['Temperature'], name='Raw Temp (°C)', line=dict(color='#3b82f6', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=mach_data['Timestamp'], y=mach_data['ma_Temperature'], name='Moving Avg (w=5)', line=dict(color='#f97316', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=mach_data['Timestamp'], y=mach_data['thr_Temperature'], name='Threshold (μ+2σ)', line=dict(color='#ef4444', dash='dot', width=1.5)), row=1, col=1)

    # Vibration Trace
    fig.add_trace(go.Scatter(x=mach_data['Timestamp'], y=mach_data['Vibration'], name='Raw Vibration', line=dict(color='#10b981', width=1)), row=2, col=1)
    fig.add_trace(go.Scatter(x=mach_data['Timestamp'], y=mach_data['ma_Vibration'], name='Moving Avg (w=5)', line=dict(color='#8b5cf6', width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=mach_data['Timestamp'], y=mach_data['thr_Vibration'], name='Threshold (μ+2σ)', line=dict(color='#ef4444', dash='dot', width=1.5)), row=2, col=1)

    # Shaded Alarm Intervals
    alarm_points = mach_data[mach_data['Alarm']]
    if not alarm_points.empty:
        for _, row in alarm_points.iterrows():
            fig.add_vrect(
                x0=row['Timestamp'] - pd.Timedelta(minutes=2.5),
                x1=row['Timestamp'] + pd.Timedelta(minutes=2.5),
                fillcolor="red", opacity=0.25, line_width=0,
                row="all", col=1
            )

    fig.update_layout(height=600, hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig, use_container_width=True)

    # Secondary Sensor Harmonics
    with st.expander("🔬 Advanced Vibration Harmonics & Features"):
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            fig_crest = px.line(mach_data, x='Timestamp', y='Vibration_Crest_Factor', title="Vibration Crest Factor", height=280)
            st.plotly_chart(fig_crest, use_container_width=True)
        with col_v2:
            fig_p2p = px.line(mach_data, x='Timestamp', y='Vibration_P2P', title="Vibration Peak-to-Peak (P2P)", height=280)
            st.plotly_chart(fig_p2p, use_container_width=True)


# =============================================================================
# 4. PRODUCTION COUNTS & RATE
# =============================================================================
elif menu == "📦 Production Counts & Rate":
    st.title("📦 Production Counts & Manufacturing Throughput")
    st.markdown("Analyzes production volume, output rates (parts/hour), and good vs scrap yield.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Total Production Output (Good vs Rejected)")
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=kpi_summary.index, y=kpi_summary['Total_Production'] - kpi_summary['Total_Rejected'],
            name='Good Parts', marker_color='#22c55e', text=kpi_summary['Total_Production'] - kpi_summary['Total_Rejected'], textposition='inside'
        ))
        fig_bar.add_trace(go.Bar(
            x=kpi_summary.index, y=kpi_summary['Total_Rejected'],
            name='Rejected Parts', marker_color='#ef4444', text=kpi_summary['Total_Rejected'], textposition='inside'
        ))
        fig_bar.update_layout(barmode='stack', yaxis_title="Units Produced", height=420)
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        st.subheader("Production Hourly Rate (Parts / Running Hour)")
        fig_rate = px.bar(
            kpi_summary.reset_index(), x='Machine_ID', y='Production_Rate',
            text='Production_Rate', color='Production_Rate',
            color_continuous_scale='Blues',
            labels={'Production_Rate': 'Parts / Hour'}
        )
        fig_rate.update_layout(height=420)
        st.plotly_chart(fig_rate, use_container_width=True)

    st.subheader("Cumulative Output Progression Over Time")
    cum_df = df.copy()
    cum_df['Cumulative_Production'] = cum_df.groupby('Machine_ID')['Production_Count'].cumsum()
    fig_cum = px.line(
        cum_df, x='Timestamp', y='Cumulative_Production', color='Machine_ID',
        title="Fleet Cumulative Production Curves", height=380
    )
    st.plotly_chart(fig_cum, use_container_width=True)


# =============================================================================
# 5. MACHINE COMPARISON
# =============================================================================
elif menu == "⚖️ Machine Comparison":
    st.title("⚖️ Comprehensive Machine Comparison")
    st.markdown("Multi-dimensional benchmarking across mechanical, thermodynamic, operational, and quality metrics.")

    metrics = [
        'Average_Temperature', 'Average_Vibration', 'Maximum_Motor_Current',
        'Average_Cycle_Time', 'Utilization_%', 'Production_Rate', 'Operational_Variability_%'
    ]

    # Normalized radar chart
    comp_df = kpi_summary[metrics].copy()
    norm_comp = (comp_df - comp_df.min()) / (comp_df.max() - comp_df.min() + 1e-6) * 100

    categories = ['Avg Temp', 'Avg Vib', 'Max Current', 'Cycle Time', 'Utilization', 'Prod Rate', 'Variability']

    fig_radar = go.Figure()
    for m in kpi_summary.index:
        vals = norm_comp.loc[m].tolist()
        vals += vals[:1]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals,
            theta=categories + [categories[0]],
            fill='toself',
            name=m,
            opacity=0.5
        ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True, height=500, title="Normalized Machine Performance Footprint (0-100% Scale)"
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    st.subheader("Detailed KPI Comparison Matrix")
    st.dataframe(kpi_summary.style.background_gradient(cmap='Blues'), use_container_width=True)

    # Side-by-side grouped parameter comparison
    st.subheader("Direct Parameter Comparison Bar Charts")
    param = st.selectbox("Select Benchmark Parameter", metrics, index=0)
    fig_p = px.bar(
        kpi_summary.reset_index(), x='Machine_ID', y=param, color='Machine_ID',
        text=param, title=f"Comparison: {param}"
    )
    st.plotly_chart(fig_p, use_container_width=True)


# =============================================================================
# 6. UTILIZATION COMPARISON
# =============================================================================
elif menu == "⏱️ Utilization & Downtime":
    st.title("⏱️ Machine Utilization & Downtime Comparison")
    st.markdown("Evaluates overall equipment effectiveness (OEE runtime efficiency) vs scheduled/unscheduled downtime.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Machine Utilization (%) vs 80% Target")
        fig_util = px.bar(
            kpi_summary.reset_index(), x='Machine_ID', y='Utilization_%',
            text='Utilization_%', color='Utilization_%',
            color_continuous_scale='Greens'
        )
        fig_util.add_hline(y=80, line_dash='dash', line_color='red', annotation_text='Target (80%)')
        fig_util.update_layout(yaxis=dict(range=[0, 100]), height=420)
        st.plotly_chart(fig_util, use_container_width=True)

    with col2:
        st.subheader("Operating Hours vs Downtime Hours")
        fig_hours = go.Figure()
        fig_hours.add_trace(go.Bar(
            x=kpi_summary.index, y=kpi_summary['Running_Hours'],
            name='Running Hours', marker_color='#3b82f6', text=kpi_summary['Running_Hours'], textposition='inside'
        ))
        fig_hours.add_trace(go.Bar(
            x=kpi_summary.index, y=kpi_summary['Downtime_Hours'],
            name='Downtime Hours', marker_color='#f43f5e', text=kpi_summary['Downtime_Hours'], textposition='inside'
        ))
        fig_hours.update_layout(barmode='group', yaxis_title="Hours", height=420)
        st.plotly_chart(fig_hours, use_container_width=True)

    st.markdown("""
    **Utilization Insights:**
    - **Highest Utilization:** `CNC01` achieves 78.50% utilization (130.83 running hours).
    - **Highest Downtime:** `CNC02` exhibits 25.33 downtime hours (utilization 75.50%).
    - Downtime includes `Maintenance`, `Fault`, and `Stopped` states.
    """)


# =============================================================================
# 7. REJECTION-RATE COMPARISON
# =============================================================================
elif menu == "🎯 Rejection-Rate Comparison":
    st.title("🎯 Rejection-Rate & Quality Assurance")
    st.markdown("Evaluates defect frequencies, scrap percentages, and their relationships with process temperature and cycle duration.")

    col1, col2 = st.columns([3, 2])
    with col1:
        st.subheader("Rejection Rate by CNC Machine (%)")
        rej_df = kpi_summary.reset_index()
        fig_rej = px.bar(
            rej_df, x='Machine_ID', y='Rejection_Rate_%', text='Rejection_Rate_%',
            color='Rejection_Rate_%', color_continuous_scale='Reds'
        )
        fleet_mean_rej = kpi_summary['Rejection_Rate_%'].mean()
        fig_rej.add_hline(y=fleet_mean_rej, line_dash='dash', line_color='orange', annotation_text=f"Fleet Avg: {fleet_mean_rej:.2f}%")
        fig_rej.update_layout(height=420)
        st.plotly_chart(fig_rej, use_container_width=True)

    with col2:
        st.subheader("Scrap Metrics Summary")
        scrap_summary = pd.DataFrame({
            "Total Output": kpi_summary['Total_Production'],
            "Scrap Count": kpi_summary['Total_Rejected'],
            "Rejection %": kpi_summary['Rejection_Rate_%'],
            "Quality Loss Rank": kpi_summary['Rejection_Rate_%'].rank(ascending=False).astype(int)
        })
        st.dataframe(scrap_summary, use_container_width=True)
        st.info("💡 **CNC03** exhibits the highest rejection rate at **2.39%**, followed by CNC04 at 2.03%.")

    st.subheader("Correlation: Cycle Time vs Temperature vs Rejection Risk")
    fig_scatter = px.scatter(
        df[df['Machine_Status'] == 'Running'].sample(1000, random_state=42),
        x='Cycle_Time', y='Temperature', color='Machine_ID', size='Rejected_Parts',
        hover_data=['Vibration', 'Motor_Current'],
        title="Sample Telemetry: Cycle Time vs Temperature (Marker Size = Rejected Parts)",
        height=400
    )
    st.plotly_chart(fig_scatter, use_container_width=True)


# =============================================================================
# 8. DETECTED ABNORMAL PERIODS
# =============================================================================
elif menu == "⚠️ Detected Abnormal Periods":
    st.title("⚠️ Detected Abnormal Periods & Fault Intervals")
    st.markdown("Task 5 Multi-Sensor Consensus Alarm Results (Rule + Z-Score + Moving Average + Persistence Filter).")

    if not events_df.empty:
        st.success(f"✅ Detection pipeline successfully isolated {len(events_df)} confirmed abnormal operating period(s).")
        st.subheader("Event Incident Log")
        st.dataframe(events_df, use_container_width=True)

        st.subheader("🔍 Deep-Dive Telemetry Zoom: Abnormal Event on CNC03")
        cnc03 = full_task5[full_task5['Machine_ID'] == 'CNC03'].copy()
        alarm_times = cnc03.loc[cnc03['Alarm'], 'Timestamp']
        t1 = alarm_times.min() - pd.Timedelta(hours=4)
        t2 = alarm_times.max() + pd.Timedelta(hours=4)
        zoom = cnc03[(cnc03['Timestamp'] >= t1) & (cnc03['Timestamp'] <= t2)]

        fig_event = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=(
            "Vibration Spike (Baseline μ=1.47, Peak=4.36)",
            "Temperature Rise (Baseline μ=48.69°C, Peak=56.49°C)",
            "Motor Current Surge (Baseline μ=10.85A, Peak=14.77A)"
        ))

        fig_event.add_trace(go.Scatter(x=zoom['Timestamp'], y=zoom['Vibration'], name='Vibration', line=dict(color='#10b981')), row=1, col=1)
        fig_event.add_trace(go.Scatter(x=zoom['Timestamp'], y=zoom['ma_Vibration'], name='MA Vibration', line=dict(color='green', dash='dash')), row=1, col=1)
        fig_event.add_trace(go.Scatter(x=zoom['Timestamp'], y=zoom['thr_Vibration'], name='Threshold (μ+2σ)', line=dict(color='black', dash='dot')), row=1, col=1)

        fig_event.add_trace(go.Scatter(x=zoom['Timestamp'], y=zoom['Temperature'], name='Temperature', line=dict(color='#f97316')), row=2, col=1)
        fig_event.add_trace(go.Scatter(x=zoom['Timestamp'], y=zoom['ma_Temperature'], name='MA Temperature', line=dict(color='brown', dash='dash')), row=2, col=1)
        fig_event.add_trace(go.Scatter(x=zoom['Timestamp'], y=zoom['thr_Temperature'], name='Threshold (μ+2σ)', line=dict(color='black', dash='dot')), row=2, col=1)

        fig_event.add_trace(go.Scatter(x=zoom['Timestamp'], y=zoom['Motor_Current'], name='Motor Current', line=dict(color='#3b82f6')), row=3, col=1)
        fig_event.add_trace(go.Scatter(x=zoom['Timestamp'], y=zoom['ma_Motor_Current'], name='MA Current', line=dict(color='blue', dash='dash')), row=3, col=1)
        fig_event.add_trace(go.Scatter(x=zoom['Timestamp'], y=zoom['thr_Motor_Current'], name='Threshold (μ+2σ)', line=dict(color='black', dash='dot')), row=3, col=1)

        for _, r in zoom[zoom['Alarm']].iterrows():
            fig_event.add_vrect(
                x0=r['Timestamp'] - pd.Timedelta(minutes=2.5),
                x1=r['Timestamp'] + pd.Timedelta(minutes=2.5),
                fillcolor="red", opacity=0.3, line_width=0, row="all", col=1
            )

        fig_event.update_layout(height=700, hovermode="x unified")
        st.plotly_chart(fig_event, use_container_width=True)
    else:
        st.info("No abnormal periods detected under current parameter settings.")


# =============================================================================
# 9. ALARM SUMMARY & SENSITIVITY TUNING
# =============================================================================
elif menu == "🚨 Alarm Summary & Tuning":
    st.title("🚨 Multi-Method Alarm Summary & Diagnostics")
    st.markdown("Compares individual trigger methods (Rule-based, Z-score, Moving Average) and evaluates consensus alarm sensitivity.")

    col_t1, col_t2, col_t3, col_t4 = st.columns(4)
    with col_t1:
        tune_k_rule = st.slider("Rule Threshold (K_RULE * σ)", 1.0, 3.5, 2.0, 0.1)
    with col_t2:
        tune_z_limit = st.slider("Z-Score Limit", 1.0, 3.5, 2.0, 0.1)
    with col_t3:
        tune_ma_k = st.slider("Moving Avg Dev (MA_K * σ)", 1.0, 3.5, 2.0, 0.1)
    with col_t4:
        tune_persist = st.slider("Persistence Window (Samples)", 1, 5, 3, 1)

    # Re-evaluate with tuned parameters
    tuned_task5, _, tuned_events = dp.compute_task5_alarms(
        df, k_rule=tune_k_rule, z_limit=tune_z_limit, ma_k=tune_ma_k, persist=tune_persist
    )

    alarm_diag = pd.DataFrame({
        'Machine_ID': all_machines,
        'Rule_Flag_Counts': tuned_task5.groupby('Machine_ID')['Rule_Flag'].sum(),
        'Z_Flag_Counts': tuned_task5.groupby('Machine_ID')['Z_Flag'].sum(),
        'MA_Flag_Counts': tuned_task5.groupby('Machine_ID')['MA_Flag'].sum(),
        'Consensus_Flags': tuned_task5.groupby('Machine_ID')['Combined_Flag'].sum(),
        'Persisted_Alarm_Samples': tuned_task5.groupby('Machine_ID')['Alarm'].sum(),
        'Confirmed_Events': [len(tuned_events[tuned_events['Machine_ID'] == m]) for m in all_machines]
    }).set_index('Machine_ID')

    col1, col2 = st.columns([3, 2])
    with col1:
        st.subheader("Diagnostic Algorithm Triggers Breakdown")
        melted_diag = alarm_diag[['Rule_Flag_Counts', 'Z_Flag_Counts', 'MA_Flag_Counts', 'Consensus_Flags']].reset_index().melt(
            id_vars='Machine_ID', var_name='Detection_Method', value_name='Triggers'
        )
        fig_diag = px.bar(melted_diag, x='Machine_ID', y='Triggers', color='Detection_Method', barmode='group', height=400)
        st.plotly_chart(fig_diag, use_container_width=True)

    with col2:
        st.subheader("Confirmed Alarm Events per Machine")
        fig_ev = px.bar(alarm_diag.reset_index(), x='Machine_ID', y='Confirmed_Events', color='Confirmed_Events',
                        color_continuous_scale='Reds', height=400)
        st.plotly_chart(fig_ev, use_container_width=True)

    st.subheader("Detailed Diagnostic Summary Table")
    st.dataframe(alarm_diag, use_container_width=True)


# =============================================================================
# 10. REAL-TIME SENSOR SIMULATION
# =============================================================================
elif menu == "🔄 Real-Time Sensor Simulation":
    st.title("🔄 Live Dynamic Sensor Telemetry Simulation")
    st.markdown("""
    Generates dynamic new sensor telemetry every few seconds and executes the real-time multi-sensor condition monitoring algorithm.
    """)

    # Simulation Controls
    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([2, 2, 2, 2])
    with ctrl1:
        sim_machine = st.selectbox("Simulate Machine", all_machines, index=all_machines.index("CNC03"))
    with ctrl2:
        sim_speed = st.slider("Refresh Interval (seconds)", 1, 10, 2)
    with ctrl3:
        sim_auto = st.toggle("Continuous Stream Loop", value=st.session_state.sim_active)
        st.session_state.sim_active = sim_auto
    with ctrl4:
        force_anomaly = st.checkbox("🚨 Inject Fault / Anomaly Spike", value=st.session_state.sim_force_anomaly)
        st.session_state.sim_force_anomaly = force_anomaly

    btn_col1, btn_col2 = st.columns([1, 1])
    with btn_col1:
        step_btn = st.button("▶️ Generate Single Telemetry Reading", use_container_width=True)
    with btn_col2:
        reset_btn = st.button("🗑️ Clear Simulation History", use_container_width=True)

    if reset_btn:
        st.session_state.sim_history = []
        st.rerun()

    # Generator Logic
    def produce_reading(inject):
        new_row = dp.generate_simulated_reading(sim_machine, baseline, inject_anomaly=inject)
        new_row['Timestamp'] = pd.Timestamp.now()
        
        # Real-time baseline thresholds
        m = sim_machine
        s_mean = baseline.xs("mean", level=1, axis=1).loc[m]
        s_std = baseline.xs("std", level=1, axis=1).loc[m]
        
        t_thr = s_mean['Temperature'] + 2.0 * s_std['Temperature']
        v_thr = s_mean['Vibration'] + 2.0 * s_std['Vibration']
        c_thr = s_mean['Motor_Current'] + 2.0 * s_std['Motor_Current']
        
        # Flags
        is_run = new_row['Machine_Status'] == "Running"
        rule_f = is_run and (new_row['Temperature'] > t_thr) and (new_row['Vibration'] > v_thr) and (new_row['Motor_Current'] > c_thr)
        
        z_t = (new_row['Temperature'] - s_mean['Temperature']) / s_std['Temperature']
        z_v = (new_row['Vibration'] - s_mean['Vibration']) / s_std['Vibration']
        z_c = (new_row['Motor_Current'] - s_mean['Motor_Current']) / s_std['Motor_Current']
        z_f = is_run and ((z_t > 2.0) + (z_v > 2.0) + (z_c > 2.0) >= 2)
        
        comb_f = (int(rule_f) + int(z_f)) >= 1  # Instantaneous flag
        
        new_row['Rule_Flag'] = rule_f
        new_row['Z_Flag'] = z_f
        new_row['Combined_Flag'] = comb_f
        new_row['Alarm'] = comb_f and (len([r for r in st.session_state.sim_history[-2:] if r.get('Combined_Flag')]) >= 1)
        
        st.session_state.sim_history.append(new_row)
        if len(st.session_state.sim_history) > 100:
            st.session_state.sim_history.pop(0)

    if step_btn:
        produce_reading(force_anomaly)

    # Real-time Display Fragment
    @st.fragment(run_every=f"{sim_speed}s" if st.session_state.sim_active else None)
    def render_simulation_stream():
        if st.session_state.sim_active:
            produce_reading(st.session_state.sim_force_anomaly)
            
        history = st.session_state.sim_history
        if not history:
            st.info("Simulation queue empty. Click **'Generate Single Telemetry Reading'** or activate **'Continuous Stream Loop'** above.")
            return

        latest = history[-1]
        hist_df = pd.DataFrame(history)

        # Alarm Banner
        if latest['Alarm']:
            st.error(f"🚨 **CRITICAL ALARM DETECTED ON {latest['Machine_ID']}!** Multiple sensors exceed dynamic safety envelope. Immediate inspection required!")
        elif latest['Combined_Flag']:
            st.warning(f"⚠️ **ANOMALY WARNING ON {latest['Machine_ID']}:** Multi-sensor deviation flagged. Monitoring for persistence.")
        else:
            st.success(f"🟢 **NORMAL OPERATION ON {latest['Machine_ID']}:** All dynamic telemetry streams nominal.")

        # Live Gauges
        g1, g2, g3, g4 = st.columns(4)
        with g1:
            fig_g1 = go.Figure(go.Indicator(
                mode="gauge+number", value=latest['Temperature'],
                title={'text': "Temperature (°C)"},
                gauge={
                    'axis': {'range': [20, 70]},
                    'bar': {'color': "#ef4444" if latest['Temperature'] > 53 else "#3b82f6"},
                    'steps': [
                        {'range': [20, 50], 'color': "#ecfdf5"},
                        {'range': [50, 55], 'color': "#fef3c7"},
                        {'range': [55, 70], 'color': "#fee2e2"}
                    ],
                    'threshold': {'line': {'color': "red", 'width': 3}, 'thickness': 0.75, 'value': 54.0}
                }
            ))
            fig_g1.update_layout(height=230, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_g1, use_container_width=True)

        with g2:
            fig_g2 = go.Figure(go.Indicator(
                mode="gauge+number", value=latest['Vibration'],
                title={'text': "Vibration"},
                gauge={
                    'axis': {'range': [0, 5.0]},
                    'bar': {'color': "#ef4444" if latest['Vibration'] > 2.0 else "#10b981"},
                    'steps': [
                        {'range': [0, 1.8], 'color': "#ecfdf5"},
                        {'range': [1.8, 2.5], 'color': "#fef3c7"},
                        {'range': [2.5, 5.0], 'color': "#fee2e2"}
                    ],
                    'threshold': {'line': {'color': "red", 'width': 3}, 'thickness': 0.75, 'value': 2.0}
                }
            ))
            fig_g2.update_layout(height=230, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_g2, use_container_width=True)

        with g3:
            fig_g3 = go.Figure(go.Indicator(
                mode="gauge+number", value=latest['Motor_Current'],
                title={'text': "Motor Current (A)"},
                gauge={
                    'axis': {'range': [0, 20]},
                    'bar': {'color': "#f97316"},
                    'steps': [
                        {'range': [0, 12], 'color': "#ecfdf5"},
                        {'range': [12, 14], 'color': "#fef3c7"},
                        {'range': [14, 20], 'color': "#fee2e2"}
                    ],
                    'threshold': {'line': {'color': "red", 'width': 3}, 'thickness': 0.75, 'value': 13.5}
                }
            ))
            fig_g3.update_layout(height=230, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_g3, use_container_width=True)

        with g4:
            fig_g4 = go.Figure(go.Indicator(
                mode="gauge+number", value=latest['Spindle_RPM'],
                title={'text': "Spindle RPM"},
                gauge={
                    'axis': {'range': [0, 4000]},
                    'bar': {'color': "#8b5cf6"},
                    'steps': [
                        {'range': [0, 2000], 'color': "#f1f5f9"},
                        {'range': [2000, 3500], 'color': "#ecfdf5"},
                        {'range': [3500, 4000], 'color': "#fef3c7"}
                    ]
                }
            ))
            fig_g4.update_layout(height=230, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_g4, use_container_width=True)

        # Dynamic Streaming Line Chart
        st.subheader("Dynamic Streaming Telemetry (Last 50 Samples)")
        fig_dyn = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Live Temperature (°C)", "Live Vibration"))
        
        fig_dyn.add_trace(go.Scatter(x=hist_df['Timestamp'], y=hist_df['Temperature'], name='Temp (°C)', line=dict(color='#3b82f6', width=2)), row=1, col=1)
        fig_dyn.add_trace(go.Scatter(x=hist_df['Timestamp'], y=hist_df['Vibration'], name='Vibration', line=dict(color='#10b981', width=2)), row=2, col=1)

        # Highlight alarms in stream
        for _, row in hist_df[hist_df['Alarm']].iterrows():
            fig_dyn.add_vrect(
                x0=row['Timestamp'] - pd.Timedelta(seconds=sim_speed*0.5),
                x1=row['Timestamp'] + pd.Timedelta(seconds=sim_speed*0.5),
                fillcolor="red", opacity=0.3, line_width=0, row="all", col=1
            )
        fig_dyn.update_layout(height=420, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_dyn, use_container_width=True)

        # Real-Time Telemetry Feed Table
        st.subheader("Live Telemetry Event Log")
        display_feed = hist_df[['Timestamp', 'Machine_ID', 'Machine_Status', 'Temperature', 'Vibration', 'Motor_Current', 'Spindle_RPM', 'Rule_Flag', 'Z_Flag', 'Alarm']].tail(10).iloc[::-1]
        st.dataframe(display_feed, use_container_width=True)

    render_simulation_stream()
