# 🏭 CNC Smart Fleet Diagnostics, Condition Monitoring & Real-Time Simulation Dashboard

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.66%2B-FF4B4B?logo=streamlit)
![Plotly](https://img.shields.io/badge/Plotly-Interactive-3F4F75?logo=plotly)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-PCA%20%26%20Scaling-F7931E?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green)

An end-to-end industrial IoT telemetry analysis, degradation modeling, multi-sensor anomaly detection, and interactive dashboard system for five CNC machines (**CNC01** through **CNC05**).

---

## 📌 Project Overview

This repository provides a complete diagnostic suite covering 10,000 telemetry observations across 5 CNC units over 7 days (5-minute sampling interval). It integrates:
- **Task 3: Preprocessing & Dimensionality Reduction**: Grouped IQR outlier filtering by operational state, degradation health evaluation (Monotonicity, Trendability, Robustness), Standardization, and PCA retaining 95% variance.
- **Task 4: Fleet Operational Benchmarking**: Comprehensive KPI formulation (Utilization %, Running Hours, Production Rate, Rejection Rate %, Downtime Hours, and Operational Variability CV %).
- **Task 5: Multi-Sensor Anomaly & Alarm System**: Consensus ensemble voting (Rule-based $\mu+2\sigma$, Z-Score $>2$, Moving Average deviation $>2\sigma$) paired with a 3-sample persistence filter.
- **Interactive Python Dashboard (`app.py`)**: Built with Streamlit and Plotly covering all 8 deliverables with rich interactivity.
- **Real-Time Dynamic Sensor Simulation**: Streaming telemetry generator executing online consensus anomaly detection with interactive fault injection.

---

## 🚀 Quick Start

### 1. Installation
Clone or download the repository, then install dependencies:
```bash
git clone https://github.com/your-username/cnc-fleet-diagnostics.git
cd cnc-fleet-diagnostics
pip install -r requirements.txt
```

### 2. Launch the Interactive Dashboard
Run the Streamlit application:
```bash
streamlit run app.py
```
*Or on Windows, simply double-click **`run_dashboard.bat`**.*  
Access the dashboard at **`http://localhost:8501`**.

### 3. Run the Unified Jupyter Notebook
Open and run **`final_code.ipynb`** in Jupyter Lab, Jupyter Notebook, or VS Code:
```bash
jupyter lab final_code.ipynb
```

### 4. Regenerate Figures & Static HTML Report
```bash
python generate_visualizations.py
```
Open **`visualizations/dashboard_report.html`** in any web browser for an offline interactive view.

---

## 📊 Core Visualization Deliverables

All high-resolution visualization charts are generated and stored in the [`visualizations/`](visualizations/) directory:

| Deliverable | Description | Visualization |
| :--- | :--- | :--- |
| **1. Machine Operating Status** | Stacked percentage distribution of time spent in Running, Idle, Maintenance, Fault, and Stopped states. | `visualizations/1_machine_operating_status.png` |
| **2. Temp & Vibration Trends** | Continuous time series with 5-sample rolling averages, $\mu+2\sigma$ thresholds, and shaded alarm regions. | `visualizations/2_temperature_vibration_trends.png` |
| **3. Production Counts & Rate** | Stacked good vs. scrap output bars and production rates (parts per running hour). | `visualizations/3_production_counts_and_rate.png` |
| **4. Machine Comparison** | Normalized 7-dimensional radar benchmark footprint and comparative metrics matrix. | `visualizations/4_machine_comparison_radar.png` |
| **5. Utilization Comparison** | Utilization % vs. 80% industrial target paired with downtime hours breakdown. | `visualizations/5_utilization_downtime_comparison.png` |
| **6. Rejection-Rate Comparison** | Scrap defect rate % ranking (CNC03 highest at 2.39%) and cycle-time quality correlations. | `visualizations/6_rejection_rate_comparison.png` |
| **7. Detected Abnormal Periods** | Detail view of the 45-minute sustained anomaly on CNC03 (vibration spike to 4.36, temp to 56.5°C). | `visualizations/7_detected_abnormal_period_detail.png` |
| **8. Alarm Summary & Diagnostics** | Breakdown of trigger counts across detection algorithms (Rule, Z-Score, MA, Consensus). | `visualizations/8_alarm_summary_diagnostics.png` |

---

## 🔄 Live Dynamic Sensor Simulation

The dashboard features a live telemetry streaming engine:
- **Dynamic Auto-Refresh**: Uses `@st.fragment` to update gauges and charts every 1–10 seconds without full page reload.
- **Physical Gauges**: Plotly radial dials for **Temperature**, **Vibration**, **Motor Current**, and **Spindle RPM** with color-coded safety zones.
- **Fault Injection**: An **"🚨 Inject Fault / Anomaly Spike"** button to simulate physical bearing failure and watch the alarm system trigger live.
- **Rolling Strip Chart & Event Feed**: 50-sample moving strip chart with vertical alarm shading and tabular telemetry log.

---

## 📁 Repository Structure

```
├── app.py                                   # Main Streamlit web dashboard application
├── data_pipeline.py                         # Core analytics, KPI calculator & simulation generator
├── generate_visualizations.py               # Standalone script to export PNGs and HTML report
├── run_dashboard.bat                        # Windows one-click batch launcher
├── final_code.ipynb                         # Unified Jupyter Notebook (Tasks 3, 4, 5 consolidated)
├── CNC_Dashboard_and_Visualization_Report.docx # 1-Page executive DOCX summary report
├── initial data.csv                         # Raw 10,000-row CNC telemetry dataset
├── CNC_Task4_Complete_Data.csv              # Processed dataset with merged machine KPIs
├── requirements.txt                         # Python dependencies
├── .gitignore                               # Git ignore configuration
└── visualizations/                          # Pre-rendered high-res figures & interactive HTML report
    ├── 1_machine_operating_status.png
    ├── 2_temperature_vibration_trends.png
    ├── 3_production_counts_and_rate.png
    ├── 4_machine_comparison_radar.png
    ├── 5_utilization_downtime_comparison.png
    ├── 6_rejection_rate_comparison.png
    ├── 7_detected_abnormal_period_detail.png
    ├── 8_alarm_summary_diagnostics.png
    └── dashboard_report.html
```

---

## 📈 Fleet KPI Benchmark Summary

| Machine ID | Utilization (%) | Running Hours | Production Rate (pts/h) | Total Parts | Rejection Rate (%) | Downtime (h) | Confirmed Alarms |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CNC01** | **78.50%** | **130.8 h** | 72.8 | 9,528 | **1.95%** | 22.0 h | 0 (Nominal) |
| **CNC02** | 75.50% | 125.8 h | 65.2 | 8,210 | 1.96% | **25.3 h** | 0 (High Downtime) |
| **CNC03** | 77.35% | 128.9 h | **75.5** | **9,729** | **2.39%** | 22.7 h | **1 Critical Event (45m)** |
| **CNC04** | 78.25% | 130.4 h | 63.5 | 8,285 | 2.03% | 22.0 h | 0 (Peak Thermal/Stress) |
| **CNC05** | 77.45% | 129.1 h | 74.8 | 9,651 | 1.99% | 24.3 h | 0 (Nominal Stable) |
| **Fleet Mean** | **77.41%** | **645.1 h** | **70.4** | **45,403** | **2.07%** | **116.3 h** | **1 Event (CNC03)** |

---

## 📄 License
This project is licensed under the MIT License.
