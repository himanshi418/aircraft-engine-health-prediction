"""
app.py - Aircraft Engine Health Prediction (Streamlit)

Loads the trained three-feature Random Forest saved by train_model.py and
predicts Remaining Useful Life (RUL) from:
    time_in_cycles, sensor_2, sensor_3

Run with:  streamlit run app.py
"""

from __future__ import annotations

import math
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "rul_rf_3feat.joblib"
FEATURES = ["time_in_cycles", "sensor_2", "sensor_3"]  # must match train_model.py

st.set_page_config(
    page_title="Aircraft Engine Health Prediction",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CSS = """
<style>
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1150px;}
html, body, [class*="css"] {font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;}
.hero {padding: 0.5rem 0 1.25rem 0; border-bottom: 1px solid #1E3250; margin-bottom: 1.25rem;}
.hero .tag {color: #22D3EE; font-size: 0.78rem; letter-spacing: 0.16em; text-transform: uppercase; font-weight: 600;}
.hero h1 {color: #F1F5FB; font-size: 2.1rem; margin: 0.35rem 0 0.4rem 0; font-weight: 650; line-height: 1.2;}
.hero p {color: #8FA3BF; font-size: 1.02rem; margin: 0; max-width: 46rem; line-height: 1.55;}
.card {background: #0F1D33; border: 1px solid #1E3250; border-radius: 12px; padding: 1.4rem 1.5rem;}
.eyebrow {color: #8FA3BF; font-size: 0.78rem; letter-spacing: 0.1em; text-transform: uppercase; font-weight: 600;}
.big {color: #F1F5FB; font-size: 4rem; font-weight: 700; line-height: 1.1; margin-top: 0.4rem;}
.big .unit {color: #22D3EE; font-size: 1.3rem; font-weight: 500; margin-left: 0.6rem;}
.meter-track {background: #0A1424; border: 1px solid #1E3250; border-radius: 999px; height: 14px; overflow: hidden; margin-top: 1.1rem;}
.meter-fill {height: 100%; border-radius: 999px;}
.meter-fill.ok {background: #2DD4BF;}
.meter-fill.warn {background: #F5B544;}
.meter-fill.alert {background: #F2766B;}
.meter-label {display: flex; justify-content: space-between; color: #8FA3BF; font-size: 0.78rem; margin-top: 0.4rem;}
.badge {display: inline-block; padding: 0.25rem 0.7rem; border-radius: 999px; font-size: 0.82rem; font-weight: 600; border: 1px solid;}
.badge.ok {color: #2DD4BF; border-color: #2DD4BF55; background: #2DD4BF14;}
.badge.warn {color: #F5B544; border-color: #F5B54455; background: #F5B54414;}
.badge.alert {color: #F2766B; border-color: #F2766B55; background: #F2766B14;}
.fine {color: #8FA3BF; font-size: 0.85rem; line-height: 1.5;}
.body {color: #D5DEEB; font-size: 0.98rem; line-height: 1.6;}
.step {background: #0F1D33; border: 1px solid #1E3250; border-radius: 12px; padding: 1.1rem 1.3rem; margin-bottom: 0.8rem;}
.step .n {color: #22D3EE; font-weight: 700; font-size: 0.85rem; letter-spacing: 0.1em;}
.step .t {color: #F1F5FB; font-weight: 600; font-size: 1.05rem; margin: 0.15rem 0 0.25rem 0;}
.step .d {color: #B7C5D9; font-size: 0.95rem; line-height: 1.55;}
div.stButton > button {width: 100%; border-radius: 8px; font-weight: 600; padding: 0.6rem 1rem;}
div[data-testid="stTabs"] button {font-weight: 600;}
@media (max-width: 640px) {
  .hero h1 {font-size: 1.6rem;}
  .big {font-size: 3rem;}
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_bundle(path_str: str, modified_time: float):
    """Load the saved model bundle. `modified_time` makes the cache refresh after retraining."""
    return joblib.load(path_str)


def get_bundle():
    """Return (bundle, error_message). Never raises."""
    if not MODEL_PATH.is_file():
        return None, (
            "The trained model file was not found.\n\n"
            f"Expected location: {MODEL_PATH}\n\n"
            "Create it by running this in the terminal, from the project folder:\n"
            "    python3 train_model.py"
        )
    try:
        bundle = load_bundle(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
    except Exception as exc:  # noqa: BLE001 - show any load problem in a friendly way
        return None, (
            f"The model file exists but could not be loaded ({type(exc).__name__}: {exc}).\n\n"
            "Re-create it with:  python3 train_model.py"
        )

    required = {"model", "features", "ranges", "max_rul"}
    if not isinstance(bundle, dict) or not required.issubset(bundle):
        return None, "The model file has an unexpected format. Re-create it with:  python3 train_model.py"
    if list(bundle["features"]) != FEATURES:
        return None, (
            f"The model was trained on {bundle['features']} but the app expects {FEATURES}. "
            "Re-create the model with:  python3 train_model.py"
        )
    return bundle, None


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def validate_inputs(cycles, s2, s3, ranges):
    errors, warnings = [], []
    named = [("Operating cycles", cycles), ("Sensor 2", s2), ("Sensor 3", s3)]

    for label, value in named:
        if value is None:
            errors.append(f"{label} is empty. Please enter a number.")
        elif not math.isfinite(float(value)):
            errors.append(f"{label} must be a finite number.")
    if errors:
        return errors, warnings

    if cycles < 1:
        errors.append("Operating cycles must be at least 1.")
    elif float(cycles) != int(cycles):
        errors.append("Operating cycles must be a whole number.")
    if s2 <= 0:
        errors.append("Sensor 2 must be greater than 0.")
    if s3 <= 0:
        errors.append("Sensor 3 must be greater than 0.")
    if errors:
        return errors, warnings

    for label, key, value in [
        ("Operating cycles", "time_in_cycles", cycles),
        ("Sensor 2", "sensor_2", s2),
        ("Sensor 3", "sensor_3", s3),
    ]:
        lo, hi = ranges[key]["min"], ranges[key]["max"]
        if value < lo or value > hi:
            warnings.append(
                f"{label} ({value:g}) is outside the range seen in the training data ({lo:g} to {hi:g}). "
                "A Random Forest cannot extrapolate beyond the data it learned from, so treat this estimate with extra caution."
            )
    return errors, warnings


def status_for(rul: float):
    """Illustrative categories only. These thresholds are NOT validated maintenance limits."""
    if rul >= 100:
        return "Extended life remaining", "ok"
    if rul >= 50:
        return "Monitor", "warn"
    return "Maintenance planning advised", "alert"


def render_result(res: dict, max_rul: float) -> str:
    rul = res["rul"]
    pct = 0.0 if max_rul <= 0 else max(0.0, min(rul / max_rul, 1.0)) * 100
    label, tone = status_for(rul)
    return "".join([
        '<div class="card">',
        '<div class="eyebrow">Predicted remaining useful life</div>',
        f'<div class="big">{rul:,.0f}<span class="unit">cycles</span></div>',
        f'<div class="meter-track"><div class="meter-fill {tone}" style="width:{pct:.1f}%"></div></div>',
        f'<div class="meter-label"><span>0</span><span>{max_rul:,.0f} (longest RUL in training data)</span></div>',
        '<div style="margin-top:1.1rem">',
        f'<span class="badge {tone}">{label}</span>',
        '<div class="fine" style="margin-top:0.5rem">Illustrative category only. These bands are not validated maintenance thresholds.</div>',
        '</div>',
        '<div class="body" style="margin-top:1.1rem">',
        f'In simple terms: based on the three readings entered, the model estimates this simulated engine could run for roughly '
        f'<b>{rul:,.0f} more operating cycles</b> before reaching the dataset-defined end-of-life condition. '
        'This is a statistical estimate, not a guaranteed failure date.',
        '</div>',
        f'<div class="fine" style="margin-top:0.9rem">Inputs used: {res["cycles"]} cycles, '
        f'sensor 2 = {res["s2"]:g}, sensor 3 = {res["s3"]:g}</div>',
        '</div>',
    ])


def placeholder_card() -> str:
    return (
        '<div class="card"><div class="eyebrow">Predicted remaining useful life</div>'
        '<div class="big" style="color:#3A506F">--<span class="unit" style="color:#3A506F">cycles</span></div>'
        '<div class="body" style="margin-top:1rem">Enter the three readings and press '
        '<b>Predict Remaining Useful Life</b> to see the estimate.</div></div>'
    )


def fmt_metrics(m: dict) -> str:
    return f"MAE {m['MAE']:.2f} | RMSE {m['RMSE']:.2f} | R\u00b2 {m['R2']:.3f}"


# Consistent dark, editorial chart styling across the dashboard.
CHART_BG = "#101B2D"
PLOT_BG = "#101B2D"
TEXT = "#E8EEF7"
MUTED = "#9AAAC0"
GRID = "#26364D"
PALETTE = ["#55D6E8", "#A78BFA", "#F4B860", "#F28C8C", "#75C9A5"]

def style_chart(fig, height=390):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=CHART_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(family="Inter, -apple-system, sans-serif", color=TEXT, size=12),
        title=dict(font=dict(size=16, color=TEXT), x=0.02, xanchor="left"),
        height=height,
        margin=dict(l=24, r=24, t=64, b=42),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED)),
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, tickfont=dict(color=MUTED), title_font=dict(color=MUTED)),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, tickfont=dict(color=MUTED), title_font=dict(color=MUTED)),
    )
    return fig


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.markdown(
    '<div class="hero">'
    '<div class="tag">Predictive maintenance</div>'
    '<h1>Aircraft Engine Health Prediction</h1>'
    '<p>Predictive maintenance uses data from an engine to estimate how much useful life remains, '
    'so that maintenance can be planned before problems develop. This tool estimates the Remaining '
    'Useful Life (RUL) of a simulated turbofan engine, measured in operating cycles.</p>'
    '</div>',
    unsafe_allow_html=True,
)

bundle, load_error = get_bundle()
if load_error:
    st.error(load_error)
    st.stop()

model = bundle["model"]
ranges = bundle["ranges"]
max_rul = float(bundle["max_rul"])

DEFAULTS = {
    "in_cycles": max(1, int(round(ranges["time_in_cycles"]["median"]))),
    "in_s2": round(float(ranges["sensor_2"]["median"]), 2),
    "in_s3": round(float(ranges["sensor_3"]["median"]), 2),
}


def reset_inputs():
    for key, value in DEFAULTS.items():
        st.session_state[key] = value
    st.session_state["result"] = None


if "result" not in st.session_state:
    st.session_state["result"] = None

tab_eda, tab_models, tab_dash, tab_how, tab_about, tab_limits = st.tabs(
    ["EDA Dashboard", "Model Performance", "RUL Prediction", "How it works", "About the project", "Model limitations"]
)

# --------------------------------------------------------------------------
# A. Exploratory Data Analysis dashboard
# --------------------------------------------------------------------------
with tab_eda:
    st.markdown("### Explore the NASA C-MAPSS FD001 training data")
    st.markdown(
        "Upload **train_FD001.csv** (the same training file used in the notebook) "
        "to view the five EDA visualisations from your analysis."
    )
    uploaded_train = st.file_uploader(
        "Upload train_FD001.csv", type=["csv"], key="eda_train_upload",
        help="Use the original NASA C-MAPSS FD001 training CSV, not the test file."
    )
    if uploaded_train is None:
        st.info("The EDA charts will appear here after you upload train_FD001.csv. "
                "Your prediction tool remains available in the RUL Prediction tab.")
    else:
        try:
            eda_df = pd.read_csv(uploaded_train)
            required = {"unit_number", "time_in_cycles", "RUL", "sensor_2", "sensor_3",
                        "sensor_4", "sensor_7", "sensor_11", "sensor_12"}
            missing = sorted(required - set(eda_df.columns))
            if missing:
                st.error("This file is missing required columns: " + ", ".join(missing))
            else:
                c1, c2, c3 = st.columns(3)
                c1.metric("Training observations", f"{len(eda_df):,}")
                c2.metric("Engine units", f"{eda_df['unit_number'].nunique():,}")
                c3.metric("Features / columns", f"{eda_df.shape[1]:,}")

                st.markdown("#### 1. Distribution of Remaining Useful Life")
                fig = px.histogram(eda_df, x="RUL", nbins=30,
                                   labels={"RUL": "Remaining Useful Life (cycles)", "count": "Number of observations"},
                                   title="Distribution of Remaining Useful Life")
                fig.update_layout(bargap=0.08)
                fig.update_traces(marker_color=PALETTE[0])  
                style_chart(fig, 350)
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("#### 2. Engine degradation over operating cycles")
                engine = eda_df[eda_df["unit_number"] == 1].sort_values("time_in_cycles")
                fig = px.line(engine, x="time_in_cycles", y="sensor_2",
                              labels={"time_in_cycles": "Operating Cycles", "sensor_2": "Sensor 2 Reading"},
                              title="Sensor 2 Measurements Across Engine Operating Cycles")
                

                

                
                fig.update_traces(line=dict(color=PALETTE[1], width=2.8), fill="tozeroy", fillcolor="rgba(167,139,250,0.10)")
                fig.update_traces(marker_line_width=0, opacity=0.92)
                fig.update_yaxes(range=[641.5, 644.4], autorange=False)
                style_chart(fig, 520)
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("#### 3. Correlation between selected variables")
                corr_cols = ["RUL", "time_in_cycles", "sensor_2", "sensor_3",
                             "sensor_4", "sensor_7", "sensor_11", "sensor_12"]
                corr = eda_df[corr_cols].corr(numeric_only=True)
                fig = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale=["#F28C8C", "#18263A", "#55D6E8"],
                                zmin=-1, zmax=1, title="Correlation Between Selected Variables")
                style_chart(fig, 430)
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("#### 4. Twenty engines with the lowest average RUL")
                engine_rul = (eda_df.groupby("unit_number")["RUL"].mean()
                              .nsmallest(20).sort_values())
                plot_df = engine_rul.rename("Average RUL").rename_axis("Engine Number").reset_index()
                plot_df["Engine Number"] = plot_df["Engine Number"].astype(str)
                fig = px.bar(plot_df, x="Average RUL", y="Engine Number", orientation="h",
                             title="20 Engines with the Lowest Average RUL", color_discrete_sequence=[PALETTE[2]])
                fig.update_traces(marker_line_width=0, opacity=0.92)
                style_chart(fig, 520)
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("#### 5. Correlation of variables with RUL")
                correlation = (eda_df.corr(numeric_only=True)["RUL"].drop("RUL")
                               .dropna().sort_values().rename("Correlation with RUL")
                               .rename_axis("Variable").reset_index())
                fig = px.bar(correlation, x="Correlation with RUL", y="Variable", orientation="h",
                             title="Correlation of Variables with RUL", color="Correlation with RUL",
                             color_continuous_scale=["#F28C8C", "#26364D", "#55D6E8"], range_color=[-1, 1])
                fig.update_traces(marker_line_width=0)
                style_chart(fig, 440)
                st.plotly_chart(fig, use_container_width=True)

                with st.expander("Dataset quality checks"):
                    q1, q2 = st.columns(2)
                    q1.metric("Missing values", f"{int(eda_df.isna().sum().sum()):,}")
                    q2.metric("Duplicate rows", f"{int(eda_df.duplicated().sum()):,}")
                    st.caption("These charts reproduce the EDA topics from the project notebook. "
                               "Correlation indicates association, not causation.")
        except Exception as exc:
            st.error(f"The CSV could not be read: {type(exc).__name__}: {exc}")

# --------------------------------------------------------------------------
# B. Original model performance (18-feature notebook models)
# --------------------------------------------------------------------------
with tab_models:
    st.markdown("### Model performance comparison")
    st.markdown("These are the results from the **original 18-feature models** evaluated on the separate NASA test dataset in the notebook. They are distinct from the simplified three-feature model used in the live prediction demo.")

    metric_cols = st.columns(3)
    metric_cols[0].metric("Random Forest MAE", "20.186 cycles", "Lower is better")
    metric_cols[1].metric("Random Forest RMSE", "26.847 cycles", "Lower is better")
    metric_cols[2].metric("Random Forest R²", "0.583", "Higher is better")

    model_results = pd.DataFrame({
        "Model": ["Linear Regression", "Random Forest"],
        "MAE": [25.975, 20.186],
        "RMSE": [31.250, 26.847],
        "R²": [0.434, 0.583],
    })
    left_chart, right_chart = st.columns(2, gap="large")
    with left_chart:
        st.markdown("#### Prediction error")
        error_long = model_results.melt(id_vars="Model", value_vars=["MAE", "RMSE"], var_name="Metric", value_name="Error (cycles)")
        fig = px.bar(error_long, x="Metric", y="Error (cycles)", color="Model", barmode="group",
                     color_discrete_map={"Linear Regression": PALETTE[2], "Random Forest": PALETTE[0]},
                     text="Error (cycles)", title="MAE and RMSE — lower is better")
        fig.update_traces(texttemplate="%{text:.2f}", textposition="outside", cliponaxis=False)
        style_chart(fig, 390)
        st.plotly_chart(fig, use_container_width=True)
    with right_chart:
        st.markdown("#### Goodness of fit")
        fig = px.bar(model_results, x="Model", y="R²", color="Model",
                     color_discrete_map={"Linear Regression": PALETTE[2], "Random Forest": PALETTE[0]},
                     text="R²", title="R² score — higher is better", range_y=[0, 1])
        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside", cliponaxis=False, showlegend=False)
        style_chart(fig, 390)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Exact test-set results")
    st.dataframe(model_results.set_index("Model"), use_container_width=True)
    st.caption("MAE = Mean Absolute Error; RMSE = Root Mean Squared Error; R² = coefficient of determination. The Random Forest has lower error and a higher R² on this test set. The metrics describe model performance on this dataset, not guaranteed real-world aircraft performance.")

# --------------------------------------------------------------------------
# C. RUL prediction dashboard
# --------------------------------------------------------------------------
with tab_dash:
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown('<div class="eyebrow">Engine readings</div>', unsafe_allow_html=True)
        cycles = st.number_input(
            "Operating cycles (time_in_cycles)",
            min_value=1, max_value=1000, value=DEFAULTS["in_cycles"], step=1, key="in_cycles",
            help="How many operating cycles the engine has completed so far.",
        )
        s2 = st.number_input(
            "Sensor 2 reading (sensor_2)",
            min_value=0.0, max_value=10000.0, value=DEFAULTS["in_s2"], step=0.01, format="%.2f", key="in_s2",
            help="Reading from sensor 2 in the C-MAPSS FD001 data.",
        )
        s3 = st.number_input(
            "Sensor 3 reading (sensor_3)",
            min_value=0.0, max_value=10000.0, value=DEFAULTS["in_s3"], step=0.01, format="%.2f", key="in_s3",
            help="Reading from sensor 3 in the C-MAPSS FD001 data.",
        )
        st.caption(
            f"Training data ranges: cycles {ranges['time_in_cycles']['min']:g} to {ranges['time_in_cycles']['max']:g}, "
            f"sensor 2 {ranges['sensor_2']['min']:g} to {ranges['sensor_2']['max']:g}, "
            f"sensor 3 {ranges['sensor_3']['min']:g} to {ranges['sensor_3']['max']:g}."
        )

        predict_clicked = st.button("Predict Remaining Useful Life", type="primary")
        st.button("Reset", on_click=reset_inputs)

        if predict_clicked:
            errors, warnings = validate_inputs(cycles, s2, s3, ranges)
            if errors:
                st.session_state["result"] = None
                for message in errors:
                    st.error(message)
            else:
                for message in warnings:
                    st.warning(message)
                try:
                    X = pd.DataFrame(
                        [{"time_in_cycles": int(cycles), "sensor_2": float(s2), "sensor_3": float(s3)}],
                        columns=FEATURES,
                    )
                    prediction = float(model.predict(X)[0])
                    st.session_state["result"] = {
                        "rul": max(prediction, 0.0),
                        "cycles": int(cycles),
                        "s2": float(s2),
                        "s3": float(s3),
                    }
                except Exception as exc:  # noqa: BLE001
                    st.session_state["result"] = None
                    st.error(f"The prediction could not be completed ({type(exc).__name__}: {exc}).")

    with right:
        result = st.session_state.get("result")
        if result:
            st.markdown(render_result(result, max_rul), unsafe_allow_html=True)
        else:
            st.markdown(placeholder_card(), unsafe_allow_html=True)

    with st.expander("About this demo model"):
        st.markdown(
            "This is a **simplified three-feature Random Forest** trained only on "
            "`time_in_cycles`, `sensor_2` and `sensor_3`. It is a separate demonstration model and is "
            "**not** the original 18-feature model from the notebook."
        )
        info = [f"Training rows: {bundle.get('train_rows', 'n/a'):,}" if isinstance(bundle.get("train_rows"), int) else None]
        if bundle.get("n_engines"):
            info.append(f"Training engines: {bundle['n_engines']}")
        st.markdown(" | ".join(i for i in info if i))
        if bundle.get("validation_metrics"):
            st.markdown(f"Validation ({bundle.get('validation_note', 'held-out data')}): {fmt_metrics(bundle['validation_metrics'])}")
        if bundle.get("test_metrics"):
            st.markdown(f"Test set (last cycle of each test engine): {fmt_metrics(bundle['test_metrics'])}")
        st.caption("These figures belong to the three-feature demo model only. They are not the original model's results.")

# --------------------------------------------------------------------------
# C. How it works
# --------------------------------------------------------------------------
with tab_how:
    steps = [
        ("STEP 1", "Operating cycles",
         "Operating cycles represent the accumulated operation of the engine. One cycle is one run of the engine, and the more cycles it has completed, the more it has aged."),
        ("STEP 2", "Sensor readings",
         "Sensors measure conditions inside the engine while it runs, such as temperatures and pressures. Their readings carry information about how the engine is operating."),
        ("STEP 3", "Learning from history",
         "The machine-learning model (a Random Forest) was trained on historical simulated engine data. It learned which combinations of cycles and sensor values tend to appear at different points in an engine's life."),
        ("STEP 4", "Estimating remaining life",
         "Given the readings you enter, the model compares them with the patterns it learned and estimates how many operating cycles remain before the dataset-defined end of life."),
    ]
    for n, title, desc in steps:
        st.markdown(
            f'<div class="step"><div class="n">{n}</div><div class="t">{title}</div><div class="d">{desc}</div></div>',
            unsafe_allow_html=True,
        )

# --------------------------------------------------------------------------
# D. About the project
# --------------------------------------------------------------------------
with tab_about:
    st.markdown(
        '<div class="card body">'
        'This is a college Predictive Analytics project that estimates the Remaining Useful Life of an aircraft '
        'engine using the <b>NASA C-MAPSS FD001</b> simulated turbofan degradation dataset. The workflow covered '
        '<b>data preprocessing</b> (including creating the RUL target), <b>exploratory data analysis</b> of engine '
        'sensors and cycles, and two regression models: <b>Linear Regression</b> and <b>Random Forest</b>.'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown("&nbsp;", unsafe_allow_html=True)
    st.markdown("**Original project results (18-feature models, from the Jupyter notebook)**")
    original = pd.DataFrame(
        {
            "Metric": ["MAE", "RMSE", "R\u00b2"],
            "Linear Regression": ["25.975", "31.250", "0.434"],
            "Random Forest": ["20.186", "26.847", "0.583"],
        }
    )
    st.table(original.set_index("Metric"))
    st.caption(
        "Random Forest performed better on the separate test dataset. These results belong to the original 18-feature "
        "models only. The live dashboard uses a simplified three-feature Random Forest, which is a separate demonstration "
        "model with its own, different accuracy."
    )

# --------------------------------------------------------------------------
# E. Model limitations
# --------------------------------------------------------------------------
with tab_limits:
    st.markdown(
        '<div class="card body">'
        '<ul style="margin:0; padding-left:1.2rem; line-height:1.8">'
        '<li>C-MAPSS contains <b>simulated</b> engine degradation data, not records from real aircraft engines.</li>'
        '<li>This is an educational project. It is <b>not certified</b> for real aircraft maintenance or flight-safety decisions.</li>'
        '<li>The prediction is a statistical estimate and <b>cannot guarantee an actual failure date</b>.</li>'
        '<li>The dashboard model uses only three inputs, so it is a simplified demonstration and is expected to be less informative than the original 18-feature model.</li>'
        '<li>The status categories are illustrative and are not validated maintenance thresholds.</li>'
        '<li>Inputs far outside the training data ranges produce unreliable estimates.</li>'
        '</ul></div>',
        unsafe_allow_html=True,
    )
