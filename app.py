from pathlib import Path
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Gas Turbine ML Dashboard",
    page_icon="⚙️",
    layout="wide",
)

ARTIFACT_PATH = Path(__file__).parent / "model" / "gas_turbine_artifacts.joblib"

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; padding-bottom: 2rem;}
      [data-testid="stMetric"] {background: rgba(128,128,128,0.08); padding: 14px; border-radius: 12px;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("⚙️ Gas Turbine Emissions & Performance Dashboard")
st.caption("Industrial Process Parameter Optimization Using Machine Learning")


@st.cache_resource
def load_artifacts(path: str):
    return joblib.load(path)


if not ARTIFACT_PATH.exists():
    st.error("The trained model artifact is missing.")
    st.markdown(
        "Place **`gas_turbine_artifacts.joblib`** inside the **`model/`** folder. "
        "Export it from the final, executed cell in your Google Colab notebook."
    )
    st.stop()

try:
    bundle = load_artifacts(str(ARTIFACT_PATH))
except Exception as exc:
    st.error("The model artifact could not be loaded.")
    st.exception(exc)
    st.stop()

model = bundle["model"]
input_columns = bundle["input_columns"]
target_columns = bundle["target_columns"]
input_defaults = bundle["input_defaults"]
input_ranges = bundle["input_ranges"]
test_metrics = pd.DataFrame(bundle.get("test_metrics", []))
feature_importance = pd.DataFrame(bundle.get("feature_importance", []))
optimization = bundle.get("optimization", {})
validation_summary = bundle.get("validation_model_summary", {})

# Top-level project summary
m1, m2, m3, m4 = st.columns(4)
m1.metric("Dataset records", "36,733")
m2.metric("Input features", str(len(input_columns)))
m3.metric("Predicted outputs", str(len(target_columns)))
m4.metric("Selected model", bundle.get("selected_model", "Random Forest"))

st.info(
    "This dashboard uses the trained model exported from the Colab notebook. "
    "Predictions are model estimates based on historical sensor data, not live turbine readings."
)

overview_tab, predict_tab, performance_tab, optimize_tab = st.tabs(
    ["Project overview", "Interactive prediction", "Model evaluation", "Optimization results"]
)

with overview_tab:
    st.subheader("Project overview")
    st.write(
        "The project uses eight gas-turbine operating variables to estimate three outputs: "
        "Turbine Energy Yield (TEY), carbon monoxide (CO), and nitrogen oxides (NOX). "
        "Four regression approaches were compared, and Random Forest was selected using "
        "the average validation R²."
    )
    st.markdown("**Input variables**")
    st.write(", ".join(input_columns))
    st.markdown("**Prediction targets**")
    st.write(", ".join(target_columns))

    if validation_summary:
        val_df = pd.DataFrame(
            [{"Model": name, "Average validation R²": value}
             for name, value in validation_summary.items()]
        )
        st.markdown("**Validation model comparison**")
        st.dataframe(val_df, use_container_width=True, hide_index=True)
        fig = px.bar(
            val_df,
            x="Model",
            y="Average validation R²",
            title="Average validation R² by model",
        )
        st.plotly_chart(fig, use_container_width=True)

with predict_tab:
    st.subheader("Predict turbine performance and emissions")
    st.write(
        "Change the operating-condition inputs below and select **Run prediction**. "
        "The available input range is based on the 5th–95th percentile of the training data."
    )

    with st.form("prediction_form"):
        left, right = st.columns(2)
        values = {}
        for idx, feature in enumerate(input_columns):
            lo, hi = map(float, input_ranges[feature])
            default = min(max(float(input_defaults[feature]), lo), hi)
            step = max((hi - lo) / 500.0, 0.001)
            container = left if idx % 2 == 0 else right
            with container:
                values[feature] = st.number_input(
                    feature,
                    min_value=lo,
                    max_value=hi,
                    value=default,
                    step=step,
                    format="%.4f",
                    help="Input range is limited to the training-data 5th–95th percentile.",
                )
        submitted = st.form_submit_button("Run prediction", type="primary")

    if submitted:
        row = pd.DataFrame([[values[col] for col in input_columns]], columns=input_columns)
        prediction = model.predict(row)[0]
        result = dict(zip(target_columns, [float(x) for x in prediction]))
        st.markdown("#### Predicted outputs")
        cols = st.columns(len(target_columns))
        pretty_names = {
            "TEY": "Turbine Energy Yield",
            "CO": "Carbon Monoxide (CO)",
            "NOX": "Nitrogen Oxides (NOX)",
        }
        for col, target in zip(cols, target_columns):
            col.metric(pretty_names.get(target, target), f"{result[target]:.4f}")
        prediction_df = pd.DataFrame(
            {"Output": list(result.keys()), "Predicted value": list(result.values())}
        )
        fig = px.bar(
            prediction_df,
            x="Output",
            y="Predicted value",
            title="Predicted output values (dataset units)",
            text_auto=".3f",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("These are model predictions, not a guarantee of real operating performance.")

with performance_tab:
    st.subheader("Final test-set performance")
    if not test_metrics.empty:
        st.dataframe(test_metrics.round(4), use_container_width=True, hide_index=True)
        if "Target" in test_metrics.columns and "R2" in test_metrics.columns:
            fig = px.bar(
                test_metrics,
                x="Target",
                y="R2",
                title="R² on the 2014–2015 held-out test data",
                text_auto=".4f",
            )
            fig.add_hline(y=0, line_dash="dash", annotation_text="R² = 0")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No test metrics were included in the exported artifact.")

    if not feature_importance.empty:
        st.markdown("#### Feature importance")
        fig = px.bar(
            feature_importance.sort_values("Importance", ascending=True),
            x="Importance",
            y="Feature",
            orientation="h",
            title="Random Forest feature importance",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Feature importance describes the model's use of each feature; it does not prove causation."
        )

    if not test_metrics.empty and "Target" in test_metrics.columns:
        nox = test_metrics[test_metrics["Target"] == "NOX"]
        if not nox.empty and float(nox.iloc[0]["R2"]) < 0:
            st.warning(
                "NOX has a negative test R² in this experiment. Treat its predictions as unreliable "
                "on the held-out period; this limitation is shown rather than hidden."
            )

with optimize_tab:
    st.subheader("Exploratory optimization")
    st.write(
        "Differential Evolution searched selected process variables to reduce predicted CO, "
        "subject to a minimum TEY of 95% of the baseline and non-negative predicted CO and NOX."
    )
    if optimization:
        status_cols = st.columns(3)
        status_cols[0].metric("Optimizer succeeded", "Yes" if optimization.get("success") else "No")
        status_cols[1].metric("Constraints feasible", "Yes" if optimization.get("feasible") else "No")
        status_cols[2].metric(
            "Candidate distance",
            f"{optimization.get('candidate_distance', float('nan')):.4f}"
            if optimization.get("candidate_distance") is not None else "N/A",
        )

        before = optimization.get("baseline", {})
        after = optimization.get("optimized", {})
        if before and after:
            comparison_df = pd.DataFrame(
                [{"Output": key, "Baseline": before[key], "Optimized": after[key]}
                 for key in target_columns if key in before and key in after]
            )
            chart_df = comparison_df.melt(
                id_vars="Output", var_name="Condition", value_name="Predicted value"
            )
            fig = px.bar(
                chart_df,
                x="Output",
                y="Predicted value",
                color="Condition",
                barmode="group",
                title="Baseline vs optimized model predictions",
                text_auto=".3f",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(comparison_df.round(4), use_container_width=True, hide_index=True)

        parameter_rows = optimization.get("parameters", [])
        if parameter_rows:
            st.markdown("#### Optimized parameter values")
            st.dataframe(pd.DataFrame(parameter_rows).round(4), use_container_width=True, hide_index=True)

        if optimization.get("unusual_candidate", False):
            st.warning(
                "The optimized candidate was flagged as unusual compared with historical conditions. "
                "This is an exploratory model result, not a validated operating recommendation."
            )
    else:
        st.warning("Optimization results were not included in the exported artifact.")

st.divider()
st.caption(
    "Academic demonstration only. The model is trained on historical data; it is not connected "
    "to a live turbine or a safety-critical control system."
)
