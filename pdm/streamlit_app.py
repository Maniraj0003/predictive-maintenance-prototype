import os

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide")

DEFAULT_API_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ Settings")
api_url = st.sidebar.text_input("Backend API URL", value=DEFAULT_API_URL).rstrip("/")
st.sidebar.caption("Point this at your deployed FastAPI backend (e.g. Vercel URL).")

st.title("🔧 Intelligent Predictive Maintenance Dashboard")
st.caption(
    "Autoencoder-based anomaly detection + selectable RUL prediction model, "
    "trained on the NASA C-MAPSS Turbofan Engine Degradation dataset (FD001)."
)
selected_model = st.radio(
    "Choose the RUL prediction model used by the dashboard",
    options=["random_forest", "xgboost"],
    format_func=lambda model: "Random Forest" if model == "random_forest" else "XGBoost",
    horizontal=True,
    index=0,
)
model_label = "Random Forest" if selected_model == "random_forest" else "XGBoost"
st.info(f"All RUL predictions below use **{model_label}**.")


def api_get(path):
    try:
        r = requests.get(f"{api_url}{path}", timeout=15)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)


def api_post(path, payload):
    try:
        r = requests.post(f"{api_url}{path}", json=payload, timeout=15)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)


def get_maintenance_recommendation(risk_level, predicted_rul, is_anomaly):
    """Convert model outputs into an actionable maintenance recommendation."""
    if risk_level == "Critical":
        return (
            "🔴 Immediate Maintenance Recommended",
            "The engine has very low remaining useful life and/or abnormal behavior. "
            "Inspect the engine and schedule maintenance before continued operation."
        )

    if risk_level == "Warning":
        return (
            "🟡 Maintenance Inspection Recommended",
            "The engine shows signs of degradation. "
            "Schedule an inspection and continue monitoring the engine condition."
        )

    if is_anomaly:
        return (
            "🟡 Inspect Engine / Sensor Condition",
            "Abnormal sensor behavior has been detected. Inspect the sensor readings "
            "and continue monitoring the engine."
        )

    return (
        "🟢 Continue Normal Monitoring",
        "The engine currently has sufficient remaining useful life "
        "and does not require immediate maintenance."
    )


meta, meta_err = api_get("/api/meta")
if meta_err:
    st.error(
        f"Could not reach backend API at `{api_url}`. "
        f"Make sure it's running/deployed. ({meta_err})"
    )
    st.stop()

with st.expander("ℹ️ Model info", expanded=False):
    c1, c2, c3 = st.columns(3)
    c1.metric("Dataset", "NASA C-MAPSS (FD001)")
    c2.metric("RUL validation MAE", f"{meta['val_mae_cycles']:.1f} cycles")
    c3.metric("Anomaly threshold (recon. error)", f"{meta['anomaly_threshold']:.5f}")
    st.write("**Anomaly detector:**", meta["model"]["anomaly_detector"])
    st.write("**Failure predictor:**", meta["model"]["failure_predictor"])

tab1, tab2, tab3 = st.tabs(["📈 Engine Lifecycle Demo", "🧪 Manual Sensor Input", "📊 Model Evaluation"])

# ---------------------------------------------------------------------------
# Tab 1: full lifecycle simulation for a demo engine
# ---------------------------------------------------------------------------
with tab1:
    units_resp, err = api_get("/api/demo-units")
    if err:
        st.error(err)
    else:
        unit_id = st.selectbox("Select a demo engine (unit)", units_resp["units"])
        history_resp, err2 = api_get(f"/api/demo/{unit_id}?model={selected_model}")
        if err2:
            st.error(err2)
        else:
            hist = pd.DataFrame(history_resp["history"])

            # Let the evaluator move through the engine lifecycle instead of
            # always showing only the final (Critical) cycle.
            min_cycle = int(hist["cycle"].min())
            max_cycle = int(hist["cycle"].max())
            default_cycle = min(max_cycle, max(min_cycle, int(min_cycle + 0.75 * (max_cycle - min_cycle))))

            selected_cycle = st.slider(
                "Select cycle to inspect",
                min_value=min_cycle,
                max_value=max_cycle,
                value=default_cycle,
                step=1,
                key=f"cycle_slider_{unit_id}",
                help="Move through the lifecycle to see the transition from Healthy to Warning/Critical.",
            )

            selected_rows = hist[hist["cycle"] == selected_cycle]
            if selected_rows.empty:
                # Defensive fallback in case the dataset has a gap in cycle numbers.
                selected_idx = (hist["cycle"] - selected_cycle).abs().idxmin()
                current = hist.loc[selected_idx]
            else:
                current = selected_rows.iloc[0]

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Current cycle", int(current["cycle"]))
            k2.metric(f"Predicted RUL ({model_label})", f"{current['predicted_rul']:.0f} cycles")
            k3.metric("Risk level", current["risk_level"])
            k4.metric("Anomaly flagged?", "Yes 🔴" if current["is_anomaly"] else "No 🟢")
            st.caption(
                f"RUL estimate from {model_label}. "
                f"Reference RUL for this demo cycle: {current['true_rul']:.0f} cycles."
            )

            recommendation, explanation = get_maintenance_recommendation(
                current["risk_level"],
                current["predicted_rul"],
                current["is_anomaly"],
            )

            st.subheader("🛠️ Maintenance Recommendation")
            if current["risk_level"] == "Critical":
                st.error(f"**{recommendation}**\n\n{explanation}")
            elif current["risk_level"] == "Warning" or current["is_anomaly"]:
                st.warning(f"**{recommendation}**\n\n{explanation}")
            else:
                st.success(f"**{recommendation}**\n\n{explanation}")

            fig_rul = go.Figure()
            fig_rul.add_trace(go.Scatter(x=hist["cycle"], y=hist["true_rul"], name="True RUL", line=dict(dash="dot")))
            fig_rul.add_trace(go.Scatter(x=hist["cycle"], y=hist["predicted_rul"], name=f"Predicted RUL ({model_label})"))
            fig_rul.add_vline(x=int(current["cycle"]), line_dash="dash", annotation_text=f"Selected cycle: {int(current['cycle'])}")
            fig_rul.update_layout(title="Remaining Useful Life over engine lifecycle", xaxis_title="Cycle", yaxis_title="RUL (cycles)", height=380)
            st.plotly_chart(fig_rul, use_container_width=True)

            fig_anom = go.Figure()
            fig_anom.add_trace(go.Scatter(x=hist["cycle"], y=hist["reconstruction_error"], name="Reconstruction error"))
            fig_anom.add_hline(y=meta["anomaly_threshold"], line_dash="dash", line_color="red", annotation_text="Anomaly threshold")
            fig_anom.add_vline(x=int(current["cycle"]), line_dash="dash", annotation_text=f"Selected cycle: {int(current['cycle'])}")
            fig_anom.update_layout(title="Anomaly score (autoencoder reconstruction error)", xaxis_title="Cycle", yaxis_title="Reconstruction error", height=380)
            st.plotly_chart(fig_anom, use_container_width=True)

            st.dataframe(hist[["cycle", "true_rul", "predicted_rul", "reconstruction_error", "is_anomaly", "risk_level"]], use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 2: manual sensor input -> single prediction
# ---------------------------------------------------------------------------
with tab2:
    st.write("Enter a sensor reading manually to get a live prediction.")
    feats = meta["feature_cols"]
    defaults = {
        "op1": 0.0, "op2": 0.0, "s2": 642.0, "s3": 1590.0, "s4": 1400.0,
        "s7": 554.0, "s8": 2388.0, "s9": 9050.0, "s11": 47.3, "s12": 522.0,
        "s13": 2388.0, "s14": 8140.0, "s15": 8.4, "s17": 393.0, "s20": 38.9, "s21": 23.4,
    }
    cols = st.columns(4)
    values = {}
    for i, feat in enumerate(feats):
        with cols[i % 4]:
            values[feat] = st.number_input(feat, value=float(defaults.get(feat, 0.0)), format="%.3f")

    if st.button("Predict", type="primary"):
        pred, err3 = api_post(f"/api/predict?model={selected_model}", values)
        if err3:
            st.error(err3)
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric(f"Predicted RUL ({model_label})", f"{pred['predicted_rul']:.0f} cycles")
            c2.metric("Risk level", pred["risk_level"])
            c3.metric("Anomaly?", "Yes 🔴" if pred["is_anomaly"] else "No 🟢")

            recommendation, explanation = get_maintenance_recommendation(
                pred["risk_level"],
                pred["predicted_rul"],
                pred["is_anomaly"],
            )

            st.subheader("🛠️ Maintenance Recommendation")
            if pred["risk_level"] == "Critical":
                st.error(f"**{recommendation}**\n\n{explanation}")
            elif pred["risk_level"] == "Warning" or pred["is_anomaly"]:
                st.warning(f"**{recommendation}**\n\n{explanation}")
            else:
                st.success(f"**{recommendation}**\n\n{explanation}")

            st.json(pred)

# ---------------------------------------------------------------------------
# Tab 3: model evaluation
# ---------------------------------------------------------------------------
with tab3:
    eval_resp, err4 = api_get("/api/evaluation")
    if err4:
        st.error(err4)
    else:
        st.caption(eval_resp.get("methodology", ""))

        def render_rul_comparison(m):
            rf = m.get("random_forest", {})
            xgb = m.get("xgboost", {})
            mae_key = "rul_mae" if "rul_mae" in rf else "rul_mae_all_cycles"
            rmse_key = "rul_rmse" if "rul_rmse" in rf else "rul_rmse_all_cycles"

            comparison = pd.DataFrame(
                {
                    "Model": ["Random Forest", "XGBoost"],
                    "MAE (cycles)": [rf.get(mae_key), xgb.get(mae_key)],
                    "RMSE (cycles)": [rf.get(rmse_key), xgb.get(rmse_key)],
                }
            )
            st.write("**RUL model comparison**")
            st.dataframe(
                comparison.style.format(
                    {"MAE (cycles)": "{:.2f}", "RMSE (cycles)": "{:.2f}"}
                ),
                hide_index=True,
                use_container_width=True,
            )

            if "rul_mae_last_cycle" in rf:
                last_cycle = pd.DataFrame(
                    {
                        "Model": ["Random Forest", "XGBoost"],
                        "MAE (cycles)": [
                            rf["rul_mae_last_cycle"],
                            xgb["rul_mae_last_cycle"],
                        ],
                        "RMSE (cycles)": [
                            rf["rul_rmse_last_cycle"],
                            xgb["rul_rmse_last_cycle"],
                        ],
                    }
                )
                st.write("**Last-cycle RUL comparison**")
                st.dataframe(
                    last_cycle.style.format(
                        {"MAE (cycles)": "{:.2f}", "RMSE (cycles)": "{:.2f}"}
                    ),
                    hide_index=True,
                    use_container_width=True,
                )

        def render_split(title, m, source_note):
            st.subheader(title)
            st.caption(source_note)
            render_rul_comparison(m)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("RUL MAE", f"{m.get('rul_mae', m.get('rul_mae_all_cycles')):.2f} cycles")
            c2.metric("Accuracy", f"{m['anomaly_accuracy']:.1%}")
            c3.metric("Precision", f"{m['anomaly_precision']:.1%}")
            c4.metric("Recall", f"{m['anomaly_recall']:.1%}")
            st.write(f"**F1 score:** {m['anomaly_f1']:.3f}  |  **Rows evaluated:** {m['n_rows']}  |  **Engines:** {m['n_engines']}")

            cm = m["confusion_matrix"]
            cm_df = pd.DataFrame(
                cm,
                index=["Actual: Healthy", "Actual: Anomaly"],
                columns=["Predicted: Healthy", "Predicted: Anomaly"],
            )
            fig = go.Figure(data=go.Heatmap(
                z=cm, x=cm_df.columns, y=cm_df.index,
                text=cm, texttemplate="%{text}", colorscale="Blues", showscale=False,
            ))
            fig.update_layout(title=f"{title} — Confusion Matrix", height=320)
            st.plotly_chart(fig, use_container_width=True)

        val = eval_resp.get("held_out_validation")
        if val:
            render_split(
                "Held-out Validation (20 engines, from training data)",
                val,
                "These 20 engines were removed before training and never seen by either model.",
            )

        st.divider()

        test_eval = eval_resp.get("official_test_set")
        if test_eval:
            render_split(
                "NASA Official Test Set (100 engines, completely separate dataset)",
                test_eval,
                test_eval.get("note", ""),
            )
            st.info(
                "Note the low precision here: only 0.9% of test cycles are true "
                "near-failures (test trajectories are truncated early), so the "
                "anomaly detector's false positives dominate the precision score "
                "even though recall stays high. This is a known trade-off of "
                "threshold-based anomaly detectors on imbalanced data."
            )
        else:
            st.warning("No official test-set report found. Run `python evaluate.py` to generate one.")