import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import load_models

st.set_page_config(page_title="Model Performance | EMIPredict AI", page_icon="🏆", layout="wide")
st.title("🏆 Model Performance & MLflow Experiment Comparison")

clf, reg, scaler, clf_encoder, meta = load_models()

st.markdown("### 🥇 Selected Production Models")
c1, c2 = st.columns(2)
with c1:
    st.success(f"**Classifier:** {meta['best_classifier']['name']}")
    tm = meta["best_classifier"]["test_metrics"]
    st.metric("Test Accuracy", f"{tm['accuracy']*100:.2f}%")
    st.metric("Test F1 (macro)", f"{tm['f1_macro']:.4f}")
    st.metric("Test ROC-AUC (OVR)", f"{tm['roc_auc_ovr']:.4f}")
with c2:
    st.success(f"**Regressor:** {meta['best_regressor']['name']}")
    tm = meta["best_regressor"]["test_metrics"]
    st.metric("Test RMSE", f"₹{tm['rmse']:,.0f}")
    st.metric("Test MAE", f"₹{tm['mae']:,.0f}")
    st.metric("Test R²", f"{tm['r2']:.4f}")

st.caption("Metrics computed on a held-out test set (15% of data, unseen during training or "
           "model selection) for an unbiased final evaluation.")

st.markdown("---")
st.markdown("### 📈 All Experiment Runs (MLflow-tracked)")

tab1, tab2 = st.tabs(["Classification Models", "Regression Models"])

with tab1:
    clf_df = pd.DataFrame(meta["all_classification_results"])
    clf_df = clf_df.sort_values("f1_macro", ascending=False).reset_index(drop=True)
    st.dataframe(
        clf_df.style.format({
            "accuracy": "{:.4f}", "precision_macro": "{:.4f}", "recall_macro": "{:.4f}",
            "f1_macro": "{:.4f}", "roc_auc_ovr": "{:.4f}", "train_time_sec": "{:.1f}s",
        }).highlight_max(subset=["f1_macro"], color="#2ecc7133"),
        use_container_width=True, hide_index=True,
    )

    melted = clf_df.melt(id_vars="name", value_vars=["accuracy", "f1_macro", "roc_auc_ovr"],
                          var_name="metric", value_name="score")
    fig = px.bar(melted, x="name", y="score", color="metric", barmode="group",
                 title="Classification Model Comparison")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    reg_df = pd.DataFrame(meta["all_regression_results"])
    reg_df = reg_df.sort_values("rmse").reset_index(drop=True)
    st.dataframe(
        reg_df.style.format({
            "rmse": "₹{:,.0f}", "mae": "₹{:,.0f}", "r2": "{:.4f}", "mape": "{:.4f}", "train_time_sec": "{:.1f}s",
        }).highlight_min(subset=["rmse"], color="#2ecc7133"),
        use_container_width=True, hide_index=True,
    )

    fig = px.bar(reg_df, x="name", y="r2", title="Regression Model Comparison — R² Score",
                 color="r2", color_continuous_scale="Viridis")
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.bar(reg_df, x="name", y="rmse", title="Regression Model Comparison — RMSE (lower is better)",
                  color="rmse", color_continuous_scale="Reds")
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")
with st.expander("ℹ️ About experiment tracking"):
    st.markdown("""
All training runs shown above were logged with **MLflow** — including
hyperparameters, metrics, training time, and the serialized model artifact
for every run. To browse the full MLflow UI (parameters, artifacts, model
registry) locally, run from the project root:

```bash
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

Then open http://localhost:5000 in your browser. Two experiments were logged:
`EMIPredict_Classification` and `EMIPredict_Regression`, and the best run of
each was registered to the MLflow Model Registry as `EMIPredict-Classifier`
and `EMIPredict-Regressor`.
""")
