import streamlit as st
import pandas as pd
import plotly.express as px
import json

st.set_page_config(
    page_title="Model Performance",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Model Performance")

st.caption(
    "Evaluate the forecasting models using different performance metrics."
)

st.divider()

# --------------------------------------
# Load Files
# --------------------------------------

Path(__file__).resolve().parent.parent
importance_df = pd.read_csv(BASE_DIR / "feature_importance.csv")

with open(BASE_DIR / "metrics.json", "r") as f:
    metrics = json.load(f)
# --------------------------------------
# KPI Cards
# --------------------------------------

st.subheader("📈 Hourly Model")

c1, c2, c3 = st.columns(3)

c1.metric(
    "MAE",
    f"{metrics['hourly_mae']:.2f}"
)

c2.metric(
    "RMSE",
    f"{metrics['hourly_rmse']:.2f}"
)

c3.metric(
    "MAPE",
    f"{metrics['hourly_mape']:.2f}%"
)

st.divider()

st.subheader("💰 Daily Model")

c1, c2, c3 = st.columns(3)

c1.metric(
    "MAE",
    f"{metrics['daily_mae']:.2f}"
)

c2.metric(
    "RMSE",
    f"{metrics['daily_rmse']:.2f}"
)

c3.metric(
    "MAPE",
    f"{metrics['daily_mape']:.2f}%"
)

st.divider()

st.subheader("🎯 Overall Performance")

c1, c2 = st.columns(2)

c1.metric(
    "Forecast Accuracy",
    f"{metrics['forecast_accuracy']:.2f}%"
)

c2.metric(
    "Peak Capture Rate",
    f"{metrics['peak_capture_rate']:.2f}%"
)

st.divider()

with st.container(border=True):

    st.subheader("⭐ Feature Importance")

    fig = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance"
    )

    fig.update_layout(
        template="plotly_dark",
        height=600
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

st.divider()

st.markdown("""
### 📌 Interpretation

- Lower MAE indicates better prediction accuracy.
- Lower RMSE indicates fewer large prediction errors.
- Lower MAPE means lower percentage error.
- Higher Forecast Accuracy indicates better model performance.
- Feature Importance shows which variables contributed most to the prediction.
""")
