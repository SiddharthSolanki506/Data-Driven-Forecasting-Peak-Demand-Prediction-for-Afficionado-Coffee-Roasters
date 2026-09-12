import streamlit as st
import pandas as pd
import plotly.express as px
import json
from pathlib import Path

st.set_page_config(
    page_title="Demand Forecasting",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Coffee Demand Forecasting")

st.caption(
    "Visualize model predictions and forecasting performance."
)

st.divider()



BASE_DIR = Path(__file__).resolve().parent.parent

@st.cache_data
def load_files():

    hourly = pd.read_csv(BASE_DIR / "hourly_forecast.csv")
    daily = pd.read_csv(BASE_DIR / "daily_forecast.csv")
    importance = pd.read_csv(BASE_DIR / "feature_importance.csv")

    with open(BASE_DIR / "metrics.json", "r") as f:
        metrics = json.load(f)

    return hourly, daily, importance, metrics
    
hourly_df, daily_df, importance_df, metrics = load_files()


st.subheader("📊 Forecast Performance")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Hourly MAE",
    f"{metrics['hourly_mae']:.2f}"
)

col2.metric(
    "Hourly RMSE",
    f"{metrics['hourly_rmse']:.2f}"
)

col3.metric(
    "Hourly MAPE",
    f"{metrics['hourly_mape']:.2f}%"
)

col4.metric(
    "Forecast Accuracy",
    f"{metrics['forecast_accuracy']:.2f}%"
)

st.divider()



col1, col2 = st.columns(2)

with col1:

    with st.container(border=True):

        st.subheader("📈 Hourly Transactions Forecast")

        fig = px.line(
    hourly_df,
    x="datetime",
    y=[
        "hourly_transaction_volume",
        "predictions"
    ],
    labels={
        "value":"Transactions",
        "datetime":"Time",
        "variable":"Series"
    }
)

        fig.update_layout(
            template="plotly_dark",
            height=450
        )

        st.plotly_chart(fig, use_container_width=True)


with col2:

    with st.container(border=True):

        st.subheader("💰 Daily Revenue Forecast")

        fig = px.line(
    daily_df,
    x="datetime",
    y=[
        "daily_revenue",
        "predictions"
    ],
    labels={
        "value":"Revenue",
        "datetime":"Date",
        "variable":"Series"
    }
)

        fig.update_layout(
            template="plotly_dark",
            height=450
        )

        st.plotly_chart(fig, use_container_width=True)



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
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)


st.divider()

st.subheader("📋 Forecast Summary")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Daily MAE",
    f"{metrics['daily_mae']:.2f}"
)

col2.metric(
    "Daily RMSE",
    f"{metrics['daily_rmse']:.2f}"
)

col3.metric(
    "Daily MAPE",
    f"{metrics['daily_mape']:.2f}%"
)