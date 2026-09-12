import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Dataset Explorer",
    page_icon="📋",
    layout="wide"
)

st.title("📋 Dataset Explorer")

df = pd.read_excel(
    "Afficionado Coffee Roasters.xlsx"
)

df["revenue"] = df["transaction_qty"] * df["unit_price"]

df["hour"] = pd.to_datetime(
    df["transaction_time"].astype(str)
).dt.hour

st.subheader("Dataset Preview")

st.dataframe(df)

st.divider()

c1, c2 = st.columns(2)

with c1:

    st.subheader("Dataset Shape")

    st.write(df.shape)

    st.subheader("Missing Values")

    st.write(df.isnull().sum())

with c2:

    st.subheader("Column Information")

    info = pd.DataFrame({
        "Column": df.columns,
        "Datatype": df.dtypes.astype(str)
    })

    st.dataframe(info)

st.divider()

csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    "📥 Download Dataset",
    csv,
    "coffee_dataset.csv",
    "text/csv",
    use_container_width=True
)