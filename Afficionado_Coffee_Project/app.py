import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
from pathlib import Path

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Coffee Demand Forecasting Dashboard",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# Custom CSS
# --------------------------------------------------
st.markdown("""
<style>

/* Main Background */
.stApp{
    background-color:#0E1117;
}

/* Main Title */
h1{
    color:white;
    font-size:42px !important;
    font-weight:700;
}

/* Section Titles */
h2,h3{
    color:white;
}

/* KPI Cards */
[data-testid="metric-container"]{
    background:#1E1E1E;
    border:1px solid #3a3a3a;
    padding:18px;
    border-radius:15px;
    box-shadow:0px 4px 12px rgba(0,0,0,0.35);
}

/* Metric Labels */
[data-testid="metric-container"] label{
    font-size:16px;
    font-weight:600;
}

/* Sidebar */
section[data-testid="stSidebar"]{
    background:#161B22;
}

/* Chart Container */
.chart-box{
    border:1px solid #3a3a3a;
    border-radius:15px;
    padding:15px;
    background:#1E1E1E;
    margin-bottom:20px;
}

</style>
""", unsafe_allow_html=True)
# --------------------------------------------------
# Dashboard Title
# --------------------------------------------------
st.title("☕ Coffee Demand Forecasting Dashboard")

st.caption(
    "Interactive dashboard for sales analytics and demand forecasting using Machine Learning."
)

st.divider()

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------
@st.cache_data
def load_data():
    base_dir = Path(__file__).resolve().parent
    df = pd.read_excel(base_dir / "Afficionado Coffee Roasters.xlsx")

    # Create Revenue Column
    df["revenue"] = df["transaction_qty"] * df["unit_price"]

    # Create Hour Column
    df["hour"] = pd.to_datetime(
        df["transaction_time"].astype(str)
    ).dt.hour

    return df

df = load_data()


# --------------------------------------------------
# Sidebar Filters
# --------------------------------------------------
st.sidebar.header("🔎 Filters")


if st.sidebar.button("🔄 Reset Filters"):
    st.rerun()

selected_store = st.sidebar.multiselect(
    "Store",
    options=df["store_location"].unique(),
    default=df["store_location"].unique()
)

selected_category = st.sidebar.multiselect(
    "Product Category",
    options=df["product_category"].unique(),
    default=df["product_category"].unique()
)

selected_hour = st.sidebar.slider(
    "Hour Range",
    min_value=int(df["hour"].min()),
    max_value=int(df["hour"].max()),
    value=(int(df["hour"].min()), int(df["hour"].max()))
)

filtered_df = df[
    (df["store_location"].isin(selected_store)) &
    (df["product_category"].isin(selected_category)) &
    (df["hour"].between(selected_hour[0], selected_hour[1]))
]


# --------------------------------------------------
# KPI Calculations
# --------------------------------------------------
total_revenue = filtered_df["revenue"].sum()
total_transactions = filtered_df["transaction_id"].nunique()
total_stores = filtered_df["store_location"].nunique()
total_products = filtered_df["product_type"].nunique()
avg_order_value = filtered_df["revenue"].mean()

# --------------------------------------------------
# KPI Cards
# --------------------------------------------------

def format_currency(value):
    if pd.isna(value):
        return "$0.00"
    if value >= 1000000:
        return f"${value/1000000:.2f}M"
    elif value >= 1000:
        return f"${value/1000:.2f}K"
    else:
        return f"${value:,.2f}"

st.markdown("## 📊 Dashboard Overview")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("💰 Revenue",format_currency(total_revenue))
col2.metric("🧾 Transactions", f"{total_transactions:,}")
col3.metric("🏪 Stores", total_stores)
col4.metric("☕ Products", total_products)
col5.metric("📦 Avg Order", format_currency(avg_order_value))


st.markdown("---")

col1, col2 = st.columns(2)
with col1:

    with st.container(border=True):

        st.subheader("🏪 Revenue by Store")

        store_rev = (
            filtered_df
            .groupby("store_location")["revenue"]
            .sum()
            .reset_index()
        )

        fig = px.bar(
            store_rev,
            x="store_location",
            y="revenue",
            color="store_location",
            text_auto=".2s"
        )

        fig.update_layout(
            template="plotly_dark",
            showlegend=False,
            height=420
        )

        st.plotly_chart(fig, use_container_width=True)

with col2:

    with st.container(border=True):

        st.subheader("🥧 Revenue by Category")

        category_rev = (
            filtered_df
            .groupby("product_category")["revenue"]
            .sum()
            .reset_index()
        )

        fig = px.pie(
            category_rev,
            names="product_category",
            values="revenue",
            hole=0.55
        )

        fig.update_traces(
    textposition="inside",
    textinfo="percent"
)
        fig.update_layout(
    legend=dict(
        orientation="h",
        y=-0.25
    )
)

        st.plotly_chart(fig, use_container_width=True)

col3, col4 = st.columns(2)

with col3:

    with st.container(border=True):

        st.subheader("⏰ Revenue by Hour")

        hour_rev = (
            filtered_df
            .groupby("hour")["revenue"]
            .sum()
            .reset_index()
        )

        fig = px.area(
            hour_rev,
            x="hour",
            y="revenue"
        )

        fig.update_layout(
            template="plotly_dark",
            height=420
        )

        st.plotly_chart(fig, use_container_width=True)



with col4:

    with st.container(border=True):

        st.subheader("🏆 Top 10 Products")

        top_products = (
            filtered_df
            .groupby("product_type")["revenue"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )

        fig = px.bar(
            top_products,
            x="revenue",
            y="product_type",
            orientation="h",
            color="revenue",
            text_auto=".2s"
        )

        fig.update_layout(
            template="plotly_dark",
            height=420
        )

        st.plotly_chart(fig, use_container_width=True)



st.markdown("---")

col5, col6 = st.columns(2)

with col5:

    with st.container(border=True):

        st.subheader("🔥 Store Revenue Heatmap")

        heatmap = filtered_df.pivot_table(
            values="revenue",
            index="hour",
            columns="store_location",
            aggfunc="sum"
        )

        fig = px.imshow(
            heatmap,
            aspect="auto",
            text_auto=".2s"
        )

        fig.update_layout(
            template="plotly_dark",
            height=420
        )

        st.plotly_chart(fig, use_container_width=True)


with col6:
     with st.container(border=True):

        st.subheader("💵 Average Unit Price by Category")

        avg_price = (
            filtered_df
            .groupby("product_category")["unit_price"]
            .mean()
            .reset_index()
        )

        fig = px.bar(
            avg_price,
            x="product_category",
            y="unit_price",
            color="product_category",
            text_auto=".2f"
        )

        fig.update_layout(
            template="plotly_dark",
            showlegend=False,
            height=420
        )

        st.plotly_chart(fig, use_container_width=True)


col7, col8 = st.columns(2)

with col7:

    with st.container(border=True):

        st.subheader("☕ Top Product Details")

        details = (
            filtered_df
            .groupby("product_detail")["revenue"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )

        fig = px.bar(
            details,
            x="revenue",
            y="product_detail",
            orientation="h",
            color="revenue",
            text_auto=".2s"
        )

        fig.update_layout(
            template="plotly_dark",
            height=420
        )

        st.plotly_chart(fig, use_container_width=True)


with col8:
    with st.container(border=True):
        st.subheader("💡 Business Insights")

        if filtered_df.empty:
            st.info("No data matches the selected filters. Please broaden your filters.")
        else:
            best_store = (
                filtered_df.groupby("store_location")["revenue"]
                .sum()
                .idxmax()
            )

            best_category = (
                filtered_df.groupby("product_category")["revenue"]
                .sum()
                .idxmax()
            )

            best_product = (
                filtered_df.groupby("product_type")["revenue"]
                .sum()
                .idxmax()
            )

            peak_hour = (
                filtered_df.groupby("hour")["revenue"]
                .sum()
                .idxmax()
            )

            st.metric("🏪 Best Store", best_store)
            st.metric("🥇 Best Category", best_category)
            st.metric("☕ Best Product", best_product)
            st.metric("⏰ Peak Hour", f"{peak_hour}:00")


st.markdown("---")

csv = filtered_df.to_csv(index=False).encode("utf-8")

st.download_button(
    "📥 Download Filtered Dataset",
    data=csv,
    file_name="coffee_dashboard.csv",
    mime="text/csv",
    use_container_width=True
)

st.markdown("---")

st.caption(
    "Coffee Demand Forecasting Dashboard | Developed using Streamlit, Plotly, Pandas & Machine Learning"
)
