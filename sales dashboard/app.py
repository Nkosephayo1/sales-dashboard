import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet

st.set_page_config("Sales Dashboard")
st.title("Interactive Sales Dashboard")
st.markdown("Analyse sales data by product,region, and time")


@st.cache_data
def load_data():
    df = pd.read_csv("dummy_sales_data.csv", parse_dates=["Date"])
    return df


df = load_data()

st.sidebar.header("Filter your data below 👇")
product = st.sidebar.multiselect(
    "Select your product(s)", options=df["Product"].unique(), default=df["Product"].unique()
)
region = st.sidebar.multiselect("Select the region(s)", options=df["Region"].unique(), default=df["Region"].unique())

date_range = st.sidebar.date_input("Select Date Range of Sales", [df["Date"].min(), df["Date"].max()])


filtered_df = df[(df["Product"].isin(product)) & (df["Region"].isin(region))]
filtered_df = filtered_df[
    (filtered_df["Date"] >= pd.to_datetime(date_range[0])) & (filtered_df["Date"] <= pd.to_datetime(date_range[1]))
]

tab1, tab2 = st.tabs(["📶Overview", "Predictions🔜"])

with tab1:
    st.subheader("Summary Metrics 📈📊")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Sales", f"R{filtered_df['Sales'].sum():,.0f}")
    col2.metric("Average Daily Sales", f"R{filtered_df.groupby('Date')['Sales'].sum().mean():,.2f}")
    col3.metric("Number of Transactions", f"{len(filtered_df)}")

    st.subheader("📅 Daily Sales Trend")
    daily_sales_filtered = filtered_df.groupby("Date")["Sales"].sum().reset_index()
    fig1 = px.line(daily_sales_filtered, x="Date", y="Sales", title="Daily Sales")
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("📦 Sales by Product")
    product_sales = filtered_df.groupby("Product")["Sales"].sum().reset_index()
    fig2 = px.bar(product_sales, x="Product", y="Sales", color="Product", title="Total Sales by Product")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("🌍 Sales by Region")
    region_sales = filtered_df.groupby("Region")["Sales"].sum().reset_index()
    fig3 = px.pie(region_sales, names="Region", values="Sales", title="Sales Distribution by Region")
    st.plotly_chart(fig3, use_container_width=True)

with tab2:
    header_placeholder = st.empty()
    forecast_days = st.slider("Forecast Day Range", min_value=7, max_value=90, value=30)
    header_placeholder.subheader(f"Future Sales for {forecast_days} Days📈 ")
    if len(filtered_df["Date"].unique()) < 2:
        st.warning("Not enough data to perform forecasting. Please adjust your filters.")
    else:

        @st.cache_data
        def train(df, forecast_days):
            daily_sales = df.groupby("Date")["Sales"].sum().reset_index()
            daily_sales.columns = ["ds", "y"]
            model = Prophet()
            model.fit(daily_sales)
            future = model.make_future_dataframe(periods=forecast_days)
            forecast = model.predict(future)
            return daily_sales, forecast

        daily_sales, forecast = train(filtered_df, forecast_days)

        forecast = forecast.rename(columns={"yhat": "Predicted Sales", "ds": "Date"})

        fig_forecast = px.line(
            forecast, x="Date", y="Predicted Sales", title=f"Future Sales for the next {forecast_days} days"
        )
        fig_forecast.update_traces(line=dict(color="green"))
        st.plotly_chart(fig_forecast, use_container_width=True)

        forecast_table = forecast[forecast["Date"] > daily_sales["ds"].max()]
        forecast_table = forecast_table[["Date", "Predicted Sales"]].copy().reset_index(drop=True)
        forecast_table["Predicted Sales"] = forecast_table["Predicted Sales"].round(2)

        st.subheader("Forecast Table")
        st.dataframe(forecast_table)
        st.download_button(
            label="Download Forecast Table",
            data=forecast_table.to_csv(index=False),
            file_name="Predicted Sales.csv",
            mime="text/csv",
        )
