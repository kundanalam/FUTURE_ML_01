import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Sales Forecasting", layout="wide")

st.title("📊 Sales Forecasting Dashboard")

# Upload File
file = st.file_uploader("Upload Sales CSV File", type=["csv"])

if file is None:
    st.warning("Please upload a dataset")
    st.stop()

# Load Data
@st.cache_data
def load_data(file):
    df = pd.read_csv(file, encoding="latin1")
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    return df

df = load_data(file)

# Aggregate Daily Sales
daily_sales = df.groupby("Order Date")["Sales"].sum().reset_index()

# Prepare Data for Prophet
prophet_df = daily_sales.rename(columns={
    "Order Date": "ds",
    "Sales": "y"
})

# Forecast Period
period = st.slider("Forecast Days", 30, 180, 90)

# Train/Test Split
train = prophet_df[:-period]
test = prophet_df[-period:]

# Train Model
model = Prophet(weekly_seasonality=True, yearly_seasonality=True)
model.fit(train)

# Future Dates
future = model.make_future_dataframe(periods=period)
forecast = model.predict(future)

# Predictions
pred = forecast.tail(period)["yhat"].values
actual = test["y"].values

# Metrics
mae = mean_absolute_error(actual, pred)
rmse = np.sqrt(mean_squared_error(actual, pred))

col1, col2 = st.columns(2)
col1.metric("MAE", round(mae,2))
col2.metric("RMSE", round(rmse,2))

# Plot Forecast
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=daily_sales["Order Date"],
    y=daily_sales["Sales"],
    name="Actual Sales"
))

fig.add_trace(go.Scatter(
    x=forecast["ds"],
    y=forecast["yhat"],
    name="Forecast",
    line=dict(dash="dash")
))

st.plotly_chart(fig, use_container_width=True)

# Forecast Table
st.subheader("Forecast Data")
st.dataframe(forecast[['ds','yhat','yhat_lower','yhat_upper']].tail(period))