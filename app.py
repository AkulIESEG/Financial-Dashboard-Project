

# Import Libraries
import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt

# -------------------------------------------------------
# Header Section
# -------------------------------------------------------
st.title("Financial Dashboard")
st.markdown("""
Welcome to the Financial Dashboard! This app provides a platform to:
- **Explore stock performance**: View historical data and trends.
- **Analyze financial metrics**: Investigate rolling volatility and cumulative returns.
- **Compare stocks**: Visualize comparisons between multiple stocks.
- **Simulate price movements**: Use Monte Carlo simulations to forecast potential future prices.

Use the tabs above to navigate and the sidebar to customize inputs.
""")

# -------------------------------------------------------
# Sidebar Configuration
# -------------------------------------------------------
st.sidebar.title("Dashboard Settings")
st.sidebar.markdown("Use the inputs below to customize your analysis.")

# Stock ticker and date range inputs
stock_ticker = st.sidebar.text_input("Enter a Stock Ticker (e.g., AAPL, TSLA):", value="AAPL", key="ticker_input_main")
start_date = st.sidebar.date_input("Start Date:", value=pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("End Date:", value=pd.to_datetime("2024-01-01"))

# Monte Carlo simulation parameters
n_simulations = st.sidebar.slider("Number of Simulations:", min_value=100, max_value=1000, step=100, value=500)
n_days = st.sidebar.slider("Days to Simulate:", min_value=30, max_value=365, step=30, value=252)

# Rolling window slider for Tab 2
rolling_window = st.sidebar.slider("Rolling Window (days):", min_value=5, max_value=60, value=20)

# -------------------------------------------------------
# Tabs Setup
# -------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Overview", 
    "Metrics", 
    "Comparison", 
    "Monte Carlo Simulation"
])

# -------------------------------------------------------
# Tab 1: Overview
# -------------------------------------------------------
with tab1:
    st.header("Overview")
    st.write("""
    Explore historical stock performance and view key summary statistics. 
    Adjust the stock ticker and date range in the sidebar.
    """)
    stock_data = yf.download(stock_ticker, start=start_date, end=end_date)

    if stock_data.empty:
        st.error("No data available for the selected stock ticker or date range.")
    else:
        stock_data.reset_index(inplace=True)
        st.subheader("Stock Closing Prices Over Time")
        fig = px.line(stock_data, x='Date', y='Close', title="Closing Prices Over Time", template="plotly_white")
        st.plotly_chart(fig)

        st.subheader("Summary Statistics")
        st.write(stock_data[['Close']].describe())

# -------------------------------------------------------
# Tab 2: Metrics
# -------------------------------------------------------
with tab2:
    st.header("Financial Metrics")
    if not stock_data.empty:
        stock_data['Daily Return'] = stock_data['Close'].pct_change()
        stock_data['Cumulative Return'] = (1 + stock_data['Daily Return']).cumprod()
        stock_data['Rolling Volatility'] = stock_data['Daily Return'].rolling(rolling_window).std()

        st.subheader("Cumulative Returns Over Time")
        fig = px.line(stock_data, x=stock_data.index, y='Cumulative Return', title="Cumulative Returns", template="plotly_white")
        st.plotly_chart(fig)

        st.subheader("Rolling Volatility Over Time")
        fig = px.line(stock_data, x=stock_data.index, y='Rolling Volatility', title="Rolling Volatility", template="plotly_white")
        st.plotly_chart(fig)

# -------------------------------------------------------
# Tab 3: Comparison
# -------------------------------------------------------
with tab3:
    st.header("Comparative Analysis")
    stock2 = st.text_input("Enter the second stock ticker (e.g., TSLA):", value="TSLA", key="ticker_input_comparison")
    data1 = yf.download(stock_ticker, start=start_date, end=end_date)
    data2 = yf.download(stock2, start=start_date, end=end_date)

    if not data1.empty and not data2.empty:
        data1 = data1[['Close']].rename(columns={'Close': stock_ticker})
        data2 = data2[['Close']].rename(columns={'Close': stock2})
        comparison_data = pd.concat([data1, data2], axis=1).dropna().reset_index()

        st.subheader("Closing Prices Comparison")
        fig = px.line(comparison_data, x='Date', y=[stock_ticker, stock2], title="Closing Prices Comparison", template="plotly_white")
        st.plotly_chart(fig)

        st.subheader("Cumulative Returns Comparison")
        comparison_data[f"{stock_ticker}_Cumulative"] = (comparison_data[stock_ticker].pct_change() + 1).cumprod()
        comparison_data[f"{stock2}_Cumulative"] = (comparison_data[stock2].pct_change() + 1).cumprod()
        fig = px.line(comparison_data, x='Date', y=[f"{stock_ticker}_Cumulative", f"{stock2}_Cumulative"], title="Cumulative Returns Comparison", template="plotly_white")
        st.plotly_chart(fig)

# -------------------------------------------------------
# Tab 4: Monte Carlo Simulation
# -------------------------------------------------------
with tab4:
    st.header("Monte Carlo Simulation for Stock Prices")
    stock_data = yf.download(stock_ticker, start=start_date, end=end_date)

    if stock_data.empty:
        st.error(f"No data available for {stock_ticker} in the selected date range.")
    else:
        returns = stock_data['Close'].pct_change()
        mean_return = returns.mean()
        volatility = returns.std()

        simulations = np.zeros((n_simulations, n_days))
        for i in range(n_simulations):
            prices = [stock_data['Close'].iloc[-1]]
            for _ in range(1, n_days):
                prices.append(prices[-1] * (1 + np.random.normal(mean_return, volatility)))
            simulations[i] = prices

        st.subheader("Simulated Price Paths")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(simulations.T, alpha=0.1)
        st.pyplot(fig)

        st.subheader("Final Price Distribution")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.hist(simulations[:, -1], bins=50)
        st.pyplot(fig2)
