

# Import Libraries
import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

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
stock_ticker = st.sidebar.text_input("Enter a Stock Ticker (e.g., AAPL, TSLA):", value="AAPL")
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

#Tab 1

with tab1:
    st.header("Overview")
    st.write("""
    Explore historical stock performance and view key summary statistics. 
    Adjust the stock ticker and date range in the sidebar.
    """)

    # Fetch stock data using yfinance
    stock_data = yf.download(stock_ticker, start=start_date, end=end_date)

    # Debugging information
    st.write("### Debug: Displaying stock_data DataFrame")
    st.write("**Columns in stock_data:**")
    st.write(list(stock_data.columns))  # Show columns in the DataFrame
    st.write("**Head of stock_data:**")
    st.write(stock_data.head())  # Show the first few rows for debugging

    # Handle edge cases
    if stock_data.empty:
        st.error("No data available for the selected stock ticker or date range.")
    elif 'Close' not in stock_data.columns:
        st.error("'Close' column is missing in the data. Cannot proceed with analysis.")
    elif stock_data['Close'].dropna().empty:
        st.error("The 'Close' column contains no valid data. Cannot display meaningful results.")
    else:
        # Ensure that the 'Close' column is a 1D Series
        stock_data['Close'] = stock_data['Close'].squeeze()  # Flatten any multi-dimensional data

        # Reset the index for Plotly compatibility
        stock_data.reset_index(inplace=True)

        st.subheader("Stock Closing Prices Over Time")
        try:
            fig = px.line(
                stock_data,
                x='Date',  # Explicitly use the 'Date' column for the x-axis
                y='Close',  # Explicitly use the 'Close' column for the y-axis
                title="Closing Prices Over Time",
                template="plotly_white"
            )
            st.plotly_chart(fig)
        except Exception as e:
            st.error(f"An error occurred while generating the plot: {e}")

        # Display summary statistics
        st.subheader("Summary Statistics")
        try:
            summary_stats = stock_data[['Close']].describe()
            st.write(summary_stats)
        except Exception as e:
            st.error(f"An error occurred while displaying summary statistics: {e}")


# -------------------------------------------------------
# Tab 2: Metrics
# -------------------------------------------------------
with tab2:
    st.header("Financial Metrics")
    st.write("""
    Investigate financial metrics such as cumulative returns and rolling volatility.
    Adjust the rolling window size in the sidebar.
    """)

    if stock_data.empty:
        st.error("No data available for the selected stock ticker or date range.")
    else:
        # Calculate daily percentage returns
        stock_data['Daily Return'] = stock_data['Close'].pct_change()

        # Calculate cumulative returns
        stock_data['Cumulative Return'] = (1 + stock_data['Daily Return']).cumprod()

        # Calculate rolling volatility
        stock_data['Rolling Volatility'] = stock_data['Daily Return'].rolling(rolling_window).std()

        # Line chart: Cumulative returns
        st.subheader("Cumulative Returns Over Time")
        fig = px.line(
            stock_data, 
            x=stock_data.index, 
            y='Cumulative Return', 
            title="Cumulative Returns", 
            template="plotly_white"
        )
        st.plotly_chart(fig)

        # Line chart: Rolling volatility
        st.subheader("Rolling Volatility Over Time")
        fig = px.line(
            stock_data, 
            x=stock_data.index, 
            y='Rolling Volatility', 
            title="Rolling Volatility", 
            template="plotly_white"
        )
        st.plotly_chart(fig)

# -------------------------------------------------------
# Tab 3: Comparison
# -------------------------------------------------------

with tab3:
    st.header("Comparative Analysis")
    st.write("""
    Compare the performance of two stocks side by side. 
    Enter a second stock ticker below for comparison.
    """)

    # Fetch data for the main stock
    data1 = yf.download(stock_ticker, start=start_date, end=end_date)

    # Input for the second stock ticker
    stock2 = st.text_input("Enter the second stock ticker (e.g., TSLA):", value="TSLA")
    data2 = yf.download(stock2, start=start_date, end=end_date)

    if data1.empty or data2.empty:
        st.error("One or both stock tickers returned no data. Try again with valid tickers.")
    else:
        # Align indices
        data1 = data1[['Close']].rename(columns={'Close': stock_ticker})
        data2 = data2[['Close']].rename(columns={'Close': stock2})
        comparison_data = pd.concat([data1, data2], axis=1).dropna()

        # Visualization: Closing Prices
        st.subheader("Closing Prices Comparison")
        try:
            fig = px.line(
                comparison_data.reset_index(),
                x='Date',
                y=comparison_data.columns,
                title="Closing Prices Comparison",
                template="plotly_white"
            )
            st.plotly_chart(fig)
        except Exception as e:
            st.error(f"An error occurred while generating the plot: {e}")

        # Visualization: Cumulative Returns
        st.subheader("Cumulative Returns Comparison")
        comparison_data[stock_ticker] = (comparison_data[stock_ticker].pct_change() + 1).cumprod()
        comparison_data[stock2] = (comparison_data[stock2].pct_change() + 1).cumprod()
        try:
            fig = px.line(
                comparison_data.reset_index(),
                x='Date',
                y=comparison_data.columns,
                title="Cumulative Returns Comparison",
                template="plotly_white"
            )
            st.plotly_chart(fig)
        except Exception as e:
            st.error(f"An error occurred while generating the plot: {e}")



# -------------------------------------------------------
# Tab 4: Monte Carlo Simulation
# -------------------------------------------------------


# Tab 4: Monte Carlo Simulation
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Tab 4 content
st.header("Monte Carlo Simulation")
st.write("Forecast potential future stock prices using Monte Carlo simulations. Adjust the simulation parameters in the sidebar.")

# Sidebar Inputs
st.sidebar.subheader("Monte Carlo Parameters")
num_simulations = st.sidebar.number_input("Number of Simulations", min_value=1, max_value=1000, value=100, step=10)
num_days = st.sidebar.number_input("Number of Days", min_value=1, max_value=365, value=30, step=1)
initial_price = st.sidebar.number_input("Initial Stock Price", min_value=0.01, value=100.0, step=1.0)
volatility = st.sidebar.number_input("Volatility (%)", min_value=0.01, max_value=100.0, value=20.0, step=0.1) / 100
drift = st.sidebar.number_input("Drift (%)", min_value=-100.0, max_value=100.0, value=5.0, step=0.1) / 100

# Threshold Input
threshold = st.number_input("Enter a threshold price:", min_value=0.0, value=150.0, step=1.0)

# Generate Monte Carlo Simulation
st.subheader("Simulated Price Paths")
np.random.seed(42)  # For reproducibility
daily_returns = np.random.normal(loc=drift / num_days, scale=volatility / np.sqrt(num_days), size=(num_days, num_simulations))
price_paths = np.zeros_like(daily_returns)
price_paths[0] = initial_price

# Simulate Price Paths
for t in range(1, num_days):
    price_paths[t] = price_paths[t - 1] * (1 + daily_returns[t])

# Plot Simulated Paths
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(price_paths, alpha=0.6)
ax.set_title("Monte Carlo Simulation of Stock Price Paths")
ax.set_xlabel("Days")
ax.set_ylabel("Price")
st.pyplot(fig)

# Distribution of Final Prices
st.subheader("Distribution of Final Prices")
final_prices = price_paths[-1]

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(final_prices, bins=30, color='blue', alpha=0.7)
ax.axvline(initial_price, color='red', linestyle='dashed', linewidth=2, label="Initial Price")
ax.set_title("Distribution of Final Stock Prices")
ax.set_xlabel("Price")
ax.set_ylabel("Frequency")
ax.legend()
st.pyplot(fig)

# Probability Below Threshold
st.subheader("Probability Below Threshold")
try:
    # Ensure final_prices is 1D and compatible with threshold
    final_prices_series = pd.Series(final_prices.flatten()) if len(final_prices.shape) > 1 else pd.Series(final_prices)

    # Check if threshold is valid
    if not isinstance(threshold, (int, float)):
        st.error("Threshold must be a numeric value.")
    else:
        probability_below_threshold = (final_prices_series < threshold).mean() * 100
        st.write(f"Probability of prices falling below {threshold}: **{probability_below_threshold:.2f}%**")
except Exception as e:
    st.error(f"An error occurred while calculating the probability: {str(e)}")

