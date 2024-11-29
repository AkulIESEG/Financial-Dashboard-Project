

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


# Monte Carlo Simulation Tab
elif selected_tab == "Monte Carlo Simulation":
    st.title("Monte Carlo Simulation")
    st.write("Forecast potential future stock prices using Monte Carlo simulations. Adjust the simulation parameters in the sidebar.")

    # Parameters for Monte Carlo Simulation
    simulation_runs = st.sidebar.number_input("Number of Simulations", min_value=100, max_value=10000, value=500, step=100)
    time_horizon = st.sidebar.number_input("Time Horizon (in days)", min_value=30, max_value=365, value=250, step=10)
    threshold = st.sidebar.number_input("Enter a threshold price:", min_value=0.0, value=150.0)

    # Monte Carlo Simulation Logic
    if not stock_data.empty and 'Close' in stock_data.columns:
        # Log Returns Calculation
        daily_returns = stock_data['Close'].pct_change().dropna()
        mean_return = daily_returns.mean()
        std_dev = daily_returns.std()

        # Simulating Price Paths
        np.random.seed(42)  # Ensure reproducibility
        price_paths = np.zeros((time_horizon, simulation_runs))
        initial_price = stock_data['Close'].iloc[-1]

        for sim in range(simulation_runs):
            simulated_prices = [initial_price]
            for day in range(1, time_horizon):
                price_today = simulated_prices[-1]
                simulated_return = np.random.normal(mean_return, std_dev)
                simulated_price = price_today * (1 + simulated_return)
                simulated_prices.append(simulated_price)
            price_paths[:, sim] = simulated_prices

        # Plot Simulated Price Paths
        st.subheader("Simulated Price Paths")
        try:
            fig_simulated_paths = px.line(
                pd.DataFrame(price_paths),
                title="Simulated Stock Price Paths",
                labels={"index": "Days", "value": "Price", "variable": "Simulation"}
            )
            st.plotly_chart(fig_simulated_paths)
        except Exception as e:
            st.error(f"An error occurred while generating the simulated price paths plot: {e}")

        # Plot Distribution of Final Prices
        final_prices = price_paths[-1]
        st.subheader("Distribution of Final Prices")
        try:
            fig_final_prices = px.histogram(
                final_prices,
                nbins=50,
                title="Distribution of Simulated Final Prices",
                labels={"value": "Price", "count": "Frequency"}
            )
            st.plotly_chart(fig_final_prices)
        except Exception as e:
            st.error(f"An error occurred while generating the final prices distribution plot: {e}")

        # Probability Calculation
        st.subheader("Probability Analysis")
        st.write("Enter a threshold price in the sidebar to calculate the probability of the final price falling below the threshold.")
        
        # Debugging and probability calculation
        st.write("Debug: Final Prices Shape:", final_prices.shape if hasattr(final_prices, "shape") else "N/A")
        st.write("Debug: Final Prices Data Type:", final_prices.dtype if hasattr(final_prices, "dtype") else type(final_prices))
        st.write("Debug: Threshold Value:", threshold)

        try:
            if hasattr(final_prices, "ravel"):
                final_prices = final_prices.ravel()
            probability_below_threshold = (final_prices < threshold).mean() * 100
            st.write(f"Probability that final price is below ${threshold:.2f}: {probability_below_threshold:.2f}%")
        except Exception as e:
            st.error(f"Error while calculating probability: {e}")
    else:
        st.error("Stock data is unavailable or incomplete. Please select a valid stock ticker and ensure data is loaded properly.")


