

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
with tab4:
    st.header("Monte Carlo Simulation")
    st.write("""
    Forecast potential future stock prices using Monte Carlo simulations. 
    Adjust the simulation parameters in the sidebar.
    """)

    # Fetch stock data using yfinance
    stock_data = yf.download(stock_ticker, start=start_date, end=end_date)

    if stock_data.empty or 'Close' not in stock_data.columns:
        st.error("No data available for the selected stock ticker or the 'Close' column is missing.")
    else:
        # Calculate daily returns
        daily_returns = stock_data['Close'].pct_change().dropna()
        mean_return = daily_returns.mean()
        std_dev_return = daily_returns.std()

        # Monte Carlo Simulation
        last_price = stock_data['Close'].iloc[-1]
        simulated_paths = []
        for _ in range(n_simulations):
            path = [last_price]
            for _ in range(n_days):
                next_price = path[-1] * (1 + np.random.normal(mean_return, std_dev_return))
                path.append(next_price)
            simulated_paths.append(path)

        # Convert results to DataFrame
        simulated_df = pd.DataFrame(simulated_paths).T

        # Simulated paths chart
        st.subheader("Simulated Price Paths")
        fig = go.Figure()
        for col in simulated_df.columns:
            fig.add_trace(go.Scatter(
                x=list(range(len(simulated_df))), 
                y=simulated_df[col], 
                mode="lines", 
                line=dict(width=1), 
                opacity=0.5
            ))
        fig.update_layout(
            title="Monte Carlo Simulated Price Paths", 
            xaxis_title="Days", 
            yaxis_title="Price", 
            template="plotly_white"
        )
        st.plotly_chart(fig)

        # Distribution of final prices
        st.subheader("Distribution of Final Prices")
        final_prices = pd.Series(simulated_df.iloc[-1].values)  # Explicitly convert to Series
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=final_prices, 
            nbinsx=20, 
            histnorm='probability', 
            marker_color='blue', 
            opacity=0.75
        ))
        fig.update_layout(
            title="Final Simulated Prices", 
            xaxis_title="Price", 
            yaxis_title="Probability", 
            template="plotly_white"
        )
        st.plotly_chart(fig)

        # Probability for threshold
        threshold = st.number_input("Enter a threshold price:", value=150.0)
        probability_below_threshold = (final_prices < threshold).mean() * 100
        st.write(f"Probability of falling below ${threshold}: {probability_below_threshold:.2f}%")

        # Download results
        if st.button("Download Simulation Results"):
            simulated_df.to_csv("MonteCarloSimulationResults.csv", index=False)
            st.success("Results saved as MonteCarloSimulationResults.csv")





