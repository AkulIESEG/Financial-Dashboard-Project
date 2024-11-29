

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


# Import necessary libraries
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# Function to simulate Monte Carlo paths
def simulate_monte_carlo(stock_data, num_simulations=1000, num_days=252, threshold=150.00):
    returns = stock_data['Close'].pct_change()
    mean_return = returns.mean()
    volatility = returns.std()

    # Simulating the future price paths
    simulations = np.zeros((num_simulations, num_days))
    for i in range(num_simulations):
        price_path = [stock_data['Close'].iloc[-1]]
        for t in range(1, num_days):
            price_path.append(price_path[-1] * (1 + np.random.normal(mean_return, volatility)))
        simulations[i] = price_path

    return simulations

# Function to calculate probability of crossing threshold
def calculate_probability(final_prices, threshold):
    # Ensure final_prices is 1-dimensional
    final_prices = final_prices.flatten() if isinstance(final_prices, np.ndarray) and final_prices.ndim > 1 else final_prices

    # Ensure threshold is a scalar
    if not isinstance(threshold, (float, int)):
        raise ValueError("Threshold must be a scalar (float or int).")

    # Calculating the probability of prices below the threshold
    probability_below_threshold = (final_prices < threshold).mean() * 100
    return probability_below_threshold

# Streamlit UI components
st.title("Monte Carlo Simulation for Stock Prices")
st.sidebar.header("Simulation Parameters")

# Stock data input (example using AAPL)
stock_ticker = st.sidebar.text_input("Enter a Stock Ticker (e.g., AAPL, TSLA):", value="AAPL")
start_date = st.sidebar.date_input("Start Date:", value=pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("End Date:", value=pd.to_datetime("2024-01-01"))

# Fetch stock data using yfinance
stock_data = yf.download(stock_ticker, start=start_date, end=end_date)

if stock_data.empty:
    st.error(f"No data available for {stock_ticker} in the selected date range.")
else:
    st.sidebar.markdown("### Stock Data")
    st.sidebar.write(stock_data.tail())  # Display last 5 rows of stock data in sidebar

    # Inputs for simulation
    num_simulations = st.sidebar.number_input("Number of Simulations", min_value=100, max_value=5000, value=1000)
    num_days = st.sidebar.number_input("Number of Days (Forecast)", min_value=10, max_value=500, value=252)
    threshold = st.sidebar.number_input("Threshold Price", min_value=0.0, value=150.00)

    # Simulate price paths
    simulations = simulate_monte_carlo(stock_data, num_simulations, num_days, threshold)

    # Plot the simulated price paths
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(simulations.T, color='blue', alpha=0.1)
    ax.set_title('Monte Carlo Simulations of Stock Price Paths')
    ax.set_xlabel('Days')
    ax.set_ylabel('Price ($)')
    st.pyplot(fig)

    # Distribution of final prices
    final_prices = simulations[:, -1]
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.hist(final_prices, bins=50, color='green', edgecolor='black')
    ax2.set_title('Distribution of Final Prices')
    ax2.set_xlabel('Price ($)')
    ax2.set_ylabel('Frequency')
    st.pyplot(fig2)

    # Calculate and display probability of price crossing the threshold
    probability_below_threshold = calculate_probability(final_prices, threshold)
    st.write(f"Probability of the stock price falling below {threshold} in {num_days} days: {probability_below_threshold:.2f}%")
