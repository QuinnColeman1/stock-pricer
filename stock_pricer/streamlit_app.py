"""Streamlit web application for stock price prediction and analysis.

This module provides an interactive web interface for analyzing stock prices
and making predictions using the StockPredictor class.
"""

from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from stock_pricer.fetch_stocks import StockPredictor

# Page configuration
st.set_page_config(page_title="Stock Price Predictor", page_icon="📈", layout="wide")


def initialize_session_state() -> None:
    """Initialize session state variables.

    Sets up the initial state for the Streamlit application.
    """
    if "predictor" not in st.session_state:
        st.session_state.predictor = None
    if "predictions" not in st.session_state:
        st.session_state.predictions = None
    if "ticker" not in st.session_state:
        st.session_state.ticker = ""
    if "days" not in st.session_state:
        st.session_state.days = 30
    if "simulations" not in st.session_state:
        st.session_state.simulations = 1000
    if "trend_weight" not in st.session_state:
        st.session_state.trend_weight = 0.5


def page_inputs() -> None:
    """Display the input parameters page.

    This page allows users to input stock tickers and configure prediction parameters.
    """
    st.title("📊 Stock Price Predictor with Trend Detection")
    st.markdown("---")

    st.header("🔧 Input Parameters")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Stock Selection")
        ticker = (
            st.text_input(
                "Enter Stock Ticker:",
                value=st.session_state.ticker,
                placeholder="e.g., GOOGL, AAPL, MSFT",
                help="Enter a valid stock ticker symbol",
            )
            .upper()
            .strip()
        )

        period = st.selectbox(
            "Historical Data Period:",
            options=["1y", "2y", "5y", "10y", "max"],
            index=2,  # Default to 5y
            help="Amount of historical data to use for calibration",
        )

        trend_weight = st.slider(
            "Trend Weight:",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.trend_weight,
            step=0.1,
            help="0 = Use only historical average, 1 = Use only recent trend",
        )

    with col2:
        st.subheader("Prediction Settings")
        days = st.number_input(
            "Prediction Days:",
            min_value=1,
            max_value=365,
            value=st.session_state.days,
            help="Number of days to predict into the future",
        )

        simulations = st.selectbox(
            "Monte Carlo Simulations:",
            options=[500, 1000, 2000, 5000, 10000],
            index=1,  # Default to 1000
            help="More simulations = more accurate but slower",
        )

        use_trend = st.checkbox(
            "Use Trend Adjustment",
            value=True,
            help="Apply detected trend patterns to predictions",
        )

    st.markdown("---")

    # Analyze button
    if st.button("🚀 Analyze Stock", type="primary", use_container_width=True):
        if not ticker:
            st.error("Please enter a stock ticker!")
            return

        try:
            with st.spinner(f"Fetching data and analyzing trends for {ticker}..."):
                # Create predictor instance
                predictor = StockPredictor(ticker)

                # Fetch data
                data = predictor.fetch_data(period=period)

                # Calibrate parameters with trend detection
                params = predictor.calibrate_parameters(trend_weight=trend_weight)

                # Generate predictions
                predictions = predictor.predict_prices(
                    days=int(days),
                    num_simulations=int(simulations),
                    use_trend_adjustment=use_trend,
                )

                # Store in session state
                st.session_state.predictor = predictor
                st.session_state.predictions = predictions
                st.session_state.ticker = ticker
                st.session_state.days = int(days)
                st.session_state.simulations = int(simulations)
                st.session_state.trend_weight = trend_weight

                st.success(f"✅ Analysis complete for {ticker}!")

                # Show trend analysis summary
                if hasattr(predictor, "trend_info"):
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        if predictor.trend_info["daily_trend"] < -0.001:
                            st.error("⚠️ DECLINING TREND DETECTED")
                        elif predictor.trend_info["daily_trend"] > 0.001:
                            st.success("📈 UPWARD TREND DETECTED")
                        else:
                            st.info("➡️ SIDEWAYS MOVEMENT")

                    with col2:
                        st.metric(
                            "Daily Trend",
                            f"{predictor.trend_info['daily_trend'] * 100:.3f}%",
                            f"{predictor.trend_info['annualized_trend'] * 100:.1f}% annualized",
                        )

                    with col3:
                        st.metric(
                            "Trend Strength (R²)",
                            f"{predictor.trend_info['trend_strength']:.3f}",
                            help="Higher values indicate stronger trend",
                        )

                st.info(
                    "Navigate to the 'Price Analysis' or 'Monte Carlo' tabs to view detailed results."
                )

        except Exception as e:
            st.error(f"Error analyzing {ticker}: {str(e)}")
            return

    # Display current parameters if available
    if st.session_state.predictor is not None:
        st.markdown("---")
        st.header("📋 Current Analysis")

        predictor = st.session_state.predictor

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Current Price", f"${predictor.current_price:.2f}")

        with col2:
            st.metric(
                "Mu (Drift)",
                f"{predictor.mu:.4f}",
                f"{predictor.mu * 100:.2f}% annual",
                help="Annual expected return",
            )

        with col3:
            st.metric(
                "Sigma (Volatility)",
                f"{predictor.sigma:.4f}",
                f"{predictor.sigma * 100:.2f}% annual",
                help="Annual volatility",
            )

        with col4:
            st.metric(
                "Theta (Mean Reversion)",
                f"{predictor.theta:.4f}",
                help="Speed of mean reversion",
            )


def page_price_analysis() -> None:
    """Display the price analysis page.

    This page shows historical price data, technical indicators, and trend analysis.
    """
    st.title("📈 Price Analysis & Predictions")
    st.markdown("---")

    if st.session_state.predictor is None:
        st.warning("⚠️ Please run analysis on the 'Input Parameters' page first!")
        return

    predictor = st.session_state.predictor
    predictions = st.session_state.predictions

    # Trend Analysis Section
    if hasattr(predictor, "trend_info"):
        st.header("🔍 Trend Analysis")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Daily Trend",
                f"{predictor.trend_info['daily_trend'] * 100:.4f}%/day",
                f"{predictor.trend_info['annualized_trend'] * 100:.2f}%/year",
            )

        with col2:
            st.metric(
                "Trend Strength",
                f"{predictor.trend_info['trend_strength']:.3f}",
                "R-squared value",
            )

        with col3:
            st.metric("MA Trend (5d vs 20d)", f"{predictor.trend_info['ma_trend'] * 100:.2f}%")

        with col4:
            st.metric("Volume Trend", f"{predictor.trend_info['volume_trend'] * 100:.2f}%")

    # Historical price chart with trend overlay
    st.header("📊 Historical Price Data with Trend")

    historical_data = predictor.data.copy()

    # Calculate trend line for historical data
    if hasattr(predictor, "trend_info"):
        n = min(60, len(historical_data))
        recent_data = historical_data.iloc[-n:]
        x = np.arange(n)
        trend_line = recent_data["Close"].iloc[0] * np.exp(predictor.trend_info["daily_trend"] * x)

    fig_hist = go.Figure()

    # Add historical prices
    fig_hist.add_trace(
        go.Scatter(
            x=historical_data.index,
            y=historical_data["Close"],
            mode="lines",
            name="Historical Close Price",
            line=dict(color="blue", width=2),
        )
    )

    # Add trend line
    if hasattr(predictor, "trend_info"):
        fig_hist.add_trace(
            go.Scatter(
                x=recent_data.index,
                y=trend_line,
                mode="lines",
                name=f"Trend ({predictor.trend_info['daily_trend'] * 100:.3f}%/day)",
                line=dict(color="orange", width=2, dash="dash"),
            )
        )

    # Add moving averages
    if len(historical_data) >= 20:
        ma_20 = historical_data["Close"].rolling(window=20).mean()
        ma_5 = historical_data["Close"].rolling(window=5).mean()

        fig_hist.add_trace(
            go.Scatter(
                x=historical_data.index,
                y=ma_20,
                mode="lines",
                name="MA 20",
                line=dict(color="green", width=1, dash="dot"),
            )
        )

        fig_hist.add_trace(
            go.Scatter(
                x=historical_data.index,
                y=ma_5,
                mode="lines",
                name="MA 5",
                line=dict(color="red", width=1, dash="dot"),
            )
        )

    fig_hist.update_layout(
        title=f"{predictor.ticker} - Historical Close Prices with Trend Analysis",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode="x unified",
        height=500,
    )

    st.plotly_chart(fig_hist, use_container_width=True)

    # Prediction chart
    st.header("🔮 Price Predictions")

    # Create future dates
    last_date = predictor.data.index[-1].date()
    future_dates = [last_date + timedelta(days=i) for i in range(st.session_state.days + 1)]

    fig_pred = go.Figure()

    # Add historical context
    historical_days = min(30, len(predictor.data))
    hist_dates = predictor.data.index[-historical_days:].date
    hist_prices = predictor.data["Close"].iloc[-historical_days:].values

    fig_pred.add_trace(
        go.Scatter(
            x=hist_dates,
            y=hist_prices,
            mode="lines",
            name="Historical",
            line=dict(color="blue", width=2),
        )
    )

    # Add prediction mean
    fig_pred.add_trace(
        go.Scatter(
            x=future_dates,
            y=predictions["daily_means"],
            mode="lines",
            name="Expected Price",
            line=dict(color="red", width=3),
        )
    )

    # Add confidence intervals
    fig_pred.add_trace(
        go.Scatter(
            x=future_dates,
            y=predictions["daily_95th"],
            mode="lines",
            name="95th Percentile",
            line=dict(color="gray", width=1, dash="dash"),
            showlegend=False,
        )
    )

    fig_pred.add_trace(
        go.Scatter(
            x=future_dates,
            y=predictions["daily_5th"],
            mode="lines",
            name="5th Percentile",
            line=dict(color="gray", width=1, dash="dash"),
            fill="tonexty",
            fillcolor="rgba(128,128,128,0.2)",
            showlegend=False,
        )
    )

    # Add current price line
    fig_pred.add_hline(
        y=predictor.current_price,
        line_dash="dash",
        line_color="black",
        annotation_text=f"Current: ${predictor.current_price:.2f}",
    )

    # Add trend projection if available
    if hasattr(predictor, "trend_info"):
        trend_projection = predictor.current_price * np.exp(
            predictor.trend_info["daily_trend"] * np.arange(st.session_state.days + 1)
        )
        fig_pred.add_trace(
            go.Scatter(
                x=future_dates,
                y=trend_projection,
                mode="lines",
                name="Pure Trend Projection",
                line=dict(color="orange", width=2, dash="dot"),
            )
        )

    fig_pred.update_layout(
        title=f"{predictor.ticker} - {st.session_state.days} Day Price Prediction",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode="x unified",
        height=500,
    )

    st.plotly_chart(fig_pred, use_container_width=True)

    # Summary statistics
    st.header("📊 Prediction Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Expected Outcome")
        expected_return = ((predictions["final_mean"] / predictor.current_price) - 1) * 100
        st.metric(
            "Expected Price",
            f"${predictions['final_mean']:.2f}",
            f"{expected_return:+.2f}%",
        )
        st.metric("Standard Deviation", f"${predictions['final_std']:.2f}")

    with col2:
        st.subheader("Risk Metrics")
        var_5 = predictor.current_price - predictions["final_percentiles"]["5th"]
        prob_increase = np.sum(predictions["all_paths"][:, -1] > predictor.current_price) / len(
            predictions["all_paths"][:, -1]
        )

        st.metric(
            "Value at Risk (5%)",
            f"${var_5:.2f}",
            f"{var_5 / predictor.current_price * 100:.1f}%",
        )
        st.metric("Probability of Gain", f"{prob_increase * 100:.1f}%")

    with col3:
        st.subheader("Price Ranges")
        st.metric(
            "95% Confidence Interval",
            f"${predictions['final_percentiles']['5th']:.2f} - "
            f"${predictions['final_percentiles']['95th']:.2f}",
        )
        st.metric(
            "50% Confidence Interval",
            f"${predictions['final_percentiles']['25th']:.2f} - "
            f"${predictions['final_percentiles']['75th']:.2f}",
        )


def page_monte_carlo() -> None:
    """Display the Monte Carlo simulation page.

    This page shows the results of Monte Carlo simulations for price prediction,
    including confidence intervals and probability distributions.
    """
    st.title("🎲 Monte Carlo Simulation")
    st.markdown("---")

    if st.session_state.predictor is None:
        st.warning("⚠️ Please run analysis on the 'Input Parameters' page first!")
        return

    predictor = st.session_state.predictor
    predictions = st.session_state.predictions

    # Simulation overview
    st.header("🔢 Simulation Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Simulations Run", f"{st.session_state.simulations:,}")

    with col2:
        st.metric("Prediction Days", st.session_state.days)

    with col3:
        st.metric("Current Price", f"${predictor.current_price:.2f}")

    with col4:
        st.metric("Trend Weight", f"{st.session_state.trend_weight:.1f}")

    # Monte Carlo paths visualization
    st.header("📈 Simulated Price Paths")

    # Create future dates
    last_date = predictor.data.index[-1].date()
    future_dates = [last_date + timedelta(days=i) for i in range(st.session_state.days + 1)]

    fig_mc = go.Figure()

    # Show subset of paths to avoid overcrowding
    num_paths_to_show = min(100, st.session_state.simulations)
    step = max(1, st.session_state.simulations // num_paths_to_show)

    for i in range(0, st.session_state.simulations, step):
        fig_mc.add_trace(
            go.Scatter(
                x=future_dates,
                y=predictions["all_paths"][i],
                mode="lines",
                line=dict(color="lightblue", width=0.5),
                opacity=0.3,
                showlegend=False,
                hovertemplate="<extra></extra>",
            )
        )

    # Add mean path
    fig_mc.add_trace(
        go.Scatter(
            x=future_dates,
            y=predictions["daily_means"],
            mode="lines",
            name="Mean Path",
            line=dict(color="red", width=3),
        )
    )

    # Add percentile paths
    fig_mc.add_trace(
        go.Scatter(
            x=future_dates,
            y=predictions["daily_5th"],
            mode="lines",
            name="5th Percentile",
            line=dict(color="darkgreen", width=2, dash="dash"),
        )
    )

    fig_mc.add_trace(
        go.Scatter(
            x=future_dates,
            y=predictions["daily_95th"],
            mode="lines",
            name="95th Percentile",
            line=dict(color="darkred", width=2, dash="dash"),
        )
    )

    # Add current price
    fig_mc.add_hline(
        y=predictor.current_price,
        line_dash="dash",
        line_color="black",
        annotation_text=f"Starting Price: ${predictor.current_price:.2f}",
    )

    fig_mc.update_layout(
        title=f"{predictor.ticker} - Monte Carlo Simulation ({num_paths_to_show} paths shown)",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        height=600,
        hovermode="x unified",
    )

    st.plotly_chart(fig_mc, use_container_width=True)

    # Final price distribution
    st.header("📊 Final Price Distribution")

    final_prices = predictions["all_paths"][:, -1]
    final_returns = (final_prices / predictor.current_price - 1) * 100

    col1, col2 = st.columns(2)

    with col1:
        # Histogram of prices
        fig_hist = px.histogram(
            x=final_prices,
            nbins=50,
            title=f"Distribution of Final Prices (Day {st.session_state.days})",
        )
        fig_hist.add_vline(
            x=predictor.current_price,
            line_dash="dash",
            line_color="black",
            annotation_text="Current Price",
        )
        fig_hist.add_vline(
            x=predictions["final_mean"],
            line_dash="dash",
            line_color="red",
            annotation_text="Expected Price",
        )
        fig_hist.update_layout(xaxis_title="Price ($)", yaxis_title="Frequency", height=400)
        st.plotly_chart(fig_hist, use_container_width=True)

    with col2:
        # Histogram of returns
        fig_returns = px.histogram(
            x=final_returns,
            nbins=50,
            title=f"Distribution of {st.session_state.days}-Day Returns",
        )
        fig_returns.add_vline(x=0, line_dash="dash", line_color="black", annotation_text="Break-even")
        fig_returns.add_vline(
            x=np.mean(final_returns),
            line_dash="dash",
            line_color="red",
            annotation_text=f"Expected: {np.mean(final_returns):.1f}%",
        )

        # Color negative returns differently
        fig_returns.update_traces(marker_color=["red" if x < 0 else "green" for x in final_returns])

        fig_returns.update_layout(xaxis_title="Return (%)", yaxis_title="Frequency", height=400)
        st.plotly_chart(fig_returns, use_container_width=True)

    # Detailed statistics table
    st.header("📋 Detailed Statistics")

    # Calculate additional statistics
    from scipy import stats as scipy_stats

    skewness = scipy_stats.skew(final_prices)
    kurtosis = scipy_stats.kurtosis(final_prices)

    stats_data = {
        "Metric": [
            "Mean",
            "Median",
            "Standard Deviation",
            "Minimum",
            "Maximum",
            "5th Percentile",
            "25th Percentile",
            "75th Percentile",
            "95th Percentile",
            "Skewness",
            "Kurtosis",
            "Probability of Gain",
            "Sharpe Ratio",
        ],
        "Value": [
            f"${predictions['final_mean']:.2f}",
            f"${predictions['final_percentiles']['50th']:.2f}",
            f"${predictions['final_std']:.2f}",
            f"${final_prices.min():.2f}",
            f"${final_prices.max():.2f}",
            f"${predictions['final_percentiles']['5th']:.2f}",
            f"${predictions['final_percentiles']['25th']:.2f}",
            f"${predictions['final_percentiles']['75th']:.2f}",
            f"${predictions['final_percentiles']['95th']:.2f}",
            f"{skewness:.3f}",
            f"{kurtosis:.3f}",
            f"{np.sum(final_prices > predictor.current_price) / len(final_prices) * 100:.1f}%",
            f"{(predictor.mu - 0.02) / predictor.sigma:.2f}",
        ],
    }

    stats_df = pd.DataFrame(stats_data)
    st.dataframe(stats_df, hide_index=True, use_container_width=True)


def main() -> None:
    """Run the main application.

    This function initializes the application state and handles page navigation
    between different sections of the stock price prediction tool.
    """
    initialize_session_state()

    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.selectbox(
        "Select Page:",
        ["🔧 Input Parameters", "📈 Price Analysis", "🎲 Monte Carlo Simulation"],
    )

    # Display selected page
    if page == "🔧 Input Parameters":
        page_inputs()
    elif page == "📈 Price Analysis":
        page_price_analysis()
    elif page == "🎲 Monte Carlo Simulation":
        page_monte_carlo()

    # Sidebar info
    if st.session_state.predictor is not None:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📊 Current Analysis")
        st.sidebar.write(f"**Ticker:** {st.session_state.ticker}")
        st.sidebar.write(f"**Days:** {st.session_state.days}")
        st.sidebar.write(f"**Simulations:** {st.session_state.simulations:,}")
        st.sidebar.write(f"**Trend Weight:** {st.session_state.trend_weight}")

        if hasattr(st.session_state.predictor, "trend_info"):
            trend = st.session_state.predictor.trend_info["daily_trend"]
            if trend < -0.001:
                st.sidebar.error("⚠️ Declining Trend")
            elif trend > 0.001:
                st.sidebar.success("📈 Upward Trend")
            else:
                st.sidebar.info("➡️ Sideways Movement")


if __name__ == "__main__":
    main()
