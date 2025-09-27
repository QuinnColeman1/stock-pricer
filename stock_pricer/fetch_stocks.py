import argparse
import sys
from datetime import timedelta

import numpy as np
import pandas as pd
import yfinance as yf  # type: ignore[import-untyped]
from numpy.typing import NDArray
from sklearn.linear_model import LinearRegression  # type: ignore[import-untyped]

# Type aliases
FloatArray = NDArray[np.float64]
PredictionResult = dict[str, FloatArray | float | dict[str, float]]


class StockPredictor:
    """
    Enhanced stock price predictor using Geometric Brownian Motion model with trend detection.
    Calibrates mu (drift), sigma (volatility), and theta (mean reversion) parameters with
    time-weighted calculations and trend analysis.
    """

    def __init__(self, ticker: str) -> None:
        self.ticker = ticker
        self.data: pd.DataFrame | None = None
        self.returns: pd.Series | None = None
        self.mu: float | None = None  # drift parameter
        self.sigma: float | None = None  # volatility parameter
        self.theta: float | None = None  # mean reversion parameter
        self.current_price: float = 0.0
        self.trend_strength: float = 0.0
        self.recent_trend: float = 0.0
        self.trend_info: dict[str, float] = {}

    def fetch_data(self, period: str = "5y") -> pd.DataFrame:
        """Fetch historical stock data."""
        print(f"Fetching data for {self.ticker}...")
        stock = yf.Ticker(self.ticker)
        self.data = stock.history(period=period, interval="1d")

        if self.data.empty:
            error_msg = f"No data found for {self.ticker}"
            raise ValueError(error_msg)

        self.current_price = self.data["Close"].iloc[-1]
        return self.data[["Open", "Close", "Volume"]]

    def calculate_returns(self) -> pd.Series:
        """Calculate daily log returns.

        Returns:
            pd.Series: Daily log returns

        Raises:
            ValueError: If data is not available
        """
        if self.data is None or self.data.empty:
            error_msg = "No data available. Run fetch_data() first."
            raise ValueError(error_msg)

        prices = self.data["Close"]
        if len(prices) < 2:
            error_msg = "Insufficient data points to calculate returns"
            raise ValueError(error_msg)

        # Explicitly convert to pandas Series and handle types
        returns_series = pd.Series(np.log(prices / prices.shift(1)), dtype=np.float64)
        self.returns = returns_series.dropna()

        if self.returns.empty:
            error_msg = "Failed to calculate returns (empty result)"
            raise ValueError(error_msg)

        return self.returns

    def calculate_time_weights(self, lookback_days: int | None = None) -> FloatArray:
        """
        Calculate exponential time weights for recent data.
        More recent data gets higher weight.

        Returns:
            FloatArray: Array of weights that sum to 1

        Raises:
            ValueError: If returns data is not available
        """
        if self.returns is None or len(self.returns) == 0:
            error_msg = "No returns data available. Run calculate_returns() first."
            raise ValueError(error_msg)

        if lookback_days and lookback_days > 0:
            n = min(lookback_days, len(self.returns))
            returns_subset = self.returns.iloc[-n:]
        else:
            returns_subset = self.returns
            n = len(returns_subset)

        if n == 0:
            return np.array([1.0], dtype=np.float64)  # Return single weight if no data

        # Exponential decay with half-life of n/4 days
        half_life = max(n / 4, 1)  # Ensure half_life is at least 1
        decay_factor = np.log(2) / half_life

        # Create weights with explicit type
        weights: FloatArray = np.exp(-decay_factor * np.arange(n, 0, -1, dtype=np.float64))

        # Handle potential numerical stability issues
        weight_sum = float(np.sum(weights))
        if weight_sum <= 0:
            return np.ones(n, dtype=np.float64) / n  # Fallback to equal weights if sum is non-positive

        return weights / weight_sum  # Normalize

    def detect_trend(self, lookback_days: int = 60) -> dict[str, float]:
        """
        Detect recent trend using linear regression and moving averages.

        Returns:
            dict: Dictionary containing trend metrics

        Raises:
            ValueError: If data is not available or insufficient
        """
        if self.data is None or self.data.empty:
            error_msg = "No data available. Run fetch_data() first."
            raise ValueError(error_msg)

        if len(self.data) < 2:
            error_msg = "Insufficient data points for trend detection"
            raise ValueError(error_msg)

        # Get recent price data
        n = min(max(lookback_days, 2), len(self.data))  # Ensure at least 2 points
        price_series = self.data["Close"].iloc[-n:]

        if price_series.isna().any() or price_series.le(0).any():
            error_msg = "Invalid price data (contains zeros or NaNs)"
            raise ValueError(error_msg)

        recent_prices = price_series.values

        try:
            # Linear regression for trend
            X = np.arange(n).reshape(-1, 1)
            # Ensure recent_prices is a numpy array with float64 dtype before taking log
            y = np.log(np.asarray(recent_prices, dtype=np.float64))  # Log prices for percentage trend

            model = LinearRegression()
            model.fit(X, y)

            # Daily trend (slope)
            daily_trend = float(model.coef_[0])

            # R-squared for trend strength
            y_pred = model.predict(X)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

            # Moving average analysis
            ma_trend = 0.0
            if n >= 20:
                ma_20 = price_series.iloc[-20:].mean()
                ma_5 = price_series.iloc[-5:].mean()
                ma_trend = float((ma_5 - ma_20) / ma_20) if ma_20 > 0 else 0.0

            # Volume trend (declining volume often accompanies trend exhaustion)
            volume_trend = 0.0
            if "Volume" in self.data.columns and n >= 20:
                recent_vol = self.data["Volume"].iloc[-10:].mean()
                older_vol = self.data["Volume"].iloc[-20:-10].mean()
                volume_trend = float((recent_vol - older_vol) / older_vol) if older_vol > 0 else 0.0

            self.trend_strength = r_squared
            self.recent_trend = daily_trend

            return {
                "daily_trend": daily_trend,
                "annualized_trend": daily_trend * 252,
                "trend_strength": r_squared,
                "ma_trend": ma_trend,
                "volume_trend": volume_trend,
            }

        except Exception as e:
            error_msg = "Error detecting trend"
            raise ValueError(error_msg) from e

    def calculate_dynamic_volatility(self, lookback_days: int = 30) -> float:
        """
        Calculate volatility with emphasis on recent data using EWMA
        (Exponentially Weighted Moving Average).

        Returns:
            float: Annualized volatility

        Raises:
            ValueError: If returns data is not available or insufficient
        """
        if self.returns is None or len(self.returns) == 0:
            self.calculate_returns()
            if self.returns is None or len(self.returns) == 0:
                error_msg = "Failed to calculate returns for volatility calculation"
                raise ValueError(error_msg)

        # Use exponentially weighted standard deviation
        n = min(lookback_days, len(self.returns))
        if n < 2:
            error_msg = "Insufficient data points to calculate volatility"
            raise ValueError(error_msg)

        recent_returns = self.returns.iloc[-n:]

        # EWMA with span of max(1, n//2) to avoid division by zero
        span = max(1, n // 2)
        ewm_std = recent_returns.ewm(span=span, adjust=False).std().iloc[-1]

        # Handle potential NaN values
        if pd.isna(ewm_std):
            # Fallback to simple standard deviation if EWMA fails
            ewm_std = recent_returns.std()
            if pd.isna(ewm_std):
                error_msg = "Failed to calculate volatility (all values might be identical)"
                raise ValueError(error_msg)

        # Annualize (252 trading days)
        return float(ewm_std * np.sqrt(252))

    def calibrate_parameters(self, trend_weight: float = 0.5) -> dict[str, float | dict[str, float]]:
        """
        Calibrate GBM parameters with trend detection and time-weighting.

        Args:
            trend_weight: Weight given to recent trend (0-1).
                         0 = pure historical average, 1 = pure recent trend

        Returns:
            Dict containing:
                - mu: float - Drift parameter
                - sigma: float - Volatility parameter
                - theta: float - Mean reversion parameter
                - trend_info: Dict[str, float] - Trend analysis metrics

        Raises:
            ValueError: If data is not available or insufficient for calibration
        """
        if self.returns is None or len(self.returns) == 0:
            self.calculate_returns()
            if self.returns is None or len(self.returns) == 0:
                error_msg = "Failed to calculate returns for parameter calibration"
                raise ValueError(error_msg)

        dt = 1.0 / 252.0  # Daily time step (252 trading days in a year)
        trend_info: dict[str, float] = {}

        # Detect recent trend
        try:
            trend_info = self.detect_trend()
        except Exception as e:
            error_msg = "Error detecting trend"
            raise ValueError(error_msg) from e

        # Calculate time-weighted historical drift
        try:
            weights = self.calculate_time_weights()
            weighted_mean_return = float(np.average(self.returns, weights=weights))
            historical_mu = weighted_mean_return / dt
        except Exception as e:
            error_msg = f"Error calculating historical drift: {str(e)}"
            raise ValueError(error_msg) from e

        # Combine historical drift with recent trend
        try:
            annualized_trend = float(trend_info.get("annualized_trend", 0.0))
            self.mu = (1.0 - trend_weight) * historical_mu + trend_weight * annualized_trend
        except (KeyError, TypeError) as e:
            error_msg = f"Invalid trend data: {str(e)}"
            raise ValueError(error_msg) from e

        # Dynamic volatility (recent volatility matters more)
        try:
            self.sigma = self.calculate_dynamic_volatility()
        except Exception as e:
            error_msg = f"Error calculating volatility: {str(e)}"
            raise ValueError(error_msg) from e

        # Enhanced mean reversion calculation
        # Stronger mean reversion when trend is strong but opposite to historical average
        returns_array = self.returns.values
        if len(returns_array) > 2:
            try:
                # Calculate autocorrelation with multiple lags
                autocorr_lags = min(5, len(returns_array) - 1)
                autocorr_values = [
                    pd.Series(returns_array).autocorr(lag=i) for i in range(1, autocorr_lags + 1)
                ]
                valid_autocorrs = [
                    ac for ac in autocorr_values if ac is not None and not np.isnan(ac) and ac != 0
                ]

                if valid_autocorrs:
                    avg_autocorr = float(np.mean(valid_autocorrs))
                    base_theta = -np.log(abs(avg_autocorr))
                else:
                    base_theta = 0.1

                # Adjust mean reversion based on trend strength
                # Stronger mean reversion when trend is unsustainable
                if abs(trend_info.get("daily_trend", 0)) > 0.002:  # Strong trend
                    trend_strength = float(trend_info.get("trend_strength", 0.0))
                    self.theta = base_theta * (1.0 + trend_strength)
                else:
                    self.theta = base_theta
            except Exception as e:
                error_msg = f"Error calculating mean reversion: {str(e)}"
                raise ValueError(error_msg) from e
        else:
            self.theta = 0.1

        # Store additional info
        self.trend_info = trend_info

        # Ensure all values are float before returning
        result: dict[str, float | dict[str, float]] = {
            "mu": float(self.mu) if self.mu is not None else 0.0,
            "sigma": float(self.sigma) if self.sigma is not None else 0.0,
            "theta": float(self.theta) if self.theta is not None else 0.1,
            "trend_info": trend_info,
        }
        return result

    def predict_prices(
        self,
        days: int = 30,
        num_simulations: int = 1000,
        use_trend_adjustment: bool = True,
    ) -> PredictionResult:
        """
        Predict stock prices using Monte Carlo simulation with trend-aware GBM.

        Args:
            days: Number of days to predict (must be positive)
            num_simulations: Number of Monte Carlo simulations (must be positive)
            use_trend_adjustment: Whether to adjust for detected trends

        Returns:
            Dictionary containing prediction results with keys:
                - all_paths: np.ndarray of shape (num_simulations, days+1) with all simulated paths
                - daily_means: np.ndarray of shape (days+1,) with mean price for each day
                - daily_stds: np.ndarray of shape (days+1,) with standard deviation for each day
                - daily_5th: np.ndarray of shape (days+1,) with 5th percentile for each day
                - daily_95th: np.ndarray of shape (days+1,) with 95th percentile for each day
                - final_mean: float, mean of final prices
                - final_std: float, standard deviation of final prices
                - final_percentiles: dict with 5th, 25th, 50th, 75th, 95th percentiles of final prices

        Raises:
            ValueError: If parameters are invalid or simulation fails
        """
        # Input validation
        if days <= 0:
            raise ValueError("Number of days must be positive")
        if num_simulations <= 0:
            raise ValueError("Number of simulations must be positive")

        # Ensure parameters are calibrated
        if any(param is None for param in [self.mu, self.sigma, self.theta]):
            self.calibrate_parameters()
            if any(param is None for param in [self.mu, self.sigma, self.theta]):
                error_msg = "Failed to calibrate model parameters"
                raise ValueError(error_msg)

        dt = 1.0 / 252.0  # Daily time step (252 trading days in a year)

        # Initialize price paths
        try:
            prices: FloatArray = np.zeros((num_simulations, days + 1), dtype=np.float64)
            prices[:, 0] = float(self.current_price) if self.current_price is not None else 0.0
        except Exception as e:
            error_msg = f"Failed to initialize price paths: {str(e)}"
            raise ValueError(error_msg) from e

        # Generate random shocks
        try:
            np.random.seed(42)  # For reproducible results
            random_shocks: FloatArray = np.random.normal(0, 1, (num_simulations, days))
        except Exception as e:
            error_msg = f"Failed to generate random numbers: {str(e)}"
            raise ValueError(error_msg) from e

        # Trend decay factor (trends don't continue forever)
        initial_trend_boost = 0.0
        trend_decay = 1.0

        if use_trend_adjustment and hasattr(self, "trend_info"):
            try:
                trend_decay = 0.95  # Trend impact decays by 5% per day
                initial_trend_boost = float(
                    self.trend_info.get("daily_trend", 0.0) * self.trend_info.get("trend_strength", 0.0)
                )
            except (TypeError, KeyError):
                # Fall back to no trend adjustment if trend info is invalid
                initial_trend_boost = 0.0
                trend_decay = 1.0

        # Ensure we have valid parameters
        mu = float(self.mu) if self.mu is not None else 0.0
        sigma = float(self.sigma) if self.sigma is not None else 0.0
        theta = float(self.theta) if self.theta is not None else 0.1
        current_price = float(self.current_price) if self.current_price is not None else 0.0

        # Simulate price paths
        try:
            for t in range(days):
                # Base drift with decaying trend adjustment
                trend_adjustment = initial_trend_boost * (trend_decay**t)
                adjusted_drift = (mu * dt) + trend_adjustment

                # Volatility can increase in strong trends (volatility clustering)
                volatility_multiplier = 1.0
                if hasattr(self, "recent_trend") and abs(self.recent_trend) > 0.001:
                    volatility_multiplier = 1 + 0.2 * min(t / max(1, days), 1)  # Up to 20% increase

                diffusion = sigma * volatility_multiplier * np.sqrt(dt) * random_shocks[:, t]

                # Mean reversion becomes stronger as prices deviate from trend
                log_prices = np.log(prices[:, t])
                expected_log_price = np.log(current_price) + mu * dt * (t + 1)
                mean_reversion = -theta * (log_prices - expected_log_price) * dt

                # Calculate next day prices
                log_returns = adjusted_drift + diffusion + mean_reversion
                prices[:, t + 1] = prices[:, t] * np.exp(log_returns)
        except Exception as e:
            error_msg = f"Error during price simulation: {str(e)}"
            raise ValueError(error_msg) from e

        # Calculate statistics
        try:
            final_prices = prices[:, -1]
            mean_final_price = float(np.mean(final_prices))
            std_final_price = float(np.std(final_prices, ddof=1))  # Use ddof=1 for sample std dev

            # Calculate percentiles
            percentiles = np.percentile(final_prices, [5, 25, 50, 75, 95])

            # Daily statistics
            daily_means: FloatArray = np.mean(prices, axis=0).astype(np.float64)
            daily_stds: FloatArray = np.std(prices, axis=0, ddof=1).astype(np.float64)
            daily_5th: FloatArray = np.percentile(prices, 5, axis=0).astype(np.float64)
            daily_95th: FloatArray = np.percentile(prices, 95, axis=0).astype(np.float64)

            # Create result dictionary with explicit types
            result: PredictionResult = {
                "all_paths": prices.astype(np.float64),
                "daily_means": daily_means,
                "daily_stds": daily_stds,
                "daily_5th": daily_5th,
                "daily_95th": daily_95th,
                "final_mean": mean_final_price,
                "final_std": std_final_price,
                "final_percentiles": {
                    "5th": float(percentiles[0]),
                    "25th": float(percentiles[1]),
                    "50th": float(percentiles[2]),
                    "75th": float(percentiles[3]),
                    "95th": float(percentiles[4]),
                },
            }
            return result
        except Exception as e:
            error_msg = "Error calculating statistics"
            raise ValueError(error_msg) from e

    def print_calibration_results(self) -> None:
        """
        Print calibrated parameters and trend analysis.

        Raises:
            ValueError: If required data is not available
        """
        if not hasattr(self, "ticker") or not self.ticker:
            raise ValueError("Ticker information not available")

        if not hasattr(self, "current_price") or not isinstance(self.current_price, (int, float)):
            raise ValueError("Current price not available")

        if self.mu is None or self.sigma is None or self.theta is None:
            raise ValueError("Model parameters not calibrated. Run calibrate_parameters() first.")

        print(f"\n=== Parameter Calibration Results for {self.ticker} ===")
        print(f"Current Price: ${self.current_price:.2f}")
        print(f"Mu (Annual Drift): {self.mu:.4f} ({self.mu * 100:.2f}%)")
        print(f"Sigma (Annual Volatility): {self.sigma:.4f} ({self.sigma * 100:.2f}%)")
        print(f"Theta (Mean Reversion): {self.theta:.4f}")

        # Trend analysis
        if hasattr(self, "trend_info") and isinstance(self.trend_info, dict):
            print("\n=== Trend Analysis ===")
            try:
                daily_trend = self.trend_info.get("daily_trend", 0.0)
                annualized_trend = self.trend_info.get("annualized_trend", 0.0)
                trend_strength = self.trend_info.get("trend_strength", 0.0)
                ma_trend = self.trend_info.get("ma_trend", 0.0)
                volume_trend = self.trend_info.get("volume_trend", 0.0)

                print(f"Recent Daily Trend: {daily_trend:.6f} ({daily_trend * 100:.4f}%/day)")
                print(f"Annualized Trend: {annualized_trend:.4f} ({annualized_trend * 100:.2f}%/year)")
                print(f"Trend Strength (R²): {trend_strength:.3f}")
                print(f"MA Trend (5d vs 20d): {ma_trend * 100:.2f}%")
                print(f"Volume Trend: {volume_trend * 100:.2f}%")

                # Interpretation
                if daily_trend < -0.001:
                    print("\n⚠️  WARNING: Stock is in a DECLINING trend")
                elif daily_trend > 0.001:
                    print("\n📈 Stock is in an UPWARD trend")
                else:
                    print("\n➡️  Stock is moving SIDEWAYS")
            except Exception as e:
                print(f"\n⚠️  Could not display trend analysis: {str(e)}")

        # Historical statistics
        if hasattr(self, "returns") and self.returns is not None and len(self.returns) > 0:
            try:
                returns_mean = float(np.mean(self.returns))
                returns_std = float(np.std(self.returns, ddof=1))
                print("\nHistorical Statistics:")
                print(f"Daily Return Mean: {returns_mean:.6f} ({returns_mean * 100:.4f}%)")
                print(f"Daily Return Std: {returns_std:.6f} ({returns_std * 100:.4f}%)")

                if self.sigma != 0:  # Avoid division by zero
                    sharpe_ratio = (self.mu - 0.02) / self.sigma if self.mu is not None else 0.0
                    print(f"Sharpe Ratio (approx): {sharpe_ratio:.2f}")
            except Exception as e:
                print(f"\n⚠️  Could not display historical statistics: {str(e)}")
        else:
            print("\n⚠️  No historical return data available for statistics")

    def print_predictions(self, predictions: dict, days: int = 30) -> None:
        """
        Print prediction results with trend-aware insights.

        Args:
            predictions: Dictionary containing prediction results from predict_prices()
            days: Number of days in the prediction horizon

        Raises:
            ValueError: If predictions are invalid or missing required data
        """
        if not isinstance(predictions, dict):
            raise ValueError("Predictions must be a dictionary")

        if not hasattr(self, "ticker") or not self.ticker:
            raise ValueError("Ticker information not available")

        if not hasattr(self, "current_price") or not isinstance(self.current_price, (int, float)):
            raise ValueError("Current price not available")

        # Check for required keys in predictions
        required_keys = [
            "final_mean",
            "final_std",
            "final_percentiles",
            "all_paths",
            "daily_means",
            "daily_stds",
            "daily_5th",
            "daily_95th",
        ]

        missing_keys = [key for key in required_keys if key not in predictions]
        if missing_keys:
            raise ValueError(f"Missing required prediction keys: {', '.join(missing_keys)}")

        # Check percentiles
        if not isinstance(predictions["final_percentiles"], dict):
            raise ValueError("final_percentiles must be a dictionary")

        required_percentiles = ["5th", "25th", "50th", "75th", "95th"]
        missing_percentiles = [
            p for p in required_percentiles if p not in predictions["final_percentiles"]
        ]
        if missing_percentiles:
            raise ValueError(f"Missing required percentiles: {', '.join(missing_percentiles)}")

        try:
            # Print header and basic predictions
            print(f"\n==={days}-Day Price Predictions for {self.ticker}===")
            print(f"Expected Price (Mean): ${float(predictions['final_mean']):.2f}")
            print(f"Standard Deviation: ${float(predictions['final_std']):.2f}")

            # Calculate and print expected return
            expected_return = ((float(predictions["final_mean"]) / float(self.current_price)) - 1) * 100
            print(f"Expected Return: {expected_return:.2f}%")

            # Print confidence intervals
            print("\nConfidence Intervals:")
            print(
                f"95% CI: ${float(predictions['final_percentiles']['5th']):.2f} - "
                f"${float(predictions['final_percentiles']['95th']):.2f}"
            )
            print(
                f"50% CI: ${float(predictions['final_percentiles']['25th']):.2f} - "
                f"${float(predictions['final_percentiles']['75th']):.2f}"
            )
            print(f"Median: ${float(predictions['final_percentiles']['50th']):.2f}")

            # Probability analysis
            if "all_paths" in predictions and predictions["all_paths"].size > 0:
                prob_increase = np.sum(predictions["all_paths"][:, -1] > self.current_price) / len(
                    predictions["all_paths"][:, -1]
                )
                print(f"\nProbability of price increase: {float(prob_increase) * 100:.1f}%")

                # Risk metrics
                var_5 = float(self.current_price - predictions["final_percentiles"]["5th"])
                print(
                    f"Value at Risk (5%): ${var_5:.2f} ("
                    f"{(var_5 / float(self.current_price)) * 100:.1f}%)"
                )

                # Trend-based insights
                if (
                    hasattr(self, "trend_info")
                    and isinstance(self.trend_info, dict)
                    and self.trend_info.get("daily_trend", 0) < -0.001
                ):
                    print("\n⚠️  Note: Predictions account for the detected DECLINING trend.")
                    print("   The model gives more weight to recent negative performance.")
        except (KeyError, IndexError, TypeError, ValueError) as e:
            raise ValueError(f"Error processing prediction results: {str(e)}")


def plot_predictions(predictor: "StockPredictor", predictions: dict, days: int = 30) -> None:
    """
    Plot enhanced prediction results with trend visualization.

    Args:
        predictor: StockPredictor instance with data and parameters
        predictions: Dictionary containing prediction results from predict_prices()
        days: Number of days in the prediction horizon

    Raises:
        ValueError: If inputs are invalid or required data is missing
        ImportError: If matplotlib is not installed
    """
    # Input validation
    if not hasattr(predictor, "data") or predictor.data is None:
        raise ValueError("Predictor is missing required data")

    if not isinstance(predictions, dict):
        raise ValueError("Predictions must be a dictionary")

    required_keys = ["all_paths", "daily_means", "daily_5th", "daily_95th"]
    missing_keys = [key for key in required_keys if key not in predictions]
    if missing_keys:
        raise ValueError(f"Missing required prediction keys: {', '.join(missing_keys)}")

    # Check if predictions contain valid data
    if not isinstance(predictions["all_paths"], np.ndarray) or predictions["all_paths"].size == 0:
        raise ValueError("No prediction data available to plot")

    try:
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("Plotting requires matplotlib. Install with: pip install matplotlib") from e

    try:
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[2, 1])

        # Get historical data
        if len(predictor.data) == 0:
            raise ValueError("No historical data available for plotting")

        last_date = predictor.data.index[-1].date()
        future_dates = [last_date + timedelta(days=i) for i in range(days + 1)]

        # Historical context (last 60 days or available data)
        historical_days = min(60, len(predictor.data))
        hist_dates = predictor.data.index[-historical_days:]
        hist_prices = predictor.data["Close"].iloc[-historical_days:].values

        # Plot historical prices
        ax1.plot(hist_dates, hist_prices, "b-", linewidth=2, label="Historical")

        # Plot connection line between historical and predicted
        if len(hist_dates) > 0 and len(future_dates) > 0 and len(predictions["daily_means"]) > 0:
            ax1.plot(
                [hist_dates[-1], future_dates[0]],
                [hist_prices[-1], predictions["daily_means"][0]],
                "k--",
                alpha=0.5,
            )

        # Plot sample paths (up to 50 for clarity)
        sample_paths = predictions["all_paths"][:: max(1, len(predictions["all_paths"]) // 50)]
        for path in sample_paths:
            if len(path) == len(future_dates):
                ax1.plot(future_dates, path, alpha=0.1, color="green", linewidth=0.5)

        # Plot mean and confidence intervals
        if len(predictions["daily_means"]) == len(future_dates):
            ax1.plot(
                future_dates,
                predictions["daily_means"],
                "r-",
                linewidth=2,
                label="Expected Price",
            )

        if len(predictions["daily_5th"]) == len(future_dates) and len(predictions["daily_95th"]) == len(
            future_dates
        ):
            ax1.fill_between(
                future_dates,
                predictions["daily_5th"],
                predictions["daily_95th"],
                alpha=0.3,
                color="gray",
                label="90% Confidence Interval",
            )

        # Mark current price if available
        if hasattr(predictor, "current_price") and isinstance(predictor.current_price, (int, float)):
            ax1.axhline(
                y=predictor.current_price,
                color="black",
                linestyle="--",
                label=f"Current Price: ${predictor.current_price:.2f}",
            )

        # Add trend line if available
        if (
            hasattr(predictor, "trend_info")
            and isinstance(predictor.trend_info, dict)
            and "daily_trend" in predictor.trend_info
            and hasattr(predictor, "current_price")
        ):
            try:
                daily_trend = float(predictor.trend_info["daily_trend"])
                trend_line = float(predictor.current_price) * np.exp(daily_trend * np.arange(days + 1))
                ax1.plot(
                    future_dates,
                    trend_line,
                    "orange",
                    linestyle=":",
                    label=f"Current Trend ({daily_trend * 100:.3f}%/day)",
                )
            except (TypeError, ValueError) as e:
                print(f"Warning: Could not plot trend line: {str(e)}")

        # Set plot titles and labels
        ticker = getattr(predictor, "ticker", "UNKNOWN")
        ax1.set_title(
            f"{ticker} - {days} Day Price Prediction (with Historical Context)",
            fontsize=14,
        )
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Price ($)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Format x-axis dates
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax1.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, (historical_days + days) // 10)))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

        # Bottom plot: Return distribution
        try:
            final_prices = predictions["all_paths"][:, -1]
            if (
                len(final_prices) > 0
                and hasattr(predictor, "current_price")
                and predictor.current_price > 0
            ):
                final_returns = (final_prices / predictor.current_price - 1) * 100
                ax2.hist(final_returns, bins=50, alpha=0.7, color="blue", edgecolor="black")
                ax2.axvline(x=0, color="black", linestyle="--", label="Break-even")

                mean_return = float(np.mean(final_returns))
                ax2.axvline(
                    x=mean_return,
                    color="red",
                    linestyle="-",
                    label=f"Expected: {mean_return:.1f}%",
                )

                ax2.set_title(f"Distribution of {days}-Day Returns", fontsize=12)
                ax2.set_xlabel("Return (%)")
                ax2.set_ylabel("Frequency")
                ax2.legend()
                ax2.grid(True, alpha=0.3)
        except (IndexError, ValueError) as e:
            print(f"Warning: Could not plot return distribution: {str(e)}")
            # Hide the second subplot if we can't plot returns
            ax2.axis("off")

        plt.tight_layout()
        plt.show()

    except Exception as e:
        plt.close("all")  # Clean up any figures
        raise ValueError(f"Error generating plot: {str(e)}") from e


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch stock data, calibrate parameters with trend detection, and predict future prices."
        )
    )
    parser.add_argument(
        "--tickers",
        "-t",
        nargs="*",
        help="Ticker symbols (e.g. GOOGL MSFT AAPL). If omitted, you'll be prompted.",
    )
    parser.add_argument(
        "--days",
        "-d",
        type=int,
        default=30,
        help="Number of days to predict (default: 30)",
    )
    parser.add_argument(
        "--simulations",
        "-s",
        type=int,
        default=1000,
        help="Number of Monte Carlo simulations (default: 1000)",
    )
    parser.add_argument(
        "--plot",
        "-p",
        action="store_true",
        help="Generate prediction plots (requires matplotlib)",
    )
    parser.add_argument(
        "--trend-weight",
        "-w",
        type=float,
        default=0.5,
        help="Weight given to recent trend vs historical average (0-1, default: 0.5)",
    )

    args = parser.parse_args()

    # Validate trend weight
    if not 0 <= args.trend_weight <= 1:
        print("Error: trend-weight must be between 0 and 1")
        sys.exit(1)

    # Determine ticker list
    if args.tickers:
        tickers = args.tickers
    else:
        raw = input("Enter ticker symbol(s), comma-separated (e.g. GOOGL,MSFT): ").strip()
        if not raw:
            print("No ticker provided. Exiting.")
            sys.exit(1)
        tickers = [t.strip().upper() for t in raw.split(",") if t.strip()]

    # Process each ticker
    for ticker in tickers:
        print(f"\n{'=' * 60}")
        print(f"ANALYZING: {ticker}")
        print("=" * 60)

        try:
            # Initialize predictor
            predictor = StockPredictor(ticker)

            # Fetch data and display recent prices
            recent_data = predictor.fetch_data()
            print("\nRecent Data (last 5 days):")
            print(recent_data.tail())

            # Calibrate parameters with trend detection
            print(f"\nCalibrating with trend weight: {args.trend_weight}")
            predictor.calibrate_parameters(trend_weight=args.trend_weight)
            predictor.print_calibration_results()

            # Generate predictions
            print(f"\nGenerating {args.days}-day predictions with {args.simulations} simulations...")
            predictions = predictor.predict_prices(days=args.days, num_simulations=args.simulations)
            predictor.print_predictions(predictions, args.days)

            # Plot if requested
            if args.plot:
                plot_predictions(predictor, predictions, args.days)

        except Exception as e:
            print(f"Error analyzing {ticker}: {e}")
            continue

    print(f"\n{'=' * 60}")
    print("Analysis complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
