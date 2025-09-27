"""
Tests for the StockPredictor class.
These tests are focused on core functionality and are excluded from type checking.
"""

# type: ignore  # Ignore type checking for this test file
import os
import sys
from datetime import datetime
from unittest import TestCase, mock

import pandas as pd
import pytest

# Import the module to test
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from stock_pricer.fetch_stocks import StockPredictor  # noqa: E402

# Mock data for testing
MOCK_HISTORY = {
    "Open": [100, 101, 102, 103, 104],
    "Close": [101, 102, 103, 104, 105],
    "High": [102, 103, 104, 105, 106],
    "Low": [99, 100, 101, 102, 103],
    "Volume": [1000, 2000, 1500, 2500, 3000],
}


class TestStockPredictor(TestCase):
    """Test cases for the StockPredictor class."""

    def setUp(self):
        """Set up test fixtures."""
        self.ticker = "AAPL"
        self.predictor = StockPredictor(self.ticker)

        # Create a mock dataframe with test data
        dates = pd.date_range(end=datetime.now(), periods=5)
        self.mock_df = pd.DataFrame(MOCK_HISTORY, index=dates)
        self.predictor.data = self.mock_df

    def test_initialization(self):
        """Test that the StockPredictor initializes correctly."""
        self.assertEqual(self.predictor.ticker, self.ticker)
        self.assertIsNone(self.predictor.mu)
        self.assertIsNone(self.predictor.sigma)
        self.assertIsNone(self.predictor.theta)
        self.assertEqual(self.predictor.current_price, 0.0)

    def test_calculate_returns(self):
        """Test the calculate_returns method."""
        # Test with valid data
        returns = self.predictor.calculate_returns()
        self.assertIsNotNone(returns)
        self.assertEqual(len(returns), len(self.mock_df) - 1)  # One less than data points

        # Test with no data
        self.predictor.data = None
        with self.assertRaises(ValueError):
            self.predictor.calculate_returns()

    def test_calculate_time_weights(self):
        """Test the calculate_time_weights method."""
        # First need to set up returns
        self.predictor.calculate_returns()

        # Test with default parameters
        weights = self.predictor.calculate_time_weights()
        self.assertIsNotNone(weights)
        self.assertAlmostEqual(weights.sum(), 1.0, places=6)

        # Test with lookback_days
        weights = self.predictor.calculate_time_weights(lookback_days=3)
        self.assertEqual(len(weights), 3)
        self.assertAlmostEqual(weights.sum(), 1.0, places=6)

    @mock.patch("yfinance.Ticker")
    def test_fetch_data(self, mock_ticker):
        """Test the fetch_data method with a mock."""
        # Setup mock
        mock_history = mock.MagicMock()
        mock_history.history.return_value = self.mock_df
        mock_ticker.return_value = mock_history

        # Test successful fetch
        result = self.predictor.fetch_data(period="5y")
        self.assertIsNotNone(result)
        self.assertEqual(self.predictor.current_price, 105)  # Last close price

        # Test with no data
        mock_history.history.return_value = pd.DataFrame()
        with self.assertRaises(ValueError):
            self.predictor.fetch_data()

    def test_detect_trend(self):
        """Test the detect_trend method."""
        # First need to set up data
        self.predictor.calculate_returns()

        # Test trend detection
        trend_info = self.predictor.detect_trend()
        self.assertIsNotNone(trend_info)
        self.assertIn("daily_trend", trend_info)
        self.assertIn("trend_strength", trend_info)

        # Test with insufficient data
        self.predictor.data = self.mock_df.iloc[:1]  # Only one data point
        with self.assertRaises(ValueError):
            self.predictor.detect_trend()

    def test_predict_prices(self):
        """Test the predict_prices method."""
        # First need to set up data and calculate parameters
        self.predictor.calculate_returns()
        self.predictor.mu = 0.0005  # Small drift
        self.predictor.sigma = 0.01  # 1% daily volatility

        # Test prediction
        predictions = self.predictor.predict_prices(days=30, num_simulations=100)
        self.assertIsNotNone(predictions)
        # Check for expected keys in the predictions
        expected_keys = [
            "all_paths",
            "daily_means",
            "daily_stds",
            "daily_5th",
            "daily_95th",
            "final_mean",
            "final_std",
            "final_percentiles",
        ]
        for key in expected_keys:
            self.assertIn(key, predictions)
        # Verify the shape of all_paths matches expectations (simulations x days)
        self.assertEqual(predictions["all_paths"].shape, (100, 31))  # 100 sims, 31 days (includes day 0)


if __name__ == "__main__":
    pytest.main()
