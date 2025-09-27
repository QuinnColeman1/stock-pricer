"""Stock Price Predictor with Trend Detection.

A comprehensive stock analysis tool that uses Geometric Brownian Motion (GBM)
with trend detection and Monte Carlo simulations to predict future stock prices.

Main modules:
    - fetch_stocks: Core prediction engine with StockPredictor class.
    - streamlit_app: Interactive web interface for analysis.

Example usage:
    >>> from stock_pricer import StockPredictor
    >>> predictor = StockPredictor("AAPL")
    >>> predictor.fetch_data("5y")
    >>> predictor.calibrate_parameters(trend_weight=0.6)
    >>> predictions = predictor.predict_prices(days=30, num_simulations=1000)
"""

from .fetch_stocks import StockPredictor  # noqa: F401

__version__ = "0.1.0"
__author__ = "quinn"
PACKAGE_NAME = "stock-pricer"
DESCRIPTION = "A stock price prediction tool with trend detection"

# Define what should be imported with "from stock_pricer import *"
__all__ = ["StockPredictor", "__version__", "__author__"]
