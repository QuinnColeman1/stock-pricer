"""
Test suite for Stock Price Predictor

This package contains all unit and integration tests for the stock_pricer module.

Test modules:
- test_basic: Basic functionality tests
- test_fetch_stocks: Tests for the StockPredictor class and related functions
- test_streamlit_app: Tests for the Streamlit web interface components

Running tests:
    pytest                          # Run all tests
    pytest -v                       # Verbose output
    pytest --cov=stock_pricer      # With coverage report
    pytest -k "trend"              # Run tests matching "trend"
"""

import sys
from pathlib import Path

# Add parent directory to path for imports during testing
# This ensures tests can import from stock_pricer regardless of where pytest is run
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
