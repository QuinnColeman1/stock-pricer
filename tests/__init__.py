"""Test suite for the stock_pricer package.

Modules:
- conftest: shared fixtures (synthetic price data, mocked yfinance, calibrated predictors)
- test_stock_predictor: unit tests for every StockPredictor method
- test_plot_and_cli: tests for plot_predictions() and the command-line main()
- test_streamlit_app: headless tests of the Streamlit pages via streamlit.testing.AppTest

No test touches the network: yfinance is always mocked.

Running tests:
    make test                       # Full run with coverage
    pytest -v                       # Verbose output
    pytest -k "trend"               # Run tests matching "trend"
"""
