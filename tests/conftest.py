"""Shared fixtures for the stock_pricer test suite.

All price data is synthetic and generated with a fixed seed, and yfinance is
always mocked, so the suite is deterministic and never touches the network.
"""

from collections.abc import Callable, Iterator
from unittest import mock

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")  # Headless backend: never open a window during tests

from stock_pricer.fetch_stocks import StockPredictor  # noqa: E402


def make_price_frame(
    n: int = 120,
    daily_drift: float = 0.0,
    noise: float = 0.01,
    start_price: float = 100.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Build an OHLCV DataFrame shaped like yfinance's ``Ticker.history`` output.

    Close prices follow a geometric random walk with the given daily log drift.
    """
    rng = np.random.default_rng(seed)
    log_returns = daily_drift + noise * rng.standard_normal(n - 1)
    close = start_price * np.exp(np.concatenate([[0.0], np.cumsum(log_returns)]))
    index = pd.bdate_range(end="2026-10-30", periods=n)
    return pd.DataFrame(
        {
            "Open": close * 0.995,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": np.full(n, 1_000_000.0),
        },
        index=index,
    )


@pytest.fixture
def price_frame() -> Callable[..., pd.DataFrame]:
    """Factory fixture so tests can request custom synthetic data."""
    return make_price_frame


@pytest.fixture
def uptrend_data() -> pd.DataFrame:
    return make_price_frame(daily_drift=0.003, noise=0.005)


@pytest.fixture
def downtrend_data() -> pd.DataFrame:
    return make_price_frame(daily_drift=-0.003, noise=0.005)


def _loaded(ticker: str, data: pd.DataFrame) -> StockPredictor:
    """Predictor in the state fetch_data() would leave it: data + current price."""
    predictor = StockPredictor(ticker)
    predictor.data = data
    predictor.current_price = float(data["Close"].iloc[-1])
    return predictor


@pytest.fixture
def loaded_predictor(uptrend_data: pd.DataFrame) -> StockPredictor:
    return _loaded("TEST", uptrend_data)


@pytest.fixture
def calibrated_predictor(loaded_predictor: StockPredictor) -> StockPredictor:
    loaded_predictor.calibrate_parameters()
    return loaded_predictor


@pytest.fixture
def declining_predictor(downtrend_data: pd.DataFrame) -> StockPredictor:
    predictor = _loaded("DOWN", downtrend_data)
    predictor.calibrate_parameters()
    return predictor


@pytest.fixture
def small_predictions(calibrated_predictor: StockPredictor) -> dict:
    """Fast prediction run (10 days x 200 paths) from the calibrated predictor."""
    return calibrated_predictor.predict_prices(days=10, num_simulations=200)


@pytest.fixture
def mock_yf_ticker(uptrend_data: pd.DataFrame) -> Iterator[mock.MagicMock]:
    """Patch yfinance.Ticker so every ticker returns ``uptrend_data``.

    Tests can change ``mock_yf_ticker.return_value.history`` to customise behaviour.
    """
    with mock.patch("stock_pricer.fetch_stocks.yf.Ticker") as ticker_cls:
        ticker_cls.return_value.history.return_value = uptrend_data
        yield ticker_cls
