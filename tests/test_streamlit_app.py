"""Headless tests for the Streamlit app using streamlit.testing.v1.AppTest.

The app script runs in-process, so the yfinance mock from conftest applies to it.
"""

from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from stock_pricer.fetch_stocks import StockPredictor

APP_PATH = str(Path(__file__).resolve().parents[1] / "stock_pricer" / "streamlit_app.py")
TIMEOUT = 60

INPUT_PAGE = "🔧 Input Parameters"
ANALYSIS_PAGE = "📈 Price Analysis"
MONTE_CARLO_PAGE = "🎲 Monte Carlo Simulation"


def by_label(widgets, label: str):
    return next(w for w in widgets if w.label == label)


@pytest.fixture
def app() -> AppTest:
    return AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)


@pytest.fixture
def app_with_results(app, calibrated_predictor) -> AppTest:
    """App whose session state already holds a finished analysis (10 days x 200 paths)."""
    days, sims = 10, 200
    app.session_state["predictor"] = calibrated_predictor
    predictions = calibrated_predictor.predict_prices(days=days, num_simulations=sims)
    app.session_state["predictions"] = predictions
    app.session_state["ticker"] = calibrated_predictor.ticker
    app.session_state["days"] = days
    app.session_state["simulations"] = sims
    app.session_state["trend_weight"] = 0.5
    return app


def go_to(app: AppTest, page: str) -> AppTest:
    app.run()
    by_label(app.sidebar.selectbox, "Select Page:").select(page)
    return app.run()


def assert_no_exception(app: AppTest) -> None:
    assert not app.exception, [e.value for e in app.exception]


# --------------------------------------------------------------------------- #
# Session state & navigation
# --------------------------------------------------------------------------- #
class TestStartup:
    def test_initial_state_and_input_page(self, app):
        app.run()
        assert_no_exception(app)
        assert app.session_state["predictor"] is None
        assert app.session_state["predictions"] is None
        assert app.session_state["ticker"] == ""
        assert app.session_state["days"] == 30
        assert app.session_state["simulations"] == 1000
        assert app.session_state["trend_weight"] == 0.5
        assert "Stock Price Predictor" in app.title[0].value
        assert len(app.sidebar.markdown) == 0  # no "Current Analysis" block yet

    @pytest.mark.parametrize("page", [ANALYSIS_PAGE, MONTE_CARLO_PAGE])
    def test_result_pages_warn_without_analysis(self, app, page):
        go_to(app, page)
        assert_no_exception(app)
        assert "Please run analysis" in app.warning[0].value


# --------------------------------------------------------------------------- #
# Input page
# --------------------------------------------------------------------------- #
class TestInputPage:
    def test_empty_ticker_shows_error(self, app):
        app.run()
        app.button[0].click().run()
        assert "Please enter a stock ticker!" in app.error[0].value
        assert app.session_state["predictor"] is None

    def test_successful_analysis(self, app, mock_yf_ticker):
        app.run()
        app.text_input[0].input(" aapl ")
        by_label(app.number_input, "Prediction Days:").set_value(7)
        by_label(app.selectbox, "Monte Carlo Simulations:").select(500)
        by_label(app.selectbox, "Historical Data Period:").select("2y")
        app.button[0].click().run()

        assert_no_exception(app)
        mock_yf_ticker.assert_called_with("AAPL")
        mock_yf_ticker.return_value.history.assert_called_with(period="2y", interval="1d")

        assert any("Analysis complete for AAPL" in s.value for s in app.success)
        assert any("UPWARD TREND" in s.value for s in app.success)
        assert app.session_state["ticker"] == "AAPL"
        assert app.session_state["days"] == 7
        assert app.session_state["simulations"] == 500
        assert app.session_state["predictions"]["all_paths"].shape == (500, 8)

        labels = {m.label for m in app.metric}
        assert {"Current Price", "Mu (Drift)", "Sigma (Volatility)", "Theta (Mean Reversion)"} <= labels
        assert any("Ticker:** AAPL" in m.value for m in app.sidebar.markdown)

    def test_fetch_failure_shows_error(self, app, mock_yf_ticker):
        mock_yf_ticker.return_value.history.return_value = pd.DataFrame()
        app.run()
        app.text_input[0].input("BAD")
        app.button[0].click().run()

        assert_no_exception(app)
        assert "Error analyzing BAD" in app.error[0].value
        assert app.session_state["predictor"] is None

    def test_declining_trend_banner(self, app, mock_yf_ticker, downtrend_data):
        mock_yf_ticker.return_value.history.return_value = downtrend_data
        app.run()
        app.text_input[0].input("DOWN")
        by_label(app.selectbox, "Monte Carlo Simulations:").select(500)
        app.button[0].click().run()

        assert_no_exception(app)
        assert any("DECLINING TREND" in e.value for e in app.error)
        assert any("Declining Trend" in e.value for e in app.sidebar.error)


# --------------------------------------------------------------------------- #
# Result pages
# --------------------------------------------------------------------------- #
class TestResultPages:
    def test_price_analysis_page(self, app_with_results):
        go_to(app_with_results, ANALYSIS_PAGE)
        assert_no_exception(app_with_results)

        labels = {m.label for m in app_with_results.metric}
        assert {
            "Daily Trend",
            "Trend Strength",
            "MA Trend (5d vs 20d)",
            "Volume Trend",
            "Expected Price",
            "Value at Risk (5%)",
            "Probability of Gain",
            "95% Confidence Interval",
        } <= labels
        assert len(app_with_results.get("plotly_chart")) == 2

    def test_monte_carlo_page(self, app_with_results):
        go_to(app_with_results, MONTE_CARLO_PAGE)
        assert_no_exception(app_with_results)

        metrics = {m.label: m.value for m in app_with_results.metric}
        assert metrics["Simulations Run"] == "200"
        assert metrics["Prediction Days"] == "10"
        assert len(app_with_results.get("plotly_chart")) == 3

        stats = app_with_results.dataframe[0].value
        assert list(stats["Metric"]) == [
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
        ]

    def test_input_page_shows_current_analysis(self, app_with_results):
        app_with_results.run()
        assert_no_exception(app_with_results)
        assert any(h.value == "📋 Current Analysis" for h in app_with_results.header)
        assert any("Upward Trend" in s.value for s in app_with_results.sidebar.success)

    def test_sidebar_sideways(self, app_with_results, calibrated_predictor):
        calibrated_predictor.trend_info["daily_trend"] = 0.0
        app_with_results.run()
        assert any("Sideways" in i.value for i in app_with_results.sidebar.info)


class TestWithoutTrendInfo:
    """Regression: a predictor with an empty trend_info used to crash with KeyError('daily_trend')."""

    @pytest.fixture
    def app_no_trend(self, app, uptrend_data) -> AppTest:
        predictor = StockPredictor("NOTREND")
        predictor.data = uptrend_data
        predictor.current_price = float(uptrend_data["Close"].iloc[-1])
        predictor.mu, predictor.sigma, predictor.theta = 0.1, 0.2, 0.1
        days, sims = 5, 50
        app.session_state["predictor"] = predictor
        app.session_state["predictions"] = predictor.predict_prices(days=days, num_simulations=sims)
        app.session_state["ticker"] = predictor.ticker
        app.session_state["days"] = days
        app.session_state["simulations"] = sims
        return app

    @pytest.mark.parametrize("page", [INPUT_PAGE, ANALYSIS_PAGE, MONTE_CARLO_PAGE])
    def test_pages_render(self, app_no_trend, page):
        go_to(app_no_trend, page)
        assert_no_exception(app_no_trend)
        # No trend banner in the sidebar when there is no trend data
        assert not any("Trend" in w.value for w in app_no_trend.sidebar.success)
        assert not any("Trend" in w.value for w in app_no_trend.sidebar.error)
        assert not any("Sideways" in w.value for w in app_no_trend.sidebar.info)

    def test_price_analysis_skips_trend_sections(self, app_no_trend):
        go_to(app_no_trend, ANALYSIS_PAGE)
        labels = {m.label for m in app_no_trend.metric}
        assert "Daily Trend" not in labels
        assert "Expected Price" in labels
        assert not any(h.value == "🔍 Trend Analysis" for h in app_no_trend.header)
