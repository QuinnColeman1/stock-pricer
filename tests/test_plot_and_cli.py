"""Tests for plot_predictions() and the command-line entry point main()."""

import sys
from collections.abc import Iterator
from unittest import mock

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from stock_pricer import fetch_stocks
from stock_pricer.fetch_stocks import StockPredictor, plot_predictions


@pytest.fixture(autouse=True)
def _close_figures() -> Iterator[None]:
    yield
    plt.close("all")


# --------------------------------------------------------------------------- #
# plot_predictions
# --------------------------------------------------------------------------- #
class TestPlotPredictions:
    def test_builds_both_panels_and_shows(self, calibrated_predictor, small_predictions):
        with mock.patch("matplotlib.pyplot.show") as show:
            plot_predictions(calibrated_predictor, small_predictions, days=10)

        show.assert_called_once()
        price_ax, returns_ax = plt.gcf().axes
        assert "TEST - 10 Day Price Prediction" in price_ax.get_title()
        legend = [t.get_text() for t in price_ax.get_legend().get_texts()]
        assert "Historical" in legend
        assert "Expected Price" in legend
        assert "90% Confidence Interval" in legend
        assert any(label.startswith("Current Trend") for label in legend)
        assert "Distribution of 10-Day Returns" in returns_ax.get_title()

    def test_no_trend_line_without_trend_info(self, uptrend_data):
        predictor = StockPredictor("T")
        predictor.data = uptrend_data
        predictor.current_price = float(uptrend_data["Close"].iloc[-1])
        predictor.mu, predictor.sigma, predictor.theta = 0.1, 0.2, 0.1  # set manually, no trend_info
        predictions = predictor.predict_prices(days=5, num_simulations=20)

        with mock.patch("matplotlib.pyplot.show"):
            plot_predictions(predictor, predictions, days=5)

        legend = [t.get_text() for t in plt.gcf().axes[0].get_legend().get_texts()]
        assert "Expected Price" in legend
        assert not any(label.startswith("Current Trend") for label in legend)

    def test_missing_data_raises(self, calibrated_predictor, small_predictions):
        calibrated_predictor.data = None
        with pytest.raises(ValueError, match="missing required data"):
            plot_predictions(calibrated_predictor, small_predictions)

    def test_non_dict_raises(self, calibrated_predictor):
        with pytest.raises(ValueError, match="must be a dictionary"):
            plot_predictions(calibrated_predictor, "not a dict")

    def test_missing_keys_raise(self, calibrated_predictor, small_predictions):
        del small_predictions["daily_5th"]
        with pytest.raises(ValueError, match="daily_5th"):
            plot_predictions(calibrated_predictor, small_predictions)

    def test_empty_paths_raise(self, calibrated_predictor, small_predictions):
        small_predictions["all_paths"] = np.empty((0, 0))
        with pytest.raises(ValueError, match="No prediction data"):
            plot_predictions(calibrated_predictor, small_predictions)

    def test_empty_history_is_wrapped_and_figures_closed(self, calibrated_predictor, small_predictions):
        calibrated_predictor.data = pd.DataFrame(columns=["Close"])
        with pytest.raises(ValueError, match="Error generating plot"):
            plot_predictions(calibrated_predictor, small_predictions, days=10)
        assert plt.get_fignums() == []


# --------------------------------------------------------------------------- #
# main (CLI)
# --------------------------------------------------------------------------- #
def run_cli(monkeypatch: pytest.MonkeyPatch, *args: str) -> None:
    monkeypatch.setattr(sys, "argv", ["fetch_stocks", *args])
    fetch_stocks.main()


class TestMain:
    @pytest.mark.usefixtures("mock_yf_ticker")
    def test_full_run(self, monkeypatch, capsys):
        run_cli(monkeypatch, "-t", "AAPL", "-d", "5", "-s", "50")
        out = capsys.readouterr().out

        assert "ANALYZING: AAPL" in out
        assert "Calibrating with trend weight: 0.5" in out
        assert "Parameter Calibration Results for AAPL" in out
        assert "5-Day Price Predictions for AAPL" in out
        assert "Analysis complete!" in out
        assert "Error" not in out

    def test_failing_ticker_does_not_stop_others(
        self, monkeypatch, capsys, mock_yf_ticker, uptrend_data
    ):
        def history_for(ticker: str) -> mock.MagicMock:
            stock = mock.MagicMock()
            stock.history.return_value = pd.DataFrame() if ticker == "BAD" else uptrend_data
            return stock

        mock_yf_ticker.side_effect = history_for
        run_cli(monkeypatch, "-t", "BAD", "GOOD", "-d", "3", "-s", "20")
        out = capsys.readouterr().out

        assert "Error analyzing BAD: No data found for BAD" in out
        assert "3-Day Price Predictions for GOOD" in out
        assert "Analysis complete!" in out

    @pytest.mark.usefixtures("mock_yf_ticker")
    def test_plot_flag_calls_plot(self, monkeypatch):
        with mock.patch.object(fetch_stocks, "plot_predictions") as plot:
            run_cli(monkeypatch, "-t", "AAPL", "-d", "3", "-s", "20", "--plot")
        plot.assert_called_once()
        assert plot.call_args.args[2] == 3

    @pytest.mark.usefixtures("mock_yf_ticker")
    def test_no_plot_by_default(self, monkeypatch):
        with mock.patch.object(fetch_stocks, "plot_predictions") as plot:
            run_cli(monkeypatch, "-t", "AAPL", "-d", "3", "-s", "20")
        plot.assert_not_called()

    @pytest.mark.usefixtures("mock_yf_ticker")
    def test_trend_weight_is_passed_through(self, monkeypatch, capsys):
        run_cli(monkeypatch, "-t", "AAPL", "-d", "3", "-s", "20", "-w", "0.8")
        assert "Calibrating with trend weight: 0.8" in capsys.readouterr().out

    @pytest.mark.parametrize("weight", ["-0.1", "1.5"])
    def test_invalid_trend_weight_exits(self, monkeypatch, capsys, weight):
        with pytest.raises(SystemExit) as exc:
            run_cli(monkeypatch, "-t", "AAPL", "-w", weight)
        assert exc.value.code == 1
        assert "trend-weight must be between 0 and 1" in capsys.readouterr().out

    def test_prompts_for_tickers(self, monkeypatch, capsys, mock_yf_ticker):
        monkeypatch.setattr("builtins.input", lambda _prompt: " aapl, msft ,")
        run_cli(monkeypatch, "-d", "3", "-s", "20")
        out = capsys.readouterr().out
        assert "ANALYZING: AAPL" in out
        assert "ANALYZING: MSFT" in out
        assert mock_yf_ticker.call_count == 2

    def test_empty_prompt_exits(self, monkeypatch, capsys):
        monkeypatch.setattr("builtins.input", lambda _prompt: "   ")
        with pytest.raises(SystemExit) as exc:
            run_cli(monkeypatch)
        assert exc.value.code == 1
        assert "No ticker provided" in capsys.readouterr().out
