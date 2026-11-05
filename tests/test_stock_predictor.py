"""Unit tests for the StockPredictor class in stock_pricer.fetch_stocks."""

import numpy as np
import pandas as pd
import pytest

from stock_pricer.fetch_stocks import StockPredictor

DT = 1.0 / 252.0
PREDICTION_KEYS = {
    "all_paths",
    "daily_means",
    "daily_stds",
    "daily_5th",
    "daily_95th",
    "final_mean",
    "final_std",
    "final_percentiles",
}


# --------------------------------------------------------------------------- #
# __init__
# --------------------------------------------------------------------------- #
class TestInit:
    def test_defaults(self):
        predictor = StockPredictor("AAPL")
        assert predictor.ticker == "AAPL"
        assert predictor.data is None
        assert predictor.returns is None
        assert predictor.mu is None
        assert predictor.sigma is None
        assert predictor.theta is None
        assert predictor.current_price == 0.0
        assert predictor.trend_strength == 0.0
        assert predictor.recent_trend == 0.0
        assert predictor.trend_info == {}


# --------------------------------------------------------------------------- #
# fetch_data
# --------------------------------------------------------------------------- #
class TestFetchData:
    def test_returns_subset_and_sets_current_price(self, mock_yf_ticker, uptrend_data):
        predictor = StockPredictor("AAPL")
        result = predictor.fetch_data(period="1y")

        mock_yf_ticker.assert_called_once_with("AAPL")
        mock_yf_ticker.return_value.history.assert_called_once_with(period="1y", interval="1d")
        assert list(result.columns) == ["Open", "Close", "Volume"]
        assert predictor.current_price == pytest.approx(uptrend_data["Close"].iloc[-1])
        # Full frame (incl. High/Low) is kept on the instance
        pd.testing.assert_frame_equal(predictor.data, uptrend_data)

    def test_default_period_is_5y(self, mock_yf_ticker):
        StockPredictor("AAPL").fetch_data()
        mock_yf_ticker.return_value.history.assert_called_once_with(period="5y", interval="1d")

    def test_empty_data_raises(self, mock_yf_ticker):
        mock_yf_ticker.return_value.history.return_value = pd.DataFrame()
        with pytest.raises(ValueError, match="No data found for BAD"):
            StockPredictor("BAD").fetch_data()


# --------------------------------------------------------------------------- #
# calculate_returns
# --------------------------------------------------------------------------- #
class TestCalculateReturns:
    def test_values_are_daily_log_returns(self):
        predictor = StockPredictor("T")
        predictor.data = pd.DataFrame({"Close": [100.0, 110.0, 99.0]})
        returns = predictor.calculate_returns()

        np.testing.assert_allclose(returns.values, [np.log(1.1), np.log(0.9)])
        assert predictor.returns is returns

    def test_length_is_one_less_than_prices(self, loaded_predictor):
        returns = loaded_predictor.calculate_returns()
        assert len(returns) == len(loaded_predictor.data) - 1
        assert not returns.isna().any()

    @pytest.mark.parametrize(
        "data",
        [None, pd.DataFrame(), pd.DataFrame({"Close": [100.0]})],
        ids=["none", "empty", "single-row"],
    )
    def test_invalid_data_raises(self, data):
        predictor = StockPredictor("T")
        predictor.data = data
        with pytest.raises(ValueError):
            predictor.calculate_returns()


# --------------------------------------------------------------------------- #
# calculate_time_weights
# --------------------------------------------------------------------------- #
class TestCalculateTimeWeights:
    def test_requires_returns(self):
        with pytest.raises(ValueError, match="No returns data"):
            StockPredictor("T").calculate_time_weights()

    def test_weights_sum_to_one_and_favour_recent_days(self, loaded_predictor):
        loaded_predictor.calculate_returns()
        weights = loaded_predictor.calculate_time_weights()

        assert len(weights) == len(loaded_predictor.returns)
        assert weights.sum() == pytest.approx(1.0)
        assert (weights > 0).all()
        assert (np.diff(weights) > 0).all(), "weights must increase towards the most recent day"

    def test_half_life_is_quarter_of_window(self, price_frame):
        predictor = StockPredictor("T")
        predictor.data = price_frame(n=101)  # -> 100 returns, half-life 25 days
        predictor.calculate_returns()
        weights = predictor.calculate_time_weights()
        assert weights[-1] / weights[-26] == pytest.approx(2.0)

    def test_lookback_limits_window(self, loaded_predictor):
        loaded_predictor.calculate_returns()
        weights = loaded_predictor.calculate_time_weights(lookback_days=10)
        assert len(weights) == 10
        assert weights.sum() == pytest.approx(1.0)

    def test_lookback_larger_than_data_uses_all(self, loaded_predictor):
        loaded_predictor.calculate_returns()
        weights = loaded_predictor.calculate_time_weights(lookback_days=10_000)
        assert len(weights) == len(loaded_predictor.returns)

    @pytest.mark.parametrize("lookback", [None, 0, -5])
    def test_non_positive_lookback_uses_all(self, loaded_predictor, lookback):
        loaded_predictor.calculate_returns()
        weights = loaded_predictor.calculate_time_weights(lookback_days=lookback)
        assert len(weights) == len(loaded_predictor.returns)


# --------------------------------------------------------------------------- #
# detect_trend
# --------------------------------------------------------------------------- #
class TestDetectTrend:
    EXPECTED_KEYS = {"daily_trend", "annualized_trend", "trend_strength", "ma_trend", "volume_trend"}

    def test_uptrend(self, loaded_predictor):
        info = loaded_predictor.detect_trend()
        assert set(info) == self.EXPECTED_KEYS
        assert info["daily_trend"] > 0
        assert info["annualized_trend"] == pytest.approx(info["daily_trend"] * 252)
        assert 0.0 <= info["trend_strength"] <= 1.0
        assert info["ma_trend"] > 0  # 5-day MA above 20-day MA in an uptrend

    def test_downtrend(self, declining_predictor):
        info = declining_predictor.detect_trend()
        assert info["daily_trend"] < 0
        assert info["ma_trend"] < 0

    def test_perfect_exponential_growth(self):
        predictor = StockPredictor("T")
        predictor.data = pd.DataFrame({"Close": 100.0 * np.exp(0.01 * np.arange(60))})
        info = predictor.detect_trend()
        assert info["daily_trend"] == pytest.approx(0.01)
        assert info["trend_strength"] == pytest.approx(1.0)

    def test_flat_prices_have_no_trend(self):
        predictor = StockPredictor("T")
        predictor.data = pd.DataFrame({"Close": np.full(30, 50.0)})
        info = predictor.detect_trend()
        assert info["daily_trend"] == pytest.approx(0.0)
        assert info["trend_strength"] == 0.0

    def test_stores_trend_attributes(self, loaded_predictor):
        info = loaded_predictor.detect_trend()
        assert loaded_predictor.recent_trend == info["daily_trend"]
        assert loaded_predictor.trend_strength == info["trend_strength"]

    def test_lookback_uses_only_recent_window(self):
        # 60 days falling, then 20 days rising
        falling = 100.0 * np.exp(-0.01 * np.arange(60))
        rising = 55.0 * np.exp(0.01 * np.arange(20))
        close = np.concatenate([falling, rising])
        predictor = StockPredictor("T")
        predictor.data = pd.DataFrame({"Close": close})
        assert predictor.detect_trend(lookback_days=80)["daily_trend"] < 0
        assert predictor.detect_trend(lookback_days=20)["daily_trend"] > 0

    def test_ma_and_volume_trend_need_20_points(self, price_frame):
        predictor = StockPredictor("T")
        predictor.data = price_frame(n=10, daily_drift=0.01)
        info = predictor.detect_trend()
        assert info["ma_trend"] == 0.0
        assert info["volume_trend"] == 0.0

    def test_volume_trend(self, price_frame):
        data = price_frame(n=60)
        data.loc[data.index[-10:], "Volume"] = 2_000_000.0  # volume doubles in the last 10 days
        predictor = StockPredictor("T")
        predictor.data = data
        assert predictor.detect_trend()["volume_trend"] == pytest.approx(1.0)

    @pytest.mark.parametrize(
        "data",
        [None, pd.DataFrame(), pd.DataFrame({"Close": [100.0]})],
        ids=["none", "empty", "single-row"],
    )
    def test_invalid_data_raises(self, data):
        predictor = StockPredictor("T")
        predictor.data = data
        with pytest.raises(ValueError):
            predictor.detect_trend()


# --------------------------------------------------------------------------- #
# calculate_dynamic_volatility
# --------------------------------------------------------------------------- #
class TestCalculateDynamicVolatility:
    def test_matches_annualised_ewma_std(self, loaded_predictor):
        loaded_predictor.calculate_returns()
        expected = loaded_predictor.returns.iloc[-30:].ewm(span=15, adjust=False).std().iloc[-1]
        assert loaded_predictor.calculate_dynamic_volatility() == pytest.approx(expected * np.sqrt(252))

    def test_computes_returns_if_missing(self, loaded_predictor):
        assert loaded_predictor.returns is None
        assert loaded_predictor.calculate_dynamic_volatility() > 0
        assert loaded_predictor.returns is not None

    def test_noisier_stock_has_higher_volatility(self, price_frame):
        calm, wild = StockPredictor("CALM"), StockPredictor("WILD")
        calm.data = price_frame(noise=0.005)
        wild.data = price_frame(noise=0.03)
        assert wild.calculate_dynamic_volatility() > calm.calculate_dynamic_volatility()

    def test_insufficient_data_raises(self):
        predictor = StockPredictor("T")
        predictor.data = pd.DataFrame({"Close": [100.0, 101.0]})  # one return only
        with pytest.raises(ValueError, match="Insufficient data"):
            predictor.calculate_dynamic_volatility()

    def test_no_data_raises(self):
        with pytest.raises(ValueError):
            StockPredictor("T").calculate_dynamic_volatility()


# --------------------------------------------------------------------------- #
# calibrate_parameters
# --------------------------------------------------------------------------- #
class TestCalibrateParameters:
    def test_sets_and_returns_parameters(self, loaded_predictor):
        result = loaded_predictor.calibrate_parameters()

        assert set(result) == {"mu", "sigma", "theta", "trend_info"}
        assert result["mu"] == loaded_predictor.mu
        assert result["sigma"] == loaded_predictor.sigma
        assert result["theta"] == loaded_predictor.theta
        assert result["trend_info"] is loaded_predictor.trend_info
        assert all(np.isfinite([loaded_predictor.mu, loaded_predictor.sigma, loaded_predictor.theta]))
        assert loaded_predictor.sigma > 0
        assert loaded_predictor.theta >= 0

    def test_sigma_is_dynamic_volatility(self, calibrated_predictor):
        assert calibrated_predictor.sigma == pytest.approx(
            calibrated_predictor.calculate_dynamic_volatility()
        )

    def test_trend_weight_blends_historical_and_trend_drift(self, loaded_predictor):
        loaded_predictor.calculate_returns()
        weights = loaded_predictor.calculate_time_weights()
        historical_mu = np.average(loaded_predictor.returns, weights=weights) / DT
        trend_mu = loaded_predictor.detect_trend()["annualized_trend"]

        def mu_for(weight: float) -> float:
            return loaded_predictor.calibrate_parameters(trend_weight=weight)["mu"]

        assert mu_for(0.0) == pytest.approx(historical_mu)
        assert mu_for(1.0) == pytest.approx(trend_mu)
        assert mu_for(0.5) == pytest.approx(0.5 * historical_mu + 0.5 * trend_mu)

    def test_uptrend_gives_positive_drift(self, calibrated_predictor, declining_predictor):
        assert calibrated_predictor.mu > 0
        assert declining_predictor.mu < 0

    def test_theta_defaults_when_too_few_returns(self):
        predictor = StockPredictor("T")
        predictor.data = pd.DataFrame({"Close": [100.0, 102.0, 101.0]})  # 2 returns
        assert predictor.calibrate_parameters()["theta"] == 0.1

    def test_no_data_raises(self):
        with pytest.raises(ValueError):
            StockPredictor("T").calibrate_parameters()


# --------------------------------------------------------------------------- #
# predict_prices
# --------------------------------------------------------------------------- #
class TestPredictPrices:
    def test_output_structure(self, calibrated_predictor):
        days, sims = 30, 100
        pred = calibrated_predictor.predict_prices(days=days, num_simulations=sims)

        assert set(pred) == PREDICTION_KEYS
        assert pred["all_paths"].shape == (sims, days + 1)
        for key in ("daily_means", "daily_stds", "daily_5th", "daily_95th"):
            assert pred[key].shape == (days + 1,)
        assert set(pred["final_percentiles"]) == {"5th", "25th", "50th", "75th", "95th"}

    def test_paths_start_at_current_price_and_stay_valid(self, calibrated_predictor, small_predictions):
        paths = small_predictions["all_paths"]
        np.testing.assert_allclose(paths[:, 0], calibrated_predictor.current_price)
        assert np.isfinite(paths).all()
        assert (paths > 0).all()

    def test_statistics_are_consistent_with_paths(self, small_predictions):
        paths = small_predictions["all_paths"]
        final = paths[:, -1]
        assert small_predictions["final_mean"] == pytest.approx(final.mean())
        assert small_predictions["final_std"] == pytest.approx(final.std(ddof=1))
        np.testing.assert_allclose(small_predictions["daily_means"], paths.mean(axis=0))
        assert small_predictions["final_percentiles"]["50th"] == pytest.approx(np.median(final))

        pct = small_predictions["final_percentiles"]
        assert pct["5th"] <= pct["25th"] <= pct["50th"] <= pct["75th"] <= pct["95th"]
        assert (small_predictions["daily_5th"] <= small_predictions["daily_95th"]).all()

    def test_is_reproducible(self, calibrated_predictor):
        first = calibrated_predictor.predict_prices(days=5, num_simulations=50)
        second = calibrated_predictor.predict_prices(days=5, num_simulations=50)
        np.testing.assert_array_equal(first["all_paths"], second["all_paths"])

    def test_zero_volatility_zero_drift_stays_flat(self):
        predictor = StockPredictor("T")
        predictor.current_price = 100.0
        predictor.mu, predictor.sigma, predictor.theta = 0.0, 0.0, 0.1
        pred = predictor.predict_prices(days=10, num_simulations=5)
        np.testing.assert_allclose(pred["all_paths"], 100.0)

    def test_deterministic_drift_matches_gbm_formula(self):
        predictor = StockPredictor("T")
        predictor.current_price = 100.0
        predictor.mu, predictor.sigma, predictor.theta = 0.252, 0.0, 0.0
        pred = predictor.predict_prices(days=20, num_simulations=3)
        expected = 100.0 * np.exp(0.252 * DT * np.arange(21))
        np.testing.assert_allclose(pred["all_paths"], np.tile(expected, (3, 1)))

    def test_trend_adjustment_shifts_mean_in_trend_direction(self, calibrated_predictor):
        with_trend = calibrated_predictor.predict_prices(days=20, num_simulations=200)
        without = calibrated_predictor.predict_prices(
            days=20, num_simulations=200, use_trend_adjustment=False
        )
        assert with_trend["final_mean"] > without["final_mean"]

    def test_auto_calibrates_when_parameters_missing(self, loaded_predictor):
        assert loaded_predictor.mu is None
        pred = loaded_predictor.predict_prices(days=5, num_simulations=20)
        assert loaded_predictor.mu is not None
        assert np.isfinite(pred["all_paths"]).all()

    @pytest.mark.parametrize(("days", "sims"), [(0, 10), (-1, 10), (10, 0), (10, -5)])
    def test_invalid_arguments_raise(self, calibrated_predictor, days, sims):
        with pytest.raises(ValueError, match="must be positive"):
            calibrated_predictor.predict_prices(days=days, num_simulations=sims)

    @pytest.mark.parametrize("price", [0.0, -5.0, float("nan"), float("inf")])
    def test_invalid_current_price_is_rejected(self, calibrated_predictor, price):
        # Regression: a zero start price used to return all-NaN paths silently
        calibrated_predictor.current_price = price
        with pytest.raises(ValueError, match="Current price must be a positive number"):
            calibrated_predictor.predict_prices(days=5, num_simulations=10)

    def test_current_price_checked_before_calibration(self, loaded_predictor):
        loaded_predictor.current_price = 0.0  # e.g. data set manually, fetch_data() never run
        with pytest.raises(ValueError, match="Current price"):
            loaded_predictor.predict_prices(days=5, num_simulations=10)
        assert loaded_predictor.mu is None, "should fail fast without calibrating"

    def test_no_trend_adjustment_without_trend_info(self):
        predictor = StockPredictor("T")
        predictor.current_price = 100.0
        predictor.mu, predictor.sigma, predictor.theta = 0.1, 0.2, 0.1
        assert predictor.trend_info == {}
        with_flag = predictor.predict_prices(days=10, num_simulations=50, use_trend_adjustment=True)
        without = predictor.predict_prices(days=10, num_simulations=50, use_trend_adjustment=False)
        np.testing.assert_array_equal(with_flag["all_paths"], without["all_paths"])


# --------------------------------------------------------------------------- #
# print_calibration_results
# --------------------------------------------------------------------------- #
class TestPrintCalibrationResults:
    def test_prints_parameters_and_upward_trend(self, calibrated_predictor, capsys):
        calibrated_predictor.print_calibration_results()
        out = capsys.readouterr().out

        assert "Parameter Calibration Results for TEST" in out
        assert f"Current Price: ${calibrated_predictor.current_price:.2f}" in out
        assert f"Mu (Annual Drift): {calibrated_predictor.mu:.4f}" in out
        assert "Trend Analysis" in out
        assert "UPWARD trend" in out
        assert "Sharpe Ratio" in out

    def test_prints_declining_warning(self, declining_predictor, capsys):
        declining_predictor.print_calibration_results()
        assert "DECLINING trend" in capsys.readouterr().out

    def test_prints_sideways(self, calibrated_predictor, capsys):
        calibrated_predictor.trend_info["daily_trend"] = 0.0
        calibrated_predictor.print_calibration_results()
        assert "SIDEWAYS" in capsys.readouterr().out

    def test_without_returns_or_trend(self, capsys):
        # Regression: an empty trend_info used to print a fake all-zero "SIDEWAYS" trend
        predictor = StockPredictor("T")
        predictor.current_price = 10.0
        predictor.mu, predictor.sigma, predictor.theta = 0.1, 0.2, 0.3
        predictor.print_calibration_results()
        out = capsys.readouterr().out
        assert "No historical return data" in out
        assert "No trend analysis available" in out
        assert "=== Trend Analysis ===" not in out
        assert "SIDEWAYS" not in out

    def test_uncalibrated_raises(self, loaded_predictor):
        with pytest.raises(ValueError, match="not calibrated"):
            loaded_predictor.print_calibration_results()

    def test_missing_ticker_raises(self, calibrated_predictor):
        calibrated_predictor.ticker = ""
        with pytest.raises(ValueError, match="Ticker"):
            calibrated_predictor.print_calibration_results()


# --------------------------------------------------------------------------- #
# print_predictions
# --------------------------------------------------------------------------- #
class TestPrintPredictions:
    def test_prints_summary(self, calibrated_predictor, small_predictions, capsys):
        calibrated_predictor.print_predictions(small_predictions, days=10)
        out = capsys.readouterr().out

        price = calibrated_predictor.current_price
        expected_return = (small_predictions["final_mean"] / price - 1) * 100
        prob = (small_predictions["all_paths"][:, -1] > price).mean() * 100
        assert "10-Day Price Predictions for TEST" in out
        assert f"Expected Price (Mean): ${small_predictions['final_mean']:.2f}" in out
        assert f"Expected Return: {expected_return:.2f}%" in out
        assert f"Probability of price increase: {prob:.1f}%" in out
        assert "Value at Risk (5%)" in out
        assert "DECLINING" not in out

    def test_declining_note(self, declining_predictor, capsys):
        pred = declining_predictor.predict_prices(days=5, num_simulations=50)
        declining_predictor.print_predictions(pred, days=5)
        assert "DECLINING trend" in capsys.readouterr().out

    def test_non_dict_raises(self, calibrated_predictor):
        with pytest.raises(ValueError, match="must be a dictionary"):
            calibrated_predictor.print_predictions([1, 2, 3])

    def test_missing_keys_raises(self, calibrated_predictor, small_predictions):
        del small_predictions["final_std"]
        with pytest.raises(ValueError, match="Missing required prediction keys: final_std"):
            calibrated_predictor.print_predictions(small_predictions)

    def test_bad_percentiles_raise(self, calibrated_predictor, small_predictions):
        small_predictions["final_percentiles"] = [1, 2, 3]
        with pytest.raises(ValueError, match="final_percentiles must be a dictionary"):
            calibrated_predictor.print_predictions(small_predictions)

    def test_missing_percentiles_raise(self, calibrated_predictor, small_predictions):
        del small_predictions["final_percentiles"]["95th"]
        with pytest.raises(ValueError, match="Missing required percentiles: 95th"):
            calibrated_predictor.print_predictions(small_predictions)

    def test_missing_ticker_raises(self, calibrated_predictor, small_predictions):
        calibrated_predictor.ticker = ""
        with pytest.raises(ValueError, match="Ticker"):
            calibrated_predictor.print_predictions(small_predictions)
