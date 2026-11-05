# Stock Price Predictor 📈

<div align="center">

<img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.11">
<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white" alt="Streamlit">
<a href="https://github.com/QuinnColeman1/stock-pricer/actions/workflows/ci.yml"><img src="https://github.com/QuinnColeman1/stock-pricer/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
<img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="MIT License">

</div>

## Overview

Stock Price Predictor fetches historical prices from Yahoo Finance, detects the
stock's recent trend, and runs Monte Carlo simulations to estimate a range of
possible future prices. Use it as an interactive Streamlit app, a command-line
tool, or a Python library.

> ⚠️ **Not financial advice.** This is an educational project. Simulated
> prices are statistical scenarios, not forecasts you should trade on. See
> [Limitations](#-limitations).

## 🚀 Quick Start

**Requirements:** Python **3.11** (the project pins `>=3.11,<3.12`),
[Poetry](https://python-poetry.org/), and an internet connection to download
price data.

1. **Install dependencies**
   ```bash
   pip install poetry
   poetry install
   ```
   `poetry.toml` sets `in-project = true`, so Poetry creates the environment at
   `./.venv`.

2. **Launch the app**
   ```bash
   poetry run streamlit run stock_pricer/streamlit_app.py
   ```

3. **Open your browser** at `http://localhost:8501`.

## 🖥️ Using the App

Switch between the three pages from the sidebar:

| Page | What it shows |
|------|---------------|
| **🔧 Input Parameters** | Choose the ticker and settings (see below), then click **🚀 Analyze Stock**. Shows the current price and calibrated model parameters. |
| **📈 Price Analysis** | Trend metrics (regression slope, trend strength, 5-day vs 20-day moving-average trend, volume trend), a historical price chart with the fitted trend, and the prediction chart with its 5th–95th percentile band. Also shows a summary: expected price, standard deviation, probability of gain, and price ranges. |
| **🎲 Monte Carlo Simulation** | A sample of simulated price paths and the distribution of final prices. |

### Settings

| Setting | Options | Default | Meaning |
|---------|---------|---------|---------|
| Ticker | Any Yahoo Finance symbol | *(empty)* | Stock to analyze (e.g. `AAPL`, `MSFT`, `GOOGL`) |
| Historical Data Period | `1y`, `2y`, `5y`, `10y`, `max` | `5y` | History used to calibrate the model |
| Trend Weight | 0.0 – 1.0 | 0.5 | 0 = long-run historical drift only, 1 = recent trend only |
| Prediction Days | 1 – 365 | 30 | How far ahead to simulate (trading days) |
| Monte Carlo Simulations | 500, 1k, 2k, 5k, 10k | 1,000 | More paths = smoother estimates, slower run |
| Use Trend Adjustment | on / off | on | Add a decaying trend boost to the simulated drift |

The sidebar also shows a summary of the current analysis and labels the trend
as upward, declining, or sideways.

## ⌨️ Command-Line Usage

```bash
poetry run python stock_pricer/fetch_stocks.py -t AAPL MSFT -d 30 -s 5000 -w 0.6 --plot
```

| Flag | Default | Description |
|------|---------|-------------|
| `-t`, `--tickers` | *(prompt)* | One or more ticker symbols. If omitted, you are prompted for a comma-separated list. |
| `-d`, `--days` | `30` | Number of days to predict |
| `-s`, `--simulations` | `1000` | Number of Monte Carlo simulations |
| `-w`, `--trend-weight` | `0.5` | Weight of recent trend vs historical average (0–1) |
| `-p`, `--plot` | off | Show matplotlib charts of the predictions |

For each ticker, the CLI prints the most recent prices, the calibrated
parameters and trend metrics, and a prediction summary.

## 🐍 Python API

```python
from stock_pricer import StockPredictor

predictor = StockPredictor("AAPL")
predictor.fetch_data("5y")
predictor.calibrate_parameters(trend_weight=0.6)
predictions = predictor.predict_prices(days=30, num_simulations=1000)

print(predictions["final_mean"])
print(predictions["final_percentiles"])   # {"5th": ..., "25th": ..., "50th": ..., "75th": ..., "95th": ...}
```

## 🧮 How It Works

1. **Data** — Daily open, close and volume come from Yahoo Finance via
   [`yfinance`](https://github.com/ranaroussi/yfinance). Daily log returns are
   computed from closing prices.
2. **Historical drift** — The model takes an average of all historical returns
   with exponential time weights (half-life = ¼ of the window), so recent data
   counts more.
3. **Trend detection** (last 60 days) — A linear regression on log prices gives
   the daily trend and its strength (R²). Supporting signals: the 5-day vs
   20-day moving-average trend and the change in trading volume.
4. **Blending** — The model drift is
   `μ = (1 − trend_weight) · historical_drift + trend_weight · annualized_trend`.
5. **Volatility** — An exponentially weighted standard deviation of the last 30
   days of returns, annualized.
6. **Mean reversion** — The model estimates a reversion strength θ from return
   autocorrelation and increases it when the recent trend is strong. This pulls
   simulated prices back toward the expected drift path.
7. **Simulation** — Each day, every path takes a step based on Geometric
   Brownian Motion (GBM), with these adjustments:
   - an optional trend boost that decays by 5% per day,
   - volatility that rises gradually by up to 20% when the trend is strong,
   - the mean-reversion pull from step 6.
8. **Output** — Mean and standard deviation of the final price, the 5th, 25th,
   50th, 75th and 95th percentiles of final prices, the 5th–95th percentile
   band for each day, and the probability of a gain.

## ⚠️ Limitations

- **Model assumptions:** GBM assumes normally distributed returns. It does not
  capture sudden price jumps, earnings events, crashes, or volatility regimes.
- **Past performance:** the detected trend and historical drift do not predict
  the future.
- **Data source:** results depend on Yahoo Finance availability and accuracy.
  `yfinance` is an unofficial API and may rate-limit or break.
- **No transaction costs, dividends or splits** are modeled in the simulations.

## 🛠️ Development

Install with the dev tools:

```bash
poetry install --with dev
```

Common tasks (run `make help` for the full list):

| Command | What it does |
|---------|--------------|
| `make test` | Run pytest with coverage; HTML report written to `htmlcov/` |
| `make lint` / `make lint-fix` | Ruff lint checks / auto-fix |
| `make format` | Ruff formatting |
| `make check-types` | mypy static type checking |
| `make security` | Bandit security scan |
| `make clean` | Remove caches and build artifacts |

Run the tests directly with:

```bash
poetry run pytest
```

The tests use synthetic, seeded price data and mock `yfinance`, so they run
offline and give the same results every time.

GitHub Actions ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs
format checks, linting and tests on pushes to `main` and `develop`, and on pull
requests to `main`.

## 📁 Project Structure

```
stock_pricer/
├── stock_pricer/
│   ├── __init__.py          # Exports StockPredictor, version info
│   ├── fetch_stocks.py      # Prediction engine, plotting, CLI entry point
│   └── streamlit_app.py     # Streamlit web interface
├── tests/                   # pytest suite (fixtures in conftest.py)
├── .github/workflows/ci.yml # CI pipeline
├── Makefile                 # Dev task shortcuts
├── pyproject.toml           # Poetry config, dependencies, tool settings
├── poetry.toml              # Keeps the virtualenv in ./.venv
└── LICENSE
```

## 🩺 Troubleshooting

| Problem | Fix |
|---------|-----|
| `streamlit: command not found` or `pytest: command not found` | These are installed in the project's virtualenv. Prefix with `poetry run`, or activate it first: `source .venv/bin/activate`. Don't install them with `apt`. |
| `poetry run pytest run` collects 0 tests | Drop the trailing `run`. pytest treats it as a path. |
| Poetry complains about the Python version | Use Python 3.11: `poetry env use python3.11`. |
| No data or an error for a ticker | Check the symbol on Yahoo Finance (e.g. `BRK-B`, not `BRK.B`) and your internet connection. |

## 📜 License

MIT License. See [LICENSE](LICENSE) for details.
