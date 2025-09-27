# Stock Price Predictor 📈

<div align="center">

<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white" alt="Streamlit">

</div>

## Overview

A simple yet powerful tool for analyzing stock trends and predicting future prices using historical data. Make more informed investment decisions with clear visualizations and risk metrics.

## 🚀 Quick Start

### Option 1: Using Docker (Recommended)

1. **Build the Docker image**
   ```bash
   docker build -t stock-pricer .
   ```

2. **Run the container**
   ```bash
   docker run -p 8501:8501 stock-pricer
   ```

3. **Open in your browser** at `http://localhost:8501`

### Option 2: Local Installation

1. **Install dependencies**
   ```bash
   pip install poetry
   poetry install
   ```

2. **Launch the app**
   ```bash
   poetry run streamlit run stock_pricer/streamlit_app.py
   ```

3. **Open in your browser** at `http://localhost:8501`

## 📊 Features

- **Stock Analysis**
  - Historical price charts with moving averages
  - Volume and trend indicators
  - Customizable time periods

- **Price Predictions**
  - Future price estimates
  - Confidence intervals
  - Risk assessment metrics

- **User-Friendly**
  - Clean, intuitive interface
  - Mobile-responsive design
  - No technical knowledge required

## 📝 How to Use

1. **Enter a stock ticker** (e.g., AAPL, MSFT, GOOGL)
2. **Select analysis options**
   - Time period
   - Confidence level
   - Technical indicators
3. **Review the results**
   - Price charts with predictions
   - Risk metrics
   - Key statistics

## 📋 Requirements

- Python 3.11 or later
- Modern web browser
- Internet connection

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.
