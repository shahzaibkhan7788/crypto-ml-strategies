# CryptoMLStrategies 🚀

A comprehensive, modular framework for building, testing, and executing machine-learning-driven cryptocurrency trading strategies. This project integrates high-frequency data fetching, advanced technical analysis, diverse ML models, and live execution on exchanges like Bybit.

---

## 📑 Table of Contents
- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Core Components](#core-components)
    - [Data Acquisition](#1-data-acquisition)
    - [Technical Indicators (TA-Lib)](#2-technical-indicators-ta-lib)
    - [Machine Learning Library](#3-machine-learning-library)
    - [Signal Generation](#4-signal-generation)
    - [Execution & Backtesting](#5-execution--backtesting)
- [Getting Started](#getting-started)
- [Workflow](#workflow)

---

## 🌟 Project Overview
CryptoMLStrategies is designed for quantitative traders who want to leverage machine learning for market predictions. It provides a full pipeline from raw data to executed trades, allowing for rapid experimentation with different indicators and model architectures.

---

## 📂 Project Structure
```text
crypto-ml-strategies/
├── data/               # Data downloaders (Binance/Bybit) and storage
├── execution/          # Live execution scripts (Bybit integration)
├── indicator/          # TA-Lib based technical indicator calculator
├── ml/                 # ML Models (Learners) and trained weights (Trainer)
│   ├── learner/        # Model definitions (LSTM, XGBoost, Random Forest, etc.)
│   └── trainer/        # Saved model metadata and weights (BTC, ETH, SOL)
├── signalgenerator/    # Logic to convert indicators into trade signals
├── strategies/         # Strategy pipeline and generator logic
├── trading-simulator/  # Backtesting and simulation environment
├── utils.py            # Shared utility functions
└── requirements.txt    # Project dependencies
```

---

## 🛠 Core Components

### 1. Data Acquisition
The project includes robust data fetchers for Binance and Bybit.
- Supports multiple time horizons (1m, 1h, 1d, etc.).
- Includes auto-interpolation for missing values and duplicate detection.
- Stores data in PostgreSQL for high-speed retrieval.

### 2. Technical Indicators (TA-Lib)
The `indicator_calculator.py` module wraps the powerful TA-Lib library, offering a massive array of indicators categorized into:
- **Trend**: SMA, EMA, MACD, ADX, SAR.
- **Volume**: OBV, AD, MFI, Custom Volume.
- **Momentum**: RSI, Stochastic, CCI, WILLR, MOM, BOP, CMO, DX, PPO, ROC.
- **Volatility**: ATR, NATR, Bollinger Bands, Standard Deviation.
- **Cycle & Patterns**: Hilbert Transform, Candlestick pattern recognition (CDL*).

### 3. Machine Learning Library
Located in `ml/learner/`, the project implements a wide variety of regression and classification models:
- **Deep Learning**: LSTM for time-series forecasting.
- **Ensemble**: XGBoost, AdaBoost, Random Forest, Gradient Boosting.
- **Classic Models**: Linear/Ridge/Bayesian Regression, Decision Trees, KNN.
- **Base Architecture**: Standardized saving/loading of model weights and metadata.

### 4. Signal Generation
The `SignalGenerator` class in `signalgenerator/` processes the output of technical indicators to create discrete trading signals (Buy: 1, Sell: -1, Neutral: 0). 
- Implements custom logic for each indicator (e.g., RSI overbought/oversold, MACD crossovers).

### 5. Execution & Backtesting
- **Live Execution**: The `execution/bybit/` module handles authenticated API calls to Bybit for placing orders.
- **Backtesting**: Integrated backtest scripts allow for evaluating strategies on historical data before going live.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- TA-Lib C++ Library (Required for the `talib` python wrapper)
- PostgreSQL (Optional, but recommended for data storage)

### Installation
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🔄 Workflow
1. **Fetch Data**: Use `auto_fetch.py` or script in `data/` to pull historical OHLCV data.
2. **Train Models**: Use scripts in `ml/` to train one of the learners on your data.
3. **Generate Signals**: The `SignalGenerator` consumes data and technical indicators.
4. **Execute/Backtest**: Run `execution/bybit/main.py` for live trading or use the `trading-simulator`.

---

## 📈 Supported Models
| Category | Models |
| :--- | :--- |
| **Deep Learning** | LSTM |
| **Boosting** | XGBoost, Gradient Boosting, AdaBoost |
| **Regression** | Linear, Ridge, Bayesian Ridge |
| **Trees** | Random Forest, Decision Tree |
| **Other** | KNN |

---

*“Trade smarter, not harder, with Machine Learning.”*
