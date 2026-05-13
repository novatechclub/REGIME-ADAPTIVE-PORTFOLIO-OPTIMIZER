# Regime-Adaptive Portfolio Optimizer

A quantitative framework for regime-conditioned allocation and capacity stress testing.
Nova Tech Club | 
## Project Team

| Name | Role |
|---|---|
| Sengul Seyda Yilmaz | Project Supervisor, Team Lead of the Tech & Innovation Department |
| Moritz Martin Bauer | Finance Project Team Member |
| Maximilian Haussmann | Finance Project Team Member |
| Alessandro De Riccardis | Finance Project Team Member |

## Overview

Five-module pipeline:

1. **Data pipeline** — ETF prices (yfinance) + macro series (FRED). *Moritz*
2. **Regime classifier** — 3-state Gaussian HMM on macro features. *Alessandro*
3. **Signal layer** — regime-conditioned expected returns. *Max*
4. **Convex optimizer** — constrained mean-variance with vol-target and turnover. *Moritz*
5. **Backtester + stress test** — walk-forward 2019–2024, capacity frontier. *Alessandro*

## Tech stack

Python 3.10+ · yfinance · FRED API · hmmlearn · cvxpy · pandas / numpy / matplotlib

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export FRED_API_KEY=your_fred_key_here   # get a free one at https://fred.stlouisfed.org/docs/api/api_key.html
```

## Smoke tests

Run from the repo root:

```bash
python -m src.data_loader     # loads prices + macro, prints shapes
python -m src.optimizer       # random-input optimization, prints weights
```

## Repository layout

| Path | Description |
| --- | --- |
| `src/data_loader.py` | M1 — ETF prices + FRED macro features (Moritz) |
| `src/regime_classifier.py` | M2 — HMM regime labels (Alessandro) |
| `src/signals.py` | M3 — regime-conditioned expected returns (Max) |
| `src/optimizer.py` | M4 — cvxpy portfolio optimizer (Moritz) |
| `src/backtester.py` | M5a — walk-forward backtest (Alessandro) |
| `src/stress_test.py` | M5b — capacity stress test (Alessandro) |
| `notebooks/demo.ipynb` | End-to-end demo notebook |
| `results/figures/` | Chart PNGs (Max) |
| `results/metrics.csv` | Summary performance table |
| `models/hmm_model.pkl` | Fitted HMM model |

## Interface contract (binding)

| Module | Function | Returns |
| --- | --- | --- |
| `data_loader.py` | `get_prices(start, end)` | DataFrame: dates × 8 tickers |
| `data_loader.py` | `get_features(start, end)` | DataFrame: dates × 3 macro features |
| `regime_classifier.py` | `get_regime(date)` | int: 0 Risk-On / 1 Risk-Off / 2 Transitional |
| `signals.py` | `get_expected_returns(date, regime_label)` | np.ndarray shape (8,) |
| `optimizer.py` | `solve_portfolio(mu, Sigma, ...)` | np.ndarray shape (8,) |

## Asset universe

`SPY`, `TLT`, `GLD`, `EEM`, `VNQ`, `LQD`, `DBC`, `IEF` — full window 2010-01-01 → 2024-12-31.
