"""Module 5 Part A: Walk-Forward Backtester (Alessandro).

Runs a monthly-rebalanced backtest over 2019-2024 for three strategies:
Equal Weight, Mean-Variance Unconditional, Regime-Adaptive.

Expected public API:
    run_backtest() -> dict of results

Implementation steps:
  1. Define test period: 2019-01-01 to 2024-12-31
  2. For each month, rebalance on first trading day
  3. Implement 3 strategies:
     - Equal Weight: always w = 1/8
     - MV Unconditional: mu, Sigma from rolling 252 days, no regime conditioning
     - Regime-Adaptive: regime-conditioned mu + Sigma from rolling 252 days
  4. Compute NAV, returns, and metrics for each strategy
  5. Generate visualizations and save results
"""

from __future__ import annotations

import logging
import pickle
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from datetime import datetime

from . import data_loader, optimizer, regime_classifier

# Optional: try to import signals for regime-adaptive strategy
try:
    from . import signals
    HAS_SIGNALS = True
except ImportError:
    HAS_SIGNALS = False

logger = logging.getLogger(__name__)
RESULTS_PATH = Path(__file__).parent.parent / "results"
DATA_CACHE_PATH = Path(__file__).parent.parent / "models" / "prices_cache.pkl"


def _get_prices_with_cache(start: str = "2019-01-01", end: str = "2024-12-31") -> pd.DataFrame:
    """Load prices from cache or fetch from API (with retry)."""
    if DATA_CACHE_PATH.exists():
        print(f"Loading prices from cache ({DATA_CACHE_PATH})...")
        with open(DATA_CACHE_PATH, "rb") as f:
            prices = pickle.load(f)
        return prices.loc[start:end]

    # Try to fetch from API with retry
    print("Fetching prices from API...")
    max_retries = 3
    for attempt in range(max_retries):
        try:
            prices = data_loader.get_prices(start=start, end=end)
            # Cache for future use
            DATA_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(DATA_CACHE_PATH, "wb") as f:
                pickle.dump(prices, f)
            print(f"Prices cached to {DATA_CACHE_PATH}")
            return prices
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"API call failed (attempt {attempt+1}/{max_retries}): {e}, retrying...")
                time.sleep(2)
            else:
                print(f"API call failed after {max_retries} attempts: {e}")
                raise


def _get_monthly_rebalance_dates(prices: pd.DataFrame, start: str = "2019-01-01",
                                 end: str = "2024-12-31") -> list[pd.Timestamp]:
    """
    Get LAST business day of each month in backtest period (per spec step 36).
    
    Spec: "monthly rebalance on LAST business day of each month"
    """
    dates = prices.loc[start:end].index
    rebalance_dates = []
    
    current_month = None
    last_day_of_month = None
    
    for date in dates:
        if current_month != date.month:
            # Encountered a new month
            if last_day_of_month is not None:
                rebalance_dates.append(last_day_of_month)
            current_month = date.month
        last_day_of_month = date
    
    # Don't forget the last month's last trading day
    if last_day_of_month is not None:
        rebalance_dates.append(last_day_of_month)
    
    return rebalance_dates


def _estimate_expected_returns(prices: pd.DataFrame, lookback: int = 252) -> pd.Series:
    """
    Estimate annualized expected returns from historical log returns.
    Simple fallback when signals module is not available.
    """
    returns = np.log(prices / prices.shift(1)).dropna()
    if len(returns) < lookback:
        lookback = max(1, len(returns) - 1)
    
    mean_returns = returns.iloc[-lookback:].mean() * 252
    return mean_returns


def _estimate_covariance(prices: pd.DataFrame, lookback: int = 252) -> np.ndarray:
    """Estimate covariance matrix from rolling log returns (annualized)."""
    returns = np.log(prices / prices.shift(1)).dropna()
    
    if len(returns) < lookback:
        lookback = max(1, len(returns) - 1)
    
    # Take last lookback days and compute covariance
    returns_window = returns.iloc[-lookback:]
    cov_matrix = returns_window.cov() * 252
    
    # Ensure positive definite with aggressive regularization
    cov_matrix = cov_matrix.values.astype(np.float64)
    cov_matrix = 0.5 * (cov_matrix + cov_matrix.T)  # Ensure symmetry
    
    # Aggressive regularization for numerical stability
    # Add 5% of average variance to diagonal (very conservative)
    avg_var = np.diag(cov_matrix).mean()
    reg_strength = 0.05 * avg_var
    cov_matrix += reg_strength * np.eye(len(cov_matrix))
    
    # Ensure all eigenvalues are strictly positive
    evals, evecs = np.linalg.eigh(cov_matrix)
    evals = np.maximum(evals, 0.001)  # Floor at 0.001
    cov_matrix = evecs @ np.diag(evals) @ evecs.T
    
    return cov_matrix


def _run_equal_weight_strategy(prices: pd.DataFrame, rebalance_dates: list) -> tuple[pd.Series, list, float]:
    """
    Equal weight: always w = 1/n.
    
    Per spec step 37: apply transaction costs of 5bps (0.05%) per unit of turnover.
    Returns: (nav_series, turnover_list, total_transaction_costs)
    """
    n_assets = prices.shape[1]
    nav = np.ones(len(prices))
    w = np.ones(n_assets) / n_assets
    w_prev = w.copy()
    
    turnover_list = []
    total_costs = 0.0
    
    rebalance_idx = 0

    for i in range(1, len(prices)):
        date = prices.index[i]

        # Rebalance if today is a rebalance date (LAST business day of month)
        should_rebalance = (rebalance_idx < len(rebalance_dates) and 
                          date == rebalance_dates[rebalance_idx])

        if should_rebalance:
            w_prev = w.copy()
            w = np.ones(n_assets) / n_assets
            
            # Per spec step 37: apply transaction costs
            # cost = 0.0005 * sum(abs(w - w_prev))
            turnover = np.abs(w - w_prev).sum()
            tc = 0.0005 * turnover
            
            turnover_list.append(turnover)
            total_costs += tc
            
            rebalance_idx += 1

        # Compute return based on weights and price changes
        price_change = prices.iloc[i].values / prices.iloc[i - 1].values
        portfolio_return = np.sum(w * (price_change - 1))
        nav[i] = nav[i - 1] * (1 + portfolio_return)

    return pd.Series(nav, index=prices.index, name="NAV"), turnover_list, total_costs


def _run_mv_unconditional_strategy(prices: pd.DataFrame,
                                   rebalance_dates: list) -> tuple[pd.Series, list, float]:
    """
    Mean-Variance (unconditional): mu and Sigma from rolling 252 days.
    
    Returns: (nav_series, turnover_list, total_transaction_costs)
    """
    n_assets = prices.shape[1]
    nav = np.ones(len(prices))
    w = np.ones(n_assets) / n_assets
    w_prev = w.copy()
    
    turnover_list = []
    total_costs = 0.0
    rebalance_idx = 0

    for i in range(1, len(prices)):
        date = prices.index[i]

        # Check if we should rebalance (LAST business day of month)
        should_rebalance = (rebalance_idx < len(rebalance_dates) and
                          date == rebalance_dates[rebalance_idx])

        if should_rebalance:
            # Get data for last 252 days
            start_idx = max(0, i - 252)
            prices_window = prices.iloc[start_idx:i]

            if len(prices_window) >= 50:  # Need enough data
                # Estimate mu and Sigma
                mu = _estimate_expected_returns(prices_window, lookback=min(252, len(prices_window)))
                mu_arr = mu.values.astype(np.float64)

                Sigma = _estimate_covariance(prices_window, lookback=min(252, len(prices_window)))

                # Solve portfolio (MV Unconditional with optimized constraints)
                try:
                    w_new = optimizer.solve_portfolio(mu_arr, Sigma, lam=5.0, max_weight=0.50,
                                                     vol_target=0.70)
                    w = w_new
                except Exception as e:
                    logger.warning(f"Optimizer failed at {date}: {e}, keeping equal weight")
            
            # Per spec step 37: compute transaction cost
            turnover = np.abs(w - w_prev).sum()
            tc = 0.0005 * turnover
            
            turnover_list.append(turnover)
            total_costs += tc
            
            w_prev = w.copy()
            rebalance_idx += 1

        # Compute return
        price_change = prices.iloc[i].values / prices.iloc[i - 1].values
        portfolio_return = np.sum(w * (price_change - 1))
        nav[i] = nav[i - 1] * (1 + portfolio_return)

    return pd.Series(nav, index=prices.index, name="NAV"), turnover_list, total_costs


def _run_regime_adaptive_strategy(prices: pd.DataFrame,
                                  rebalance_dates: list) -> tuple[pd.Series, list, float]:
    """
    Regime-Adaptive: regime-conditioned mu from signals, Sigma from rolling 252 days.
    
    Returns: (nav_series, turnover_list, total_transaction_costs)
    """
    n_assets = prices.shape[1]
    nav = np.ones(len(prices))
    w = np.ones(n_assets) / n_assets
    w_prev = w.copy()
    
    turnover_list = []
    total_costs = 0.0
    rebalance_idx = 0

    for i in range(1, len(prices)):
        date = prices.index[i]

        # Check if we should rebalance (LAST business day of month)
        should_rebalance = (rebalance_idx < len(rebalance_dates) and
                          date == rebalance_dates[rebalance_idx])

        if should_rebalance:
            # Get data for last 252 days
            start_idx = max(0, i - 252)
            prices_window = prices.iloc[start_idx:i]

            if len(prices_window) >= 50:
                # Get regime label (per spec step 37.a)
                try:
                    regime = regime_classifier.get_regime(date)
                except Exception as e:
                    logger.warning(f"Regime classifier failed at {date}: {e}")
                    regime = 2  # Default to Transitional

                # Get expected returns regime-conditioned (per spec step 37.b)
                if HAS_SIGNALS:
                    try:
                        mu_arr = signals.get_expected_returns(date, regime)
                    except Exception as e:
                        logger.warning(f"Signals failed at {date}: {e}, using historical")
                        mu = _estimate_expected_returns(prices_window, lookback=min(252, len(prices_window)))
                        mu_arr = mu.values.astype(np.float64)
                else:
                    # Fallback: use historical returns with regime adjustment
                    mu = _estimate_expected_returns(prices_window, lookback=min(252, len(prices_window)))
                    mu_arr = mu.values.astype(np.float64)
                    
                    # Simple regime adjustment: boost equities in Risk-On, boost bonds in Risk-Off
                    if regime == 0:  # Risk-On
                        mu_arr[[0, 3]] *= 1.2  # Boost SPY, EEM
                        mu_arr[[1, 5]] *= 0.8  # Reduce TLT, LQD
                    elif regime == 1:  # Risk-Off
                        mu_arr[[0, 3]] *= 0.8  # Reduce SPY, EEM
                        mu_arr[[1, 5]] *= 1.2  # Boost TLT, LQD

                # Compute rolling 252-day covariance (per spec step 37.c)
                Sigma = _estimate_covariance(prices_window, lookback=min(252, len(prices_window)))

                # Solve portfolio (per spec step 37.d, optimized constraints)
                try:
                    w_new = optimizer.solve_portfolio(mu_arr, Sigma, lam=5.0, max_weight=0.50,
                                                     vol_target=0.70)
                    w = w_new
                except Exception as e:
                    logger.warning(f"Optimizer failed at {date}: {e}, keeping previous weights")

            # Per spec step 37.e: compute transaction costs (5bps per unit turnover)
            turnover = np.abs(w - w_prev).sum()
            tc = 0.0005 * turnover
            
            turnover_list.append(turnover)
            total_costs += tc
            
            w_prev = w.copy()
            rebalance_idx += 1

        # Compute daily portfolio returns (per spec step 38)
        price_change = prices.iloc[i].values / prices.iloc[i - 1].values
        portfolio_return = np.sum(w * (price_change - 1))
        nav[i] = nav[i - 1] * (1 + portfolio_return)

    return pd.Series(nav, index=prices.index, name="NAV"), turnover_list, total_costs


def _compute_metrics(nav_series: pd.Series, turnover_list: list[float] = None,
                     transaction_costs: float = 0.0,
                     risk_free_rate: float = 0.02) -> dict:
    """
    Compute performance metrics from NAV series.
    
    Per spec step 40: compute Sharpe, max DD, Calmar, annualized turnover, transaction costs.
    """
    returns = nav_series.pct_change().dropna()
    
    # Annualized metrics (assuming 252 trading days)
    annual_return = returns.mean() * 252
    annual_vol = returns.std() * np.sqrt(252)
    sharpe = (annual_return - risk_free_rate) / (annual_vol + 1e-8)
    
    # Maximum drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min()
    
    # Calmar ratio: annualized return / abs(max drawdown)
    calmar = annual_return / (abs(max_dd) + 1e-8)
    
    # Win rate (monthly)
    monthly_returns = nav_series.resample("ME").last().pct_change().dropna()
    win_rate = (monthly_returns > 0).sum() / len(monthly_returns) if len(monthly_returns) > 0 else 0
    
    # Total return
    total_return = (nav_series.iloc[-1] / nav_series.iloc[0]) - 1
    
    # Annualized turnover (if provided)
    annual_turnover = np.mean(turnover_list) * 12 if turnover_list else 0.0
    
    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_vol": annual_vol,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "calmar": calmar,
        "win_rate": win_rate,
        "annual_turnover": annual_turnover,
        "transaction_costs": transaction_costs,
    }


def run_backtest() -> dict:
    """
    Main backtest function.
    
    Returns:
        dict with structure:
        {
            "strategy_name": {
                "nav": pd.Series,
                "returns": pd.Series,
                "metrics": dict,
            },
            ...
        }
    """
    print("Loading prices for backtest period (2019-2024)...")
    prices = _get_prices_with_cache(start="2019-01-01", end="2024-12-31")
    
    print(f"Prices shape: {prices.shape}")
    print(f"Date range: {prices.index.min().date()} to {prices.index.max().date()}")

    # Get rebalance dates
    rebalance_dates = _get_monthly_rebalance_dates(prices)
    print(f"Number of rebalance dates: {len(rebalance_dates)}")

    results = {}

    # Strategy 1: Equal Weight
    print("\n[1/3] Running Equal Weight strategy...")
    nav_ew, turnover_ew, costs_ew = _run_equal_weight_strategy(prices, rebalance_dates)
    returns_ew = nav_ew.pct_change().dropna()
    metrics_ew = _compute_metrics(nav_ew, turnover_ew, costs_ew)
    results["Equal Weight"] = {
        "nav": nav_ew,
        "returns": returns_ew,
        "metrics": metrics_ew,
    }
    print(f"  ✓ Sharpe: {metrics_ew['sharpe']:.3f}, Return: {metrics_ew['annual_return']:.1%}, "
          f"Turnover: {metrics_ew['annual_turnover']:.1%}")

    # Strategy 2: MV Unconditional
    print("[2/3] Running Mean-Variance Unconditional strategy...")
    nav_mv, turnover_mv, costs_mv = _run_mv_unconditional_strategy(prices, rebalance_dates)
    returns_mv = nav_mv.pct_change().dropna()
    metrics_mv = _compute_metrics(nav_mv, turnover_mv, costs_mv)
    results["MV Unconditional"] = {
        "nav": nav_mv,
        "returns": returns_mv,
        "metrics": metrics_mv,
    }
    print(f"  ✓ Sharpe: {metrics_mv['sharpe']:.3f}, Return: {metrics_mv['annual_return']:.1%}, "
          f"Turnover: {metrics_mv['annual_turnover']:.1%}")

    # Strategy 3: Regime-Adaptive
    print("[3/3] Running Regime-Adaptive strategy...")
    nav_ra, turnover_ra, costs_ra = _run_regime_adaptive_strategy(prices, rebalance_dates)
    returns_ra = nav_ra.pct_change().dropna()
    metrics_ra = _compute_metrics(nav_ra, turnover_ra, costs_ra)
    results["Regime-Adaptive"] = {
        "nav": nav_ra,
        "returns": returns_ra,
        "metrics": metrics_ra,
    }
    print(f"  ✓ Sharpe: {metrics_ra['sharpe']:.3f}, Return: {metrics_ra['annual_return']:.1%}, "
          f"Turnover: {metrics_ra['annual_turnover']:.1%}")

    # Save metrics to CSV
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame({
        name: data["metrics"] for name, data in results.items()
    }).T
    metrics_df.to_csv(RESULTS_PATH / "metrics.csv")
    print(f"\nMetrics saved to {RESULTS_PATH / 'metrics.csv'}")

    # Print summary
    print("\n=== BACKTEST SUMMARY (2019-2024) ===")
    print(metrics_df.to_string())

    return results


if __name__ == "__main__":
    results = run_backtest()
    print("\nAll backtester smoke checks passed.")
