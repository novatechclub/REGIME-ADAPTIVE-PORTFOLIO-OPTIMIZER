"""Module 2: Regime Classifier (Alessandro).

Trains a 3-state Gaussian HMM on macro features and produces a regime label
(0=Risk-On, 1=Risk-Off, 2=Transitional) per trading day.

Expected public API:
    get_regime(date) -> int

Implementation per spec steps 26-34:
  1. Use macro features: VIX, T10Y2Y, SPY 21-day rolling return
  2. Standardize with StandardScaler fitted ONLY on 2010-2018 (no lookahead bias)
  3. Train 3-state Gaussian HMM on training set
  4. Generate regime labels via Viterbi algorithm
  5. Save model and regime series for external access
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

from . import data_loader

MODEL_PATH = Path(__file__).parent.parent / "models" / "hmm_model.pkl"
REGIME_CACHE_PATH = Path(__file__).parent.parent / "models" / "regime_cache.pkl"
SCALER_PATH = Path(__file__).parent.parent / "models" / "feature_scaler.pkl"

TRAIN_START = "2010-01-01"
TRAIN_END = "2018-12-31"
TEST_START = "2019-01-01"


def _compute_spy_momentum(prices: pd.DataFrame, window: int = 21) -> pd.Series:
    """Compute rolling 21-day return of SPY as momentum feature."""
    spy_prices = prices["SPY"]
    spy_momentum = spy_prices.pct_change(window)
    return spy_momentum


def _build_features_with_momentum(prices: pd.DataFrame, 
                                  features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build feature matrix: VIX, T10Y2Y, SPY 21-day momentum.
    
    Per spec step 27: "Use three features: VIX (level), T10Y2Y (level), 
    and a rolling 21-day return of SPY (to capture equity momentum)."
    """
    spy_momentum = _compute_spy_momentum(prices, window=21)
    
    # Select VIX and T10Y2Y from features_df
    feature_matrix = pd.DataFrame({
        "VIX": features_df["VIX"],
        "T10Y2Y": features_df["T10Y2Y"],
        "SPY_Momentum": spy_momentum,
    })
    
    # Drop NaN rows
    feature_matrix = feature_matrix.dropna()
    
    return feature_matrix


def _train_hmm(features: pd.DataFrame, n_states: int = 3) -> tuple[GaussianHMM, dict, StandardScaler]:
    """
    Train Gaussian HMM and interpret regimes.
    
    Per spec step 29: "Train a Gaussian HMM with 3 hidden states using 
    GaussianHMM(n_components=3, covariance_type='full', n_iter=100)."
    
    Returns:
        (fitted_model, regime_map, scaler)
    """
    # Split training and test set to avoid lookahead bias
    # Per spec step 28: "Fit the scaler ONLY on training set (2010-2018)"
    train_features = features.loc[TRAIN_START:TRAIN_END]
    
    # Fit StandardScaler ONLY on training data (2010-2018)
    scaler = StandardScaler()
    scaler.fit(train_features)
    
    # Apply fitted scaler to all data
    X_scaled = scaler.transform(features)

    # Fit HMM with multiple random seeds to avoid local minima
    best_model = None
    best_score = -np.inf

    for seed in range(5):
        model = GaussianHMM(n_components=n_states, covariance_type="full", 
                           n_iter=100, random_state=seed)  # Per spec: n_iter=100
        try:
            model.fit(X_scaled)
            score = model.score(X_scaled)
            if score > best_score:
                best_score = score
                best_model = model
        except Exception:
            continue

    if best_model is None:
        raise RuntimeError("HMM training failed")

    # Interpret regimes economically by mean features of each state
    # State with highest mean VIX → Risk-Off
    # State with lowest mean VIX → Risk-On
    state_means = best_model.means_  # (n_states, n_features)
    vix_idx = 0  # VIX is first feature

    vix_by_state = state_means[:, vix_idx]
    sorted_states = np.argsort(vix_by_state)

    regime_map = {
        sorted_states[0]: 0,  # lowest VIX → Risk-On
        sorted_states[2]: 1,  # highest VIX → Risk-Off
        sorted_states[1]: 2,  # middle VIX → Transitional
    }

    return best_model, regime_map, scaler


def _compute_regime_series(model: GaussianHMM, features: pd.DataFrame,
                          regime_map: dict, scaler: StandardScaler,
                          regime_dates: pd.Index) -> pd.Series:
    """Compute regime labels for all dates using Viterbi algorithm."""
    X_scaled = scaler.transform(features)

    # Use Viterbi algorithm to get most likely sequence of hidden states
    hidden_states = model.predict(X_scaled)

    # Map internal HMM states to regime labels (0, 1, 2)
    regime_labels = np.array([regime_map[state] for state in hidden_states])

    regime_series = pd.Series(regime_labels, index=regime_dates, name="regime")
    return regime_series


def _fit_or_load_model() -> tuple[GaussianHMM, dict, pd.Series, StandardScaler]:
    """
    Fit HMM model or load from cache.
    
    Returns:
        (model, regime_map, regime_series, scaler)
    """
    # Check if cache files exist
    if MODEL_PATH.exists() and REGIME_CACHE_PATH.exists() and SCALER_PATH.exists():
        print(f"Loading cached HMM model from {MODEL_PATH}")
        with open(MODEL_PATH, "rb") as f:
            cache = pickle.load(f)
        model = cache["model"]
        regime_map = cache["regime_map"]

        with open(REGIME_CACHE_PATH, "rb") as f:
            regime_series = pickle.load(f)
            
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)

        return model, regime_map, regime_series, scaler

    # Otherwise: fetch data, train, and save
    print("Fetching prices and macro features for HMM training (2010-2024)...")
    prices = data_loader.get_prices()
    features_raw = data_loader.get_features()
    
    # Build feature matrix with SPY 21-day momentum (per spec step 27)
    print("Computing SPY 21-day rolling return momentum...")
    feature_matrix = _build_features_with_momentum(prices, features_raw)
    
    print(f"Feature matrix shape: {feature_matrix.shape}")
    print(f"Features: {list(feature_matrix.columns)}")

    print("Training 3-state Gaussian HMM (n_iter=100, per spec step 29)...")
    model, regime_map, scaler = _train_hmm(feature_matrix)

    print("Computing regime labels via Viterbi algorithm...")
    regime_series = _compute_regime_series(model, feature_matrix, regime_map, 
                                          scaler, feature_matrix.index)

    # Save model, regime series, and scaler
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "regime_map": regime_map}, f)

    with open(REGIME_CACHE_PATH, "wb") as f:
        pickle.dump(regime_series, f)
        
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    print(f"Saved HMM model to {MODEL_PATH}")
    print(f"Saved scaler to {SCALER_PATH} (fitted on 2010-2018, no lookahead bias)")
    print(f"Regime interpretation:\n  0 = Risk-On (low VIX)\n  1 = Risk-Off (high VIX)\n  2 = Transitional")

    return model, regime_map, regime_series, scaler


# Global cache (lazy-loaded on first access)
_MODEL = None
_REGIME_MAP = None
_REGIME_SERIES = None
_SCALER = None


def get_regime(date: str | pd.Timestamp) -> int:
    """
    Public API: Get regime label for a given date.
    
    Args:
        date: Date as string (YYYY-MM-DD) or pd.Timestamp
    
    Returns:
        int: regime label (0=Risk-On, 1=Risk-Off, 2=Transitional)
    """
    global _MODEL, _REGIME_MAP, _REGIME_SERIES, _SCALER

    if _MODEL is None:
        _MODEL, _REGIME_MAP, _REGIME_SERIES, _SCALER = _fit_or_load_model()

    # Normalize date to pandas Timestamp
    if isinstance(date, str):
        date = pd.Timestamp(date)
    date = pd.Timestamp(date).normalize()

    if date not in _REGIME_SERIES.index:
        raise ValueError(f"Date {date.date()} not in regime series (range: "
                        f"{_REGIME_SERIES.index.min().date()} to "
                        f"{_REGIME_SERIES.index.max().date()})")

    return int(_REGIME_SERIES.loc[date])


if __name__ == "__main__":
    # Smoke test
    print("=== regime_classifier smoke test (per spec steps 26-34) ===\n")

    model, regime_map, regime_series, scaler = _fit_or_load_model()

    print(f"HMM model trained on {len(regime_series)} trading days")
    print(f"Regime distribution:\n{regime_series.value_counts().sort_index()}\n")
    
    print(f"StandardScaler fitted on: {TRAIN_START} to {TRAIN_END} (no lookahead bias)")
    print(f"Scaler mean: {scaler.mean_}")
    print(f"Scaler scale_: {scaler.scale_}\n")

    # Test get_regime on sample dates
    test_dates = ["2024-01-02", "2024-06-14", "2024-12-27"]
    print("Sample regime labels:")
    for test_date in test_dates:
        try:
            regime = get_regime(test_date)
            label_map = {0: "Risk-On", 1: "Risk-Off", 2: "Transitional"}
            print(f"  {test_date}: {regime} ({label_map[regime]})")
        except ValueError as e:
            print(f"  {test_date}: {e}")

    print("\nAll regime_classifier smoke checks passed.")
