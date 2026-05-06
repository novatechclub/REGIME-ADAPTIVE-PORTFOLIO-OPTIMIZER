"""Module 3: Signal Layer (Max).

Produces a regime-conditioned expected-return vector for the 8 ETFs.

Expected public API:
    get_expected_returns(date, regime_label) -> np.ndarray of shape (8,)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from . import data_loader
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src import data_loader

TICKERS = ["SPY", "TLT", "GLD", "EEM", "VNQ", "LQD", "DBC", "IEF"]

REGIME_SCALING = {0: 1.0, 1: 0.3, 2: 0.6}

# Risk-Off defensive tilt applied on top of regime-scaled z-score (spec step 23)
_RISK_OFF_TILT: dict[int, float] = {
    TICKERS.index("TLT"): +0.10,
    TICKERS.index("GLD"): +0.10,
    TICKERS.index("EEM"): -0.10,
    TICKERS.index("SPY"): -0.10,
}

_PRICES: pd.DataFrame | None = None


def _get_prices() -> pd.DataFrame:
    global _PRICES
    if _PRICES is None:
        _PRICES = data_loader.get_prices()
    return _PRICES


def get_expected_returns(
    date: str | pd.Timestamp,
    regime_label: int,
) -> np.ndarray:
    """Compute regime-conditioned expected returns for all 8 ETFs.

    Args:
        date: Rebalancing date (string 'YYYY-MM-DD' or pd.Timestamp).
        regime_label: 0=Risk-On, 1=Risk-Off, 2=Transitional.

    Returns:
        np.ndarray of shape (8,): expected returns ordered as TICKERS.

    Raises:
        ValueError: If there is insufficient price history (< 252 trading days).
    """
    date = pd.Timestamp(date).normalize()
    prices = _get_prices()

    # Find last available trading day on or before the requested date
    valid_dates = prices.index[prices.index <= date]
    if len(valid_dates) == 0:
        raise ValueError(f"No price data available on or before {date.date()}.")
    t = valid_dates[-1]
    t_pos = prices.index.get_loc(t)

    # Need t_pos >= 252 so that prices.iloc[t_pos - 252] is a valid row
    if t_pos < 252:
        raise ValueError(
            f"Insufficient history at {date.date()}: need at least 252 trading days "
            f"before this date, but only {t_pos} are available."
        )

    # 12-1 month cross-sectional momentum: log(P[t-21] / P[t-252])
    # Excludes the last month (21 days) to avoid short-term reversal noise.
    p_start = prices.iloc[t_pos - 252]
    p_end = prices.iloc[t_pos - 21]
    momentum = np.log(p_end / p_start)  # pd.Series, shape (8,)

    # Cross-sectional standardization: z-score across the 8 assets at this date
    mu_cross = momentum.mean()
    sigma_cross = momentum.std()

    if np.isnan(sigma_cross) or sigma_cross < 1e-10:
        z_scores = np.zeros(len(TICKERS))
    else:
        z_scores = ((momentum - mu_cross) / sigma_cross).values.astype(float)

    # Regime scaling
    scale = REGIME_SCALING[regime_label]
    expected_returns = z_scores * scale

    # Risk-Off defensive tilt (spec step 23)
    if regime_label == 1:
        for asset_idx, tilt in _RISK_OFF_TILT.items():
            expected_returns[asset_idx] += tilt

    return expected_returns


if __name__ == "__main__":
    print("=== signals.py smoke test (spec steps 19-25) ===\n")

    LABEL_NAMES = {0: "Risk-On", 1: "Risk-Off", 2: "Transitional"}

    # Three test dates covering one date per regime (regime passed in manually here;
    # in the full pipeline these labels come from regime_classifier.get_regime()).
    test_cases = [
        ("2015-06-30", 0),   # mid-cycle, Risk-On
        ("2020-03-31", 1),   # COVID crash, Risk-Off
        ("2022-09-30", 2),   # rate-hike uncertainty, Transitional
    ]

    for date_str, regime in test_cases:
        label = LABEL_NAMES[regime]
        print(f"--- {date_str} | Regime {regime}: {label} ---")

        mu = get_expected_returns(date_str, regime)
        ranked = sorted(zip(TICKERS, mu), key=lambda x: x[1], reverse=True)

        print("Expected returns (raw):")
        for ticker, ret in zip(TICKERS, mu):
            print(f"  {ticker}: {ret:+.4f}")

        print("Ranking (highest to lowest expected return):")
        for rank, (ticker, ret) in enumerate(ranked, 1):
            print(f"  {rank}. {ticker}: {ret:+.4f}")

        if regime == 1:
            tlt_rank = next(r for r, (t, _) in enumerate(ranked, 1) if t == "TLT")
            gld_rank = next(r for r, (t, _) in enumerate(ranked, 1) if t == "GLD")
            spy_rank = next(r for r, (t, _) in enumerate(ranked, 1) if t == "SPY")
            eem_rank = next(r for r, (t, _) in enumerate(ranked, 1) if t == "EEM")
            print(
                f"\n[Risk-Off check] TLT rank={tlt_rank}, GLD rank={gld_rank}, "
                f"SPY rank={spy_rank}, EEM rank={eem_rank}"
            )
            safe_ok = tlt_rank <= 4 and gld_rank <= 4
            risky_ok = spy_rank >= 5 and eem_rank >= 5
            print(f"  Safe havens in top half:  {'OK' if safe_ok else 'WARNING'}")
            print(f"  Risk assets in bot. half: {'OK' if risky_ok else 'WARNING'}")

        print()

    print("Smoke test complete.")
