"""Module 1: Data Pipeline (Moritz).

Produces the single clean master data source that all other modules import.
No other module should touch raw data directly.

Public API:
    get_prices(start, end)   -> DataFrame of daily adjusted close prices.
    get_features(start, end) -> DataFrame of macro features (VIX, T10Y2Y, NFCI).
    get_returns(start, end)  -> DataFrame of daily log returns.

Requires the environment variable FRED_API_KEY to be set.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred

TICKERS = ["SPY", "TLT", "GLD", "EEM", "VNQ", "LQD", "DBC", "IEF"]

DEFAULT_START = "2010-01-01"
DEFAULT_END = "2024-12-31"

FRED_SERIES = {
    "VIX": "VIXCLS",
    "T10Y2Y": "T10Y2Y",
    "NFCI": "NFCI",
}

MAX_NAN_FRACTION = 0.05


def _fetch_prices(start: str, end: str) -> pd.DataFrame:
    raw = yf.download(
        TICKERS,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
    )
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]]
        prices.columns = TICKERS

    prices = prices[TICKERS]
    prices.index = pd.DatetimeIndex(prices.index).normalize()
    prices.index.name = "date"
    return prices

# FRED key: 1f3d1e74a3a723558bbe8123849401e0
def _fetch_macro(start: str, end: str) -> pd.DataFrame:
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        raise RuntimeError(
            "FRED_API_KEY environment variable is not set. "
            "Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html "
            "and export it before running."
        )
    fred = Fred(api_key=api_key)

    series = {}
    for name, code in FRED_SERIES.items():
        s = fred.get_series(code, observation_start=start, observation_end=end)
        s.index = pd.DatetimeIndex(s.index).normalize()
        series[name] = s

    macro = pd.DataFrame(series)
    macro = macro.asfreq("B").ffill()
    macro.index.name = "date"
    return macro


def _align_and_clean(
    prices: pd.DataFrame, macro: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    combined = pd.concat([prices, macro], axis=1, join="inner")

    nan_fractions = combined.isna().mean()
    bad = nan_fractions[nan_fractions > MAX_NAN_FRACTION]
    if not bad.empty:
        raise AssertionError(
            f"Columns exceed {MAX_NAN_FRACTION:.0%} missing: {bad.to_dict()}"
        )

    combined = combined.ffill().bfill()

    prices_clean = combined[TICKERS]
    macro_clean = combined[list(FRED_SERIES.keys())]
    return prices_clean, macro_clean


def _load(start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    prices = _fetch_prices(start, end)
    macro = _fetch_macro(start, end)
    return _align_and_clean(prices, macro)


def _slice(df: pd.DataFrame, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    if start is None and end is None:
        return df
    return df.loc[start:end]


def get_prices(
    start: Optional[str] = None, end: Optional[str] = None
) -> pd.DataFrame:
    prices, _ = _load(start or DEFAULT_START, end or DEFAULT_END)
    return _slice(prices, start, end)


def get_features(
    start: Optional[str] = None, end: Optional[str] = None
) -> pd.DataFrame:
    _, macro = _load(start or DEFAULT_START, end or DEFAULT_END)
    return _slice(macro, start, end)


def get_returns(
    start: Optional[str] = None, end: Optional[str] = None
) -> pd.DataFrame:
    prices = get_prices(start, end)
    returns = np.log(prices / prices.shift(1)).dropna()
    return returns


if __name__ == "__main__":
    prices = get_prices()
    features = get_features()
    returns = get_returns()

    for name, df in [("prices", prices), ("features", features), ("returns", returns)]:
        print(f"\n=== {name} ===")
        print(f"shape:   {df.shape}")
        print(f"range:   {df.index.min().date()} -> {df.index.max().date()}")
        print(f"columns: {list(df.columns)}")
        print(df.tail(3))
        assert df.isna().sum().sum() == 0, f"{name} still contains NaNs"

    print("\nAll data_loader smoke checks passed.")
