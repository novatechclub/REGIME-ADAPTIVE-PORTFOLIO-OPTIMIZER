---
name: Project context - Regime-Adaptive Portfolio Optimizer
description: Team structure, module ownership, integration contracts, and FRED API key location
type: project
---

Nova Tech Club quant project. Team: Max, Moritz, Alessandro.

**Max owns:** Module 3 (signals.py), all 4 visualizations (results/figures/), 10-slide presentation.
**Moritz owns:** Module 1 (data_loader.py), Module 4 (optimizer.py).
**Alessandro owns:** Module 2 (regime_classifier.py), Module 5 (backtester.py + stress_test.py).

**Key interface contracts (never change signatures):**
- `data_loader.get_prices(start, end)` → DataFrame dates×tickers
- `data_loader.get_returns(start, end)` → DataFrame of daily log returns
- `regime_classifier.get_regime(date)` → int 0/1/2
- `signals.get_expected_returns(date, regime_label)` → np.ndarray shape (8,)
- `optimizer.solve_portfolio(mu, Sigma, lam, max_weight, vol_target, turnover_budget, w_prev)` → np.ndarray shape (8,)

**FRED API key:** 1f3d1e74a3a723558bbe8123849401e0 (set as env var FRED_API_KEY before running)
**Asset order:** ["SPY","TLT","GLD","EEM","VNQ","LQD","DBC","IEF"] (indices 0-7)
**Regime labels:** 0=Risk-On, 1=Risk-Off, 2=Transitional

**Why:** Module 3 (signals.py) is complete as of 2026-05-06. Next for Max: Step 3b visualizations (4 PNGs at 300 DPI).
**How to apply:** When working on visualizations, backtester output (returns series + weights timeseries) from Alessandro is needed as input.
