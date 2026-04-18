"""Module 4: Convex Optimizer (Moritz).

Solves the regime-conditioned mean-variance portfolio optimization problem.

    max   w.T @ mu  -  lam * w.T @ Sigma @ w
    s.t.  sum(w) == 1
          0 <= w <= max_weight
          sqrt(252 * w.T @ Sigma @ w) <= vol_target
          ||w - w_prev||_1 <= turnover_budget    (if provided)

The function signature is a binding interface contract -- other modules call
solve_portfolio from their code. Do not change parameter names.
"""

from __future__ import annotations

import logging
from typing import Optional

import cvxpy as cp
import numpy as np

logger = logging.getLogger(__name__)


def solve_portfolio(
    mu: np.ndarray,
    Sigma: np.ndarray,
    lam: float = 2.0,
    max_weight: float = 0.30,
    vol_target: float = 0.10,
    turnover_budget: Optional[float] = None,
    w_prev: Optional[np.ndarray] = None,
) -> np.ndarray:
    mu = np.asarray(mu, dtype=float).ravel()
    Sigma = np.asarray(Sigma, dtype=float)
    n = mu.shape[0]
    if Sigma.shape != (n, n):
        raise ValueError(f"Sigma must be ({n},{n}), got {Sigma.shape}")

    Sigma = 0.5 * (Sigma + Sigma.T)
    # Cholesky factor for the SOCP form of the vol constraint.
    # ||L.T @ w||_2 = sqrt(w.T Sigma w) is numerically more reliable than
    # the scalar quad_form <= k^2 form in SCS.
    L = np.linalg.cholesky(Sigma + 1e-10 * np.eye(n))

    w = cp.Variable(n)
    Sigma_psd = cp.psd_wrap(Sigma)

    objective = cp.Maximize(mu @ w - lam * cp.quad_form(w, Sigma_psd))

    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        w <= max_weight,
        cp.norm(L.T @ w, 2) <= vol_target / np.sqrt(252),
    ]

    if turnover_budget is not None and w_prev is not None:
        w_prev = np.asarray(w_prev, dtype=float).ravel()
        if w_prev.shape[0] != n:
            raise ValueError(f"w_prev must have length {n}, got {w_prev.shape[0]}")
        constraints.append(cp.norm(w - w_prev, 1) <= turnover_budget)

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS)

    if problem.status not in ("optimal", "optimal_inaccurate") or w.value is None:
        logger.warning(
            "solve_portfolio: solver status=%s, falling back to equal weights",
            problem.status,
        )
        return np.ones(n) / n

    return np.asarray(w.value).ravel()


if __name__ == "__main__":
    np.random.seed(0)
    n = 8
    # Simulate 500 days of daily returns (~1% daily vol) to get a realistic
    # empirical covariance matrix; mu is the annualized sample mean.
    daily_rets = np.random.normal(loc=0.0004, scale=0.01, size=(500, n))
    Sigma = np.cov(daily_rets.T)
    mu = daily_rets.mean(axis=0) * 252

    w1 = solve_portfolio(mu, Sigma)
    print("Unconstrained (no turnover) weights:")
    print(np.round(w1, 4))
    print(f"  sum = {w1.sum():.6f}")

    w_prev = np.ones(n) / n
    w2 = solve_portfolio(mu, Sigma, turnover_budget=0.20, w_prev=w_prev)
    print("\nWith turnover_budget=0.20 from equal weights:")
    print(np.round(w2, 4))
    print(f"  sum          = {w2.sum():.6f}")
    print(f"  L1 turnover  = {np.abs(w2 - w_prev).sum():.4f}")

    assert abs(w1.sum() - 1.0) < 1e-4
    assert (w1 >= -1e-6).all()
    assert (w1 <= 0.30 + 1e-6).all()
    assert abs(w2.sum() - 1.0) < 1e-4
    assert np.abs(w2 - w_prev).sum() <= 0.20 + 1e-4

    print("\nAll optimizer smoke checks passed.")
