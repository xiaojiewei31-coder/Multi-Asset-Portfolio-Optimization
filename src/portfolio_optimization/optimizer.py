from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _stabilize_covariance(covariance: pd.DataFrame) -> pd.DataFrame:
    matrix = (covariance + covariance.T) / 2
    matrix = matrix + np.eye(len(matrix)) * 1e-6
    return matrix


def portfolio_volatility(weights: np.ndarray, covariance: pd.DataFrame) -> float:
    variance = float(weights.T @ covariance.values @ weights)
    return float(np.sqrt(max(variance, 0)))


def minimize_volatility_for_return(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    target_return: float,
    bounds: list[tuple[float, float]],
) -> tuple[np.ndarray, float] | None:
    covariance = _stabilize_covariance(covariance)
    initial = np.repeat(1 / len(expected_returns), len(expected_returns))
    constraints = (
        {"type": "eq", "fun": lambda weights: np.sum(weights) - 1},
        {
            "type": "eq",
            "fun": lambda weights: np.dot(weights, expected_returns.values) - target_return,
        },
    )
    result = minimize(
        lambda weights: portfolio_volatility(weights, covariance),
        initial,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    if not result.success:
        return None
    return result.x, float(result.fun)


def efficient_frontier(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    points: int = 80,
    allow_short: bool = True,
) -> pd.DataFrame:
    lower_bound = -1 if allow_short else 0
    bounds = [(lower_bound, 1) for _ in expected_returns]
    target_returns = np.linspace(expected_returns.min(), expected_returns.max() * 2, points)

    rows = []
    for target in target_returns:
        result = minimize_volatility_for_return(
            expected_returns,
            covariance,
            float(target),
            bounds,
        )
        if result is None:
            continue
        weights, volatility = result
        row = {
            "target_return": float(target),
            "volatility": volatility,
        }
        row.update({f"weight_{asset}": weight for asset, weight in zip(expected_returns.index, weights)})
        rows.append(row)

    return pd.DataFrame(rows)


def tangency_portfolio(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    risk_free_rate: float,
    allow_short: bool = False,
) -> dict[str, object]:
    covariance = _stabilize_covariance(covariance)
    lower_bound = -1 if allow_short else 0
    bounds = [(lower_bound, 1) for _ in expected_returns]
    initial = np.repeat(1 / len(expected_returns), len(expected_returns))
    constraints = {"type": "eq", "fun": lambda weights: np.sum(weights) - 1}

    def negative_sharpe(weights: np.ndarray) -> float:
        ret = float(np.dot(weights, expected_returns.values))
        vol = portfolio_volatility(weights, covariance)
        if vol == 0:
            return np.inf
        return -((ret - risk_free_rate) / vol)

    result = minimize(
        negative_sharpe,
        initial,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    if not result.success:
        raise RuntimeError(f"Tangency portfolio optimization failed: {result.message}")

    weights = result.x
    ret = float(np.dot(weights, expected_returns.values))
    vol = portfolio_volatility(weights, covariance)
    sharpe = float((ret - risk_free_rate) / vol)
    return {
        "weights": pd.Series(weights, index=expected_returns.index),
        "return": ret,
        "volatility": vol,
        "sharpe_ratio": sharpe,
    }


def minimum_variance_portfolio(
    covariance: pd.DataFrame,
    lower_bound: float = 0,
    upper_bound: float = 1,
) -> pd.Series:
    covariance = _stabilize_covariance(covariance)
    initial = np.repeat(1 / len(covariance), len(covariance))
    constraints = {"type": "eq", "fun": lambda weights: np.sum(weights) - 1}
    bounds = [(lower_bound, upper_bound) for _ in covariance]

    result = minimize(
        lambda weights: portfolio_volatility(weights, covariance),
        initial,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    if not result.success:
        return pd.Series(initial, index=covariance.index)
    return pd.Series(result.x, index=covariance.index)


def max_sharpe_weights(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    risk_free_rate: float,
    lower_bound: float = 0,
    upper_bound: float = 1,
) -> pd.Series:
    try:
        result = tangency_portfolio(
            expected_returns,
            covariance,
            risk_free_rate,
            allow_short=lower_bound < 0,
        )
    except RuntimeError:
        return pd.Series(np.repeat(1 / len(expected_returns), len(expected_returns)), index=expected_returns.index)

    weights = result["weights"].clip(lower=lower_bound, upper=upper_bound)
    total = weights.sum()
    if total <= 0:
        return pd.Series(np.repeat(1 / len(expected_returns), len(expected_returns)), index=expected_returns.index)
    return weights / total


def capital_allocation_for_target_return(
    target_return: float,
    tangency_return: float,
    tangency_volatility: float,
    risk_free_rate: float,
) -> dict[str, float]:
    tangency_weight = (target_return - risk_free_rate) / (tangency_return - risk_free_rate)
    risk_free_weight = 1 - tangency_weight
    return {
        "risk_free_weight": float(risk_free_weight),
        "tangency_weight": float(tangency_weight),
        "expected_return": float(target_return),
        "expected_volatility": float(abs(tangency_weight) * tangency_volatility),
    }
