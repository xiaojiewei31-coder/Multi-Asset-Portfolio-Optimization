from __future__ import annotations

import numpy as np
import pandas as pd


def annualized_geometric_return(monthly_returns: pd.Series) -> float:
    periods = monthly_returns.dropna().shape[0]
    if periods == 0:
        return float("nan")
    return float(np.prod(1 + monthly_returns.dropna()) ** (12 / periods) - 1)


def annualized_covariance(monthly_returns: pd.DataFrame) -> pd.DataFrame:
    return monthly_returns.cov() * 12


def portfolio_return(expected_returns: pd.Series, weights: np.ndarray) -> float:
    return float(np.dot(weights, expected_returns.values))


def portfolio_volatility(covariance: pd.DataFrame, weights: np.ndarray) -> float:
    variance = float(weights.T @ covariance.values @ weights)
    return float(np.sqrt(max(variance, 0)))


def performance_summary(
    monthly_returns: pd.DataFrame,
    weights: np.ndarray,
    risk_free_rate: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    expected_returns = monthly_returns.apply(annualized_geometric_return)
    covariance = annualized_covariance(monthly_returns)
    asset_volatility = pd.Series(
        np.sqrt(np.diag(covariance.values)),
        index=expected_returns.index,
        name="annualized_volatility",
    )

    summary = pd.DataFrame(
        {
            "annualized_return": expected_returns,
            "annualized_volatility": asset_volatility,
        }
    )
    summary["sharpe_ratio"] = (
        (summary["annualized_return"] - risk_free_rate)
        / summary["annualized_volatility"]
    )

    portfolio_monthly_returns = monthly_returns.dot(weights)
    portfolio_annual_return = annualized_geometric_return(portfolio_monthly_returns)
    portfolio_annual_volatility = portfolio_volatility(covariance, weights)
    portfolio_sharpe = (
        (portfolio_annual_return - risk_free_rate) / portfolio_annual_volatility
    )

    summary.loc["Current Portfolio"] = {
        "annualized_return": portfolio_annual_return,
        "annualized_volatility": portfolio_annual_volatility,
        "sharpe_ratio": portfolio_sharpe,
    }
    return summary, covariance

