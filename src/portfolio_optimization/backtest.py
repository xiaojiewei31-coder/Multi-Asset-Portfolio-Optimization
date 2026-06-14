from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import annualized_covariance, annualized_geometric_return
from .optimizer import max_sharpe_weights, minimum_variance_portfolio


def drawdown(cumulative_returns: pd.Series) -> pd.Series:
    wealth = 1 + cumulative_returns
    peak = wealth.cummax()
    return wealth / peak - 1


def evaluate_return_series(monthly_returns: pd.Series, risk_free_rate: float) -> dict[str, float]:
    annual_return = annualized_geometric_return(monthly_returns)
    annual_volatility = float(monthly_returns.std() * np.sqrt(12))
    sharpe = (
        (annual_return - risk_free_rate) / annual_volatility
        if annual_volatility > 0
        else float("nan")
    )
    cumulative = (1 + monthly_returns).cumprod() - 1
    max_drawdown = float(drawdown(cumulative).min())
    return {
        "annualized_return": annual_return,
        "annualized_volatility": annual_volatility,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "cumulative_return": float(cumulative.iloc[-1]),
    }


def rolling_backtest(
    monthly_returns: pd.DataFrame,
    risk_free_rate: float,
    estimation_window: int = 36,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run monthly rebalanced out-of-sample backtests."""
    strategies = ["equal_weight", "minimum_variance", "max_sharpe"]
    strategy_returns = []
    weights_history = []

    for position in range(estimation_window, len(monthly_returns)):
        estimation_sample = monthly_returns.iloc[position - estimation_window : position]
        next_period = monthly_returns.iloc[position]

        expected_returns = estimation_sample.apply(annualized_geometric_return)
        covariance = annualized_covariance(estimation_sample)
        weights_by_strategy = {
            "equal_weight": pd.Series(
                np.repeat(1 / monthly_returns.shape[1], monthly_returns.shape[1]),
                index=monthly_returns.columns,
            ),
            "minimum_variance": minimum_variance_portfolio(covariance),
            "max_sharpe": max_sharpe_weights(expected_returns, covariance, risk_free_rate),
        }

        period_result = {"date": str(monthly_returns.index[position])}
        for strategy, weights in weights_by_strategy.items():
            aligned_weights = weights.reindex(monthly_returns.columns).fillna(0)
            period_result[strategy] = float(np.dot(aligned_weights.values, next_period.values))
            weights_history.append(
                {
                    "date": str(monthly_returns.index[position]),
                    "strategy": strategy,
                    **{asset: float(weight) for asset, weight in aligned_weights.items()},
                }
            )
        strategy_returns.append(period_result)

    returns = pd.DataFrame(strategy_returns).set_index("date")
    metrics = pd.DataFrame(
        [
            {"strategy": strategy, **evaluate_return_series(returns[strategy], risk_free_rate)}
            for strategy in strategies
        ]
    )
    weights = pd.DataFrame(weights_history)
    return returns, metrics, weights
