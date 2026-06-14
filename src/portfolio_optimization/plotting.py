from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_efficient_frontier(
    frontier: pd.DataFrame,
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    current_return: float,
    current_volatility: float,
    output_path: Path,
    title: str = "Efficient Frontier",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    asset_volatility = np.sqrt(np.diag(covariance.values))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(frontier["volatility"], frontier["target_return"], color="#2563eb", linewidth=2, label="Efficient frontier")
    ax.scatter(asset_volatility, expected_returns.values, color="#dc2626", label="Individual assets")
    ax.scatter(current_volatility, current_return, color="#16a34a", marker="*", s=220, label="Current portfolio")

    for asset, vol, ret in zip(expected_returns.index, asset_volatility, expected_returns.values):
        ax.annotate(asset, (vol, ret), xytext=(5, 5), textcoords="offset points", fontsize=9)

    ax.set_title(title)
    ax.set_xlabel("Annualized volatility")
    ax.set_ylabel("Annualized return")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_frontier_comparison(
    five_asset_frontier: pd.DataFrame,
    nine_asset_frontier: pd.DataFrame,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        five_asset_frontier["volatility"],
        five_asset_frontier["target_return"],
        color="#2563eb",
        linewidth=2,
        label="Five assets",
    )
    ax.plot(
        nine_asset_frontier["volatility"],
        nine_asset_frontier["target_return"],
        color="#16a34a",
        linewidth=2,
        label="Nine assets",
    )
    ax.set_title("Efficient Frontier Comparison")
    ax.set_xlabel("Annualized volatility")
    ax.set_ylabel("Annualized return")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_backtest_growth(backtest_returns: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    growth = (1 + backtest_returns).cumprod()

    fig, ax = plt.subplots(figsize=(10, 6))
    for column in growth.columns:
        ax.plot(growth.index, growth[column], linewidth=2, label=column.replace("_", " ").title())

    tick_step = max(len(growth.index) // 8, 1)
    ax.set_xticks(range(0, len(growth.index), tick_step))
    ax.set_xticklabels(growth.index[::tick_step], rotation=35, ha="right")
    ax.set_title("Rolling Backtest Growth of 1")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio value")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_tangency_portfolio(
    frontier: pd.DataFrame,
    tangency: dict[str, object],
    risk_free_rate: float,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tangency_return = float(tangency["return"])
    tangency_volatility = float(tangency["volatility"])
    sharpe_ratio = float(tangency["sharpe_ratio"])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(frontier["volatility"], frontier["target_return"], color="#dc2626", linestyle="--", linewidth=2, label="Efficient frontier")
    ax.scatter(tangency_volatility, tangency_return, s=230, color="#facc15", edgecolors="#111827", marker="*", label="Tangency portfolio")

    x_line = np.linspace(0, max(frontier["volatility"].max(), tangency_volatility) * 1.25, 100)
    y_line = risk_free_rate + sharpe_ratio * x_line
    ax.plot(x_line, y_line, color="#2563eb", linewidth=2, label="Capital market line")

    ax.annotate(
        f"Return: {tangency_return:.2%}\nVolatility: {tangency_volatility:.2%}\nSharpe: {sharpe_ratio:.2f}",
        (tangency_volatility, tangency_return),
        xytext=(-115, 22),
        textcoords="offset points",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#d1d5db"},
        fontsize=9,
    )

    ax.set_title("Tangency Portfolio and Capital Market Line")
    ax.set_xlabel("Annualized volatility")
    ax.set_ylabel("Annualized return")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
