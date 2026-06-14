from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import rolling_backtest
from .config import DATA_PATH, FIGURE_DIR, OUTPUT_DIR, RISK_FREE_RATE, TABLE_DIR
from .data import load_base_asset_returns, load_nine_asset_returns
from .metrics import performance_summary
from .optimizer import (
    capital_allocation_for_target_return,
    efficient_frontier,
    tangency_portfolio,
)
from .plotting import (
    plot_backtest_growth,
    plot_efficient_frontier,
    plot_frontier_comparison,
    plot_tangency_portfolio,
)


def _write_static_analysis(
    name: str,
    monthly_returns: pd.DataFrame,
    risk_free_rate: float,
    table_dir: Path,
    figure_dir: Path,
) -> dict[str, object]:
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    equal_weights = np.repeat(1 / monthly_returns.shape[1], monthly_returns.shape[1])
    summary, covariance = performance_summary(monthly_returns, equal_weights, risk_free_rate)
    expected_returns = summary.drop(index="Current Portfolio")["annualized_return"]
    current = summary.loc["Current Portfolio"]

    frontier = efficient_frontier(expected_returns, covariance, points=80, allow_short=True)
    tangency = tangency_portfolio(expected_returns, covariance, risk_free_rate, allow_short=False)

    same_return_allocation = capital_allocation_for_target_return(
        target_return=float(current["annualized_return"]),
        tangency_return=float(tangency["return"]),
        tangency_volatility=float(tangency["volatility"]),
        risk_free_rate=risk_free_rate,
    )
    floor_8_allocation = capital_allocation_for_target_return(
        target_return=0.08,
        tangency_return=float(tangency["return"]),
        tangency_volatility=float(tangency["volatility"]),
        risk_free_rate=risk_free_rate,
    )

    tangency_weights = tangency["weights"].rename("weight").to_frame()
    tangency_weights.loc["Tangency Portfolio"] = {
        "weight": float(tangency_weights["weight"].sum()),
    }
    tangency_metrics = pd.DataFrame(
        [
            {
                "portfolio": "Tangency Portfolio",
                "annualized_return": tangency["return"],
                "annualized_volatility": tangency["volatility"],
                "sharpe_ratio": tangency["sharpe_ratio"],
            },
            {"portfolio": "Risk-Free + Tangency: Current Return", **same_return_allocation},
            {"portfolio": "Risk-Free + Tangency: 8% Floor", **floor_8_allocation},
        ]
    )

    monthly_returns.to_csv(table_dir / "monthly_returns.csv")
    summary.to_csv(table_dir / "performance_summary.csv")
    covariance.to_csv(table_dir / "annualized_covariance.csv")
    frontier.to_csv(table_dir / "efficient_frontier.csv", index=False)
    tangency_weights.to_csv(table_dir / "tangency_weights.csv")
    tangency_metrics.to_csv(table_dir / "tangency_metrics.csv", index=False)

    plot_efficient_frontier(
        frontier,
        expected_returns,
        covariance,
        current_return=float(current["annualized_return"]),
        current_volatility=float(current["annualized_volatility"]),
        output_path=figure_dir / "efficient_frontier.png",
        title=f"Efficient Frontier: {name.replace('_', ' ').title()} Universe",
    )
    plot_tangency_portfolio(
        frontier,
        tangency,
        risk_free_rate,
        output_path=figure_dir / "tangency_portfolio.png",
    )

    return {
        "monthly_returns": monthly_returns,
        "summary": summary,
        "covariance": covariance,
        "frontier": frontier,
        "tangency": tangency,
        "tangency_weights": tangency_weights,
        "tangency_metrics": tangency_metrics,
    }


def run(data_path=DATA_PATH, risk_free_rate: float = RISK_FREE_RATE) -> dict[str, object]:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    five_asset_returns = load_base_asset_returns(data_path)
    nine_asset_returns = load_nine_asset_returns(data_path)

    five_asset = _write_static_analysis(
        "five_asset",
        five_asset_returns,
        risk_free_rate,
        TABLE_DIR / "five_asset",
        FIGURE_DIR / "five_asset",
    )
    nine_asset = _write_static_analysis(
        "nine_asset",
        nine_asset_returns,
        risk_free_rate,
        TABLE_DIR / "nine_asset",
        FIGURE_DIR / "nine_asset",
    )

    plot_frontier_comparison(
        five_asset["frontier"],
        nine_asset["frontier"],
        FIGURE_DIR / "frontier_comparison.png",
    )

    backtest_dir = TABLE_DIR / "backtests"
    backtest_dir.mkdir(parents=True, exist_ok=True)
    five_bt_returns, five_bt_metrics, five_bt_weights = rolling_backtest(
        five_asset_returns,
        risk_free_rate,
        estimation_window=36,
    )
    nine_bt_returns, nine_bt_metrics, nine_bt_weights = rolling_backtest(
        nine_asset_returns,
        risk_free_rate,
        estimation_window=36,
    )
    five_bt_returns.to_csv(backtest_dir / "five_asset_returns.csv")
    five_bt_metrics.to_csv(backtest_dir / "five_asset_metrics.csv", index=False)
    five_bt_weights.to_csv(backtest_dir / "five_asset_weights.csv", index=False)
    nine_bt_returns.to_csv(backtest_dir / "nine_asset_returns.csv")
    nine_bt_metrics.to_csv(backtest_dir / "nine_asset_metrics.csv", index=False)
    nine_bt_weights.to_csv(backtest_dir / "nine_asset_weights.csv", index=False)

    plot_backtest_growth(
        nine_bt_returns,
        FIGURE_DIR / "nine_asset_rolling_backtest.png",
    )

    with pd.ExcelWriter(OUTPUT_DIR / "portfolio_analysis.xlsx", engine="openpyxl") as writer:
        five_asset["summary"].to_excel(writer, sheet_name="five_summary")
        five_asset["covariance"].to_excel(writer, sheet_name="five_covariance")
        five_asset["frontier"].to_excel(writer, sheet_name="five_frontier", index=False)
        five_asset["tangency_weights"].to_excel(writer, sheet_name="five_tangency_weights")
        nine_asset["summary"].to_excel(writer, sheet_name="nine_summary")
        nine_asset["covariance"].to_excel(writer, sheet_name="nine_covariance")
        nine_asset["frontier"].to_excel(writer, sheet_name="nine_frontier", index=False)
        nine_asset["tangency_weights"].to_excel(writer, sheet_name="nine_tangency_weights")
        five_bt_metrics.to_excel(writer, sheet_name="five_bt_metrics", index=False)
        nine_bt_metrics.to_excel(writer, sheet_name="nine_bt_metrics", index=False)

    return {
        "five_asset": five_asset,
        "nine_asset": nine_asset,
        "five_asset_backtest": {
            "returns": five_bt_returns,
            "metrics": five_bt_metrics,
            "weights": five_bt_weights,
        },
        "nine_asset_backtest": {
            "returns": nine_bt_returns,
            "metrics": nine_bt_metrics,
            "weights": nine_bt_weights,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the portfolio optimization project.")
    parser.add_argument("--data", default=str(DATA_PATH), help="Path to the Excel workbook.")
    parser.add_argument("--risk-free-rate", type=float, default=RISK_FREE_RATE)
    args = parser.parse_args()

    results = run(data_path=args.data, risk_free_rate=args.risk_free_rate)
    five_summary = results["five_asset"]["summary"]
    nine_summary = results["nine_asset"]["summary"]
    nine_bt_metrics = results["nine_asset_backtest"]["metrics"]

    print("Analysis complete.")
    print(f"Five-asset current return: {five_summary.loc['Current Portfolio', 'annualized_return']:.2%}")
    print(f"Five-asset current volatility: {five_summary.loc['Current Portfolio', 'annualized_volatility']:.2%}")
    print(f"Nine-asset equal-weight return: {nine_summary.loc['Current Portfolio', 'annualized_return']:.2%}")
    print(f"Nine-asset equal-weight volatility: {nine_summary.loc['Current Portfolio', 'annualized_volatility']:.2%}")
    print("Nine-asset rolling backtest:")
    for _, row in nine_bt_metrics.iterrows():
        print(
            f"  {row['strategy']}: return {row['annualized_return']:.2%}, "
            f"volatility {row['annualized_volatility']:.2%}, "
            f"Sharpe {row['sharpe_ratio']:.2f}"
        )
    print(f"Outputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
