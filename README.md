# Multi-Asset Portfolio Optimization

This project studies whether a traditional Chinese stock-bond allocation can be improved by adding international equities, Hong Kong equities, REITs, and gold.

It turns a course investment report into a reproducible Python pipeline for portfolio evaluation, mean-variance optimization, efficient frontier comparison, tangency portfolio construction, and rolling-window out-of-sample backtesting.

## Research Question

Can a five-asset Chinese stock-bond portfolio achieve better risk-adjusted performance after expanding into a nine-asset global multi-asset universe?

The project compares:

- **Five-asset universe:** CSI 300, CSI 500, CSI 1000, CSI Aggregate Bond, SSE Government Bond
- **Nine-asset universe:** the five original assets plus Hang Seng REIT, Gold ETF, Hang Seng Index, and S&P 500

## Methods

- Annualized return, volatility, covariance, and Sharpe ratio
- Mean-variance efficient frontier
- Long-only tangency portfolio
- Capital allocation between the risk-free asset and tangency portfolio
- Five-asset vs nine-asset efficient frontier comparison
- Rolling 36-month out-of-sample backtest with monthly rebalancing

The backtest compares three strategies:

- Equal weight
- Long-only minimum variance
- Long-only maximum Sharpe ratio

## Project Structure

```text
.
├── data/raw/market_data.xlsx
├── src/portfolio_optimization/
│   ├── config.py
│   ├── data.py
│   ├── metrics.py
│   ├── optimizer.py
│   ├── backtest.py
│   ├── plotting.py
│   └── pipeline.py
├── outputs/
│   ├── figures/
│   ├── tables/
│   └── portfolio_analysis.xlsx
├── requirements.txt
└── run_analysis.py
```

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the full analysis:

```bash
python3 run_analysis.py
```

Generated results are written to `outputs/`.

## Current Outputs

- `outputs/portfolio_analysis.xlsx`
- `outputs/tables/five_asset/performance_summary.csv`
- `outputs/tables/nine_asset/performance_summary.csv`
- `outputs/tables/five_asset/efficient_frontier.csv`
- `outputs/tables/nine_asset/efficient_frontier.csv`
- `outputs/tables/backtests/five_asset_metrics.csv`
- `outputs/tables/backtests/nine_asset_metrics.csv`
- `outputs/figures/frontier_comparison.png`
- `outputs/figures/five_asset/efficient_frontier.png`
- `outputs/figures/nine_asset/efficient_frontier.png`
- `outputs/figures/nine_asset_rolling_backtest.png`

## Example Results

With the current dataset, the five-asset equal-weight risky portfolio has an annualized return of about 3.11% and annualized volatility of about 14.43%.

After expanding to nine assets, the equal-weight portfolio has an annualized return of about 3.47% and annualized volatility of about 11.62%.

In the nine-asset rolling backtest, the long-only minimum variance and maximum Sharpe strategies both substantially reduce realized volatility relative to equal weight. This result should be interpreted carefully because the optimized portfolios are bond-heavy in this sample, but it provides a useful starting point for more realistic constraints.

## Next Upgrades

- Add long-only and max-weight constrained portfolio comparisons.
- Add transaction costs, turnover, max drawdown, and VaR/CVaR.
- Add shrinkage covariance estimation and risk parity.
- Build a Streamlit dashboard for interactive portfolio optimization.
