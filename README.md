# trading-harness

Paper-trading harness for a buy-and-hold portfolio with drift-band rebalancing.

**Mode: paper only.** Prices are real (the Claude Code Routine fetches them from
Robinhood and writes `robinhood_input.json`); every trade is simulated by
`MockBroker` in `paper_trading_harness.py`. No real orders are placed.

## Portfolio B

| Holding | Target |
|---|---|
| VTI (US total market) | 60.0% |
| XLU (utilities / power demand) | 17.8% |
| XLI (industrials) | 14.3% |
| SMH (semiconductors) | 7.9% |

VTI is fixed at 60%. The other 40% is split by equal risk contribution
(`skfolio`), estimated on daily returns from Nov 2021 to Oct 2026. How the
holdings and weights were chosen, and how they tested out of sample, is in
`research/portfolio_candidates.py` and PR #10.

## What one run does

1. Reads prices from `robinhood_input.json`. A missing or bad price halts the
   run with no trades.
2. Skips weekends and any month that was already checked.
3. If any holding is more than 5 percentage points from its target, trades
   everything back to target; otherwise does nothing.
4. Appends to `harness_log.jsonl` and saves cash, FIFO tax lots and the last
   check date to `robinhood_state.json`.

Run the tests with `python -m unittest test_paper_trading_harness`.

## TODO before any real broker is connected

Approval is **off** in paper mode. Before connecting a real broker, set
`Config.require_human_approval = True` and build a real approval step:
proposed trades are shown to a human, and nothing is placed without an
explicit yes. Until then the harness refuses any `BROKER` other than `mock`.

## History

The momentum / mean-reversion strategy that ran until 2026-11-02 was retired
after it returned about 3.5% a year against about 28% for buy-and-hold in a
daily backtest. Its code is archived in `research/momentum_archive.py` for
the backtests that still use it.
