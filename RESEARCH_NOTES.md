# Research Notes — Trading Harness

## Data source

Real weekly OHLCV prices from two public GitHub repositories:
- `dotchev/top-stox`: 29 tickers, weekly bars, 2021–2026
- `whchien/ai-trader`: TSM daily, resampled to weekly

Common backtest window: **2022-01-17 – 2025-12-15 (205 weeks)**. Script: `research_script.py`. Output: `research_output/`.

Yahoo Finance direct access is blocked in this environment. The GitHub CSV workaround produced real historical prices, not synthetic data. PR #4 (synthetic data) was discarded and replaced by PR #5 (real data).

## Momentum ticker selection

Best Sharpe ratios within the 2–3% weekly threshold band:

| Ticker | Threshold | Sharpe | Return | Max DD | trades/wk |
|--------|-----------|--------|--------|--------|-----------|
| TSM    | 2.5%      | 0.81   | +124%  | 45.4%  | 0.239     |
| XLK    | 2.0%      | 0.77   | +64%   | 22.5%  | 0.249     |
| CAT    | 2.0%      | 0.73   | +102%  | 29.3%  | 0.244     |

XLK and CAT optimal threshold from backtest is 2.0%, not 2.5%. The deployed value of 2.5% was a simplification to use one number; the per-ticker threshold change (see below) supersedes this.

## Mean-reversion: XLE

XLE under mean-reversion (MA-50, 1% deviation) had the best Sharpe of anything tested: **1.87**, +89% return, 4.6% max drawdown, 0.063 trades/wk. XLE is in `Config.tickers` and will route to the MR path once the regime classifier has sufficient history (requires 20 ticks; MA-50 signal requires 50 ticks).

## Per-ticker thresholds (current production values)

The harness runs at a **daily-at-open cadence** (Routine fires at 9:30 AM ET weekdays). The 2.5% threshold was calibrated on weekly bars. On daily bars, 2.5% fires far too often (~37 buy signals/year for TSM vs ~6/year intended). Thresholds were recalibrated to match the weekly backtest's signal frequency on a daily cadence.

Method: annualized vol from research → daily sigma (ann_vol / √252) → threshold solving P(X ≥ t) = target_buys/252, where target = weekly trades_pw × 52 / 2.

| Ticker | Ann. vol | Daily σ | Target buys/yr | Daily threshold |
|--------|----------|---------|----------------|-----------------|
| TSM    | 37.9%    | 2.39%   | ~6.2/yr        | **4.7%**        |
| XLK    | 25.5%    | 1.61%   | ~5.3/yr        | **3.3%**        |
| CAT    | 32.5%    | 2.05%   | ~6.1/yr        | **4.0%**        |
| XLE    | 23.8%    | 1.50%   | MR primary     | **4.0%** ⚠ placeholder — see note below |

Assumptions: normal returns, 50/50 buy/sell split in weekly trades_pw, open-to-open vol ≈ close-to-close vol. Daily returns are fat-tailed, so actual extreme-move frequency is somewhat higher than the normal approximation; thresholds may be slightly low in practice.

**XLE note:** The research validated XLE exclusively as a mean-reversion ticker (Sharpe 1.87 on MR; XLE was not in the momentum candidate set). The 4.0% momentum threshold for XLE is a **placeholder** — it borrows CAT's value so the harness doesn't fire noise signals on XLE during the cold-start period before MR activates. Once the MR path is live (~50 ticks, late January 2027 at daily cadence), XLE's momentum threshold entry should be revisited: either remove it from the momentum path entirely (by routing XLE to MR-only) or set it high enough that momentum signals are effectively suppressed.

## Open items

1. **Daily-granularity backtest unvalidated.** The research covered weekly bars only. The per-ticker daily thresholds above are derived analytically, not from a daily backtest. A daily backtest against the same GitHub CSV data (resampled to daily) would validate or revise these numbers. Requires open egress to Yahoo Finance for finer-grained data, or a daily-resolution CSV source.

2. **XLE mean-reversion path cold-start.** `robinhood_state.json` currently has 1 price history entry per ticker. The regime classifier activates at 20 ticks; the MA-50 MR signal activates at 50 ticks. At once-daily cadence, XLE will run on the momentum path (4.0% threshold) until approximately late November 2026 (regime) and late January 2027 (MR signal).

3. **Cadence choice undocumented.** The daily-at-open schedule (`30 13 * * 1-5` UTC) was set when the live Routine was created and not explicitly rationalized. It compares today's open to yesterday's open — an overnight-gap + continuation signal. This is a legitimate momentum signal, but it has not been backtested.

4. **XLE MR threshold not independently calibrated.** The MR `mr_entry_threshold_pct` (1% deviation from MA-50) came from the weekly backtest (best Sharpe config). It has the same daily-vs-weekly granularity caveat as the momentum thresholds.
