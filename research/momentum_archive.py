"""Retired momentum / mean-reversion strategy, archived for research only.

This was the live paper strategy until 2026-11-02, when the harness was rebuilt
as a drift-band rebalancer. Kept so backtest_daily.py and
research/compare_strategies.py can still reproduce their results. Nothing in
the harness imports this module.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class StrategyParams:
    """Tunables for decide(). Kept separate from Config: these are
    strategy knobs (what counts as a signal), not harness/broker plumbing
    (how much cash, what account, what safety limits)."""
    # Momentum path (applied to tickers the regime filter calls "trending").
    # Per-ticker thresholds calibrated from real-data weekly vol (rv52_ann_pct)
    # scaled to daily cadence: threshold ≈ target_weekly_signal_freq_matched_daily_sigma.
    # buy_threshold_default / sell_threshold_default are used for any ticker not in the dict.
    buy_threshold_pct: dict = field(default_factory=lambda: {
        "TSM": 0.047,   # daily σ 2.39% → 4.7% ≈ 1.97σ ≈ 6 buy signals/yr
        "XLK": 0.033,   # daily σ 1.61% → 3.3% ≈ 2.05σ ≈ 5 buy signals/yr
        "CAT": 0.040,   # daily σ 2.05% → 4.0% ≈ 1.95σ ≈ 6 buy signals/yr
        "XLE": 0.040,   # PLACEHOLDER — XLE is MR-primary (research Sh 1.87); 4.0% suppresses
                        # noise momentum signals during MR cold-start (~50 ticks). Revisit once MR activates.
    })
    buy_threshold_default: float = 0.040    # fallback for tickers not in dict above
    sell_threshold_pct: dict = field(default_factory=lambda: {
        "TSM": 0.047,
        "XLK": 0.033,
        "CAT": 0.040,
        "XLE": 0.040,   # PLACEHOLDER — same rationale as buy_threshold_pct["XLE"] above
    })
    sell_threshold_default: float = 0.040   # fallback for tickers not in dict above
    buy_fraction_of_cash: float = 0.10       # size a buy as this fraction of cash
    sell_fraction_of_position: float = 0.50  # trim this fraction of the held position
    # Mean-reversion path (applied to tickers the regime filter calls "mean_reverting")
    mr_ma_window: int = 50                   # MA period for mean-reversion signal
    mr_entry_threshold_pct: float = 0.01     # enter when price is this far below MA
    mr_exit_threshold_pct: float = 0.0       # exit when price recovers to MA (0 = at MA)
    mr_buy_fraction_of_cash: float = 0.10    # size MR buy as this fraction of cash
    # Regime filter — decides, per ticker per tick, which path above applies.
    # Uses Kaufman's efficiency ratio: |net change| / sum(|tick-to-tick moves|)
    # over regime_lookback ticks. Near 1 = price moved directly (trending);
    # near 0 = price churned back and forth without going anywhere (choppy).
    regime_lookback: int = 20                # ticks of history the ratio is computed over
    regime_efficiency_threshold: float = 0.3  # >= this => trending; below => mean-reverting



# ---------------------------------------------------------------------------
# Decision logic — two trading paths (momentum, mean-reversion) gated by a
# per-ticker, per-tick regime filter. Keep this function pure: prices/
# positions/cash/history in, a decision out. No side effects, no order
# placement here — that stays in tick() so the approval gate has one single
# choke point to sit in front of.
# ---------------------------------------------------------------------------

def _trend_efficiency_ratio(history: list[float], lookback: int) -> Optional[float]:
    """Kaufman's efficiency ratio over the last `lookback` ticks: how much of
    the total tick-to-tick movement was "wasted" churning versus how much
    went toward the net move. 1.0 = price moved in a straight line (strong
    trend); near 0 = price moved a lot but round-tripped back to about where
    it started (chop). Returns None if there isn't enough history yet."""
    if len(history) < lookback + 1:
        return None
    segment = history[-(lookback + 1):]
    net_change = abs(segment[-1] - segment[0])
    total_movement = sum(abs(segment[i] - segment[i - 1]) for i in range(1, len(segment)))
    if total_movement <= 0:
        return None
    return net_change / total_movement


def _classify_regime(history: list[float], params: StrategyParams) -> str:
    """Per-ticker regime label: "trending", "mean_reverting", or
    "insufficient_data" (not enough history yet — treated as trending,
    i.e. momentum's default, same as before the regime filter existed)."""
    ratio = _trend_efficiency_ratio(history, params.regime_lookback)
    if ratio is None:
        return "insufficient_data"
    if ratio >= params.regime_efficiency_threshold:
        return "trending"
    return "mean_reverting"


# Labels written to the tick log: the path each ticker is routed down.
_REGIME_LOG_LABELS = {
    "trending": "momentum",
    "mean_reverting": "mean_reversion",
    "insufficient_data": "insufficient_data",
}


def _mr_signal(
    ticker: str,
    current_price: float,
    history: list[float],
    positions: dict[str, float],
    cash: float,
    params: StrategyParams,
) -> Optional[dict]:
    """Mean-reversion signal for a single ticker.

    Entry: price is more than mr_entry_threshold_pct below its MA → buy.
    Exit: price has recovered to within mr_exit_threshold_pct of MA and
    we hold a position → sell everything.

    Returns a trade proposal dict or None.
    """
    if len(history) < params.mr_ma_window:
        return None  # not enough history yet to compute MA

    ma = sum(history[-params.mr_ma_window:]) / params.mr_ma_window
    if ma <= 0:
        return None

    deviation = (ma - current_price) / ma  # positive when price < MA

    # A full-exit sell's dollar_amount is rounded to cents, then place_order
    # re-derives shares from that rounded amount — the two roundings rarely
    # cancel out exactly, so a dust-sized share residue is normal after an
    # exit fill, not a sign shares are still meaningfully "held". Treat
    # anything worth a cent or less as closed, or the exit signal would
    # refire every tick forever on a position that's already gone.
    held_shares = positions.get(ticker, 0.0)
    held_value = held_shares * current_price

    # Exit: price back at or above MA while we hold a (non-dust) position
    if held_value > 0.01 and deviation <= params.mr_exit_threshold_pct:
        return {
            "ticker": ticker,
            "side": "sell",
            "dollar_amount": round(held_value, 2),
            "reason": (f"{ticker} MR exit: price {current_price:.2f} recovered "
                       f"to MA {ma:.2f} ({deviation:.2%} deviation)"),
        }

    # Entry: price sufficiently below MA and no (non-dust) position open
    if held_value <= 0.01 and deviation >= params.mr_entry_threshold_pct:
        dollar_amount = round(cash * params.mr_buy_fraction_of_cash, 2)
        if dollar_amount <= 0:
            return None
        return {
            "ticker": ticker,
            "side": "buy",
            "dollar_amount": dollar_amount,
            "reason": (f"{ticker} MR entry: price {current_price:.2f} is "
                       f"{deviation:.2%} below MA-{params.mr_ma_window} {ma:.2f}"),
        }

    return None


def decide(
    prices: dict[str, float],
    positions: dict[str, float],
    cash: float,
    price_history: dict[str, list[float]],
    params: StrategyParams,
) -> Optional[dict]:
    """Two-path strategy, routed per ticker per tick by a regime filter
    instead of a fixed ticker list — a ticker trades momentum while it's
    trending and mean-reversion while it's chopping, and can switch paths
    as its own behavior changes (see _classify_regime).

    Mean-reversion path (regime == "mean_reverting"):
      Enter when price falls mr_entry_threshold_pct below its MA-{mr_ma_window}.
      Exit when price recovers to MA. Requires mr_ma_window ticks of history.
      Exits are evaluated for any ticker currently holding a position,
      regardless of its regime label this tick, so a position doesn't get
      stranded open just because the ticker relabeled as trending.

    Momentum path (regime == "trending" or "insufficient_data"):
      Compares current price to the previous tick's price. Buy on an up-move
      >= buy_threshold_pct; trim on a down-move <= -sell_threshold_pct.

    At most one trade per call. MR exit signals take priority (they are risk
    management as much as alpha); then MR entries; then the largest-move
    momentum candidate. Returns None when no signal fires.

    Must return either None or a dict shaped like:
      {"ticker": str, "side": "buy" | "sell", "dollar_amount": float, "reason": str}
    """
    regimes = {t: _classify_regime(price_history.get(t, []), params) for t in prices}

    # --- Mean-reversion path ---
    mr_exits = []
    mr_entries = []
    for ticker, current_price in prices.items():
        held_shares = positions.get(ticker, 0.0)
        is_mean_reverting = regimes[ticker] == "mean_reverting"
        if not is_mean_reverting and held_shares <= 0:
            continue  # not chopping, and nothing open to exit — skip MR entirely
        history = price_history.get(ticker, [])
        signal = _mr_signal(ticker, current_price, history, positions, cash, params)
        if signal is None:
            continue
        if signal["side"] == "sell":
            mr_exits.append(signal)
        elif is_mean_reverting:
            mr_entries.append(signal)

    if mr_exits:
        return mr_exits[0]  # first exit trumps everything else
    if mr_entries:
        return mr_entries[0]

    # --- Momentum path (tickers not currently classified mean-reverting) ---
    candidates = []
    for ticker, current_price in prices.items():
        if regimes[ticker] == "mean_reverting":
            continue
        history = price_history.get(ticker, [])
        if not history:
            continue  # no prior tick recorded yet for this ticker
        prev_price = history[-1]
        if prev_price <= 0:
            continue
        pct_change = (current_price - prev_price) / prev_price
        candidates.append((ticker, pct_change))

    if not candidates:
        return None

    candidates.sort(key=lambda c: abs(c[1]), reverse=True)
    ticker, pct_change = candidates[0]

    buy_thresh = params.buy_threshold_pct.get(ticker, params.buy_threshold_default)
    sell_thresh = params.sell_threshold_pct.get(ticker, params.sell_threshold_default)

    if pct_change >= buy_thresh:
        dollar_amount = round(cash * params.buy_fraction_of_cash, 2)
        if dollar_amount <= 0:
            return None
        return {
            "ticker": ticker,
            "side": "buy",
            "dollar_amount": dollar_amount,
            "reason": (f"{ticker} up {pct_change:.2%} since last tick, "
                       f"above +{buy_thresh:.1%} threshold"),
        }

    if pct_change <= -sell_thresh:
        held_shares = positions.get(ticker, 0.0)
        if held_shares <= 0:
            return None  # nothing held to trim
        position_value = held_shares * prices[ticker]
        dollar_amount = round(position_value * params.sell_fraction_of_position, 2)
        if dollar_amount <= 0:
            return None
        return {
            "ticker": ticker,
            "side": "sell",
            "dollar_amount": dollar_amount,
            "reason": (f"{ticker} down {pct_change:.2%} since last tick, "
                       f"past -{sell_thresh:.1%} threshold — trimming"),
        }

    return None
