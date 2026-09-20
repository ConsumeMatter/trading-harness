from __future__ import annotations

"""
Paper-Trading Harness — v1 Strategy + Persisted State
========================================================

Purpose: a scheduler-invoked "tick" function that checks (mock) market data,
runs decision logic, and executes (mock) trades — with logging and a circuit
breaker built in from day one, not bolted on later.

This is intentionally NOT a long-running process. It's meant to be invoked
once per "tick" by an external scheduler (cron, a Claude Code scheduled task,
etc.), so the scheduling layer and the trading logic stay separate concerns.

IMPORTANT — state persistence:
  Every invocation of this script runs in a fresh process (and, under the
  Claude Code Routine, a fresh container). Nothing in memory survives
  between ticks. `portfolio_state.json` is how cash, positions, recent
  price history, and the circuit breaker's day-start value survive across
  runs — it's read at startup and written back at the end, and should be
  committed to the repo the same way `harness_log.jsonl` already is.
  Without it, every run would restart from $1000 cash with no memory of
  prior trades or of what the account was worth at the start of the day.

Build order for Claude Code:
  1. Get this running end-to-end against MockBroker (no real money, no
     network calls) until the circuit breaker, logging, and approval gate
     all behave the way you expect. [DONE — v1 decide() below is real
     logic now, but still runs only against MockBroker.]
  2. Only then write a RobinhoodBroker that implements the same four
     methods as MockBroker, backed by real Robinhood Trading MCP calls.
  3. Swap MockBroker -> RobinhoodBroker in one line in __main__. Nothing
     above that line should need to change if the interface is respected.

Nothing in this file talks to a network. It's safe to run as-is.
"""

import json
import logging
import os
import random
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Configuration — the knobs a human sets, not the agent
# ---------------------------------------------------------------------------

@dataclass
class Config:
    tickers: list[str] = field(default_factory=lambda: ["TSM", "XLK", "CAT", "XLE"])
    starting_cash: float = 1000.00
    max_position_pct: float = 0.25       # no single position > 25% of account
    daily_loss_limit_pct: float = 0.05   # halt trading for the day at -5%
    require_human_approval: bool = True  # mirrors Robinhood's approval-required mode
    log_path: Path = Path("harness_log.jsonl")
    state_path: Path = Path("portfolio_state.json")
    price_history_len: int = 55          # must cover mr_ma_window (50) + a few extra ticks


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
# Broker interface — MockBroker today, RobinhoodBroker later.
# Anything that touches real money must implement this exact interface:
# get_prices(), get_account_value(), place_order(), and a .positions dict.
# ---------------------------------------------------------------------------

class MockBroker:
    """Simulated broker. No network calls. A simple random walk so the
    harness has something to react to during testing."""

    def __init__(self, tickers: list[str], starting_cash: float, seed: int | None = None):
        self._rng = random.Random(seed)
        self._prices = {t: 100.0 for t in tickers}  # arbitrary start price
        self.cash = starting_cash
        self.positions: dict[str, float] = {t: 0.0 for t in tickers}  # shares held

    def load_state(self, cash: float, positions: dict[str, float], prices: dict[str, float]) -> None:
        """Restore broker state from a previous run's persisted snapshot.
        Call this right after construction, before any ticks run."""
        self.cash = cash
        self.positions = dict(positions)
        self._prices = dict(prices)

    def current_prices(self) -> dict[str, float]:
        """Read the last-generated prices without advancing the random
        walk (unlike get_prices(), which mutates state on every call)."""
        return dict(self._prices)

    def get_prices(self) -> dict[str, float]:
        """Mock a market tick: each price randomly drifts a bit."""
        for t in self._prices:
            pct_move = self._rng.uniform(-0.03, 0.03)  # +/- 3% per tick
            self._prices[t] = round(self._prices[t] * (1 + pct_move), 2)
        return dict(self._prices)

    def get_account_value(self) -> float:
        value = self.cash
        for t, shares in self.positions.items():
            value += shares * self._prices[t]
        return round(value, 2)

    def place_order(self, ticker: str, side: str, dollar_amount: float) -> dict:
        price = self._prices[ticker]
        shares = round(dollar_amount / price, 4)
        if side == "buy":
            cost = shares * price
            if cost > self.cash:
                return {"status": "rejected", "reason": "insufficient cash"}
            self.cash -= cost
            self.positions[ticker] += shares
        elif side == "sell":
            if shares > self.positions.get(ticker, 0):
                return {"status": "rejected", "reason": "insufficient shares"}
            self.cash += shares * price
            self.positions[ticker] -= shares
        else:
            return {"status": "rejected", "reason": f"unknown side '{side}'"}
        return {"status": "filled", "ticker": ticker, "side": side,
                "shares": shares, "price": price}


class RobinhoodBroker:
    """Live broker backed by the Robinhood Trading MCP.

    Python cannot call MCP tools directly — those calls happen in the Claude
    Routine agent. This class bridges the two via JSON sidecar files:

      robinhood_input.json         ← Routine writes BEFORE running the script.
                                     Contains: account_number, prices (last_trade_price
                                     per ticker), buying_power, equity, positions
                                     (list of {symbol, quantity} for non-zero holdings).

      robinhood_pending_order.json ← This class writes when place_order() fires.
                                     Contains: ticker, side, dollar_amount.
                                     Routine reads it AFTER the script exits, calls
                                     review_equity_order then place_equity_order.

    Routine orchestration (see the Routine prompt in the repo root):
      1. MCP: get_equity_quotes + get_portfolio + get_equity_positions
      2. Write robinhood_input.json
      3. Run: BROKER=robinhood python paper_trading_harness.py
      4. If robinhood_pending_order.json exists: review + place via MCP
      5. Commit and push robinhood_state.json + harness_log.jsonl
    """

    INPUT_PATH = Path("robinhood_input.json")
    PENDING_ORDER_PATH = Path("robinhood_pending_order.json")

    def __init__(self, tickers: list[str]):
        inp = json.loads(self.INPUT_PATH.read_text())
        self.account_number: str = inp["account_number"]
        self._prices: dict[str, float] = {k: float(v) for k, v in inp["prices"].items()}
        self.cash: float = float(inp["buying_power"])
        self._equity: float = float(inp["equity"])
        self.positions: dict[str, float] = {t: 0.0 for t in tickers}
        for pos in inp.get("positions", []):
            sym = pos["symbol"]
            if sym in self.positions:
                self.positions[sym] = float(pos["quantity"])

    def load_state(self, cash: float, positions: dict[str, float],
                   prices: dict[str, float]) -> None:
        # No-op: cash/positions/prices come from robinhood_input.json (live Robinhood
        # state), not from the persisted JSON file. price_history and breaker state
        # are still loaded from the state file by the caller before this is invoked.
        pass

    def current_prices(self) -> dict[str, float]:
        return dict(self._prices)

    def get_prices(self) -> dict[str, float]:
        # Prices were already fetched by the Routine; no random walk, no mutation.
        return dict(self._prices)

    def get_account_value(self) -> float:
        return round(self._equity, 2)

    def place_order(self, ticker: str, side: str, dollar_amount: float) -> dict:
        order = {
            "ticker": ticker,
            "side": side,
            "dollar_amount": round(dollar_amount, 2),
        }
        self.PENDING_ORDER_PATH.write_text(json.dumps(order, indent=2))
        return {"status": "pending_mcp_execution", "ticker": ticker,
                "side": side, "dollar_amount": dollar_amount}


# ---------------------------------------------------------------------------
# Circuit breaker — a hard rule the code enforces, not a suggestion left
# to the strategy function to remember.
# ---------------------------------------------------------------------------

class CircuitBreaker:
    """Halts new trades once the account is down more than
    daily_loss_limit_pct from its value at the start of the trading day.
    Resets automatically on a new calendar day.

    NOTE: _day and _day_start_value must be restored via load_state() from
    persisted state at startup, or "start of day" will silently reset every
    time the script runs instead of holding for the full calendar day."""

    def __init__(self, daily_loss_limit_pct: float):
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self._day_start_value: Optional[float] = None
        self._day: Optional[date] = None

    def load_state(self, day_iso: Optional[str], day_start_value: Optional[float]) -> None:
        self._day = date.fromisoformat(day_iso) if day_iso else None
        self._day_start_value = day_start_value

    def to_state(self) -> dict:
        return {
            "day": self._day.isoformat() if self._day else None,
            "day_start_value": self._day_start_value,
        }

    def check(self, current_value: float) -> tuple[bool, str]:
        today = date.today()
        if self._day != today:
            self._day = today
            self._day_start_value = current_value
            return True, "new trading day, breaker reset"

        drawdown = (self._day_start_value - current_value) / self._day_start_value
        if drawdown >= self.daily_loss_limit_pct:
            return False, (f"HALTED: down {drawdown:.1%} today, "
                            f"limit is {self.daily_loss_limit_pct:.1%}")
        return True, f"ok, down {drawdown:.1%} today"


# ---------------------------------------------------------------------------
# Persisted portfolio state — survives across scheduled runs, since each
# run is a fresh process (and, under the Routine, a fresh container).
# ---------------------------------------------------------------------------

def load_portfolio_state(config: Config) -> dict:
    """Read the persisted snapshot, or build a fresh one on the very first
    run (or if the state file has never been committed to the repo)."""
    if config.state_path.exists():
        return json.loads(config.state_path.read_text())
    return {
        "cash": config.starting_cash,
        "positions": {t: 0.0 for t in config.tickers},
        "prices": {t: 100.0 for t in config.tickers},
        "price_history": {t: [] for t in config.tickers},
        "breaker_day": None,
        "breaker_day_start_value": None,
    }


def save_portfolio_state(config: Config, broker, breaker: CircuitBreaker,
                          price_history: dict[str, list[float]]) -> None:
    breaker_state = breaker.to_state()
    state = {
        "cash": broker.cash,
        "positions": broker.positions,
        "prices": broker.current_prices(),
        "price_history": price_history,
        "breaker_day": breaker_state["day"],
        "breaker_day_start_value": breaker_state["day_start_value"],
    }
    config.state_path.write_text(json.dumps(state, indent=2))


# ---------------------------------------------------------------------------
# Logging — every tick gets a line, whether or not it led to a trade.
# JSON Lines format so it's easy to grep, load into pandas, or feed back
# into an analysis step later.
# ---------------------------------------------------------------------------

def log_event(config: Config, event: dict) -> None:
    event = {"timestamp": datetime.now().isoformat(), **event}
    with open(config.log_path, "a") as f:
        f.write(json.dumps(event) + "\n")
    logging.info(json.dumps(event))


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


# ---------------------------------------------------------------------------
# The tick — the one function an external scheduler calls. Everything
# above is a building block; this is where they connect.
# ---------------------------------------------------------------------------

def tick(config: Config, broker: MockBroker, breaker: CircuitBreaker,
         price_history: dict[str, list[float]], params: StrategyParams) -> None:
    account_value = broker.get_account_value()
    ok, breaker_msg = breaker.check(account_value)
    log_event(config, {"event": "breaker_check", "ok": ok, "detail": breaker_msg,
                        "account_value": account_value})

    if not ok:
        return  # circuit breaker tripped — do nothing else this tick

    prices = broker.get_prices()
    decision = decide(prices, broker.positions, broker.cash, price_history, params)

    # Record this tick's prices into history AFTER decide() has run, so
    # decide() always compares "now" against "everything before now."
    for t, p in prices.items():
        price_history.setdefault(t, []).append(p)
        price_history[t] = price_history[t][-config.price_history_len:]

    if decision is None:
        log_event(config, {"event": "no_action", "prices": prices})
        return

    # Position sizing guardrail — enforced here, not left to the strategy
    # function to remember to respect.
    max_dollars = account_value * config.max_position_pct
    if decision["dollar_amount"] > max_dollars:
        log_event(config, {"event": "decision_capped",
                            "requested": decision["dollar_amount"],
                            "capped_to": max_dollars})
        decision["dollar_amount"] = max_dollars

    if config.require_human_approval:
        # In mock mode this just logs the proposal. In Claude Code, this is
        # the hook where you'd surface the proposed trade and wait for a
        # real yes/no before ever calling broker.place_order().
        log_event(config, {"event": "awaiting_approval", "proposed": decision})
        return

    result = broker.place_order(decision["ticker"], decision["side"],
                                 decision["dollar_amount"])
    log_event(config, {"event": "order_result", "decision": decision, "result": result})


# ---------------------------------------------------------------------------
# Local test run — simulates a handful of ticks so you can see the whole
# loop work end-to-end. State persists across separate runs of this
# script via portfolio_state.json; within one run, 5 ticks share the
# same in-memory broker/breaker, same as before.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = Config()
    params = StrategyParams()

    broker_mode = os.environ.get("BROKER", "mock").lower()

    if broker_mode == "robinhood":
        # Live Robinhood Agentic account. State file only carries price_history
        # and circuit-breaker state; cash/positions/prices come from the Routine
        # via robinhood_input.json. require_human_approval stays False here
        # because the Routine calls review_equity_order before place_equity_order,
        # which is the real pre-trade gate for live money.
        cfg.state_path = Path("robinhood_state.json")
        cfg.require_human_approval = False
        broker = RobinhoodBroker(cfg.tickers)
        n_ticks = 1  # one real-market tick per Routine invocation
    else:
        # Mock / paper mode. PAPER-MODE-ONLY OVERRIDE: require_human_approval
        # is set to False so signals auto-execute against MockBroker and build
        # a track record. This line must not carry over to the robinhood path.
        cfg.require_human_approval = False
        broker = MockBroker(cfg.tickers, cfg.starting_cash, seed=None)
        n_ticks = 5  # 5 ticks per local test run

    state = load_portfolio_state(cfg)
    broker.load_state(state["cash"], state["positions"], state["prices"])

    breaker = CircuitBreaker(cfg.daily_loss_limit_pct)
    breaker.load_state(state["breaker_day"], state["breaker_day_start_value"])

    price_history = state["price_history"]

    for i in range(n_ticks):
        if n_ticks > 1:
            print(f"\n--- tick {i + 1} ---")
        tick(cfg, broker, breaker, price_history, params)

    save_portfolio_state(cfg, broker, breaker, price_history)
