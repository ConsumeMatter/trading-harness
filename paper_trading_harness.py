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
    # Momentum path (applied to non-MR tickers)
    buy_threshold_pct: float = 0.025         # propose a buy on a move up this big
    sell_threshold_pct: float = 0.025        # propose trimming on a move down this big
    buy_fraction_of_cash: float = 0.10       # size a buy as this fraction of cash
    sell_fraction_of_position: float = 0.50  # trim this fraction of the held position
    # Mean-reversion path
    mr_tickers: list[str] = field(default_factory=lambda: ["XLE"])
    mr_ma_window: int = 50                   # MA period for mean-reversion signal
    mr_entry_threshold_pct: float = 0.01     # enter when price is this far below MA
    mr_exit_threshold_pct: float = 0.0       # exit when price recovers to MA (0 = at MA)
    mr_buy_fraction_of_cash: float = 0.10    # size MR buy as this fraction of cash


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


# TODO (Claude Code, later): class RobinhoodBroker with the same four
# methods, backed by real Robinhood Trading MCP tool calls instead of
# in-memory math. Nothing below this line should need to change.
#
# !! WHEN YOU ADD RobinhoodBroker: the __main__ block below sets
# !! cfg.require_human_approval = False as a paper-mode-only override, so
# !! MockBroker runs auto-execute and produces a real track record instead
# !! of every signal dead-ending at "awaiting_approval" forever. That
# !! override MUST be removed (or set back to True) the moment
# !! RobinhoodBroker replaces MockBroker in __main__. Real money must
# !! never execute without a real, interactive approval step -- and no
# !! such step exists yet. Do not skip this when doing the broker swap.


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


def save_portfolio_state(config: Config, broker: MockBroker, breaker: CircuitBreaker,
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
# Decision logic — v1: simple, deterministic threshold-on-momentum.
# Keep this function pure: prices/positions/cash/history in, a decision
# out. No side effects, no order placement here — that stays in tick() so
# the approval gate has one single choke point to sit in front of.
# ---------------------------------------------------------------------------

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

    Returns a trade proposal dict or None. Called only for tickers in
    params.mr_tickers.
    """
    if len(history) < params.mr_ma_window:
        return None  # not enough history yet to compute MA

    ma = sum(history[-params.mr_ma_window:]) / params.mr_ma_window
    if ma <= 0:
        return None

    deviation = (ma - current_price) / ma  # positive when price < MA

    held_shares = positions.get(ticker, 0.0)

    # Exit: price back at or above MA while we hold a position
    if held_shares > 0 and deviation <= params.mr_exit_threshold_pct:
        position_value = held_shares * current_price
        return {
            "ticker": ticker,
            "side": "sell",
            "dollar_amount": round(position_value, 2),
            "reason": (f"{ticker} MR exit: price {current_price:.2f} recovered "
                       f"to MA {ma:.2f} ({deviation:.2%} deviation)"),
        }

    # Entry: price sufficiently below MA and no position open
    if held_shares <= 0 and deviation >= params.mr_entry_threshold_pct:
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
    """Two-path strategy: mean-reversion for MR tickers, momentum for the rest.

    Mean-reversion path (params.mr_tickers, e.g. XLE):
      Enter when price falls mr_entry_threshold_pct below its MA-{mr_ma_window}.
      Exit when price recovers to MA. Requires mr_ma_window ticks of history.

    Momentum path (all other tickers):
      Compares current price to the previous tick's price. Buy on an up-move
      >= buy_threshold_pct; trim on a down-move <= -sell_threshold_pct.

    At most one trade per call. MR exit signals take priority (they are risk
    management as much as alpha); then MR entries; then the largest-move
    momentum candidate. Returns None when no signal fires.

    Must return either None or a dict shaped like:
      {"ticker": str, "side": "buy" | "sell", "dollar_amount": float, "reason": str}
    """
    mr_set = set(params.mr_tickers)

    # --- Mean-reversion path ---
    mr_exits = []
    mr_entries = []
    for ticker in params.mr_tickers:
        if ticker not in prices:
            continue
        current_price = prices[ticker]
        history = price_history.get(ticker, [])
        signal = _mr_signal(ticker, current_price, history, positions, cash, params)
        if signal is None:
            continue
        if signal["side"] == "sell":
            mr_exits.append(signal)
        else:
            mr_entries.append(signal)

    if mr_exits:
        return mr_exits[0]  # first exit trumps everything else
    if mr_entries:
        return mr_entries[0]

    # --- Momentum path (non-MR tickers) ---
    candidates = []
    for ticker, current_price in prices.items():
        if ticker in mr_set:
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

    if pct_change >= params.buy_threshold_pct:
        dollar_amount = round(cash * params.buy_fraction_of_cash, 2)
        if dollar_amount <= 0:
            return None
        return {
            "ticker": ticker,
            "side": "buy",
            "dollar_amount": dollar_amount,
            "reason": (f"{ticker} up {pct_change:.2%} since last tick, "
                       f"above +{params.buy_threshold_pct:.0%} threshold"),
        }

    if pct_change <= -params.sell_threshold_pct:
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
                       f"past -{params.sell_threshold_pct:.0%} threshold — trimming"),
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

    # PAPER-MODE-ONLY OVERRIDE. Config's real default is True (mirrors
    # Robinhood's approval-required mode) -- but no interactive approval
    # step exists yet, so leaving it True here just means every signal
    # logs "awaiting_approval" and nothing ever actually trades, forever.
    # Auto-execute against MockBroker so a real (fake-money) track record
    # accumulates. See the loud warning above RobinhoodBroker's TODO:
    # this line must go away the moment real money is involved.
    cfg.require_human_approval = False

    params = StrategyParams()

    state = load_portfolio_state(cfg)

    # seed=None -> real randomness each run, not the same replayed sequence
    broker = MockBroker(cfg.tickers, cfg.starting_cash, seed=None)
    broker.load_state(state["cash"], state["positions"], state["prices"])

    breaker = CircuitBreaker(cfg.daily_loss_limit_pct)
    breaker.load_state(state["breaker_day"], state["breaker_day_start_value"])

    price_history = state["price_history"]

    for i in range(5):
        print(f"\n--- tick {i + 1} ---")
        tick(cfg, broker, breaker, price_history, params)

    save_portfolio_state(cfg, broker, breaker, price_history)
