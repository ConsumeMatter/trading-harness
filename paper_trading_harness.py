from __future__ import annotations

"""
Paper-Trading Harness — Starting Skeleton
==========================================

Purpose: a scheduler-invoked "tick" function that checks (mock) market data,
runs decision logic, and executes (mock) trades — with logging and a circuit
breaker built in from day one, not bolted on later.

This is intentionally NOT a long-running process. It's meant to be invoked
once per "tick" by an external scheduler (cron, a Claude Code scheduled task,
etc.), so the scheduling layer and the trading logic stay separate concerns.

Build order for Claude Code:
  1. Get this running end-to-end against MockBroker (no real money, no
     network calls) until the circuit breaker, logging, and approval gate
     all behave the way you expect.
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
    tickers: list[str] = field(default_factory=lambda: ["CEG", "TSM", "VGT"])
    starting_cash: float = 1000.00
    max_position_pct: float = 0.25       # no single position > 25% of account
    daily_loss_limit_pct: float = 0.05   # halt trading for the day at -5%
    require_human_approval: bool = True  # mirrors Robinhood's approval-required mode
    log_path: Path = Path("harness_log.jsonl")


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


# ---------------------------------------------------------------------------
# Circuit breaker — a hard rule the code enforces, not a suggestion left
# to the strategy function to remember.
# ---------------------------------------------------------------------------

class CircuitBreaker:
    """Halts new trades once the account is down more than
    daily_loss_limit_pct from its value at the start of the trading day.
    Resets automatically on a new calendar day."""

    def __init__(self, daily_loss_limit_pct: float):
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self._day_start_value: Optional[float] = None
        self._day: Optional[date] = None

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
# Decision logic — placeholder. This is where a real strategy goes later
# (a sector thesis, an xAlphaAudit-derived signal, whatever you land on).
# Keep this function pure: prices in, a decision out. No side effects, no
# order placement here — that stays in tick() so the approval gate has one
# single choke point to sit in front of.
# ---------------------------------------------------------------------------

def decide(prices: dict[str, float], positions: dict[str, float]) -> Optional[dict]:
    """Placeholder strategy: does nothing. Replace with real logic.
    Must return either None (no action) or a dict shaped like:
      {"ticker": str, "side": "buy" | "sell", "dollar_amount": float, "reason": str}
    """
    return None


# ---------------------------------------------------------------------------
# The tick — the one function an external scheduler calls. Everything
# above is a building block; this is where they connect.
# ---------------------------------------------------------------------------

def tick(config: Config, broker: MockBroker, breaker: CircuitBreaker) -> None:
    account_value = broker.get_account_value()
    ok, breaker_msg = breaker.check(account_value)
    log_event(config, {"event": "breaker_check", "ok": ok, "detail": breaker_msg,
                        "account_value": account_value})

    if not ok:
        return  # circuit breaker tripped — do nothing else this tick

    prices = broker.get_prices()
    decision = decide(prices, broker.positions)

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
# loop work end-to-end before this goes anywhere near Claude Code or a
# real schedule.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = Config()
    broker = MockBroker(cfg.tickers, cfg.starting_cash, seed=42)
    breaker = CircuitBreaker(cfg.daily_loss_limit_pct)

    for i in range(5):
        print(f"\n--- tick {i + 1} ---")
        tick(cfg, broker, breaker)
