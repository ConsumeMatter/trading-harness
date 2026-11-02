"""
Paper-Trading Harness — Drift-Band Rebalancer (Portfolio B)
==========================================================

Holds a fixed-weight portfolio and rebalances it only when a holding drifts
too far from its target. Invoked by the Claude Code Routine; each run is a
fresh process, so cash, tax lots and the last check date live in
robinhood_state.json, which the Routine commits after every run.

Each run:
  1. Reads real prices from robinhood_input.json (written by the Routine from
     Robinhood's get_equity_quotes). Missing or non-positive prices -> halt,
     no trades, state untouched.
  2. Skips weekends, and skips if this calendar month was already checked.
  3. First run ever: buys the target weights with $1,000 of paper cash.
     Otherwise: if any holding is more than 5 percentage points away from its
     target, trades every holding back to target (sells first, then buys);
     if not, does nothing.

Execution is simulated (MockBroker): fills at the quoted price, fractional
shares, no commissions (Robinhood charges none), trades under $1 skipped.
Tax lots are FIFO; gains on shares held more than 365 days count as long-term.

Targets come from research/portfolio_candidates.py: VTI fixed at 60%; the
other 40% split across SMH/XLI/XLU by equal risk contribution (skfolio),
estimated on daily returns from 2021-11-02 to 2026-10-30.

TODO(before ANY real broker is connected): set Config.require_human_approval
back to True and build a real approval step (proposed trades shown to a human,
nothing placed without an explicit yes). Paper mode runs with approval off so
the paper track record builds unattended. The BROKER check in main() refuses
every broker except "mock" until this TODO is done.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

TARGET_WEIGHTS = {"VTI": 0.600, "SMH": 0.079, "XLI": 0.143, "XLU": 0.178}


@dataclass
class Config:
    targets: dict[str, float] = field(default_factory=lambda: dict(TARGET_WEIGHTS))
    drift_band: float = 0.05           # rebalance when any holding is >5 points off target
    starting_cash: float = 1000.00
    min_trade: float = 1.00            # skip paper trades smaller than this
    # TODO(before ANY real broker is connected): must be True. See module docstring.
    require_human_approval: bool = False
    input_path: Path = Path("robinhood_input.json")
    state_path: Path = Path("robinhood_state.json")
    log_path: Path = Path("harness_log.jsonl")


def log_event(config: Config, event: dict) -> None:
    event = {"timestamp": datetime.now().isoformat(), **event}
    with open(config.log_path, "a") as f:
        f.write(json.dumps(event) + "\n")
    logging.info(json.dumps(event))


def read_prices(config: Config) -> tuple[dict[str, float] | None, str]:
    if not config.input_path.exists():
        return None, f"{config.input_path} not found"
    try:
        raw = json.loads(config.input_path.read_text()).get("prices", {})
    except json.JSONDecodeError as exc:
        return None, f"{config.input_path} is not valid JSON: {exc}"
    prices, problems = {}, []
    for t in config.targets:
        try:
            p = float(raw[t])
        except (KeyError, TypeError, ValueError):
            problems.append(f"{t} missing")
            continue
        if p <= 0:
            problems.append(f"{t} price {p} is not positive")
            continue
        prices[t] = p
    if problems:
        return None, "; ".join(problems)
    return prices, "ok"


def fresh_state(config: Config) -> dict:
    return {"schema": "drift_band_v1", "cash": config.starting_cash,
            "lots": {t: [] for t in config.targets}, "last_check": None,
            "realized_short_term": 0.0, "realized_long_term": 0.0}


class MockBroker:
    """Simulated execution against real quoted prices. Holds FIFO tax lots."""

    def __init__(self, state: dict, prices: dict[str, float], today: date, min_trade: float):
        self.state, self.prices, self.today, self.min_trade = state, prices, today, min_trade

    def shares(self, t: str) -> float:
        return sum(lot["shares"] for lot in self.state["lots"].get(t, []))

    def holding_value(self, t: str) -> float:
        return self.shares(t) * self.prices[t]

    def account_value(self) -> float:
        return self.state["cash"] + sum(self.holding_value(t) for t in self.prices)

    def buy(self, t: str, dollars: float) -> dict | None:
        dollars = min(dollars, self.state["cash"])
        if dollars < self.min_trade:
            return None
        price = self.prices[t]
        self.state["cash"] -= dollars
        self.state["lots"].setdefault(t, []).append(
            {"shares": dollars / price, "cost": price, "date": self.today.isoformat()})
        return {"ticker": t, "side": "buy", "dollars": round(dollars, 2),
                "shares": round(dollars / price, 6), "price": price}

    def sell(self, t: str, dollars: float) -> dict | None:
        price = self.prices[t]
        shares = min(dollars / price, self.shares(t))
        if shares * price < self.min_trade:
            return None
        self.state["cash"] += shares * price
        short_gain = long_gain = 0.0
        left, lots = shares, self.state["lots"][t]
        while left > 1e-12 and lots:
            lot = lots[0]
            take = min(lot["shares"], left)
            gain = take * (price - lot["cost"])
            if (self.today - date.fromisoformat(lot["date"])).days > 365:
                long_gain += gain
            else:
                short_gain += gain
            lot["shares"] -= take
            left -= take
            if lot["shares"] <= 1e-12:
                lots.pop(0)
        self.state["realized_short_term"] += short_gain
        self.state["realized_long_term"] += long_gain
        return {"ticker": t, "side": "sell", "dollars": round(shares * price, 2),
                "shares": round(shares, 6), "price": price,
                "short_term_gain": round(short_gain, 2), "long_term_gain": round(long_gain, 2)}


def current_weights(broker: MockBroker, targets: dict[str, float]) -> dict[str, float]:
    total = broker.account_value()
    return {t: broker.holding_value(t) / total for t in targets}


def plan_rebalance(broker: MockBroker, targets: dict[str, float]) -> list[tuple[str, str, float]]:
    """Trades that bring every holding back to target: sells first, then buys."""
    total = broker.account_value()
    gaps = {t: w * total - broker.holding_value(t) for t, w in targets.items()}
    sells = [(t, "sell", -gap) for t, gap in gaps.items() if gap < 0]
    buys = [(t, "buy", gap) for t, gap in gaps.items() if gap > 0]
    return sells + buys


def execute(broker: MockBroker, plan: list[tuple[str, str, float]]) -> list[dict]:
    fills = []
    for t, side, dollars in plan:
        fill = broker.buy(t, dollars) if side == "buy" else broker.sell(t, dollars)
        if fill:
            fills.append(fill)
    return fills


def snapshot(broker: MockBroker, targets: dict[str, float]) -> dict:
    weights = current_weights(broker, targets)
    return {"account_value": round(broker.account_value(), 2),
            "cash": round(broker.state["cash"], 2),
            "weights": {t: round(w, 4) for t, w in weights.items()},
            "drift_points": {t: round((weights[t] - targets[t]) * 100, 2) for t in targets}}


def run_check(config: Config, state: dict | None, prices: dict[str, float],
              today: date) -> tuple[dict, list[dict], bool]:
    """Pure core of one run. Returns (new state, log events, whether state changed)."""
    events = []
    if state is None or state.get("schema") != "drift_band_v1":
        if state is not None:
            events.append({"event": "momentum_state_retired",
                           "old_cash": state.get("cash"), "old_positions": state.get("positions")})
        state = fresh_state(config)
        events.append({"event": "portfolio_initialized", "starting_cash": config.starting_cash,
                       "targets": config.targets})

    broker = MockBroker(state, prices, today, config.min_trade)
    month = today.strftime("%Y-%m")
    if today.weekday() >= 5:
        events.append({"event": "skipped_weekend", "prices": prices, **snapshot(broker, config.targets)})
        return state, events, len(events) > 1
    if state["last_check"] and state["last_check"][:7] == month:
        events.append({"event": "skipped_already_checked_this_month", "last_check": state["last_check"],
                       "prices": prices, **snapshot(broker, config.targets)})
        return state, events, False

    holds_anything = any(state["lots"].get(t) for t in config.targets)
    before = snapshot(broker, config.targets)
    max_drift = max(abs(d) for d in before["drift_points"].values()) / 100
    if holds_anything and max_drift <= config.drift_band:
        state["last_check"] = today.isoformat()
        events.append({"event": "within_band", "prices": prices, "max_drift_points": round(max_drift * 100, 2),
                       **before})
        return state, events, True

    plan = plan_rebalance(broker, config.targets)
    kind = "rebalance" if holds_anything else "initial_allocation"
    if config.require_human_approval:
        events.append({"event": "awaiting_approval", "kind": kind, "prices": prices,
                       "proposed": [{"ticker": t, "side": s, "dollars": round(d, 2)} for t, s, d in plan],
                       **before})
        return state, events, len(events) > 1
    fills = execute(broker, plan)
    state["last_check"] = today.isoformat()
    events.append({"event": kind, "prices": prices, "max_drift_points_before": round(max_drift * 100, 2),
                   "fills": fills, "before": before, "after": snapshot(broker, config.targets),
                   "realized_short_term": round(state["realized_short_term"], 2),
                   "realized_long_term": round(state["realized_long_term"], 2)})
    return state, events, True


def main(config: Config | None = None, today: date | None = None) -> int:
    config = config or Config()
    today = today or date.today()

    broker_mode = os.environ.get("BROKER", "mock").lower()
    if broker_mode != "mock":
        # TODO(before ANY real broker is connected): see the module docstring.
        log_event(config, {"event": "refused_broker", "broker": broker_mode,
                           "detail": "only BROKER=mock is allowed until approval is rebuilt"})
        print(f"REFUSED: BROKER={broker_mode}. Only paper mode (BROKER=mock) is allowed.")
        return 2

    prices, problem = read_prices(config)
    if prices is None:
        log_event(config, {"event": "halted_bad_prices", "detail": problem})
        print(f"HALTED, no trades: {problem}")
        return 1

    state = json.loads(config.state_path.read_text()) if config.state_path.exists() else None
    state, events, changed = run_check(config, state, prices, today)
    for e in events:
        log_event(config, e)
    if changed:
        config.state_path.write_text(json.dumps(state, indent=2))

    last = events[-1]
    snap = last.get("after", last)
    print(f"{today.isoformat()}: {last['event']}")
    print(f"  account value ${snap.get('account_value')}, cash ${snap.get('cash')}")
    for t, w in config.targets.items():
        print(f"  {t}: {snap['weights'][t]:.1%} (target {w:.1%}, drift {snap['drift_points'][t]:+.2f} points)")
    for f in last.get("fills", []):
        print(f"  paper {f['side']} {f['ticker']} ${f['dollars']} at ${f['price']}")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sys.exit(main())
