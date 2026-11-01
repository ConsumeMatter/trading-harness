"""Compare harness strategy vs monthly-rebalanced equal-weight buy-and-hold vs
200-day-MA trend filter on research_output/daily_bars.json. Research only.

Usage: python research/compare_strategies.py [cost_per_side] [start_day_index]
  e.g. python research/compare_strategies.py 0.0005      (full window, 0.05% cost)
       python research/compare_strategies.py 0.0005 199  (from first 200-day MA)"""
import json, sys
from datetime import date
from pathlib import Path

REPO = str(Path(__file__).resolve().parents[1])
COST = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0005
START_IDX = int(sys.argv[2]) if len(sys.argv) > 2 else 0
sys.path.insert(0, REPO)
from paper_trading_harness import decide, StrategyParams  # noqa: E402

BARS = json.load(open(f"{REPO}/research_output/daily_bars.json"))
T = ["TSM", "XLK", "CAT", "XLE"]
all_dates = sorted(set.intersection(*[{b["d"] for b in BARS[t]} for t in T]))
O = {t: {b["d"]: b["o"] for b in BARS[t]} for t in T}
C = {t: {b["d"]: b["c"] for b in BARS[t]} for t in T}
dates = all_dates[START_IDX:]
D = lambda s: date.fromisoformat(s)


class Book:
    def __init__(self, cash):
        self.cash, self.lots = cash, {t: [] for t in T}
        self.st = self.lt = self.costs = 0.0
        self.trades = 0
        self.cash_days = []

    def shares(self, t):
        return sum(q for q, _, _ in self.lots[t])

    def buy(self, t, dollars, px, d):
        dollars = min(dollars, self.cash / (1 + COST))
        if dollars <= 0.01:
            return
        fee = dollars * COST
        self.cash -= dollars + fee
        self.costs += fee
        self.lots[t].append([dollars / px, px * (1 + COST), d])
        self.trades += 1

    def sell_shares(self, t, sh, px, d):
        sh = min(sh, self.shares(t))
        if sh * px <= 0.01:
            return
        proceeds = sh * px
        fee = proceeds * COST
        self.cash += proceeds - fee
        self.costs += fee
        net_px = px * (1 - COST)
        left = sh
        while left > 1e-12 and self.lots[t]:
            lot = self.lots[t][0]
            take = min(lot[0], left)
            gain = take * (net_px - lot[1])
            if (D(d) - D(lot[2])).days > 365:
                self.lt += gain
            else:
                self.st += gain
            lot[0] -= take
            left -= take
            if lot[0] <= 1e-12:
                self.lots[t].pop(0)
        self.trades += 1

    def value(self, px):
        return self.cash + sum(self.shares(t) * px[t] for t in T)

    def unrealized(self, px):
        return sum(q * (px[t] - b) for t in T for q, b, _ in self.lots[t])


def harness():
    p = StrategyParams()
    b, hist, eq = Book(1000.0), {t: [] for t in T}, []
    for d in dates:
        px = {t: O[t][d] for t in T}
        pos = {t: b.shares(t) for t in T}
        dec = decide(px, pos, b.cash, hist, p)
        if dec:
            t = dec["ticker"]
            if dec["side"] == "buy":
                b.buy(t, dec["dollar_amount"], px[t], d)
            else:
                b.sell_shares(t, dec["dollar_amount"] / px[t], px[t], d)
        for t in T:
            hist[t] = (hist[t] + [px[t]])[-55:]
        close = {t: C[t][d] for t in T}
        v = b.value(close)
        eq.append(v)
        b.cash_days.append(b.cash / v)
    return b, eq


def buy_hold_monthly():
    b, eq, last_month = Book(1000.0), [], None
    for d in dates:
        px = {t: O[t][d] for t in T}
        if d[:7] != last_month:
            last_month = d[:7]
            total = b.value(px)
            target = total / (1 + COST) / 4
            for t in T:  # sells first, then buys
                diff = b.shares(t) * px[t] - target
                if diff > 0.01:
                    b.sell_shares(t, diff / px[t], px[t], d)
            for t in T:
                diff = target - b.shares(t) * px[t]
                if diff > 0.01:
                    b.buy(t, diff, px[t], d)
        close = {t: C[t][d] for t in T}
        v = b.value(close)
        eq.append(v)
        b.cash_days.append(b.cash / v)
    return b, eq


def trend_200():
    # Four independent $250 sleeves. Signal on the close, trade at the next open.
    # Before 200 closes exist for a ticker, its sleeve is held (no signal yet).
    b, eq = Book(1000.0), []
    sleeve_cash = {t: 0.0 for t in T}
    want_in = {t: True for t in T}
    first = True
    for d in dates:
        px = {t: O[t][d] for t in T}
        for t in T:
            held = b.shares(t) > 1e-9
            if first:
                before = b.cash
                b.buy(t, 250.0 / (1 + COST), px[t], d)
            elif want_in[t] and not held and sleeve_cash[t] > 0.01:
                b.buy(t, sleeve_cash[t] / (1 + COST), px[t], d)
                sleeve_cash[t] = 0.0
            elif not want_in[t] and held:
                before = b.cash
                b.sell_shares(t, b.shares(t), px[t], d)
                sleeve_cash[t] = b.cash - before
        first = False
        idx = all_dates.index(d)
        for t in T:
            window = [C[t][x] for x in all_dates[max(0, idx - 199): idx + 1]]
            if len(window) == 200:
                want_in[t] = C[t][d] > sum(window) / 200
        close = {t: C[t][d] for t in T}
        v = b.value(close)
        eq.append(v)
        b.cash_days.append(b.cash / v)
    return b, eq


def report(name, b, eq):
    yrs = (D(dates[-1]) - D(dates[0])).days / 365.25
    total = eq[-1] / 1000 - 1
    cagr = (eq[-1] / 1000) ** (1 / yrs) - 1
    peak, mdd = eq[0], 0.0
    for e in eq:
        peak = max(peak, e)
        mdd = max(mdd, 1 - e / peak)
    last = {t: C[t][dates[-1]] for t in T}
    return dict(name=name, total=total, cagr=cagr, mdd=mdd, trades_yr=b.trades / yrs,
                st=b.st, lt=b.lt, unreal=b.unrealized(last), costs=b.costs,
                avg_cash=sum(b.cash_days) / len(b.cash_days), end=eq[-1])


if __name__ == "__main__":
    out = [report("harness", *harness()), report("buyhold_monthly", *buy_hold_monthly()),
           report("trend_200dma", *trend_200())]
    print(f"window {dates[0]} -> {dates[-1]} ({len(dates)} days), cost/side {COST:.2%}")
    for r in out:
        print(f"{r['name']:16s} end ${r['end']:8.2f} total {r['total']:+7.1%} cagr {r['cagr']:+6.1%} "
              f"mdd {r['mdd']:5.1%} trades/yr {r['trades_yr']:5.1f} avg_cash {r['avg_cash']:5.1%} "
              f"realized ST ${r['st']:8.2f} LT ${r['lt']:8.2f} unrealized ${r['unreal']:8.2f} costs ${r['costs']:.2f}")
