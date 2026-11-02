"""Out-of-sample test of buy-and-hold candidate portfolios with monthly rebalancing.

Research only. Data: Robinhood daily bars (split-adjusted, price-only, no dividends),
fetched via the Robinhood MCP get_equity_historicals tool and stored in
research/data/etf_daily_2008_2026.json.gz as {symbol: [{d, o, c, v}, ...]}.

Protocol
  - Holdings were picked by criteria knowable in advance (liquidity, expense ratio,
    10+ years of history, correlation), never by returns.
  - Each portfolio has a fixed policy split declared up front. Within each sleeve,
    weights are equal-risk-contribution (skfolio RiskBudgeting, variance) estimated
    on the 5 years of daily returns before the test window. Returns are not used.
  - Window 1: weights from 2016-2020, test 2021-01 to 2026-10.
    Window 2: weights from 2010-2014, test 2015-2020.
    Bars before 2010 are not used (they contain bad prints, e.g. VTI +19.9% on 2008-10-13).
  - Rebalancing on the first trading day of each month, at the close:
      full  = always trade back to target weights
      band5 = trade back to targets only if any holding is >5 percentage points off
  - $1,000 start, fractional shares, 0.05% cost per trade, trades under $1 skipped,
    FIFO tax lots, long-term = held more than 365 days.

Usage: python research/portfolio_candidates.py
"""
import gzip
import json
import math
from datetime import date
from pathlib import Path

import pandas as pd
from skfolio import Portfolio, RiskMeasure
from skfolio.optimization import RiskBudgeting
from skfolio.preprocessing import prices_to_returns

DATA = Path(__file__).resolve().parent / "data" / "etf_daily_2008_2026.json.gz"
COST = 0.0005
MIN_TRADE = 1.0
BAND = 0.05
START_CASH = 1000.0

# Sleeves: (policy share of portfolio, holdings weighted by equal risk inside the sleeve).
PORTFOLIOS = {
    "A broad core": [(0.80, ["VTI", "VEA", "VWO"]), (0.20, ["BND"])],
    "B core+satellites": [(0.60, ["VTI"]), (0.40, ["SMH", "XLI", "XLU"])],
    "C reference": [(1.00, ["TSM", "XLK", "CAT", "XLE"])],
    "VTI only": [(1.00, ["VTI"])],
}
WINDOWS = [
    ("2021-2026", ("2016-01-01", "2020-12-31"), ("2021-01-01", "2026-12-31")),
    ("2015-2020", ("2010-01-01", "2014-12-31"), ("2015-01-01", "2020-12-31")),
]
# Trailing dividend yield (%) from Robinhood get_equity_fundamentals on 2026-10-30.
# Used only to size the "dividends excluded" caveat.
DIV_YIELD = {"VTI": 1.03, "VEA": 2.31, "VWO": 1.97, "BND": 2.69, "SMH": 0.48, "XLI": 1.14,
             "XLU": 2.92, "TSM": 0.73, "XLK": 0.47, "CAT": 0.99, "XLE": 2.31}

SCREEN = ["VTI", "SPY", "VEA", "VXUS", "VWO", "BND", "GLD", "VNQ", "SOXX", "SMH",
          "XLI", "VIS", "XLU", "NLR", "URA"]


def load():
    raw = json.load(gzip.open(DATA, "rt"))
    close = pd.DataFrame({s: pd.Series({b["d"]: b["c"] for b in rows}) for s, rows in raw.items()})
    dollar_vol = pd.DataFrame({s: pd.Series({b["d"]: b["c"] * b["v"] for b in rows}) for s, rows in raw.items()})
    close.index = pd.to_datetime(close.index)
    dollar_vol.index = pd.to_datetime(dollar_vol.index)
    return close.sort_index(), dollar_vol.sort_index()


def screen(close, dollar_vol, train):
    a, b = train
    X = prices_to_returns(close.loc[a:b, SCREEN].dropna())
    corr = X.corr()
    rows = []
    for s in SCREEN:
        p = Portfolio(X=X[[s]], weights=[1.0], compounded=True)
        rows.append({"ticker": s,
                     "median_$vol_M": round(dollar_vol.loc[a:b, s].median() / 1e6, 1),
                     "ann_vol": round(p.annualized_standard_deviation, 3),
                     "max_dd": round(p.max_drawdown, 3),
                     "corr_VTI": round(corr.loc[s, "VTI"], 2)})
    return pd.DataFrame(rows).set_index("ticker"), corr.round(2)


def settle_weights(close, spec, train):
    a, b = train
    weights = {}
    for share, tickers in spec:
        if len(tickers) == 1:
            weights[tickers[0]] = share
            continue
        X = prices_to_returns(close.loc[a:b, tickers].dropna())
        w = RiskBudgeting(risk_measure=RiskMeasure.VARIANCE).fit(X).weights_
        for t, x in zip(tickers, w):
            weights[t] = share * float(x)
    return weights


def in_sample_risk(close, weights, train):
    a, b = train
    tickers = list(weights)
    X = prices_to_returns(close.loc[a:b, tickers].dropna())
    p = Portfolio(X=X, weights=[weights[t] for t in tickers], compounded=True)
    return p.annualized_standard_deviation, p.max_drawdown


class Book:
    def __init__(self, tickers):
        self.cash = START_CASH
        self.lots = {t: [] for t in tickers}
        self.st = self.lt = self.costs = 0.0
        self.trades = 0

    def shares(self, t):
        return sum(lot[0] for lot in self.lots[t])

    def value(self, px):
        return self.cash + sum(self.shares(t) * px[t] for t in self.lots)

    def buy(self, t, dollars, price, d):
        dollars = min(dollars, self.cash / (1 + COST))
        if dollars < MIN_TRADE:
            return
        fee = dollars * COST
        self.cash -= dollars + fee
        self.costs += fee
        self.lots[t].append([dollars / price, price * (1 + COST), d])
        self.trades += 1

    def sell(self, t, dollars, price, d):
        sh = min(dollars / price, self.shares(t))
        if sh * price < MIN_TRADE:
            return
        proceeds = sh * price
        fee = proceeds * COST
        self.cash += proceeds - fee
        self.costs += fee
        net = price * (1 - COST)
        left = sh
        while left > 1e-12 and self.lots[t]:
            lot = self.lots[t][0]
            take = min(lot[0], left)
            gain = take * (net - lot[1])
            if (d - lot[2]).days > 365:
                self.lt += gain
            else:
                self.st += gain
            lot[0] -= take
            left -= take
            if lot[0] <= 1e-12:
                self.lots[t].pop(0)
        self.trades += 1

    def rebalance(self, weights, px, d):
        total = self.value(px)
        target = {t: w * total / (1 + COST) for t, w in weights.items()}
        for t in weights:
            excess = self.shares(t) * px[t] - target[t]
            if excess > 0:
                self.sell(t, excess, px[t], d)
        for t in weights:
            short = target[t] - self.shares(t) * px[t]
            if short > 0:
                self.buy(t, short, px[t], d)


def simulate(close, weights, test, rule):
    a, b = test
    tickers = list(weights)
    prices = close.loc[a:b, tickers].dropna()
    book = Book(tickers)
    equity, events, last_month = [], 0, None
    for ts, row in prices.iterrows():
        px = row.to_dict()
        d = ts.date()
        month = (d.year, d.month)
        if last_month is None:
            book.rebalance(weights, px, d)
        elif month != last_month and len(tickers) > 1:
            if rule == "full":
                book.rebalance(weights, px, d)
                events += 1
            else:
                total = book.value(px)
                drift = max(abs(book.shares(t) * px[t] / total - w) for t, w in weights.items())
                if drift > BAND:
                    book.rebalance(weights, px, d)
                    events += 1
        last_month = month
        equity.append(book.value(px))
    eq = pd.Series(equity, index=prices.index)
    years = (prices.index[-1] - prices.index[0]).days / 365.25
    last = prices.iloc[-1].to_dict()
    unreal = sum(lot[0] * (last[t] - lot[1]) for t in tickers for lot in book.lots[t])
    return {
        "end": eq.iloc[-1],
        "total": eq.iloc[-1] / START_CASH - 1,
        "cagr": (eq.iloc[-1] / START_CASH) ** (1 / years) - 1,
        "max_dd": float((1 - eq / eq.cummax()).max()),
        "trades_yr": book.trades / years,
        "rebal_events": events,
        "st": book.st, "lt": book.lt, "unrealized": unreal, "costs": book.costs,
        "start": prices.index[0].date(), "stop": prices.index[-1].date(),
    }


def main():
    close, dollar_vol = load()
    pd.set_option("display.width", 200)
    for name, train, test in WINDOWS:
        table, corr = screen(close, dollar_vol, train)
        print(f"\n=== Screen on weight-setting data {train[0]}..{train[1]} ===")
        print(table.sort_values("median_$vol_M", ascending=False).to_string())
        print("\nCorrelation of daily returns:")
        print(corr.to_string())

    for name, train, test in WINDOWS:
        print(f"\n=== Test window {name} (weights from {train[0][:4]}-{train[1][:4]}) ===")
        for pname, spec in PORTFOLIOS.items():
            w = settle_weights(close, spec, train)
            vol, dd = in_sample_risk(close, w, train)
            div = sum(w[t] * DIV_YIELD[t] for t in w)
            print(f"\n{pname}: weights " + ", ".join(f"{t} {x:.1%}" for t, x in w.items())
                  + f" | weight-setting period vol {vol:.1%}, max drop {dd:.1%} | current div yield {div:.2f}%")
            for rule in (["full", "band5"] if len(w) > 1 else ["hold"]):
                r = simulate(close, w, test, rule)
                print(f"  {rule:6s} {r['start']}..{r['stop']} end ${r['end']:8.2f} total {r['total']:+7.1%} "
                      f"cagr {r['cagr']:+6.1%} max_dd {r['max_dd']:5.1%} trades/yr {r['trades_yr']:5.1f} "
                      f"rebalances {r['rebal_events']:3d} realized ST ${r['st']:8.2f} LT ${r['lt']:8.2f} "
                      f"unrealized ${r['unrealized']:8.2f} costs ${r['costs']:.2f}")


if __name__ == "__main__":
    main()
