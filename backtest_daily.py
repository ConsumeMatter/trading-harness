"""Daily backtest of paper_trading_harness.decide() on Robinhood daily open bars
(research_output/daily_bars.json). Fills at the open, one trade per day, same
sizing as the live harness. Usage: python backtest_daily.py"""
import json, dataclasses, math
from paper_trading_harness import decide, StrategyParams, _classify_regime

BARS = json.load(open("research_output/daily_bars.json"))
T = ["TSM", "XLK", "CAT", "XLE"]
dates = sorted(set.intersection(*[{b["d"] for b in BARS[t]} for t in T]))
px = {t: {b["d"]: b["o"] for b in BARS[t]} for t in T}

def run(params, start_cash=1000.0, hist_len=55):
    cash, pos, hist = start_cash, {t: 0.0 for t in T}, {t: [] for t in T}
    eq, trades, regimes = [], [], {"momentum": 0, "mean_reversion": 0, "cold": 0}
    for d in dates:
        p = {t: px[t][d] for t in T}
        for t in T:
            r = _classify_regime(hist[t], params)
            regimes["cold" if r == "insufficient_data" else ("momentum" if r == "trending" else "mean_reversion")] += 1
        dec = decide(p, pos, cash, hist, params)
        if dec:
            t, amt = dec["ticker"], dec["dollar_amount"]
            if dec["side"] == "buy":
                amt = min(amt, cash); cash -= amt; pos[t] += amt / p[t]
            else:
                sh = min(pos[t], amt / p[t]); pos[t] -= sh; cash += sh * p[t]
            trades.append((d, t, dec["side"], "MR" if "MR" in dec["reason"] else "mom"))
        for t in T: hist[t] = (hist[t] + [p[t]])[-hist_len:]
        eq.append(cash + sum(pos[t] * p[t] for t in T))
    rets = [eq[i] / eq[i-1] - 1 for i in range(1, len(eq))]
    mu = sum(rets) / len(rets); sd = math.sqrt(sum((r-mu)**2 for r in rets) / len(rets))
    peak, mdd = eq[0], 0
    for e in eq: peak = max(peak, e); mdd = max(mdd, 1 - e / peak)
    yrs = len(dates) / 252
    return dict(ret=eq[-1]/start_cash-1, sharpe=mu/sd*math.sqrt(252) if sd else 0, mdd=mdd,
                trades_yr=len(trades)/yrs, mom=sum(1 for x in trades if x[3]=="mom"),
                mr=sum(1 for x in trades if x[3]=="MR"), regimes=regimes)

if __name__ == "__main__":
    base = StrategyParams()
    print(f"{dates[0]} -> {dates[-1]}, {len(dates)} days")
    bh = sum(px[t][dates[-1]] / px[t][dates[0]] for t in T) / 4 - 1
    print(f"equal-weight buy&hold: {bh:+.1%}")
    for scale in (0.5, 0.75, 1.0, 1.25, 1.5):
        p = dataclasses.replace(base,
            buy_threshold_pct={k: v*scale for k, v in base.buy_threshold_pct.items()},
            sell_threshold_pct={k: v*scale for k, v in base.sell_threshold_pct.items()})
        r = run(p)
        print(f"thr x{scale}: ret {r['ret']:+.1%} sharpe {r['sharpe']:.2f} mdd {r['mdd']:.1%} "
              f"trades/yr {r['trades_yr']:.1f} (mom {r['mom']}, mr {r['mr']})")
    print("regime-days (ticker-days):", run(base)["regimes"])
