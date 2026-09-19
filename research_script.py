#!/usr/bin/env python3
"""
Real-data research script for the trading harness.
Downloads weekly OHLCV data from GitHub (raw.githubusercontent.com),
computes screening metrics, runs vectorized backtests, and produces
the same output files as the PR #4 synthetic run — but with real prices.

Data source: dotchev/top-stox (weekly, 2021-2026, 29 tickers)
             whchien/ai-trader  (daily, 2020-2025, TSM only — resampled to weekly)
"""

import io
import json
import math
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import urllib.request

warnings.filterwarnings("ignore")

OUT = Path("research_output")
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Ticker universe
# ---------------------------------------------------------------------------
TICKERS = [
    "NVDA", "CEG", "TSM", "META", "AMZN",
    "XLE", "BAC", "DE", "CVX", "AAPL",
    "XOM", "XLY", "XLK", "IWM", "QQQ",
    "MSFT", "CAT", "JPM", "XLB", "XLRE",
    "NEE", "XLF", "UNH", "XLI", "LMT",
    "VTI", "XLU", "SPY", "XLV", "XLP",
]

BASE_URL = "https://raw.githubusercontent.com/dotchev/top-stox/main/data/stock_history"
TSM_URL  = "https://raw.githubusercontent.com/whchien/ai-trader/main/data/us_stock/TSM.csv"

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def fetch_csv(url: str) -> pd.DataFrame:
    req = urllib.request.Request(url, headers={"User-Agent": "research-script/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return pd.read_csv(io.StringIO(r.read().decode()))


def _normalize_date_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Strip time component so Mon 05:00 and Mon 00:00 both become Mon 00:00."""
    return pd.DatetimeIndex([d.date() for d in idx])


def load_weekly(ticker: str) -> pd.Series | None:
    """Return weekly adjusted-close series for a ticker, or None on failure."""
    try:
        if ticker == "TSM":
            df = fetch_csv(TSM_URL)
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
            # resample daily → weekly aligned to Monday (matching dotchev)
            close = df["adj_close"].resample("W-MON").last().dropna()
        else:
            url = f"{BASE_URL}/{ticker}.csv"
            df = fetch_csv(url)
            df.columns = [c.strip() for c in df.columns]
            df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_localize(None)
            df = df.set_index("Date").sort_index()
            close = df["Close"].dropna()
        close.index = _normalize_date_index(close.index)
        close.name = ticker
        return close
    except Exception as e:
        print(f"  WARN: {ticker} failed — {e}")
        return None


def load_weekly_ohlcv(ticker: str) -> pd.DataFrame | None:
    """Return weekly OHLCV (for range calculation)."""
    try:
        if ticker == "TSM":
            df = fetch_csv(TSM_URL)
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
            ohlcv = df[["open","high","low","close","volume"]].copy()
            ohlcv.columns = ["Open","High","Low","Close","Volume"]
            weekly = ohlcv.resample("W-MON").agg({
                "Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"
            }).dropna()
        else:
            url = f"{BASE_URL}/{ticker}.csv"
            df = fetch_csv(url)
            df.columns = [c.strip() for c in df.columns]
            df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_localize(None)
            df = df.set_index("Date").sort_index()
            weekly = df[["Open","High","Low","Close","Volume"]].dropna()
        weekly.index = _normalize_date_index(weekly.index)
        weekly.index.name = "Date"
        return weekly
    except Exception as e:
        print(f"  WARN: {ticker} OHLCV failed — {e}")
        return None


# ---------------------------------------------------------------------------
# Screening metrics
# ---------------------------------------------------------------------------

def screening_metrics(ohlcv: pd.DataFrame, close: pd.Series) -> dict:
    """
    avg_weekly_range_pct : mean (H-L)/C over last 52 weeks — proxy for intraday move
    rv52_ann_pct         : annualised vol from weekly returns (52-week lookback)
    return_1y_pct        : 52-week total return
    """
    recent = ohlcv.iloc[-52:]
    close_recent = close.iloc[-52:]

    weekly_range = (recent["High"] - recent["Low"]) / recent["Close"] * 100
    avg_range    = float(weekly_range.mean())

    rets = close.pct_change().dropna().iloc[-52:]
    rv   = float(rets.std() * math.sqrt(52) * 100)

    ret1y = float((close.iloc[-1] / close.iloc[-53] - 1) * 100) if len(close) >= 53 else float("nan")

    return {
        "avg_weekly_range_pct": round(avg_range, 3),
        "rv52_ann_pct":         round(rv, 1),
        "return_1y_pct":        round(ret1y, 1),
    }


# ---------------------------------------------------------------------------
# Momentum backtest (weekly bars)
# Entry: weekly return > threshold%. Hold for hold_weeks. Exit on stop too.
# ---------------------------------------------------------------------------

def momentum_backtest(close: pd.Series, threshold_pct: float,
                      hold_weeks: int = 4, stop_pct: float = 0.06) -> dict:
    prices = close.values.astype(float)
    n = len(prices)
    cash     = 1.0
    position = 0.0       # in shares (fractional)
    entry_price = 0.0
    weeks_held  = 0

    equity = np.empty(n)
    equity[0] = 1.0

    for i in range(1, n):
        weekly_ret = prices[i] / prices[i-1] - 1

        if position > 0:
            weeks_held += 1
            loss = prices[i] / entry_price - 1
            if weeks_held >= hold_weeks or loss < -stop_pct:
                cash = position * prices[i]
                position = 0.0

        elif weekly_ret > threshold_pct / 100:
            position   = cash / prices[i]
            entry_price = prices[i]
            cash       = 0.0
            weeks_held  = 0

        equity[i] = cash + position * prices[i]

    returns = np.diff(np.log(equity))
    total_ret  = (equity[-1] / equity[0] - 1) * 100
    rolls = pd.Series(equity)
    dd_series  = (rolls / rolls.cummax() - 1)
    max_dd     = float(-dd_series.min() * 100)
    sharpe     = float((returns.mean() / returns.std() * math.sqrt(52))) if returns.std() > 0 else 0.0
    n_years    = n / 52
    trades     = float(sum(
        1 for i in range(1, n) if (equity[i] != equity[i-1] and i > 1)
    ) / n_years / 52 * 52 / n_years / 52 * 52)  # rough
    # simpler trade count
    in_position = False
    trade_count = 0
    for i in range(1, n):
        weekly_ret = prices[i] / prices[i-1] - 1
        if not in_position and weekly_ret > threshold_pct / 100:
            in_position = True
            trade_count += 1
        elif in_position:
            loss = prices[i] / prices[i-1] - 1
            # approximate exit logic
            if loss < -stop_pct or trade_count > 0:
                in_position = False
    trades_pw = round(trade_count / n_years / 52, 3)

    return {
        "total_ret":  round(total_ret, 1),
        "max_dd":     round(max_dd, 1),
        "sharpe":     round(sharpe, 2),
        "trades_pw":  trades_pw,
        "equity":     equity.tolist(),
    }


# ---------------------------------------------------------------------------
# Mean-reversion backtest
# Entry: close > threshold% below MA. Exit: price returns to MA.
# ---------------------------------------------------------------------------

def mr_backtest(close: pd.Series, ma_window: int, threshold_pct: float) -> dict:
    prices = close.values.astype(float)
    n = len(prices)

    cash     = 1.0
    position = 0.0
    equity   = np.empty(n)
    equity[0]= 1.0

    for i in range(ma_window, n):
        ma = float(np.mean(prices[i-ma_window:i]))
        pct_below = (ma - prices[i]) / ma * 100

        if position > 0:
            if prices[i] >= ma:
                cash     = position * prices[i]
                position = 0.0

        elif pct_below > threshold_pct:
            position = cash / prices[i]
            cash     = 0.0

        equity[i] = cash + position * prices[i]

    for i in range(min(ma_window, n)):
        equity[i] = 1.0

    returns   = np.diff(np.log(np.where(equity > 0, equity, 1e-10)))
    total_ret = (equity[-1] - 1.0) * 100
    rolls     = pd.Series(equity)
    max_dd    = float(-(rolls / rolls.cummax() - 1).min() * 100)
    sharpe    = float(returns.mean() / returns.std() * math.sqrt(52)) if returns.std() > 0 else 0.0
    n_years   = n / 52

    in_pos = False
    tc = 0
    for i in range(ma_window, n):
        ma = float(np.mean(prices[i-ma_window:i]))
        pct_below = (ma - prices[i]) / ma * 100
        if not in_pos and pct_below > threshold_pct:
            in_pos = True; tc += 1
        elif in_pos and prices[i] >= ma:
            in_pos = False
    trades_pw = round(tc / n_years / 52, 3)

    return {
        "total_ret":  round(total_ret, 1),
        "max_dd":     round(max_dd, 1),
        "sharpe":     round(sharpe, 2),
        "trades_pw":  trades_pw,
        "equity":     equity.tolist(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Loading data for {len(TICKERS)} tickers…")
    closes = {}
    ohlcvs = {}
    for t in TICKERS:
        print(f"  {t}…", end=" ", flush=True)
        c = load_weekly(t)
        o = load_weekly_ohlcv(t)
        if c is not None and o is not None and len(c) >= 53:
            closes[t] = c
            ohlcvs[t] = o
            print(f"OK ({len(c)} weeks, {c.index[0].date()} – {c.index[-1].date()})")
        else:
            print("SKIP")

    loaded = list(closes.keys())
    print(f"\nLoaded {len(loaded)} tickers: {loaded}\n")

    # -----------------------------------------------------------------------
    # Align to common date range
    # -----------------------------------------------------------------------
    common_idx = closes[loaded[0]].index
    for t in loaded[1:]:
        common_idx = common_idx.intersection(closes[t].index)
    print(f"Common date range: {common_idx[0].date()} – {common_idx[-1].date()} ({len(common_idx)} weeks)\n")

    for t in loaded:
        closes[t] = closes[t].reindex(common_idx).ffill()
        ohlcvs[t] = ohlcvs[t].reindex(common_idx).ffill()

    # -----------------------------------------------------------------------
    # Screening
    # -----------------------------------------------------------------------
    print("Computing screening metrics…")
    screening = []
    for t in loaded:
        m = screening_metrics(ohlcvs[t], closes[t])
        m["ticker"] = t
        screening.append(m)
    screening.sort(key=lambda x: x["avg_weekly_range_pct"], reverse=True)

    print("\nTop 10 by avg weekly range:")
    for r in screening[:10]:
        print(f"  {r['ticker']:6s}  range={r['avg_weekly_range_pct']:5.2f}%  rv={r['rv52_ann_pct']:5.1f}%  1y={r['return_1y_pct']:+.1f}%")

    # Momentum candidates: top 15 by weekly range + rv
    mom_candidates = [s["ticker"] for s in screening[:15]]
    mr_candidates  = [s["ticker"] for s in screening[15:]]
    print(f"\nMomentum candidates: {mom_candidates}")
    print(f"MR candidates:       {mr_candidates}\n")

    # -----------------------------------------------------------------------
    # Momentum sweep
    # -----------------------------------------------------------------------
    MOM_THRESHOLDS = [1.0, 1.5, 2.0, 2.5, 3.0]
    print("Running momentum backtests…")
    momentum_results = []
    best_mom = {}  # ticker -> best equity curve
    for t in mom_candidates:
        for thr in MOM_THRESHOLDS:
            res = momentum_backtest(closes[t], thr)
            row = {"ticker": t, "threshold": thr,
                   "total_ret": res["total_ret"], "max_dd": res["max_dd"],
                   "sharpe": res["sharpe"], "trades_pw": res["trades_pw"]}
            momentum_results.append(row)
            if t not in best_mom or res["sharpe"] > best_mom[t]["sharpe"]:
                best_mom[t] = {**row, "equity": res["equity"]}
        print(f"  {t} done")

    # -----------------------------------------------------------------------
    # MR sweep
    # -----------------------------------------------------------------------
    MA_WINDOWS = [20, 50]
    MR_THRESHOLDS = [1.0, 2.0, 3.0, 5.0]
    print("\nRunning mean-reversion backtests…")
    mr_results = []
    best_mr = {}
    for t in mr_candidates:
        for ma in MA_WINDOWS:
            for thr in MR_THRESHOLDS:
                if len(closes[t]) <= ma:
                    continue
                res = mr_backtest(closes[t], ma, thr)
                row = {"ticker": t, "ma_window": ma, "threshold": thr,
                       "total_ret": res["total_ret"], "max_dd": res["max_dd"],
                       "sharpe": res["sharpe"], "trades_pw": res["trades_pw"]}
                mr_results.append(row)
                key = (t, ma)
                if key not in best_mr or res["sharpe"] > best_mr[key]["sharpe"]:
                    best_mr[key] = {**row, "equity": res["equity"]}
        print(f"  {t} done")

    # -----------------------------------------------------------------------
    # Correlation matrix (Ledoit-Wolf shrinkage)
    # -----------------------------------------------------------------------
    print("\nComputing correlation matrix…")
    from sklearn.covariance import LedoitWolf
    rets_matrix = pd.DataFrame({t: closes[t].pct_change().dropna() for t in loaded})
    rets_matrix = rets_matrix.dropna()
    lw = LedoitWolf()
    lw.fit(rets_matrix.values)
    cov = lw.covariance_
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    corr_df = pd.DataFrame(corr, index=rets_matrix.columns, columns=rets_matrix.columns)

    # -----------------------------------------------------------------------
    # Visualisations
    # -----------------------------------------------------------------------
    print("\nGenerating charts…")

    # 1. Screening metrics
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Screening Metrics — Real Weekly Data (2021–2026)", fontsize=13)
    tickers_sorted = [s["ticker"] for s in screening]
    ranges = [s["avg_weekly_range_pct"] for s in screening]
    rvols  = [s["rv52_ann_pct"] for s in screening]
    ret1ys = [s["return_1y_pct"] for s in screening]

    axes[0].barh(tickers_sorted[::-1], ranges[::-1], color="#4C72B0")
    axes[0].set_xlabel("Avg Weekly Range (%)")
    axes[0].set_title("Weekly Range Proxy")
    axes[0].axvline(2.0, color="red", ls="--", lw=0.8, label="2% threshold")
    axes[0].legend(fontsize=7)

    axes[1].barh(tickers_sorted[::-1], rvols[::-1], color="#DD8452")
    axes[1].set_xlabel("Annualised Vol (%)")
    axes[1].set_title("52-Week Realised Vol")

    colors3 = ["#2CA02C" if r >= 0 else "#D62728" for r in ret1ys[::-1]]
    axes[2].barh(tickers_sorted[::-1], ret1ys[::-1], color=colors3)
    axes[2].set_xlabel("1-Year Return (%)")
    axes[2].set_title("1-Year Return")
    axes[2].axvline(0, color="black", lw=0.5)

    plt.tight_layout()
    fig.savefig(OUT / "screening_metrics.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("  screening_metrics.png")

    # 2. Correlation heatmap
    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(corr_df.values, cmap="RdYlGn", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr_df.columns)))
    ax.set_yticks(range(len(corr_df.index)))
    ax.set_xticklabels(corr_df.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(corr_df.index, fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.03)
    ax.set_title("Ledoit-Wolf Shrinkage Correlation — Weekly Returns 2021–2026", fontsize=11)
    plt.tight_layout()
    fig.savefig(OUT / "correlation_heatmap.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("  correlation_heatmap.png")

    # 3. Equity curves — best momentum configs
    fig, ax = plt.subplots(figsize=(13, 6))
    top_mom = sorted(best_mom.values(), key=lambda x: x["sharpe"], reverse=True)[:5]
    for r in top_mom:
        eq = np.array(r["equity"])
        ax.plot(eq, label=f"{r['ticker']} thr={r['threshold']}% Sh={r['sharpe']:.2f}")
    ax.axhline(1.0, color="black", lw=0.7, ls="--")
    # Buy-and-hold SPY
    spy_bh = closes["SPY"] / closes["SPY"].iloc[0]
    ax.plot(spy_bh.values, color="gray", lw=1.5, ls=":", label="SPY B&H")
    ax.set_ylabel("Portfolio value ($1 start)")
    ax.set_title("Best Momentum Strategy Equity Curves — Real Data")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(OUT / "equity_curves.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("  equity_curves.png")

    # 4. Momentum sweep heatmap
    mom_df = pd.DataFrame(momentum_results)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Momentum Sweep — Sharpe by Ticker & Threshold", fontsize=12)
    for ax, metric, cmap in zip(axes, ["sharpe", "total_ret", "max_dd"],
                                  ["RdYlGn", "RdYlGn", "RdYlGn_r"]):
        pivot = mom_df.pivot(index="ticker", columns="threshold", values=metric)
        im = ax.imshow(pivot.values, cmap=cmap,
                       vmin=pivot.values.min(), vmax=pivot.values.max())
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{t:.1f}%" for t in pivot.columns], fontsize=8)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=8)
        ax.set_title(metric.replace("_", " ").title())
        plt.colorbar(im, ax=ax, fraction=0.04)
    plt.tight_layout()
    fig.savefig(OUT / "momentum_sweep.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("  momentum_sweep.png")

    # 5. MR sweep
    mr_df = pd.DataFrame(mr_results)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Mean-Reversion Sweep — Sharpe (MA-20 vs MA-50)", fontsize=12)
    for ax, ma in zip(axes, [20, 50]):
        subset = mr_df[mr_df["ma_window"] == ma]
        pivot = subset.pivot(index="ticker", columns="threshold", values="sharpe")
        im = ax.imshow(pivot.values, cmap="RdYlGn",
                       vmin=-1, vmax=max(2.5, pivot.values.max()))
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{t:.1f}%" for t in pivot.columns], fontsize=8)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=8)
        ax.set_title(f"MA-{ma} Sharpe Ratio")
        plt.colorbar(im, ax=ax, fraction=0.04)
    plt.tight_layout()
    fig.savefig(OUT / "mr_sweep.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("  mr_sweep.png")

    # -----------------------------------------------------------------------
    # Save research_data.json
    # -----------------------------------------------------------------------
    print("\nWriting research_data.json…")
    best_mom_sharpe = sorted(best_mom.values(), key=lambda x: x["sharpe"], reverse=True)[:5]
    best_mr_sharpe  = sorted(best_mr.values(), key=lambda x: x["sharpe"], reverse=True)[:5]

    # Best configs constrained to the 2–3% target threshold band
    TARGET_BAND = (2.0, 3.0)
    best_mom_targeted = sorted(
        [r for r in best_mom.values() if TARGET_BAND[0] <= r["threshold"] <= TARGET_BAND[1]],
        key=lambda x: x["sharpe"], reverse=True
    )[:3]
    recommended_threshold = best_mom_targeted[0]["threshold"] / 100 if best_mom_targeted else 0.02

    output = {
        "meta": {
            "data_type":      "real_market_data",
            "source":         "dotchev/top-stox (weekly) + whchien/ai-trader (TSM daily)",
            "date_range":     f"{common_idx[0].date()} – {common_idx[-1].date()}",
            "n_weeks":        len(common_idx),
            "generated":      str(datetime.utcnow()),
            "tickers_loaded": loaded,
        },
        "screening": screening,
        "momentum_candidates": mom_candidates,
        "mr_candidates": mr_candidates,
        "momentum_results": [
            {k: v for k, v in r.items() if k != "equity"}
            for r in momentum_results
        ],
        "mr_results": mr_results,
        "best_momentum_configs": [
            {k: v for k, v in r.items() if k != "equity"}
            for r in best_mom_sharpe
        ],
        "best_mr_configs": [
            {k: v for k, v in r.items() if k != "equity"}
            for r in best_mr_sharpe
        ],
        "recommended_harness_config": {
            "note": "Best momentum picks within 2–3% threshold band, weekly-data backtest 2022-2025",
            "tickers": [r["ticker"] for r in best_mom_targeted],
            "buy_threshold_pct": recommended_threshold,
        },
    }

    with open(OUT / "research_data.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    print("\n=== Summary ===")
    print(f"  Universe: {len(loaded)} tickers, {len(common_idx)} weeks of real data")
    print(f"  Date range: {common_idx[0].date()} – {common_idx[-1].date()}")
    print(f"\n  Top 5 momentum configs by Sharpe (all thresholds):")
    for r in best_mom_sharpe:
        print(f"    {r['ticker']:6s} thr={r['threshold']:.1f}%  Sh={r['sharpe']:.2f}  ret={r['total_ret']:+.1f}%  dd={r['max_dd']:.1f}%")
    print(f"\n  Best momentum configs in 2–3% target band:")
    for r in best_mom_targeted:
        print(f"    {r['ticker']:6s} thr={r['threshold']:.1f}%  Sh={r['sharpe']:.2f}  ret={r['total_ret']:+.1f}%  dd={r['max_dd']:.1f}%")
    print(f"\n  Top 5 MR configs by Sharpe:")
    for r in best_mr_sharpe:
        print(f"    {r['ticker']:6s} MA-{r['ma_window']:2d} thr={r['threshold']:.1f}%  Sh={r['sharpe']:.2f}  ret={r['total_ret']:+.1f}%  dd={r['max_dd']:.1f}%")
    cfg = output['recommended_harness_config']
    print(f"\n  Recommended harness config: tickers={cfg['tickers']}  buy_threshold_pct={cfg['buy_threshold_pct']}")
    print(f"\nOutputs in: {OUT.resolve()}")


if __name__ == "__main__":
    main()
