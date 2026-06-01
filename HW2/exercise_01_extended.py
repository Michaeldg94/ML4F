"""HW2 Exercise 1 (extended): wider sentiment-strategy exploration.

Reuses the exercise_01 engine and adds three signal families (z-score mean
reversion, rate-of-change momentum, stochastic oscillator), then sweeps all
seven families across ten tickers, both trade directions and both window
sizes under the same 1% cost. This is exploratory back-office work to check
the submitted rule is not a fluke; it is not part of the graded report.
Outputs land in output/ex01_ext_*.

Authors: Marco Canal, Michael Duarte Gonçalves, Olivia Peter.
"""

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

# Reuse the production engine: pricers, performance, rolling-test, strategy
# library, build_ev_catalogue, load_panel, GRIDS, make_signal. We add new
# strategy families and patch GRIDS.
import exercise_01 as ex1

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output" / "ex1" / "tables"  # shares the exercise-1 tables folder
OUT.mkdir(parents=True, exist_ok=True)

TICKERS = ["AAPL", "AMZN", "DB", "DIS", "FB",
           "GOOG", "HSBC", "JPM", "MSFT", "PFE"]
TCOST = ex1.TCOST
WINDOW_SIZES = ex1.WINDOW_SIZES
LONGSHORT = ex1.LONGSHORT_SETTINGS
EV_NAMES = ex1.EV_NAMES


# -----------------------------------------------------------------------------
# Three new strategy families
# -----------------------------------------------------------------------------

def strat_zscore_meanrev(ev, W, k, longshort):
    """Contrarian z-score: long when z = (ev - mean) / std falls below -k
    (oversold and expected to revert), exit when z rises above +k. Opposite
    direction from Bollinger breakout."""
    mu = ev.rolling(W, min_periods=W).mean()
    sd = ev.rolling(W, min_periods=W).std()
    z = (ev - mu) / sd.replace(0.0, np.nan)
    decisions = pd.Series("hold", index=ev.index, dtype=object)
    decisions[z < -k] = "long"
    decisions[z > +k] = "exit"
    return ex1._build_signal_from_decisions(decisions, longshort, ev.index)


def strat_roc_momentum(ev, L, threshold, longshort):
    """Rate-of-change momentum: long when ev rises by more than threshold
    over L days, exit when it falls by more than threshold. Computed in
    log space to make the threshold scale-invariant for positive evs."""
    # Add small constant so that ev = 0 days remain well-defined
    log_ev = np.log(ev.abs() + 1e-6)
    roc = log_ev - log_ev.shift(L)
    decisions = pd.Series("hold", index=ev.index, dtype=object)
    decisions[roc > +threshold] = "long"
    decisions[roc < -threshold] = "exit"
    return ex1._build_signal_from_decisions(decisions, longshort, ev.index)


def strat_stochastic(ev, W, D, longshort):
    """Stochastic oscillator: %K = 100 (ev - min) / (max - min) over W,
    %D = SMA of %K over D periods. Long on a %K-cross-above-%D event when
    %K is in the oversold zone (< 30), exit on the reverse cross in the
    overbought zone (> 70)."""
    rmin = ev.rolling(W, min_periods=W).min()
    rmax = ev.rolling(W, min_periods=W).max()
    rng = (rmax - rmin).replace(0.0, np.nan)
    K = 100.0 * (ev - rmin) / rng
    Dline = K.rolling(D, min_periods=D).mean()
    above = (K > Dline).astype(bool)
    above_prev = above.shift(1, fill_value=False)
    cross_up = above & (~above_prev) & (K < 30.0)
    cross_dn = (~above) & above_prev & (K > 70.0)
    decisions = pd.Series("hold", index=ev.index, dtype=object)
    decisions[cross_up] = "long"
    decisions[cross_dn] = "exit"
    return ex1._build_signal_from_decisions(decisions, longshort, ev.index)


# Extend the strategy library and grid
NEW_GRIDS = {
    "ZscoreMR": list(itertools.product([20, 40, 60], [1.0, 1.5, 2.0])),
    "RoCMomentum": list(itertools.product([5, 10, 20], [0.10, 0.20, 0.40])),
    "Stochastic": list(itertools.product([14, 21], [3, 5])),
}
ALL_GRIDS = {**ex1.GRIDS, **NEW_GRIDS}
NEW_STRATS = list(NEW_GRIDS.keys())
ALL_STRATS = ex1.STRATEGY_NAMES + NEW_STRATS


def make_signal_ext(strategy, params, ev, rvt, longshort):
    """Dispatch including the three new strategies."""
    if strategy in ex1.STRATEGY_NAMES:
        return ex1.make_signal(strategy, params, ev, rvt, longshort)
    if strategy == "ZscoreMR":
        W, k = params
        return strat_zscore_meanrev(ev, W, k, longshort)
    if strategy == "RoCMomentum":
        L, thr = params
        return strat_roc_momentum(ev, L, thr, longshort)
    if strategy == "Stochastic":
        W, D = params
        return strat_stochastic(ev, W, D, longshort)
    raise ValueError(strategy)


# -----------------------------------------------------------------------------
# Driver: tune per (ticker, strategy, ev, ls), then evaluate rolling
# -----------------------------------------------------------------------------

def tune(price, ev, rvt, strategy, longshort, min_trades=5):
    """Same tuning protocol as exercise_01: max whole-period excess with
    min trades constraint."""
    best = (None, -np.inf, None)
    for params in ALL_GRIDS[strategy]:
        try:
            sig = make_signal_ext(strategy, params, ev, rvt, longshort)
        except Exception:
            continue
        _, perf = ex1.test_strategy_whole(price, sig, tcost=TCOST)
        nt = perf.loc["nTrades", "Me"]
        if nt < min_trades:
            continue
        excess = perf.loc["CumRet", "Me"] - perf.loc["CumRet", "BH"]
        sharpe = perf.loc["Sharpe", "Me"]
        score = excess + 1e-6 * (sharpe if np.isfinite(sharpe) else 0)
        if score > best[1]:
            best = (params, score, perf)
    return best


def evaluate_rolling(price, ev, rvt, strategy, params, ls, w):
    sig = make_signal_ext(strategy, params, ev, rvt, ls)
    rw, summary = ex1.rolling_test_strategy(price, sig, w_size=w, tcost=TCOST)
    summary.attrs["n_windows"] = summary.attrs["n_windows"]
    return summary


def run_ticker(ticker):
    panel = ex1.load_panel(ticker)
    price = panel["Adj Close"].rename(ticker)
    ev_dict = ex1.build_ev_catalogue(panel)
    rvt = ev_dict["RVT"]

    whole_rows, rolling_rows = [], []
    for strategy in ALL_STRATS:
        for ev_name in EV_NAMES:
            if strategy == "SlopeRVT" and ev_name == "RVT":
                continue
            ev = ev_dict[ev_name]
            for ls in LONGSHORT:
                params, _, perf = tune(price, ev, rvt, strategy, ls)
                if params is None:
                    continue
                excess_whole = (perf.loc["CumRet", "Me"]
                                - perf.loc["CumRet", "BH"])
                whole_rows.append({
                    "ticker": ticker, "strategy": strategy, "ev": ev_name,
                    "longshort": ls, "params": str(params),
                    "Me_CumRet": perf.loc["CumRet", "Me"],
                    "BH_CumRet": perf.loc["CumRet", "BH"],
                    "excess": excess_whole,
                    "Me_Sharpe": perf.loc["Sharpe", "Me"],
                    "Me_nTrades": perf.loc["nTrades", "Me"],
                })
                for w in WINDOW_SIZES:
                    try:
                        rsum = evaluate_rolling(price, ev, rvt, strategy,
                                                params, ls, w)
                    except ValueError:
                        continue
                    rolling_rows.append({
                        "ticker": ticker, "strategy": strategy, "ev": ev_name,
                        "longshort": ls, "w_size": w, "params": str(params),
                        "n_windows": rsum.attrs["n_windows"],
                        "Me_CumRet_mean": rsum.loc["Me_CumRet", "mean"],
                        "BH_CumRet_mean": rsum.loc["BH_CumRet", "mean"],
                        "excess_mean": (rsum.loc["Me_CumRet", "mean"]
                                        - rsum.loc["BH_CumRet", "mean"]),
                        "Me_Sharpe_mean": rsum.loc["Me_Sharpe", "mean"],
                        "Me_nTrades_mean": rsum.loc["Me_nTrades", "mean"],
                        "Me_MaxDD_mean": rsum.loc["Me_MaxDD", "mean"],
                    })
    return pd.DataFrame(whole_rows), pd.DataFrame(rolling_rows)


def main():
    all_whole, all_rolling = [], []
    for ticker in TICKERS:
        print(f"\n=== {ticker} ===")
        w_df, r_df = run_ticker(ticker)
        n_pos = (w_df["excess"] > 0).sum()
        n_pos_r = (r_df[(r_df["w_size"] == 254) &
                        (r_df["longshort"] == -1)]["excess_mean"] > 0).sum()
        print(f"  {ticker}: whole-period positive-excess configs = {n_pos}, "
              f"rolling 254 long-short positive = {n_pos_r}")
        all_whole.append(w_df)
        all_rolling.append(r_df)

    big_whole = pd.concat(all_whole, ignore_index=True)
    big_rolling = pd.concat(all_rolling, ignore_index=True)
    big_whole.to_csv(OUT / "07_ext_whole.csv", index=False)
    big_rolling.to_csv(OUT / "08_ext_rolling.csv", index=False)

    # Top per (ticker, w_size, longshort): the highest mean excess that
    # also has at least one trade per window on average
    elig = big_rolling[big_rolling["Me_nTrades_mean"] >= 1.0].copy()
    top = (elig.sort_values("excess_mean", ascending=False)
           .groupby(["ticker", "w_size", "longshort"]).head(1)
           .reset_index(drop=True))
    top.to_csv(OUT / "09_ext_top.csv", index=False)

    # Headline summary: best long-short configuration per ticker at w=254
    headline = top[(top["w_size"] == 254) & (top["longshort"] == -1)].copy()
    headline = headline.sort_values("excess_mean", ascending=False)
    print("\n=== Headline: best long-short w=254 per ticker ===")
    print(headline[["ticker", "strategy", "ev", "params",
                    "excess_mean", "Me_Sharpe_mean",
                    "Me_nTrades_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
