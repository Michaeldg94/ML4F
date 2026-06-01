"""HW2 Exercise 1: Backtesting sentiment-based trading strategies.

Designs one signal that is neither MA-crossover nor RSI (a sentiment slope
confirmed by a news-volume z-score), tunes it on DB and HSBC and evaluates
long-only and long-short variants over 254- and 127-day rolling windows,
always net of the 1% trading cost. Three further sentiment families
(Bollinger, MACD, rolling-percentile) are tuned alongside it for context.
All outputs land in output/ex1/.

Authors: Marco Canal, Michael Duarte Gonçalves, Olivia Peter.
"""

import ast
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "dataSent"
OUT_DIR = ROOT / "output" / "ex1"        # exercise-1 outputs live here
FIG_DIR = OUT_DIR / "figures"            # plots, numbered within this folder
TAB_DIR = OUT_DIR / "tables"             # CSVs, numbered within this folder
for _d in (FIG_DIR, TAB_DIR):
    _d.mkdir(parents=True, exist_ok=True)

TICKERS = ["DB", "HSBC"]               # our two selected international banks
TCOST = 0.01                           # 1% trading cost, fixed by the brief
WINDOW_SIZES = [254, 127]              # one year and six months
LONGSHORT_SETTINGS = [0, -1]           # exit weight: 0 = long-only, -1 = short

EV_NAMES = ["positiveP", "negativeP", "BULL", "BEAR", "BBr", "PNlog", "RVT"]
STRATEGY_NAMES = ["BollingerSent", "MACDSent", "PctileSent", "SlopeRVT"]

# --- Performance and backtest engines ---------------------------------------
# Mirrors PyLab6_TradingStrategywCost, with one fix: its drawdown-length
# formula returned zero everywhere, so we use the textbook peak-to-trough one.


def _max_drawdown(cum):
    """Worst peak-to-trough loss of a cumulative path, and its longest run."""
    if len(cum) == 0:
        return np.nan, 0
    underwater = cum - np.maximum.accumulate(cum)
    max_dd = float(underwater.min())
    # longest underwater run, via the pad-and-diff islands trick
    under = (underwater < 0).astype(int)
    edges = np.diff(np.concatenate(([0], under, [0])))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    max_len = int((ends - starts).max()) if len(starts) else 0
    return max_dd, max_len


def performance(x, ntrades=1, cost=0.0):
    """Eight performance metrics for a daily-return series, net of the cost.

    Sharpe follows the lab convention (annualised return over daily
    volatility, times sqrt(252)), so it runs large.
    """
    x = pd.Series(x).dropna()
    cum = np.cumsum(x.values) - ntrades * cost
    # floor at -0.999 so the annualisation power never roots a negative base
    cum_final = max(float(cum[-1]), -0.999)
    ann_ret = np.power(cum_final + 1, 252.0 / len(x)) - 1
    sd = np.std(x.values, ddof=1) if len(x) > 1 else np.nan
    ann_vol = sd * np.sqrt(252)
    sharpe_lab = ann_ret / sd * np.sqrt(252) if sd and sd > 0 else np.nan
    nz = np.count_nonzero(x.values)
    win_pct = np.count_nonzero(x.values > 0) / nz if nz > 0 else np.nan
    max_dd, max_len = _max_drawdown(cum)

    return pd.Series(
        [cum[-1], ann_ret, sharpe_lab, win_pct,
         ann_vol, max_dd, max_len, ntrades],
        index=["CumRet", "AnnRet", "Sharpe", "WinPct",
               "AnnVol", "MaxDD", "MaxDDLen", "nTrades"],
        name=x.name,
    )


def count_trades(sig):
    """Count position changes in the signal (nonzero first differences)."""
    s = pd.Series(sig).fillna(0)
    return int((s.diff().fillna(0) != 0).sum())


def test_strategy_whole(price, sig, tcost=TCOST):
    """Whole-period backtest of an already-lagged signal.

    Returns the (Me, BH) daily returns and a performance table for each.
    """
    bh = (price / price.shift(1) - 1).rename("BH")
    me = (bh * sig).rename("Me")
    ret = pd.concat([me, bh], axis=1).dropna()
    nt = count_trades(sig.reindex(ret.index))
    perf = pd.concat(
        [performance(ret["Me"], ntrades=nt, cost=tcost),
         performance(ret["BH"], ntrades=1, cost=0.0)],
        axis=1,
    )
    perf.columns = ["Me", "BH"]
    return ret, perf


def rolling_test_strategy(price, sig, w_size=254, tcost=TCOST):
    """Rolling-window backtest.

    Returns the per-window metrics and their mean and std across windows,
    with the cost charged per window from the trades inside it.
    """
    bh = (price / price.shift(1) - 1).rename("BH")
    me = (bh * sig).rename("Me")
    full = pd.concat([me, bh, sig.rename("sig")], axis=1).dropna()
    n_windows = int(full.shape[0] - w_size)
    if n_windows < 1:
        raise ValueError("Window size larger than available history")

    rows = []
    for i in range(n_windows):
        sub = full.iloc[i:i + w_size]
        nt = count_trades(sub["sig"])
        pm = performance(sub["Me"].rename("Me"), ntrades=nt, cost=tcost)
        pb = performance(sub["BH"].rename("BH"), ntrades=1, cost=0.0)
        row = {f"Me_{k}": pm[k] for k in pm.index}
        row.update({f"BH_{k}": pb[k] for k in pb.index})
        rows.append(row)
    rw = pd.DataFrame(rows)
    summary = pd.DataFrame({
        "mean": rw.mean(numeric_only=True),
        "std": rw.std(numeric_only=True),
    })
    summary.attrs["n_windows"] = n_windows
    return rw, summary


# --- Sentiment indicator construction ---------------------------------------

def load_panel(ticker):
    """Load a dataSent CSV as a float DataFrame indexed by Date."""
    fp = DATA_DIR / f"{ticker}.csv"
    raw = pd.read_csv(fp)
    raw = raw.drop(columns=[c for c in ["Unnamed: 0", "downloadedDate"]
                            if c in raw.columns])
    raw["Date"] = pd.to_datetime(raw["Date"])
    raw = raw.set_index("Date").sort_index()
    raw = raw.astype(float)
    return raw


def build_ev_catalogue(panel):
    """Build the seven explanatory-variable series, following the assignment
    formulas (BULL/BEAR sums, BBr ratio, PNlog log-ratio, RVT as-is)."""
    pos = panel["positivePartscr"].rename("positiveP")
    neg = panel["negativePartscr"].rename("negativeP")
    cert = panel["certaintyPartscr"]
    unc = panel["uncertaintyPartscr"]
    findn = panel["findownPartscr"]
    finup = panel["finupPartscr"]

    bull = (pos + cert + finup).rename("BULL")
    bear = (neg + unc + findn).rename("BEAR")
    bb_sum = (bull + bear).replace(0.0, np.nan)
    bbr = (100.0 * bull / bb_sum).rename("BBr")
    # forward fill the rare BULL=BEAR=0 days (no back fill, to avoid leakage)
    bbr = bbr.ffill()
    pnlog = (0.5 * np.log((pos + 1.0) / (neg + 1.0))).rename("PNlog")
    rvt = panel["RVT"].rename("RVT")
    return {"positiveP": pos, "negativeP": neg, "BULL": bull, "BEAR": bear,
            "BBr": bbr, "PNlog": pnlog, "RVT": rvt}


# --- Strategy library --------------------------------------------------------
# Each generator returns a {-1, 0, 1} signal, lagged one day and held between
# triggers so tiny indicator wiggles do not whipsaw the position.

def _build_signal_from_decisions(decisions, longshort, index):
    """Forward-fill discrete decisions into a lagged {-1, 0, 1} signal.

    Days before the first trigger are flat. longshort is the exit weight."""
    sig = pd.Series(np.nan, index=index)
    sig[decisions == "long"] = 1
    sig[decisions == "exit"] = longshort
    sig = sig.ffill().fillna(0.0).shift(1)
    return sig


def strat_bollinger(ev, W, k, longshort):
    """Bollinger bands on the sentiment ev. Long when ev breaks above
    mean + k*std, exit when ev breaks below mean - k*std, hold otherwise."""
    mu = ev.rolling(W, min_periods=W).mean()
    sd = ev.rolling(W, min_periods=W).std()
    upper, lower = mu + k * sd, mu - k * sd
    decisions = pd.Series("hold", index=ev.index, dtype=object)
    decisions[ev > upper] = "long"
    decisions[ev < lower] = "exit"
    return _build_signal_from_decisions(decisions, longshort, ev.index)


def strat_macd(ev, fast, slow, sig_span, longshort):
    """MACD on the sentiment ev. Long when MACD crosses above its signal
    line, exit when it crosses below. Crossings are detected by sign of the
    difference series and emitted as decisions only on the crossing day."""
    if fast >= slow:
        raise ValueError("fast must be < slow")
    macd = ev.ewm(span=fast, adjust=False).mean() \
        - ev.ewm(span=slow, adjust=False).mean()
    sig_line = macd.ewm(span=sig_span, adjust=False).mean()
    above = (macd > sig_line).astype(bool)
    above_prev = above.shift(1, fill_value=False)
    cross_up = above & (~above_prev)
    cross_dn = (~above) & above_prev
    decisions = pd.Series("hold", index=ev.index, dtype=object)
    decisions[cross_up] = "long"
    decisions[cross_dn] = "exit"
    return _build_signal_from_decisions(decisions, longshort, ev.index)


def strat_percentile(ev, W, q_low, q_high, longshort):
    """Rolling-percentile breakout. Long when ev exceeds its rolling q_high
    quantile, exit when it falls below the rolling q_low quantile."""
    hi = ev.rolling(W, min_periods=W).quantile(q_high)
    lo = ev.rolling(W, min_periods=W).quantile(q_low)
    decisions = pd.Series("hold", index=ev.index, dtype=object)
    decisions[ev > hi] = "long"
    decisions[ev < lo] = "exit"
    return _build_signal_from_decisions(decisions, longshort, ev.index)


def strat_slope_rvt(ev, rvt, L, W, rvt_thr, longshort):
    """Sentiment slope confirmed by news volume. The slope is the L-day
    change in ev. RVT is z-scored over W days and used as a confirmation
    filter: only trigger when |rvt_z| > rvt_thr. The sign of the slope
    selects long versus exit."""
    slope = (ev - ev.shift(L))
    rvt_mu = rvt.rolling(W, min_periods=W).mean()
    rvt_sd = rvt.rolling(W, min_periods=W).std()
    rvt_z = (rvt - rvt_mu) / rvt_sd.replace(0.0, np.nan)
    confirmed = rvt_z.abs() > rvt_thr
    decisions = pd.Series("hold", index=ev.index, dtype=object)
    decisions[(slope > 0) & confirmed] = "long"
    decisions[(slope < 0) & confirmed] = "exit"
    return _build_signal_from_decisions(decisions, longshort, ev.index)


# Hyperparameter grids per strategy
GRIDS = {
    "BollingerSent": list(itertools.product([20, 40, 60], [1.0, 1.5, 2.0])),
    "MACDSent": [(5, 20, 9), (8, 21, 9), (12, 26, 9), (5, 35, 5)],
    "PctileSent": list(itertools.product(
        [20, 40, 60], [(0.20, 0.80), (0.30, 0.70), (0.10, 0.90)])),
    "SlopeRVT": list(itertools.product(
        [5, 10, 20], [20, 40], [0.0, 0.5, 1.0])),
}


def make_signal(strategy, params, ev, rvt, longshort):
    """Dispatch a strategy name and parameter tuple to the right generator."""
    if strategy == "BollingerSent":
        W, k = params
        return strat_bollinger(ev, W, k, longshort)
    if strategy == "MACDSent":
        f, s, sg = params
        return strat_macd(ev, f, s, sg, longshort)
    if strategy == "PctileSent":
        W, (qL, qH) = params
        return strat_percentile(ev, W, qL, qH, longshort)
    if strategy == "SlopeRVT":
        L, W, thr = params
        return strat_slope_rvt(ev, rvt, L, W, thr, longshort)
    raise ValueError(f"unknown strategy {strategy}")


# --- Whole-period tuning -----------------------------------------------------

MIN_TRADES_WHOLE = 5  # discard degenerate "almost never trade" configurations


def tune_strategy(price, ev, rvt, strategy, longshort, tcost=TCOST,
                  min_trades=MIN_TRADES_WHOLE):
    """Grid-search a strategy's hyperparameters for the best whole-period
    excess (Me - BH), subject to the minimum-trades floor. Returns the grid
    trace and the best (params, score, perf), or (None, -inf, None)."""
    rows = []
    best = (None, -np.inf, None)
    for params in GRIDS[strategy]:
        try:
            sig = make_signal(strategy, params, ev, rvt, longshort)
        except Exception as e:
            rows.append({"strategy": strategy, "params": str(params),
                         "ev": ev.name, "longshort": longshort,
                         "excess_CumRet": np.nan, "Sharpe_Me": np.nan,
                         "nTrades": np.nan, "error": str(e)})
            continue
        _, perf = test_strategy_whole(price, sig, tcost=tcost)
        excess = perf.loc["CumRet", "Me"] - perf.loc["CumRet", "BH"]
        ntrades = perf.loc["nTrades", "Me"]
        rows.append({"strategy": strategy, "params": str(params),
                     "ev": ev.name, "longshort": longshort,
                     "excess_CumRet": excess,
                     "Sharpe_Me": perf.loc["Sharpe", "Me"],
                     "nTrades": ntrades})
        if ntrades < min_trades:
            continue
        # primary key: excess, tie-break on Sharpe of Me
        sharpe_me = perf.loc["Sharpe", "Me"]
        score = excess + 1e-6 * (sharpe_me if np.isfinite(sharpe_me) else 0)
        if score > best[1]:
            best = (params, score, perf)
    return pd.DataFrame(rows), best


# --- Rolling-window evaluation of tuned configurations -----------------------

def evaluate_rolling(price, ev, rvt, strategy, params, longshort,
                     w_size=254, tcost=TCOST):
    """Build the signal with tuned params and run the rolling-window backtest.
    Returns the per-window DataFrame, the averaged summary, and the signal."""
    sig = make_signal(strategy, params, ev, rvt, longshort)
    rw, summary = rolling_test_strategy(price, sig, w_size=w_size, tcost=tcost)
    return rw, summary, sig


# --- Driver ------------------------------------------------------------------

def run_ticker(ticker):
    """Full pipeline for one ticker. Returns four DataFrames covering the
    tuning trace, whole-period performance, rolling-window summary, and the
    best per (strategy, ev, w_size, longshort) configuration."""
    print(f"\n{'=' * 70}\nProcessing ticker {ticker}\n{'=' * 70}")
    panel = load_panel(ticker)
    price = panel["Adj Close"].rename(ticker)
    ev_dict = build_ev_catalogue(panel)
    rvt = ev_dict["RVT"]

    tuning_rows = []
    whole_rows = []
    rolling_rows = []

    for strategy in STRATEGY_NAMES:
        for ev_name in EV_NAMES:
            # SlopeRVT uses RVT internally; do not also drive it by RVT-as-ev
            if strategy == "SlopeRVT" and ev_name == "RVT":
                continue
            ev = ev_dict[ev_name]
            for ls in LONGSHORT_SETTINGS:
                # 1) tune on whole period
                trace, best = tune_strategy(price, ev, rvt, strategy, ls)
                trace["ticker"] = ticker
                tuning_rows.append(trace)
                if best[0] is None:
                    continue
                params, score, perf = best
                excess = perf.loc["CumRet", "Me"] - perf.loc["CumRet", "BH"]
                whole_rows.append({
                    "ticker": ticker, "strategy": strategy, "ev": ev_name,
                    "longshort": ls, "params": str(params),
                    "Me_CumRet": perf.loc["CumRet", "Me"],
                    "BH_CumRet": perf.loc["CumRet", "BH"],
                    "excess": excess,
                    "Me_Sharpe": perf.loc["Sharpe", "Me"],
                    "Me_AnnRet": perf.loc["AnnRet", "Me"],
                    "Me_AnnVol": perf.loc["AnnVol", "Me"],
                    "Me_MaxDD": perf.loc["MaxDD", "Me"],
                    "Me_MaxDDLen": perf.loc["MaxDDLen", "Me"],
                    "Me_WinPct": perf.loc["WinPct", "Me"],
                    "Me_nTrades": perf.loc["nTrades", "Me"],
                })
                # 2) rolling-window evaluation for both window sizes
                for w in WINDOW_SIZES:
                    try:
                        rw_w, rsum, _ = evaluate_rolling(
                            price, ev, rvt, strategy, params, ls, w_size=w)
                    except ValueError:
                        continue
                    # True per-window excess (Me - BH). Its standard deviation
                    # captures the Me/BH covariance, not a quadrature sum
                    # of the two separate standard deviations.
                    excess_per_window = rw_w["Me_CumRet"] - rw_w["BH_CumRet"]
                    rolling_rows.append({
                        "ticker": ticker, "strategy": strategy, "ev": ev_name,
                        "longshort": ls, "w_size": w, "params": str(params),
                        "n_windows": rsum.attrs["n_windows"],
                        "Me_CumRet_mean": rsum.loc["Me_CumRet", "mean"],
                        "BH_CumRet_mean": rsum.loc["BH_CumRet", "mean"],
                        "excess_mean": excess_per_window.mean(),
                        "excess_std": excess_per_window.std(),
                        "excess_win": (excess_per_window > 0).mean(),
                        "Me_Sharpe_mean": rsum.loc["Me_Sharpe", "mean"],
                        "Me_AnnRet_mean": rsum.loc["Me_AnnRet", "mean"],
                        "Me_AnnVol_mean": rsum.loc["Me_AnnVol", "mean"],
                        "Me_MaxDD_mean": rsum.loc["Me_MaxDD", "mean"],
                        "Me_MaxDDLen_mean": rsum.loc["Me_MaxDDLen", "mean"],
                        "Me_WinPct_mean": rsum.loc["Me_WinPct", "mean"],
                        "Me_nTrades_mean": rsum.loc["Me_nTrades", "mean"],
                    })
    tuning_df = pd.concat(tuning_rows, ignore_index=True) if tuning_rows \
        else pd.DataFrame()
    whole_df = pd.DataFrame(whole_rows)
    rolling_df = pd.DataFrame(rolling_rows)
    return tuning_df, whole_df, rolling_df, price, ev_dict, rvt


def pick_best_rolling(rolling_df, ticker, w_size, longshort):
    """Identify the best rolling-window configuration for a given ticker,
    window size, and longshort by mean excess CumRet."""
    sub = rolling_df[(rolling_df["ticker"] == ticker)
                     & (rolling_df["w_size"] == w_size)
                     & (rolling_df["longshort"] == longshort)]
    if sub.empty:
        return None
    return sub.sort_values("excess_mean", ascending=False).iloc[0].to_dict()


def plot_diagnostics(ticker, price, ev_dict, rvt, best_cfg):
    """Save the chosen config's three figures (signal, equity curve, per-window
    excess distribution) and the per-window CSV."""
    strategy = best_cfg["strategy"]
    ev_name = best_cfg["ev"]
    ls = int(best_cfg["longshort"])
    raw_params = best_cfg["params"]
    params = ast.literal_eval(raw_params) if isinstance(raw_params, str) \
        else raw_params
    ev = ev_dict[ev_name]
    sig = make_signal(strategy, params, ev, rvt, ls)
    ret, _ = test_strategy_whole(price, sig, tcost=TCOST)

    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    ax[0].plot(price.index, price, color="black", linewidth=0.8, label=ticker)
    ax2 = ax[0].twinx()
    ax2.plot(ev.index, ev, color="tab:blue", linewidth=0.6, alpha=0.7,
             label=ev_name)
    ax[0].set_ylabel(f"{ticker} adj close")
    ax2.set_ylabel(ev_name)
    ax[0].legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax[0].set_title(f"{ticker}: chosen strategy {strategy} on {ev_name}"
                    f" with params {params}, longshort={ls}")
    ax[1].plot(sig.index, sig, color="tab:orange", linewidth=0.8)
    ax[1].set_ylabel("signal")
    ax[1].set_yticks([-1, 0, 1])
    ax[1].set_xlabel("date")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"03_signals_{ticker}.pdf", bbox_inches="tight")
    plt.close(fig)

    sig_aligned = sig.reindex(ret.index)
    cum_trades = (sig_aligned.diff().fillna(0) != 0).cumsum()
    me_curve = np.cumsum(ret["Me"]) - TCOST * cum_trades
    bh_curve = np.cumsum(ret["BH"])
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(ret.index, me_curve,
            label="Strategy (Me) cum returns, after 1% cost",
            color="tab:orange")
    ax.plot(ret.index, bh_curve,
            label="Buy and hold (BH) cum returns", color="black")
    ax.set_title(f"{ticker}: equity curve, strategy vs buy and hold "
                 f"(1% cost applied progressively)")
    ax.set_xlabel("date")
    ax.set_ylabel("cumulative return")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"01_equity_{ticker}.pdf", bbox_inches="tight")
    plt.close(fig)

    # Per-window distribution at w=254
    rw, _ = rolling_test_strategy(price, sig, w_size=254, tcost=TCOST)
    rw["excess"] = rw["Me_CumRet"] - rw["BH_CumRet"]
    rw.to_csv(TAB_DIR / f"03_rolling_windows_{ticker}.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(rw["excess"], bins=40, color="tab:blue", alpha=0.7,
            edgecolor="black", linewidth=0.4)
    ax.axvline(0.0, color="black", linewidth=1)
    ax.axvline(rw["excess"].mean(), color="tab:red", linewidth=1.2,
               linestyle="--",
               label=f"mean = {rw['excess'].mean():.3f}")
    win_pct = (rw["excess"] > 0).mean()
    ax.set_title(f"{ticker}: per-window excess Me - BH, w = 254."
                 f" Strategy beats BH in {win_pct:.1%} of windows")
    ax.set_xlabel("excess cumulative return")
    ax.set_ylabel("number of rolling windows")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"02_rolling_dist_{ticker}.pdf",
                bbox_inches="tight")
    plt.close(fig)


# --- Walk-forward out-of-sample validation -----------------------------------

def walk_forward(price, ev_dict, rvt, train_len=254, test_len=63):
    """Tune on each training window, then score the tuned rule on the next,
    unseen test window, and roll forward. Genuinely out-of-sample: each test
    window's parameters are chosen without seeing it."""
    bh = (price / price.shift(1) - 1).rename("BH")
    candidates = []
    for strategy in STRATEGY_NAMES:
        for ev_name in EV_NAMES:
            if strategy == "SlopeRVT" and ev_name == "RVT":
                continue
            for ls in LONGSHORT_SETTINGS:
                for params in GRIDS[strategy]:
                    sig = make_signal(strategy, params, ev_dict[ev_name],
                                      rvt, ls)
                    candidates.append(((strategy, ev_name, ls, params), sig))

    def window_excess(sig, idx):
        me = (bh * sig).reindex(idx).dropna()
        bh_w = bh.reindex(idx).dropna()
        nt = count_trades(sig.reindex(idx))
        pm = performance(me.rename("Me"), ntrades=nt, cost=TCOST)
        pb = performance(bh_w.rename("BH"), ntrades=1, cost=0.0)
        return pm["CumRet"] - pb["CumRet"], int(nt)

    rows = []
    idx = price.index
    start = 0
    while start + train_len + test_len <= len(idx):
        tr_idx = idx[start:start + train_len]
        te_idx = idx[start + train_len:start + train_len + test_len]
        best = (None, -np.inf, None)
        for label, sig in candidates:
            exc_tr, nt_tr = window_excess(sig, tr_idx)
            if nt_tr >= 3 and exc_tr > best[1]:
                best = (label, exc_tr, sig)
        exc_te, nt_te = window_excess(best[2], te_idx)
        rows.append({"test_start": str(te_idx[0].date()),
                     "test_end": str(te_idx[-1].date()),
                     "chosen": str(best[0]), "train_excess": best[1],
                     "test_excess": exc_te, "test_nTrades": nt_te})
        start += test_len
    return pd.DataFrame(rows)


def main():
    all_tuning, all_whole, all_rolling = [], [], []
    cache = {}
    for ticker in TICKERS:
        (tuning_df, whole_df, rolling_df,
         price, ev_dict, rvt) = run_ticker(ticker)
        tuning_df.to_csv(TAB_DIR / f"04_tuning_{ticker}.csv", index=False)
        whole_df.to_csv(TAB_DIR / f"01_whole_{ticker}.csv", index=False)
        rolling_df.to_csv(TAB_DIR / f"02_rolling_{ticker}.csv", index=False)
        all_tuning.append(tuning_df)
        all_whole.append(whole_df)
        all_rolling.append(rolling_df)
        cache[ticker] = (price, ev_dict, rvt)
        print(f"  saved whole/rolling/tuning CSVs for {ticker}")
        print(f"  whole-period configs with positive excess: "
              f"{(whole_df['excess'] > 0).sum()} of {len(whole_df)}")

    big_whole = pd.concat(all_whole, ignore_index=True)
    big_rolling = pd.concat(all_rolling, ignore_index=True)

    # Final "successful" summary: configs with positive mean excess in rolling
    successful = big_rolling[big_rolling["excess_mean"] > 0].copy()
    successful = successful.sort_values(
        ["ticker", "w_size", "longshort", "excess_mean"],
        ascending=[True, True, True, False]).reset_index(drop=True)
    successful.to_csv(TAB_DIR / "05_summary.csv", index=False)
    print(f"\nRolling configs with positive mean excess: {len(successful)}")
    print("Top configurations per (ticker, w_size, longshort):")
    top = (successful.groupby(["ticker", "w_size", "longshort"])
                     .head(1)
                     .reset_index(drop=True))
    print(top[["ticker", "w_size", "longshort", "strategy", "ev",
               "params", "excess_mean", "Me_Sharpe_mean",
               "Me_nTrades_mean"]].to_string(index=False))

    # Plot the best long-short, w=254 config per ticker (most mean excess,
    # at least one trade per window on average).
    for ticker in TICKERS:
        sub = big_rolling[(big_rolling["ticker"] == ticker)
                          & (big_rolling["w_size"] == 254)
                          & (big_rolling["longshort"] == -1)
                          & (big_rolling["Me_nTrades_mean"] >= 1.0)]
        if sub.empty:
            print(f"  no eligible (254, -1) configuration for {ticker}")
            continue
        cfg = sub.sort_values("excess_mean", ascending=False).iloc[0].to_dict()
        price, ev_dict, rvt = cache[ticker]
        plot_diagnostics(ticker, price, ev_dict, rvt, cfg)
        sign = "+" if cfg["excess_mean"] > 0 else ""
        print(f"  saved diagnostics for {ticker}: "
              f"{cfg['strategy']}/{cfg['ev']}/{cfg['params']} "
              f"excess_mean={sign}{cfg['excess_mean']:.4f}")

    # Walk-forward out-of-sample check per ticker.
    wf_all = []
    for ticker in TICKERS:
        price, ev_dict, rvt = cache[ticker]
        wf = walk_forward(price, ev_dict, rvt)
        wf.insert(0, "ticker", ticker)
        wf_all.append(wf)
        print(f"  {ticker} walk-forward: mean test excess "
              f"{wf['test_excess'].mean():+.3f}, positive in "
              f"{(wf['test_excess'] > 0).mean():.0%} of {len(wf)} windows")
    pd.concat(wf_all, ignore_index=True).to_csv(
        TAB_DIR / "06_walkforward.csv", index=False)


if __name__ == "__main__":
    main()
