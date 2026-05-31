# =============================================================================
# exercise_01.py
# HW2 ex 1: Backtesting trading strategies on sentiment indicators
#
# Team:    Marco Canal, Michael Duarte Goncalves, Olivia Peter
# Date:    2026-05-22
#
# PURPOSE:
#   Design and backtest ONE trading strategy different from MAcross and RSI
#   that uses the sentimental indicators
#   ev in {positiveP, negativeP, BULL, BEAR, BBr, PNlog} and RVT for two
#   selected stocks (DB and HSBC) over the period covered by the dataSent
#   files. The strategy is built and evaluated on both long-only and
#   long-short configurations using rolling-window analysis with window
#   sizes 254 (one year) and 127 (six months). Trading cost is fixed at
#   1 percent. We explore four candidate sentiment-driven strategies
#   (Bollinger bands, MACD, rolling-percentile breakout, sentiment-slope
#   confirmed by news volume), tune their hyperparameters on the whole
#   period, run the full rolling-window battery on each, and identify the
#   configurations whose mean excess return Me minus BH is positive.
#
# INPUTS:
#   dataSent/DB.csv
#   dataSent/HSBC.csv
#
# OUTPUTS:
#   output/ex01_whole_<TICKER>.csv         (whole-period performance per strategy)
#   output/ex01_rolling_<TICKER>.csv       (rolling-window mean/std performance)
#   output/ex01_tuning_<TICKER>.csv        (grid-search trace per strategy)
#   output/ex01_rolling_windows_<TICKER>.csv (per-window raw data for the chosen config)
#   output/ex01_summary.csv                (final summary of successful configs)
#   output/ex01_signals_<TICKER>.pdf       (chosen strategy signal vs price)
#   output/ex01_equity_<TICKER>.pdf        (chosen strategy equity vs BH)
#   output/ex01_rolling_dist_<TICKER>.pdf  (per-window excess distribution at w=254)
#
# STRUCTURE:
#   Part 1: Imports, paths, constants
#   Part 2: Performance and backtest engines (whole period and rolling window)
#   Part 3: Sentiment indicator construction
#   Part 4: Strategy library (four candidate signal generators)
#   Part 5: Tuning loop on the whole period
#   Part 6: Rolling-window evaluation of tuned configurations
#   Part 7: Reporting (summary CSV, figures)
# =============================================================================

# =============================================================================
# Part 1: Imports, paths, constants
# =============================================================================

import ast                             # safe literal parsing for params strings
import itertools                       # cartesian products for grid search
from pathlib import Path               # path handling without absolute paths

import numpy as np                     # numerical arrays
import pandas as pd                    # tabular data
import matplotlib.pyplot as plt        # plotting

SEED = 42                              # reproducibility for any random op
np.random.seed(SEED)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "dataSent"
OUT_DIR = ROOT / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TICKERS = ["DB", "HSBC"]               # our two selected stocks
TCOST = 0.01                           # 1 percent trading cost as required
WINDOW_SIZES = [254, 127]              # one year and six months of daily data
LONGSHORT_SETTINGS = [0, -1]           # 0 = long-only, -1 = long-short

# Universal sets used everywhere
EV_NAMES = ["positiveP", "negativeP", "BULL", "BEAR", "BBr", "PNlog", "RVT"]
STRATEGY_NAMES = ["BollingerSent", "MACDSent", "PctileSent", "SlopeRVT"]

# =============================================================================
# Part 2: Performance and backtest engines
# =============================================================================
# We mirror the conventions of PyLab6_TradingStrategywCost: simple daily
# returns, cumulative return via cumsum, transaction cost subtracted at the
# end as ntrades * cost, drawdowns computed on the cumulative-return path.

def performance(x, ntrades=1, cost=0.0):
    """Performance summary of a daily-return series.

    Parameters
    ----------
    x : pd.Series
        Daily simple returns of the strategy or benchmark.
    ntrades : int
        Number of trades executed by the strategy (not used for BH).
    cost : float
        Per-trade transaction cost (subtracted as ntrades * cost from the
        cumulative return, matching the lab convention).

    Returns
    -------
    pd.Series with the eight metrics used in the course.
    """
    x = pd.Series(x).dropna()
    cumRet = np.cumsum(x.values, axis=0) - ntrades * cost
    # Floor cumRet at -0.999 to avoid taking a fractional power of a negative
    # base when a degenerate strategy's losses exceed -100 percent.
    cum_final = max(float(cumRet[-1]), -0.999)
    annRet = (np.power(cum_final + 1, 252.0 / len(x)) - 1)
    sd = np.std(x.values, ddof=1) if len(x) > 1 else np.nan
    sharpe = annRet / sd * np.sqrt(252) if sd and sd > 0 else np.nan
    nz = np.count_nonzero(x.values)
    winpct = np.count_nonzero(x.values > 0) / nz if nz > 0 else np.nan
    annSD = sd * np.sqrt(252) if sd is not np.nan else np.nan
    DDs = np.minimum.accumulate(cumRet) - cumRet
    maxDD = DDs.min() if len(DDs) else np.nan
    maxLen = int((cumRet == DDs).argmax()) if len(DDs) else 0

    return pd.Series(
        [cumRet[-1], annRet, sharpe, winpct, annSD, maxDD, maxLen, ntrades],
        index=["CumRet", "AnnRet", "Sharpe", "WinPct",
               "AnnVol", "MaxDD", "MaxDDLen", "nTrades"],
        name=x.name,
    )


def count_trades(sig):
    """Number of position changes in the signal series (zero-crossings of diff)."""
    s = pd.Series(sig).fillna(0)
    return int((s.diff().fillna(0) != 0).sum())


def test_strategy_whole(price, sig, tcost=TCOST):
    """Whole-period backtest. Signal is assumed already shifted by one day.

    Returns a DataFrame with two columns Me and BH containing the daily
    return of the strategy and the buy-and-hold benchmark, plus a Series
    of performance metrics for each.
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
    """Rolling-window backtest. Returns a DataFrame whose rows are the
    averaged (and std-d) metrics across all rolling windows.

    Trading cost is applied per-window as nTrades_window * tcost, matching
    the whole-period convention. The function counts trades only inside
    each window so per-window cost is realistic. This implements the
    optional extension requested by the assignment.
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


# =============================================================================
# Part 3: Sentiment indicator construction
# =============================================================================

def load_panel(ticker):
    """Load a dataSent CSV and return a DataFrame indexed by Date with float
    columns. Drops the unnamed integer index from the file and any optional
    downloadedDate column that appears in a subset of the files."""
    fp = DATA_DIR / f"{ticker}.csv"
    raw = pd.read_csv(fp)
    raw = raw.drop(columns=[c for c in ["Unnamed: 0", "downloadedDate"]
                            if c in raw.columns])
    raw["Date"] = pd.to_datetime(raw["Date"])
    raw = raw.set_index("Date").sort_index()
    raw = raw.astype(float)
    return raw


def build_ev_catalogue(panel):
    """Build the seven explanatory-variable series used by all strategies.
    Formulas follow the assignment statement exactly: BULL and BEAR are
    plain sums (no 0.33 scaling), BBr is the percentage ratio with
    interpolation on division-by-zero days, PNlog is the log positive-to-
    negative ratio. RVT is taken as-is."""
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
    # ffill then bfill to interpolate the rare BULL=BEAR=0 days
    bbr = bbr.ffill().bfill()
    pnlog = (0.5 * np.log((pos + 1.0) / (neg + 1.0))).rename("PNlog")
    rvt = panel["RVT"].rename("RVT")
    return {"positiveP": pos, "negativeP": neg, "BULL": bull, "BEAR": bear,
            "BBr": bbr, "PNlog": pnlog, "RVT": rvt}


# =============================================================================
# Part 4: Strategy library
# =============================================================================
# Each signal generator returns a pd.Series of {-1, 0, 1} aligned to the
# input ev (and shifted by one day so that decisions are executed next-day,
# matching the lab convention). The position is held between triggers so
# the strategy is not whipsawed by tiny indicator wiggles.

def _build_signal_from_decisions(decisions, longshort, index):
    """Forward-fill discrete decisions into a continuous signal in {-1, 0, 1},
    then lag by one day. Decisions before the first trigger are filled with
    zero (out of the market). longshort selects -1 for short, 0 for flat."""
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


# =============================================================================
# Part 5: Tuning loop on the whole period
# =============================================================================

MIN_TRADES_WHOLE = 5  # discard degenerate "almost never trade" configurations


def tune_strategy(price, ev, rvt, strategy, longshort, tcost=TCOST,
                  min_trades=MIN_TRADES_WHOLE):
    """Grid-search the hyperparameters of one strategy by maximising the
    excess cumulative return Me_CumRet - BH_CumRet on the whole period,
    subject to the minimum-trades constraint. A do-nothing signal that
    happens to land above BH by chance is rejected since it is not a
    strategy in any meaningful sense. Returns the grid trace and the best
    (params, score, perf) tuple, or (None, -inf, None) if no candidate
    clears the minimum-trades floor."""
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


# =============================================================================
# Part 6: Rolling-window evaluation of tuned configurations
# =============================================================================

def evaluate_rolling(price, ev, rvt, strategy, params, longshort,
                     w_size=254, tcost=TCOST):
    """Build the signal with tuned params and run the rolling-window backtest.
    Returns the per-window DataFrame, the averaged summary, and the signal."""
    sig = make_signal(strategy, params, ev, rvt, longshort)
    rw, summary = rolling_test_strategy(price, sig, w_size=w_size, tcost=tcost)
    return rw, summary, sig


# =============================================================================
# Part 7: Driver
# =============================================================================

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
                        rw_w, rsum, _ = evaluate_rolling(price, ev, rvt, strategy,
                                                         params, ls, w_size=w)
                    except ValueError:
                        continue
                    # True per-window excess (Me - BH). Its standard deviation
                    # accounts for the Me/BH covariance, unlike a quadrature sum
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
    """Produce three diagnostic figures for the chosen configuration:
    (i) signal vs price and ev, (ii) cumulative return of strategy vs BH,
    (iii) rolling-window distribution of per-window excess Me - BH at the
    primary window size (254). Also save per-window CSV for that window."""
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
    fig.savefig(OUT_DIR / f"ex01_signals_{ticker}.pdf", bbox_inches="tight")
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
    fig.savefig(OUT_DIR / f"ex01_equity_{ticker}.pdf", bbox_inches="tight")
    plt.close(fig)

    # Per-window distribution at w=254
    rw, _ = rolling_test_strategy(price, sig, w_size=254, tcost=TCOST)
    rw["excess"] = rw["Me_CumRet"] - rw["BH_CumRet"]
    rw.to_csv(OUT_DIR / f"ex01_rolling_windows_{ticker}.csv", index=False)
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
    fig.savefig(OUT_DIR / f"ex01_rolling_dist_{ticker}.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    all_tuning, all_whole, all_rolling = [], [], []
    cache = {}
    for ticker in TICKERS:
        tuning_df, whole_df, rolling_df, price, ev_dict, rvt = run_ticker(ticker)
        tuning_df.to_csv(OUT_DIR / f"ex01_tuning_{ticker}.csv", index=False)
        whole_df.to_csv(OUT_DIR / f"ex01_whole_{ticker}.csv", index=False)
        rolling_df.to_csv(OUT_DIR / f"ex01_rolling_{ticker}.csv", index=False)
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
    successful.to_csv(OUT_DIR / "ex01_summary.csv", index=False)
    print(f"\nTotal rolling configs with positive mean excess: {len(successful)}")
    print("Top configurations per (ticker, w_size, longshort):")
    top = (successful.groupby(["ticker", "w_size", "longshort"])
                     .head(1)
                     .reset_index(drop=True))
    print(top[["ticker", "w_size", "longshort", "strategy", "ev",
               "params", "excess_mean", "Me_Sharpe_mean",
               "Me_nTrades_mean"]].to_string(index=False))

    # Pick a single representative configuration per ticker for plotting:
    # the long-short w_size = 254 configuration with the highest mean excess
    # subject to a minimum of one trade per window on average. We plot the
    # best aggressive attempt regardless of sign so that the reader sees how
    # the strategy actually behaves; for tickers without positive excess
    # this still produces a real strategy, not a near-BH degenerate.
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


if __name__ == "__main__":
    main()
