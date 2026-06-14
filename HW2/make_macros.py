#!/usr/bin/env python3
"""Generate output/values.tex: every reported figure as a LaTeX macro.

This is the single source of truth for the numbers in the prose. We read the
committed output/ tables (and the exercise-1 engine for the cross-applied
excesses), round each figure to the precision the report shows and write a
\\newcommand for it. solution.tex inputs the file, so a number printed in the
prose is defined once from the data and cannot drift from the tables.

    python make_macros.py        # writes output/values.tex

Authors: Marco Canal, Michael Duarte Gonçalves, Olivia Peter.
"""

import pandas as pd

import exercise_01 as ex1
import exercise_03 as ex3

TAB1 = "output/ex1/tables"
TAB2 = "output/ex2/tables"
TAB3 = "output/ex3/tables"
OUT = "output/values.tex"


def f(x, dp):
    """Fixed-decimal string, e.g. -0.382 -> '-0.38'."""
    return f"{float(x):.{dp}f}"


def sf(x, dp):
    """Signed fixed-decimal string, e.g. 0.231 -> '+0.231'."""
    return f"{float(x):+.{dp}f}"


def i(x):
    """Rounded integer string."""
    return str(int(round(float(x))))


def k(x):
    """Rounded integer with the report's thousands separator, e.g. 5{,}100."""
    s = str(int(round(float(x))))
    return s[:-3] + "{,}" + s[-3:] if len(s) > 3 else s


def whole_row(ticker, ev, longshort):
    """Chosen-config whole-period row for one stock."""
    df = pd.read_csv(f"{TAB1}/01_whole_{ticker}.csv")
    m = df[(df.strategy == "SlopeRVT") & (df.ev == ev)
           & (df.longshort == longshort)]
    return m.iloc[0]


def rolling_row(ticker, ev, w_size, longshort):
    """Chosen-config rolling-window row for one stock."""
    df = pd.read_csv(f"{TAB1}/02_rolling_{ticker}.csv")
    m = df[(df.strategy == "SlopeRVT") & (df.ev == ev)
           & (df.w_size == w_size) & (df.longshort == longshort)]
    return m.iloc[0]


def cross_excess(ticker, ev):
    """Whole-period long-short excess of a SlopeRVT(20,20,1) rule."""
    panel = ex1.load_panel(ticker)
    price = panel["Adj Close"].rename(ticker)
    evs = ex1.build_ev_catalogue(panel)
    sig = ex1.make_signal("SlopeRVT", (20, 20, 1.0),
                          evs[ev], evs["RVT"], -1)
    _, perf = ex1.test_strategy_whole(price, sig, tcost=ex1.TCOST)
    return perf.loc["CumRet", "Me"] - perf.loc["CumRet", "BH"]


def metrics(contract):
    """{model: row} for an exercise-2 metrics file."""
    df = pd.read_csv(f"{TAB2}/02_metrics_{contract}.csv")
    return {r["model"]: r for _, r in df.iterrows()}


def build():
    """Compute every reported figure as a {macro_name: value} mapping."""
    m = {}

    # --- Problem 1: Deutsche Bank -------------------------------------
    db_ls = whole_row("DB", "BBr", -1)
    db_lo = whole_row("DB", "BBr", 0)
    m["DBlsCumRet"] = f(db_ls.Me_CumRet, 2)
    m["DBlsExcess"] = f(db_ls.excess, 2)
    m["DBlsSharpe"] = i(db_ls.Me_Sharpe)
    m["DBlsAnnVol"] = f(db_ls.Me_AnnVol, 2)
    m["DBlsMaxDD"] = f(db_ls.Me_MaxDD, 2)
    m["DBlsDDLen"] = i(db_ls.Me_MaxDDLen)
    m["DBlsTrades"] = i(db_ls.Me_nTrades)
    m["DBbhCumRet"] = f(db_ls.BH_CumRet, 2)
    m["DBbhLossPct"] = i(abs(db_ls.BH_CumRet) * 100)
    m["DBloExcessPts"] = i(db_lo.excess * 100)
    db_r254 = rolling_row("DB", "BBr", 254, -1)
    db_r127 = rolling_row("DB", "BBr", 127, -1)
    m["DBrollExcessMean"] = f(db_r254.excess_mean, 3)
    m["DBrollExcessStd"] = f(db_r254.excess_std, 3)
    m["DBrollWindows"] = i(db_r254.n_windows)
    m["DBrollHalfExcessMean"] = f(db_r127.excess_mean, 3)

    # --- Problem 1: HSBC ---------------------------------------------
    hs_ls = whole_row("HSBC", "positiveP", -1)
    m["HSlsCumRet"] = f(hs_ls.Me_CumRet, 2)
    m["HSlsExcess"] = f(hs_ls.excess, 2)
    m["HSlsTrades"] = i(hs_ls.Me_nTrades)
    m["HSlsMaxDD"] = f(hs_ls.Me_MaxDD, 2)
    m["HSbhCumRet"] = f(hs_ls.BH_CumRet, 2)
    m["HSbhLossPct"] = i(abs(hs_ls.BH_CumRet) * 100)
    hs_r254 = rolling_row("HSBC", "positiveP", 254, -1)
    hs_lo127 = rolling_row("HSBC", "positiveP", 127, 0)
    m["HSrollExcessMean"] = f(hs_r254.excess_mean, 3)
    m["HSrollWinPct"] = i(hs_r254.Me_WinPct_mean * 100)
    m["HSloWinPct"] = f(hs_lo127.Me_WinPct_mean, 2)
    m["HSloExcessWin"] = f(hs_lo127.excess_win, 2)

    # --- Problem 1: cross-application and walk-forward ---------------
    m["DBruleOnHS"] = f(cross_excess("HSBC", "BBr"), 2)
    m["HSruleOnDB"] = f(cross_excess("DB", "positiveP"), 2)
    wf = pd.read_csv(f"{TAB1}/06_walkforward.csv")
    db_wf = wf[wf.ticker == "DB"]
    hs_wf = wf[wf.ticker == "HSBC"]
    m["DBwfMean"] = f(db_wf.test_excess.mean(), 2)
    m["HSwfMean"] = f(hs_wf.test_excess.mean(), 2)
    m["DBwfCrash"] = f(db_wf.iloc[-1].test_excess, 2)
    m["HSwfCrash"] = f(hs_wf.iloc[-1].test_excess, 2)
    m["wfTrainLo"] = f(wf.train_excess.min(), 1)
    m["wfTrainHi"] = f(wf.train_excess.max(), 1)

    # --- Problem 2: machine-learning pricing -------------------------
    am, ba = metrics("american"), metrics("barrier")
    am_data = pd.read_csv(f"{TAB2}/01_dataset_american.csv")
    ba_data = pd.read_csv(f"{TAB2}/01_dataset_barrier.csv")
    m["AmPuts"] = i((am_data.is_call == 0).sum())
    m["AmCalls"] = i((am_data.is_call == 1).sum())
    m["AmNumMs"] = f(am["GPR"].num_time_per_option_ms, 1)
    m["BaNumMs"] = f(ba["GPR"].num_time_per_option_ms, 1)
    m["BaSEmean"] = f(ba_data.mc_se.mean(), 3)
    m["BaSEpHi"] = f(ba_data.mc_se.quantile(0.90), 3)
    m["BaGPRmae"] = f(ba["GPR"].MAE, 3)
    m["BaMonoGPR"] = i(round(1 - ba["GPR"].barrier_mono_viol, 2) * 100)
    m["BaMonoNN"] = i(round(1 - ba["NN"].barrier_mono_viol, 2) * 100)
    m["AmGPRbe"] = k(round(am["GPR"].breakeven_queries, -2))
    m["AmNNbe"] = k(round(am["NN"].breakeven_queries, -1))
    m["BaGPRbe"] = k(round(ba["GPR"].breakeven_queries, -2))
    m["BaNNbe"] = k(round(ba["NN"].breakeven_queries, -2))

    # --- Problem 3: Monte Carlo --------------------------------------
    euler = pd.read_csv(f"{TAB3}/01_euler_bench.csv").set_index("variant")
    heston = pd.read_csv(f"{TAB3}/02_heston_bench.csv")
    m["BSMref"] = f(ex3.bsm_call(100.0, 100.0, 1.0, 0.03, 0.25), 3)
    m["EulerErrVone"] = sf(euler.loc["V1 vectorised", "abs_err"], 3)
    m["EulerErrVtwo"] = sf(euler.loc["V2 + antithetic", "abs_err"], 3)
    is_scaled = heston.variant.str.contains("scaled")
    is_scalar = heston.variant.str.contains("scalar")
    sc1000 = heston[is_scalar & ~is_scaled]
    scaled = heston[is_scaled]
    vec = heston[heston.variant.str.contains("vectorised")]
    m["HesScalarMs"] = i(sc1000.time_ms.iloc[0])
    m["HesScaledMs"] = k(round(scaled.time_ms.iloc[0], -2))
    m["HesVecMs"] = i(vec.time_ms.iloc[0])
    return m


def main():
    macros = build()
    head = ("% Generated by make_macros.py -- do not edit by hand.\n"
            "% Each value is read from output/ at report precision.\n\n")
    body = "\n".join(f"\\newcommand{{\\{name}}}{{{val}}}"
                     for name, val in macros.items())
    with open(OUT, "w", encoding="utf-8") as handle:
        handle.write(head + body + "\n")
    print(f"wrote {len(macros)} macros to {OUT}")


if __name__ == "__main__":
    main()
