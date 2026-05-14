# Exercise 1

# Preamble: this script builds the global asset causal network and the
# volatility spillover network for the assigned ten-asset universe. For
# every ordered pair we pick the AIC-optimal VAR lag in [1, 5] and test
# Granger causality at the 5% level, both on log returns and on EWMA
# volatility (RiskMetrics, lambda = 0.94), at weekly and monthly sampling.

# ML for Finance, 2026 (BSE): Task 1
# Marco Canal, Michael Duarte Gonçalves, Olivia Peter


import logging
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests

LOGGER = logging.getLogger(__name__)

# Wall-clock start; the matching finish line at the bottom of the script
# logs total runtime. Useful when re-running the pipeline to confirm the
# Yahoo download (the slow step on a cold cache) is not regressing.
_T_START = time.perf_counter()


# --- Part 1: setup, knobs and where the artefacts go

# Nothing in this script is random today; the seed exists so any future
# stochastic addition (e.g. a bootstrap) is reproducible.
SEED = 42
np.random.seed(SEED)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "scripts" else SCRIPT_DIR

# Asset universe specified by the assignment: broad equity, three sector
# ETFs, two precious metals, a commodity basket, two cryptos, long bonds.
ASSETS = [
    "SPY", "XLK", "XLF", "XLE", "GLD",
    "SLV", "GSG", "BTC-USD", "ETH-USD", "TLT",
]
# Display labels (BTC/ETH read better without the -USD suffix).
LABELS = {a: a.replace("-USD", "") for a in ASSETS}

# Team-assigned epoch. yfinance treats `end` as exclusive, so 2026-01-01
# correctly includes 2025-12-31 as the last trading day.
START_DATE = "2024-01-01"
END_DATE   = "2026-01-01"

# Causality and volatility settings.
MAX_LAG = 5            # AIC search range [1, MAX_LAG] per assignment
ALPHA   = 0.05         # 5% significance threshold for the Granger F-test
EWMA_LAM = 0.94        # RiskMetrics standard, as used in ML_finance_lab1
MIN_DAILY_OBSERVATIONS = 450  # sanity floor; raises if Yahoo returns less
DEBUG_TABLES = False   # gate the full decision-table debug print

# All Exercise 1 artefacts live under output/ex1/. The directory is
# created on demand so the script can be re-run from a fresh checkout
# without any manual setup.
OUTPUT_DIR = PROJECT_ROOT / "output" / "ex1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ----- Part 2: download and clean the daily price panel
def configure_logging():
    # Single-line headlines, nothing else.
    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)


def download_prices() -> pd.DataFrame:
    """Daily adjusted closes for the ten tickers from Yahoo Finance.

    ``auto_adjust=True`` returns split- and dividend-adjusted closes, which
    is what we need for return computations over a two-year window with
    several ETF distributions.
    """
    LOGGER.info(
        "Downloading %d assets from %s to %s",
        len(ASSETS),
        START_DATE,
        END_DATE,
    )
    raw = yf.download(
        ASSETS,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True,
        progress=False,
    )
    # yfinance returns a MultiIndex (field, ticker); keep only Close.
    return raw["Close"][ASSETS].copy()


def clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Align all series to the NYSE trading calendar and drop stale rows.

    Crypto trades 7 days a week, ETFs only on weekdays. We pick SPY's
    calendar as the shared clock: inner-join on rows where SPY has a
    price, forward-fill at most one day for sporadic crypto gaps, then
    drop any row that still has a NaN. We raise early if Yahoo hands back
    an obviously incomplete panel rather than overwriting saved outputs
    with garbage.
    """
    # Step 1: drop calendar days where every series is missing (typically
    # Saturdays/Sundays returned by yfinance for the crypto-only universe).
    prices = prices.dropna(how="all")
    # Step 2: pin to SPY's trading days. SPY is the dense series; crypto
    # closes on those same days are what we actually want.
    prices = prices.loc[prices["SPY"].notna()].copy()
    # Step 3: one-day forward-fill for the occasional crypto exchange
    # outage. Two days in a row almost never happens in this period; if
    # it did, the next dropna catches it.
    prices = prices.ffill(limit=1)

    n_before = len(prices)
    prices = prices.dropna(how="any")
    LOGGER.info(
        "Daily price panel has %d rows after cleaning, %d rows dropped",
        len(prices),
        n_before - len(prices),
    )

    if len(prices) < MIN_DAILY_OBSERVATIONS:
        raise RuntimeError(
            "Yahoo Finance returned an incomplete price panel. "
            f"Expected at least {MIN_DAILY_OBSERVATIONS} daily observations "
            f"for the assigned window, got {len(prices)}. Existing outputs "
            "were not overwritten."
        )
    return prices


configure_logging()

# Silence statsmodels / pandas chatter on small-sample monthly VAR fits.
# Real fitting errors are still caught inside causality_table below; this
# just kills the noisy "MaximumLikelihood failed" RuntimeWarnings that
# otherwise drown out the actual progress log.
warnings.filterwarnings("ignore")

# Download once, clean once, then persist. Re-running the whole pipeline
# does pay the Yahoo Finance round-trip again -- that is intentional, the
# point of this script is reproducibility from a clean slate.
prices = clean_prices(download_prices())
prices.rename(columns=LABELS).to_csv(OUTPUT_DIR / "01_ex1_prices_daily.csv")


# ---- Part 3: weekly and monthly closes (and the log returns)
# We read "sample the series weekly and then monthly" as: resample the
# adjusted closes to the target frequency and recompute log returns there.
def resample_last(panel: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Last observation of each period (e.g. 'W-FRI' or 'ME')."""
    # 'W-FRI' anchors weekly bins to Friday close, which lines up with
    # how the equity options world prices its weekly contracts. 'ME' is
    # the new pandas alias for month-end; the deprecated 'M' would also
    # work but raises a FutureWarning on recent pandas releases.
    return panel.resample(rule).last().dropna(how="any")


def log_returns(p):
    # Per-period log returns from a price panel. Log returns aggregate
    # additively over horizons, which is the property we lean on when
    # the resample step changes the period length. Simple returns would
    # compound multiplicatively and complicate the algebra.
    return np.log(p / p.shift(1)).dropna()


# -- Part 4: RiskMetrics EWMA volatility
def ewma_vol(
    returns: pd.DataFrame,
    lam: float = EWMA_LAM,
) -> pd.DataFrame:
    """Column-wise EWMA volatility (RiskMetrics, lambda = 0.94).

    Recursion is ``sigma2_t = lam * sigma2_{t-1} + (1 - lam) * r_{t-1}^2``,
    implemented via ``pd.ewm(alpha=1-lam, adjust=False)`` then a square
    root. We drop the first row: with ``adjust=False`` it equals r_0^2 by
    construction and is just a seeding artefact.
    """
    var = returns.pow(2).ewm(alpha=1 - lam, adjust=False).mean()
    return np.sqrt(var).iloc[1:]


# ------- Part 5: pairwise Granger machinery
def optimal_lag_aic(pair: pd.DataFrame, max_lag: int = MAX_LAG) -> int:
    """Bivariate VAR lag that minimises AIC, floored to 1.

    We pass ``trend='n'`` (no constant) to mirror the R convention
    ``VARselect(..., type='none')`` used in PyLab 3. The Granger test
    requires at least one lag, so AIC = 0 collapses to 1.
    """
    model = VAR(pair)
    sel = model.select_order(maxlags=max_lag, trend="n")
    return max(int(sel.aic), 1)


def granger_pvalue(cause_then_effect: pd.DataFrame, lag: int) -> float:
    """Granger F-test p-value for ``cause -> effect`` at the given lag.

    ``statsmodels.grangercausalitytests`` tests whether the SECOND column
    causes the FIRST, so we accept the natural [cause, effect] order and
    flip internally to avoid the convention trap.
    """
    cause, effect = cause_then_effect.columns
    data_for_test = cause_then_effect[[effect, cause]]
    res = grangercausalitytests(
        data_for_test,
        maxlag=[lag],
        addconst=True,
        verbose=False,
    )
    return float(res[lag][0]["ssr_ftest"][1])


def causality_table(returns_or_vol: pd.DataFrame,
                    max_lag: int = MAX_LAG,
                    alpha: float = ALPHA) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Directed lag table for one panel; rows are causes, columns effects.

    Off-diagonal entries are the AIC-optimal lag if the test rejects at
    ``alpha``, else 0. Diagonal is the literal string ``"-"``. The second
    return value is the matching p-value table, kept for diagnostics only.
    """
    cols = returns_or_vol.columns.tolist()
    lag_table = pd.DataFrame(0, index=cols, columns=cols, dtype=object)
    pval_table = pd.DataFrame(np.nan, index=cols, columns=cols, dtype=float)

    # TODO: the double loop is fine for n=10 assets but would be O(n^2 * lag)
    # and not worth vectorising at this size. Revisit if the universe grows.
    for a in cols:
        for b in cols:
            if a == b:
                lag_table.loc[a, b] = "-"
                continue
            # Bivariate VAR is symmetric in its two endogenous series,
            # so the [a, b] order is irrelevant for the AIC step.
            pair = returns_or_vol[[a, b]].dropna()
            if len(pair) <= max_lag + 2:
                # Not enough rows to fit VAR(max_lag); record 0.
                lag_table.loc[a, b] = 0
                continue
            try:
                lag = optimal_lag_aic(pair, max_lag=max_lag)
                p = granger_pvalue(returns_or_vol[[a, b]].dropna(), lag)
                pval_table.loc[a, b] = p
                lag_table.loc[a, b] = lag if p < alpha else 0
            except Exception as exc:
                # Tiny monthly samples can throw a singular matrix; treat
                # that as "no evidence of causality" to keep the table well
                # defined.
                lag_table.loc[a, b] = 0
                LOGGER.debug(
                    "Causality %s -> %s failed with %s, recorded 0",
                    a,
                    b,
                    type(exc).__name__,
                )
    return lag_table, pval_table


# Part 6: build the four panels, then run the tests
# Four (frequency, kind) combinations: {weekly, monthly} x {returns, EWMA vol}.
prices_W = resample_last(prices, "W-FRI")
prices_M = resample_last(prices, "ME")

ret_W = log_returns(prices_W)
ret_M = log_returns(prices_M)

# EWMA is fed the per-frequency returns directly (not the higher-frequency
# series). That keeps the volatility series aligned with the returns we
# already saved and matches Lab 1's RiskMetrics recipe at the relevant
# horizon.
vol_W = ewma_vol(ret_W)
vol_M = ewma_vol(ret_M)

# Switch to display labels (BTC, ETH) before anything hits disk. Doing the
# rename here means downstream code (plots, tables, file headers) all
# inherit the same column names with no extra plumbing.
for df in (ret_W, ret_M, vol_W, vol_M):
    df.rename(columns=LABELS, inplace=True)

ret_W.to_csv(OUTPUT_DIR / "02_ex1_returns_weekly.csv")
ret_M.to_csv(OUTPUT_DIR / "03_ex1_returns_monthly.csv")
vol_W.to_csv(OUTPUT_DIR / "04_ex1_ewma_vol_weekly.csv")
vol_M.to_csv(OUTPUT_DIR / "05_ex1_ewma_vol_monthly.csv")

LOGGER.info("Weekly sample: %d returns, %d EWMA-volatility rows", len(ret_W), len(vol_W))
LOGGER.info("Monthly sample: %d returns, %d EWMA-volatility rows", len(ret_M), len(vol_M))

LOGGER.info("Building four causality tables")
LOGGER.info("Testing weekly return causality")
lag_rW, p_rW = causality_table(ret_W)
LOGGER.info("Testing monthly return causality")
lag_rM, p_rM = causality_table(ret_M)
LOGGER.info("Testing weekly EWMA-volatility causality")
lag_vW, p_vW = causality_table(vol_W)
LOGGER.info("Testing monthly EWMA-volatility causality")
lag_vM, p_vM = causality_table(vol_M)


# --- Part 7: save the lag tables (CSV + LaTeX fragment)
def save_table(
    lag_tab: pd.DataFrame,
    csv_name: str,
    tex_name: str,
    label: str,
    caption: str,
) -> None:
    """Write the lag table as a CSV and as a booktabs LaTeX fragment.

    The .tex file is meant to be ``\\input{}`` directly from the report.
    """
    csv_path = OUTPUT_DIR / csv_name
    tex_path = OUTPUT_DIR / tex_name
    lag_tab.to_csv(csv_path)

    cols = lag_tab.columns.tolist()
    col_spec = "l" + "c" * len(cols)
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
        " & " + " & ".join(cols) + r" \\",
        r"\midrule",
    ]
    for r in lag_tab.index:
        row = [str(lag_tab.loc[r, c]) for c in cols]
        lines.append(r + " & " + " & ".join(row) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{" + caption + r"}",
        r"\label{tab:" + label + r"}",
        r"\end{table}",
    ]
    tex_path.write_text("\n".join(lines))


save_table(
    lag_rW,
    "06_ex1_causality_returns_weekly.csv",
    "07_ex1_causality_returns_weekly.tex",
    "returns_weekly",
    "Pairwise Granger causality on weekly log returns. AIC-optimal lag "
    "in $[1,5]$ if significant at the 5\\% level and 0 otherwise.",
)
save_table(
    lag_rM,
    "08_ex1_causality_returns_monthly.csv",
    "09_ex1_causality_returns_monthly.tex",
    "returns_monthly",
    "Pairwise Granger causality on monthly log returns. Entries as in "
    "the weekly returns table.",
)
save_table(
    lag_vW,
    "10_ex1_causality_vol_weekly.csv",
    "11_ex1_causality_vol_weekly.tex",
    "vol_weekly",
    "Pairwise Granger causality on weekly EWMA volatility. Entries as "
    "in the returns tables.",
)
save_table(
    lag_vM,
    "12_ex1_causality_vol_monthly.csv",
    "13_ex1_causality_vol_monthly.tex",
    "vol_monthly",
    "Pairwise Granger causality on monthly EWMA volatility. Entries as "
    "in the returns tables.",
)


# Part 8: the four network plots
# Colouring nodes by asset class makes the macro narrative immediate:
# equity-on-equity dominating looks different from cross-class arrows
# (which is what risk-off spillovers usually look like).
ASSET_CLASS = {
    "SPY": "Equity (broad)",
    "XLK": "Equity (sector)",
    "XLF": "Equity (sector)",
    "XLE": "Equity (sector)",
    "GLD": "Precious metal",
    "SLV": "Precious metal",
    "GSG": "Commodities",
    "BTC": "Crypto",
    "ETH": "Crypto",
    "TLT": "Treasuries",
}
CLASS_COLOUR = {
    "Equity (broad)":  "#4c78a8",
    "Equity (sector)": "#9ecae9",
    "Precious metal":  "#f1c453",
    "Commodities":     "#dfa454",
    "Crypto":          "#b279a2",
    "Treasuries":      "#72b095",
}
# Fixed circular order so all four plots are directly comparable. The
# ordering groups by asset class: equities first (broad, then sectors),
# then commodities, then precious metals, then Treasuries, then crypto.
# Reading clockwise gives a natural "from equities to crypto" tour.
CIRCLE_ORDER = ["SPY", "XLK", "XLF", "XLE", "GSG",
                "GLD", "SLV", "TLT", "ETH", "BTC"]


def _fixed_circular_layout(nodes):
    """Deterministic circle layout: top at 12 o'clock, clockwise from there.

    We roll our own instead of using ``nx.circular_layout`` because the
    latter (a) randomises orientation across networkx versions and
    (b) places node 0 at the 3 o'clock position; this version is stable
    and matches the eye's default reading habit.
    """
    n = len(nodes)
    pos = {}
    for i, name in enumerate(nodes):
        theta = np.pi / 2 - 2 * np.pi * i / n  # start at 12, go clockwise
        pos[name] = np.array([np.cos(theta), np.sin(theta)])
    return pos


def plot_network(lag_tab: pd.DataFrame, title: str, out_pdf: Path) -> None:
    """Render the directed graph implied by ``lag_tab`` and save as PDF.

    Nodes are coloured by asset class; we draw an edge a -> b whenever
    ``lag_tab.loc[a, b]`` is a positive integer and label it with that
    lag. Layout and colours are fixed across the four plots so the reader
    can compare them like-for-like.
    """
    G = nx.DiGraph()
    for a in lag_tab.index:
        G.add_node(a)
    for a in lag_tab.index:
        for b in lag_tab.columns:
            entry = lag_tab.loc[a, b]
            if isinstance(entry, (int, np.integer)) and entry > 0:
                G.add_edge(a, b, lag=int(entry))

    pos = _fixed_circular_layout(CIRCLE_ORDER)
    node_colours = [CLASS_COLOUR[ASSET_CLASS[n]] for n in CIRCLE_ORDER]

    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    nx.draw_networkx_nodes(G, pos, nodelist=CIRCLE_ORDER, node_size=1900,
                           node_color=node_colours, edgecolors="black",
                           linewidths=1.2, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=11, font_weight="bold", ax=ax)

    if G.number_of_edges() > 0:
        # A slight curve on every edge prevents the two arrows of a
        # reciprocal pair from drawing on top of each other.
        nx.draw_networkx_edges(
            G, pos, arrows=True, arrowsize=22, arrowstyle="-|>",
            edge_color="#333", width=1.6, alpha=0.85,
            connectionstyle="arc3,rad=0.13",
            node_size=1900, min_target_margin=18, ax=ax,
        )
        edge_labels = {(u, v): str(d["lag"]) for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=edge_labels, font_size=9, label_pos=0.45,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85),
            ax=ax,
        )

    # One legend at the bottom, asset classes only.
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=CLASS_COLOUR[c], edgecolor="black", label=c)
               for c in dict.fromkeys(ASSET_CLASS.values())]
    ax.legend(handles=handles, loc="lower center", ncol=3,
              bbox_to_anchor=(0.5, -0.06), frameon=False, fontsize=9)

    ax.set_title(title, fontsize=13)
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


plot_network(lag_rW, "Returns causality network (weekly)",
             OUTPUT_DIR / "18_ex1_network_returns_weekly.pdf")
plot_network(lag_rM, "Returns causality network (monthly)",
             OUTPUT_DIR / "19_ex1_network_returns_monthly.pdf")
plot_network(lag_vW, "Volatility spillover network (weekly, EWMA)",
             OUTPUT_DIR / "20_ex1_network_vol_weekly.pdf")
plot_network(lag_vM, "Volatility spillover network (monthly, EWMA)",
             OUTPUT_DIR / "21_ex1_network_vol_monthly.pdf")


# ------ Part 9: contemporaneous correlation snapshots
# Granger answers who leads whom; correlation answers what moves together
# right now. Putting both side by side helps the spillover narrative.
ret_W.corr().round(2).to_csv(OUTPUT_DIR / "14_ex1_contemp_corr_returns_weekly.csv")
ret_M.corr().round(2).to_csv(OUTPUT_DIR / "15_ex1_contemp_corr_returns_monthly.csv")
vol_W.corr().round(2).to_csv(OUTPUT_DIR / "16_ex1_contemp_corr_vol_weekly.csv")
vol_M.corr().round(2).to_csv(OUTPUT_DIR / "17_ex1_contemp_corr_vol_monthly.csv")


# Helper kept here so the wrap-up summary below can log edge counts; the
# macro generator reads identical counts directly from the lag-table CSVs.
def count_edges(tab):
    # Number of positive-integer entries in a lag table (significant edges).
    return sum(
        1 for a in tab.index for b in tab.columns
        if isinstance(tab.loc[a, b], (int, np.integer)) and tab.loc[a, b] > 0
    )


# --- Wrap-up: end-of-run summaries
def show(name: str, t: pd.DataFrame) -> None:
    """Optional full-table debug dump, gated by DEBUG_TABLES."""
    if not DEBUG_TABLES:
        return
    with pd.option_context("display.max_columns", None, "display.width", 200):
        LOGGER.debug("%s\n%s", name, t)


show("Returns causality lags (weekly)",  lag_rW)
show("Returns causality lags (monthly)", lag_rM)
show("Volatility causality lags (weekly)",  lag_vW)
show("Volatility causality lags (monthly)", lag_vM)

# 10 assets give 90 ordered pairs once the diagonal is excluded.
for name, tab in [("ret_W", lag_rW), ("ret_M", lag_rM),
                  ("vol_W", lag_vW), ("vol_M", lag_vM)]:
    LOGGER.info("%s has %d significant edges out of 90 ordered pairs", name, count_edges(tab))

LOGGER.info("Artefacts written to %s", OUTPUT_DIR)
LOGGER.info("Exercise 1 finished in %.1f s wall time", time.perf_counter() - _T_START)
