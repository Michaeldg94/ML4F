"""Exercise 1: who pushes whom across the asset universe.

Team:  Marco Canal, Michael Duarte Gonçalves, Olivia Peter
Date:  May 15, 2026

What this script does:
    We pull daily prices for the ten tickers the assignment lists, resample
    them to weekly and monthly closes, and then test pairwise Granger
    causality on returns and on EWMA volatility. The output is a set of
    numbered CSVs and PDFs that the LaTeX report reads back in.

    Run it once. Everything the writeup cites should land in output/ex1/.
"""

import logging
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


# --- Part 1: setup, knobs, and where the artifacts go

# Nothing in this script is random right now. The seed sits here so the day we add a bootstrap, the run still reproduces.
SEED = 42
np.random.seed(SEED)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "scripts" else SCRIPT_DIR

# The ten tickers the brief hands us: broad equity, three sector ETFs, metals, a commodity basket, two cryptos, long bonds.
ASSETS = [
    "SPY", "XLK", "XLF", "XLE", "GLD",
    "SLV", "GSG", "BTC-USD", "ETH-USD", "TLT",
]

# Friendlier names
# for tables and plots.
# BTC-USD and ETH-USD just read better without the suffix.
LABELS = {a: a.replace("-USD", "") for a in ASSETS}

# Window the team was assigned.
START_DATE = "2024-01-01"
END_DATE   = "2026-01-01"   # yfinance treats end as exclusive, so 2025-12-31 is the last day in.

# Knobs for the causality tests and the volatility model.
MAX_LAG = 5
ALPHA   = 0.05
EWMA_LAM = 0.94
MIN_DAILY_OBSERVATIONS = 450
DEBUG_TABLES = False

# Everything written by this script lands here.
OUTPUT_DIR = PROJECT_ROOT / "output" / "ex1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Filenames are numbered so the LaTeX report can \input them in a stable order without having to think about it.
OUT = {
    "prices_daily": "01_ex1_prices_daily.csv",
    "returns_weekly": "02_ex1_returns_weekly.csv",
    "returns_monthly": "03_ex1_returns_monthly.csv",
    "vol_weekly": "04_ex1_ewma_vol_weekly.csv",
    "vol_monthly": "05_ex1_ewma_vol_monthly.csv",
    "causality_returns_weekly_csv": "06_ex1_causality_returns_weekly.csv",
    "causality_returns_weekly_tex": "07_ex1_causality_returns_weekly.tex",
    "causality_returns_monthly_csv": "08_ex1_causality_returns_monthly.csv",
    "causality_returns_monthly_tex": "09_ex1_causality_returns_monthly.tex",
    "causality_vol_weekly_csv": "10_ex1_causality_vol_weekly.csv",
    "causality_vol_weekly_tex": "11_ex1_causality_vol_weekly.tex",
    "causality_vol_monthly_csv": "12_ex1_causality_vol_monthly.csv",
    "causality_vol_monthly_tex": "13_ex1_causality_vol_monthly.tex",
    "corr_returns_weekly": "14_ex1_contemp_corr_returns_weekly.csv",
    "corr_returns_monthly": "15_ex1_contemp_corr_returns_monthly.csv",
    "corr_vol_weekly": "16_ex1_contemp_corr_vol_weekly.csv",
    "corr_vol_monthly": "17_ex1_contemp_corr_vol_monthly.csv",
    "net_ret_w": "18_ex1_network_returns_weekly.pdf",
    "net_ret_m": "19_ex1_network_returns_monthly.pdf",
    "net_vol_w": "20_ex1_network_vol_weekly.pdf",
    "net_vol_m": "21_ex1_network_vol_monthly.pdf",
}


# ----- Part 2: pull and clean the daily price panel~
def configure_logging() -> None:
    """Keep the run readable. We log step headlines, not every table."""
    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)


def download_prices() -> pd.DataFrame:
    """Grab adjusted closes for the ten tickers from Yahoo."""
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
    return raw["Close"][ASSETS].copy()


def clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Line the panel up to a single trading calendar and drop stale rows.

    Crypto trades seven days a week. The ETFs don't. We pick the SPY
    calendar as the shared clock so the joint panel is rectangular and the
    VAR fits don't choke on weekend NaNs.
    """
    prices = prices.dropna(how="all")

    # SPY decides what "a trading day" means for the project.
    prices = prices.loc[prices["SPY"].notna()].copy()
    prices = prices.ffill(limit=1)

    n_before = len(prices)
    prices = prices.dropna(how="any")
    LOGGER.info(
        "Daily price panel has %d rows after cleaning, %d rows dropped",
        len(prices),
        n_before - len(prices),
    )

    # Yahoo sometimes hands back a half-populated panel on a bad day. We'd
    # rather crash early than overwrite the saved outputs with garbage.
    if len(prices) < MIN_DAILY_OBSERVATIONS:
        raise RuntimeError(
            "Yahoo Finance returned an incomplete price panel. "
            f"Expected at least {MIN_DAILY_OBSERVATIONS} daily observations "
            f"for the assigned window, got {len(prices)}. Existing outputs "
            "were not overwritten."
        )
    return prices


configure_logging()

# statsmodels and pandas
# like to chatter
# when a monthly VAR runs on a small sample. Actual fitting errors are caught further down in causality_table, so the rest is just noise.
warnings.filterwarnings("ignore")

prices = clean_prices(download_prices())
prices.rename(columns=LABELS).to_csv(OUTPUT_DIR / OUT["prices_daily"])


# ---- Part 3 ---- weekly and monthly closes (and the log returns)

# Our reading of the brief, "sample the series weekly and then monthly": resample the adjusted closes to the new
# frequency and recompute log returns there.
# We take the period-end close.
def resample_last(panel: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Last observation of each period."""
    return panel.resample(rule).last().dropna(how="any")


def log_returns(p: pd.DataFrame) -> pd.DataFrame:
    """Per-period log returns from a price panel."""
    return np.log(p / p.shift(1)).dropna()


# -- Part 4 -- RiskMetrics EWMA volatility
def ewma_vol(
    returns: pd.DataFrame,
    lam: float = EWMA_LAM,
) -> pd.DataFrame:
    """Column-wise EWMA volatility, RiskMetrics flavour (lambda = 0.94)."""
    var = returns.pow(2).ewm(alpha=1 - lam, adjust=False).mean()
    # First row is just r_0^2 by construction (no history yet). Drop it so
    # the series we keep is informative and not a seeding artefact.
    return np.sqrt(var).iloc[1:]


# ------- Part 5: pairwise Granger machinery
def optimal_lag_aic(pair: pd.DataFrame, max_lag: int = MAX_LAG) -> int:
    """Pick the VAR lag that minimises AIC.

    We follow the class convention and fit without an intercept
    (`trend="n"`). AIC can return 0 on very flat pairs, but the Granger
    test needs at least one lag, so we floor at 1.
    """
    model = VAR(pair)
    sel = model.select_order(maxlags=max_lag, trend="n")
    chosen = int(sel.aic)
    return max(chosen, 1)


def granger_pvalue(cause_then_effect: pd.DataFrame, lag: int) -> float:
    """Run the Granger F-test and return its p-value.

    Heads up: ``statsmodels.grangercausalitytests`` asks whether the SECOND
    column causes the FIRST. That convention is easy to get wrong, so this
    wrapper accepts data in the natural [cause, effect] order and flips it
    internally before handing it to statsmodels.
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
    """Build the directed lag table for one panel.

    Rows are causes, columns are effects. If a -> b is significant at
    level ``alpha``, we write the AIC-optimal lag. Otherwise we write 0.
    The diagonal gets a literal "-" so the LaTeX prints a dash.

    The second return value is the matching p-value table. We keep it for
    diagnostics; it doesn't end up in the report.
    """
    cols = returns_or_vol.columns.tolist()
    lag_table = pd.DataFrame(0, index=cols, columns=cols, dtype=object)
    pval_table = pd.DataFrame(np.nan, index=cols, columns=cols, dtype=float)

    for a in cols:
        for b in cols:
            if a == b:
                lag_table.loc[a, b] = "-"
                continue
            # The bivariate VAR is symmetric in its two endogenous series,
            # so the order of [a, b] is irrelevant for lag selection. We
            # leave it as-is.
            pair = returns_or_vol[[a, b]].dropna()
            if len(pair) <= max_lag + 2:
                # Not enough rows to fit a VAR(max_lag). Record 0 and move on.
                lag_table.loc[a, b] = 0
                continue
            try:
                lag = optimal_lag_aic(pair, max_lag=max_lag)
                # Test the direction a -> b at the chosen lag.
                p = granger_pvalue(returns_or_vol[[a, b]].dropna(), lag)
                pval_table.loc[a, b] = p
                lag_table.loc[a, b] = lag if p < alpha else 0
            except Exception as exc:
                # Tiny monthly samples occasionally throw a singular matrix.
                # We treat that as "no evidence of causality", which keeps
                # the decision table well-defined.
                lag_table.loc[a, b] = 0
                LOGGER.debug(
                    "Causality %s -> %s failed with %s, recorded 0",
                    a,
                    b,
                    type(exc).__name__,
                )
    return lag_table, pval_table


# Part 6 --- build the four panels, then run the tests ---
prices_W = resample_last(prices, "W-FRI")
prices_M = resample_last(prices, "ME")

ret_W = log_returns(prices_W)
ret_M = log_returns(prices_M)

vol_W = ewma_vol(ret_W)
vol_M = ewma_vol(ret_M)

# Switch to the prettier column names before anything is saved or plotted.
for df in (ret_W, ret_M, vol_W, vol_M):
    df.rename(columns=LABELS, inplace=True)

ret_W.to_csv(OUTPUT_DIR / OUT["returns_weekly"])
ret_M.to_csv(OUTPUT_DIR / OUT["returns_monthly"])
vol_W.to_csv(OUTPUT_DIR / OUT["vol_weekly"])
vol_M.to_csv(OUTPUT_DIR / OUT["vol_monthly"])

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


# --- Part 7 --- save the lag tables (CSV + LaTeX fragment)
def save_table(
    lag_tab: pd.DataFrame,
    csv_name: str,
    tex_name: str,
    label: str,
    caption: str,
) -> None:
    """Save the lag table twice: once as CSV, once as a booktabs LaTeX
    fragment. The .tex file is meant to be ``\\input{}`` directly from the
    report.
    """
    csv_path = OUTPUT_DIR / csv_name
    tex_path = OUTPUT_DIR / tex_name
    lag_tab.to_csv(csv_path)

    cols = lag_tab.columns.tolist()
    col_spec = "l" + "c" * len(cols)
    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")
    lines.append(" & " + " & ".join(cols) + r" \\")
    lines.append(r"\midrule")
    for r in lag_tab.index:
        row = [str(lag_tab.loc[r, c]) for c in cols]
        lines.append(r + " & " + " & ".join(row) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{" + caption + r"}")
    lines.append(r"\label{tab:" + label + r"}")
    lines.append(r"\end{table}")
    tex_path.write_text("\n".join(lines))


save_table(
    lag_rW,
    OUT["causality_returns_weekly_csv"],
    OUT["causality_returns_weekly_tex"],
    "returns_weekly",
    "Pairwise Granger causality on weekly log returns. AIC-optimal lag "
    "in $[1,5]$ if significant at the 5\\% level and 0 otherwise.",
)
save_table(
    lag_rM,
    OUT["causality_returns_monthly_csv"],
    OUT["causality_returns_monthly_tex"],
    "returns_monthly",
    "Pairwise Granger causality on monthly log returns. Entries as in "
    "the weekly returns table.",
)
save_table(
    lag_vW,
    OUT["causality_vol_weekly_csv"],
    OUT["causality_vol_weekly_tex"],
    "vol_weekly",
    "Pairwise Granger causality on weekly EWMA volatility. Entries as "
    "in the returns tables.",
)
save_table(
    lag_vM,
    OUT["causality_vol_monthly_csv"],
    OUT["causality_vol_monthly_tex"],
    "vol_monthly",
    "Pairwise Granger causality on monthly EWMA volatility. Entries as "
    "in the returns tables.",
)


# Part 8 -- the four network plots --

# Colouring nodes by asset class makes the story easier to read at a glance: equity-on-equity dominating the picture looks different from arrows that cut across asset classes (which is what risk-off spillovers usually look like).
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

# Fixed seat assignment around the circle, so the four plots are directly
# comparable to each other.
CIRCLE_ORDER = ["SPY", "XLK", "XLF", "XLE", "GSG",
                "GLD", "SLV", "TLT", "ETH", "BTC"]


def _fixed_circular_layout(nodes: list) -> dict:
    """Deterministic circle layout, 12 o'clock at the top, clockwise from there."""
    n = len(nodes)
    pos = {}
    for i, name in enumerate(nodes):
        theta = np.pi / 2 - 2 * np.pi * i / n
        pos[name] = np.array([np.cos(theta), np.sin(theta)])
    return pos


def plot_network(lag_tab: pd.DataFrame, title: str, out_pdf: Path) -> None:
    """Render the directed graph encoded by ``lag_tab`` and save it as PDF.

    Nodes are coloured by asset class. We draw an edge a -> b whenever
    ``lag_tab.loc[a, b]`` is a positive integer, and label it with that
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

    # One legend at the bottom of the figure, asset classes only.
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
             OUTPUT_DIR / OUT["net_ret_w"])
plot_network(lag_rM, "Returns causality network (monthly)",
             OUTPUT_DIR / OUT["net_ret_m"])
plot_network(lag_vW, "Volatility spillover network (weekly, EWMA)",
             OUTPUT_DIR / OUT["net_vol_w"])
plot_network(lag_vM, "Volatility spillover network (monthly, EWMA)",
             OUTPUT_DIR / OUT["net_vol_m"])


# ------ Part 9: contemporaneous correlation snapshots

# Useful for the writeup.
# Granger = who leads whom.
# Correlation = what moves together right now.
# Putting the two side by side makes the spillover story easier to read.
ret_W.corr().round(2).to_csv(OUTPUT_DIR / OUT["corr_returns_weekly"])
ret_M.corr().round(2).to_csv(OUTPUT_DIR / OUT["corr_returns_monthly"])
vol_W.corr().round(2).to_csv(OUTPUT_DIR / OUT["corr_vol_weekly"])
vol_M.corr().round(2).to_csv(OUTPUT_DIR / OUT["corr_vol_monthly"])


# --- Wrap-up: end-of-run summaries
def show(name: str, t: pd.DataFrame) -> None:
    """Optional debug dump of a full decision table. Off by default."""
    if not DEBUG_TABLES:
        return
    with pd.option_context("display.max_columns", None, "display.width", 200):
        LOGGER.debug("%s\n%s", name, t)


show("Returns causality lags (weekly)",  lag_rW)
show("Returns causality lags (monthly)", lag_rM)
show("Volatility causality lags (weekly)",  lag_vW)
show("Volatility causality lags (monthly)", lag_vM)

# Quick edge-count line. Handy for the writeup: 10 assets give 90 ordered
# pairs once the diagonal is excluded.
for name, tab in [("ret_W", lag_rW), ("ret_M", lag_rM),
                  ("vol_W", lag_vW), ("vol_M", lag_vM)]:
    n_edges = sum(1 for a in tab.index for b in tab.columns
                  if isinstance(tab.loc[a, b], (int, np.integer)) and tab.loc[a, b] > 0)
    LOGGER.info("%s has %d significant edges out of 90 ordered pairs", name, n_edges)

LOGGER.info("Artifacts written to %s", OUTPUT_DIR)
