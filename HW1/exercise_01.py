# =============================================================================
# exercise_01.py
# Global asset causal network and volatility spillover
#
# Team:    [Team member 1], [Team member 2], [Team member 3]
# Date:    2026-05-02
#
# PURPOSE:
#   Compute pairwise Granger causality tables and spillover networks for ten
#   global assets (SPY, XLK, XLF, XLE, GLD, SLV, GSG, BTC-USD, ETH-USD, TLT)
#   over the team-assigned epoch 2024-01-01 to 2025-12-31. Two series are
#   analysed for each asset: log returns and EWMA volatility (RiskMetrics
#   recipe with lambda = 0.94, as in Lab session 1). Each series is sampled
#   at weekly and monthly frequency, giving four causality tables in total.
#   For every ordered pair (A, B) the optimal VAR lag is selected from the
#   range [1, 5] using AIC on the bivariate VAR, and Granger causality is
#   evaluated at that lag at the 5% significance level.
#
# INPUTS:
#   - Asset prices retrieved from Yahoo Finance via yfinance (no local file
#     needed; requires internet access)
#
# OUTPUTS:
#   - output/table_returns_weekly.csv,  output/table_returns_weekly.tex
#   - output/table_returns_monthly.csv, output/table_returns_monthly.tex
#   - output/table_vol_weekly.csv,      output/table_vol_weekly.tex
#   - output/table_vol_monthly.csv,     output/table_vol_monthly.tex
#   - output/network_returns_weekly.pdf,  output/network_returns_monthly.pdf
#   - output/network_vol_weekly.pdf,      output/network_vol_monthly.pdf
#   - output/contemp_corr_returns_weekly.csv  (and the other three frequencies)
#
# STRUCTURE:
#   Part 1: Setup, asset list, paths, seed
#   Part 2: Data download and cleaning
#   Part 3: Resampling helpers (weekly / monthly)
#   Part 4: EWMA volatility (RiskMetrics)
#   Part 5: Pairwise causality engine (VAR.select_order + grangercausalitytests)
#   Part 6: Build the four directed lag tables
#   Part 7: Save tables as CSV and as LaTeX fragments
#   Part 8: Plot the four causality networks
#   Part 9: Contemporaneous correlations (for narrative)
# =============================================================================

# === Part 1: Setup ===========================================================
import warnings                                # to silence statsmodels noise
from pathlib import Path                       # robust relative paths

import numpy as np                             # numerical arrays
import pandas as pd                            # tabular data and time-series
import matplotlib.pyplot as plt                # plotting
import networkx as nx                          # directed graphs for spillover networks
import yfinance as yf                          # market data download

from statsmodels.tsa.api import VAR                                # bivariate VAR for lag selection
from statsmodels.tsa.stattools import grangercausalitytests        # pairwise Granger F-test

# Reproducibility: yfinance/statsmodels do not use randomness here, but we set
# numpy seed anyway so any future stochastic addition (e.g. bootstrap) is
# reproducible.
SEED = 42
np.random.seed(SEED)

# Asset universe specified by the assignment
ASSETS = ["SPY", "XLK", "XLF", "XLE", "GLD",
          "SLV", "GSG", "BTC-USD", "ETH-USD", "TLT"]

# Display labels for tables and plots (BTC/ETH read better without the -USD)
LABELS = {a: a.replace("-USD", "") for a in ASSETS}

# Team-assigned epoch
START_DATE = "2024-01-01"
END_DATE   = "2026-01-01"   # yfinance end is exclusive, so this includes 2025-12-31

# Causality settings
MAX_LAG  = 5      # search range [1, 5] per assignment
ALPHA    = 0.05   # 5% significance
EWMA_LAM = 0.94   # RiskMetrics standard, as used in ML_finance_lab1

# Output directory
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Silence the InterpolationWarning, FutureWarning, RuntimeWarning chatter that
# statsmodels and pandas emit during the small-sample monthly fits. We still
# fail loudly on actual exceptions; warnings are merely cosmetic noise.
warnings.filterwarnings("ignore")

# === Part 2: Data download and cleaning =====================================
# yfinance returns Adjusted Close when auto_adjust=True (default in newer
# versions). We request adjusted closes since splits/dividends would otherwise
# distort returns over a two-year window that includes regular ETF distributions.
print(f"Downloading daily prices for {len(ASSETS)} assets from {START_DATE} to {END_DATE}.")
raw = yf.download(ASSETS, start=START_DATE, end=END_DATE,
                  auto_adjust=True, progress=False)

# yfinance returns a MultiIndex (field, ticker); we keep the Close field only.
prices = raw["Close"][ASSETS].copy()

# Drop any rows where every asset is NaN (rare; happens only for non-trading
# global holidays where even crypto is missing in yfinance feeds).
prices = prices.dropna(how="all")

# Crypto trades 7 days per week, ETFs only on weekdays. Aligning everyone to
# the ETF trading calendar (i.e. dropping any timestamp that is not a NYSE
# trading day) keeps causality inference well-defined. We achieve this by
# inner-joining on rows where SPY (the equity benchmark) has a price.
prices = prices.loc[prices["SPY"].notna()].copy()

# Forward-fill remaining sporadic NaNs for crypto (occasional missing reads on
# exchange holidays); cap at 1 day to avoid carrying stale prices.
prices = prices.ffill(limit=1)

# After cleaning every column should be complete; if not, drop incomplete rows
# and report.
n_before = len(prices)
prices = prices.dropna(how="any")
print(f"Daily prices: {len(prices)} rows after cleaning "
      f"({n_before - len(prices)} dropped).")

# Daily log returns
log_ret_d = np.log(prices / prices.shift(1)).dropna()


# === Part 3: Resampling helpers =============================================
# Per the assignment we work at weekly and monthly frequency. We resample the
# adjusted closes and recompute log returns at the new frequency. This is the
# cleanest reading of "sampling the series weekly and then monthly".
def resample_prices(p: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample adjusted-close prices to the given pandas frequency rule
    (e.g. 'W-FRI' for weekly Friday closes, 'ME' for month-end). We use the
    last available observation in each period."""
    return p.resample(rule).last().dropna(how="any")


def log_returns(p: pd.DataFrame) -> pd.DataFrame:
    """Period log returns from a price panel."""
    return np.log(p / p.shift(1)).dropna()


# === Part 4: EWMA volatility (RiskMetrics) ==================================
def ewma_vol(returns: pd.DataFrame, lam: float = EWMA_LAM) -> pd.DataFrame:
    """RiskMetrics EWMA volatility, applied column-wise:
        sigma2_t = lam * sigma2_{t-1} + (1 - lam) * r_{t-1}^2
    Implemented via pandas .ewm(alpha=1-lam, adjust=False) on squared
    returns, then square-root. This matches ML_finance_lab1 line for line."""
    var = returns.pow(2).ewm(alpha=1 - lam, adjust=False).mean()
    # Drop the first observation: with adjust=False the recursion is seeded
    # by r_0^2 itself, so the first value is mechanical and uninformative.
    return np.sqrt(var).iloc[1:]


# === Part 5: Pairwise causality engine ======================================
def optimal_lag_aic(pair: pd.DataFrame, max_lag: int = MAX_LAG) -> int:
    """Bivariate VAR on the two columns of `pair`. Return the AIC-optimal
    lag in [1, max_lag]. We use trend='n' (no constant) to mirror the R
    convention used in PyLab3_1 (`VARselect(..., type='none')`). If the
    selected order is 0 we floor it to 1, since the assignment requires us
    to test causality at some lag in [1, 5]."""
    model = VAR(pair)
    sel = model.select_order(maxlags=max_lag, trend="n")
    chosen = int(sel.aic)
    return max(chosen, 1)


def granger_pvalue(cause_then_effect: pd.DataFrame, lag: int) -> float:
    """statsmodels.grangercausalitytests tests whether the SECOND column
    Granger-causes the FIRST. We standardise by passing data in the order
    [effect, cause] so this function's input order is the natural
    [cause, effect]. Return the p-value of the F-test at `lag`."""
    cause, effect = cause_then_effect.columns
    data_for_test = cause_then_effect[[effect, cause]]
    res = grangercausalitytests(data_for_test, maxlag=[lag], verbose=False)
    return float(res[lag][0]["ssr_ftest"][1])


def causality_table(returns_or_vol: pd.DataFrame,
                    max_lag: int = MAX_LAG,
                    alpha: float = ALPHA) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the directed causality table for the given panel. Rows are the
    cause, columns the effect, entries the AIC-optimal lag if the test
    rejects the null at level alpha, else 0. Diagonal entries are NaN
    (printed as '-' in LaTeX). Also return a parallel table of p-values for
    diagnostic purposes."""
    cols = returns_or_vol.columns.tolist()
    lag_table = pd.DataFrame(0, index=cols, columns=cols, dtype=object)
    pval_table = pd.DataFrame(np.nan, index=cols, columns=cols, dtype=float)

    for a in cols:
        for b in cols:
            if a == b:
                lag_table.loc[a, b] = "-"
                continue
            # Bivariate VAR for lag selection (order [a, b] is irrelevant for
            # AIC since the model is symmetric in its two endogenous series).
            pair = returns_or_vol[[a, b]].dropna()
            if len(pair) <= max_lag + 2:
                # Not enough observations to fit a VAR(max_lag); record 0.
                lag_table.loc[a, b] = 0
                continue
            try:
                lag = optimal_lag_aic(pair, max_lag=max_lag)
                # Granger test: a -> b at the chosen lag
                p = granger_pvalue(returns_or_vol[[a, b]].dropna(), lag)
                pval_table.loc[a, b] = p
                lag_table.loc[a, b] = lag if p < alpha else 0
            except Exception as exc:
                # Numerical singularity in tiny samples: record 0 and move on.
                lag_table.loc[a, b] = 0
                print(f"  causality {a}->{b}: failed ({type(exc).__name__}); recorded 0")
    return lag_table, pval_table


# === Part 6: Build the four directed lag tables =============================
# Build the four series we need. Returns and volatility at weekly and
# monthly frequency.
prices_W = resample_prices(prices, "W-FRI")     # weekly Friday closes
prices_M = resample_prices(prices, "ME")        # month-end closes

ret_W = log_returns(prices_W)
ret_M = log_returns(prices_M)

vol_W = ewma_vol(ret_W)
vol_M = ewma_vol(ret_M)

# Rename columns to display labels for the output artefacts
for df in (ret_W, ret_M, vol_W, vol_M):
    df.rename(columns=LABELS, inplace=True)

print(f"Weekly:  {len(ret_W)} return obs, {len(vol_W)} EWMA-vol obs.")
print(f"Monthly: {len(ret_M)} return obs, {len(vol_M)} EWMA-vol obs.")

print("\nBuilding causality tables (this fits 4 x 90 = 360 bivariate VARs)")

print("  returns, weekly")
lag_rW, p_rW = causality_table(ret_W)
print("  returns, monthly")
lag_rM, p_rM = causality_table(ret_M)
print("  EWMA volatility, weekly")
lag_vW, p_vW = causality_table(vol_W)
print("  EWMA volatility, monthly")
lag_vM, p_vM = causality_table(vol_M)


# === Part 7: Save tables as CSV and as LaTeX fragments =======================
def save_table(lag_tab: pd.DataFrame, name: str, caption: str) -> None:
    """Write the lag table to CSV and to a self-contained LaTeX fragment.
    The LaTeX uses booktabs and is intended to be \\input{} from the report."""
    csv_path = OUTPUT_DIR / f"table_{name}.csv"
    tex_path = OUTPUT_DIR / f"table_{name}.tex"
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
    lines.append(r"\label{tab:" + name + r"}")
    lines.append(r"\end{table}")
    tex_path.write_text("\n".join(lines))


save_table(lag_rW, "returns_weekly",
           "Pairwise Granger causality on weekly log returns; "
           "AIC-optimal lag in $[1,5]$ if significant at the 5\\% level, "
           "0 otherwise. Rows are the cause, columns the effect.")
save_table(lag_rM, "returns_monthly",
           "Pairwise Granger causality on monthly log returns; "
           "entries as in the weekly returns table.")
save_table(lag_vW, "vol_weekly",
           "Pairwise Granger causality on weekly EWMA volatility "
           "(RiskMetrics, $\\lambda{=}0.94$); entries as in the returns tables.")
save_table(lag_vM, "vol_monthly",
           "Pairwise Granger causality on monthly EWMA volatility "
           "(RiskMetrics, $\\lambda{=}0.94$); entries as in the returns tables.")


# === Part 8: Plot the four causality networks ===============================
# Asset-class colour map: distinguishing the macro buckets makes the network
# narrative immediate (who drives whom, where fear spills across classes).
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

# Fixed circular order so all four networks are visually comparable
CIRCLE_ORDER = ["SPY", "XLK", "XLF", "XLE", "GSG",
                "GLD", "SLV", "TLT", "ETH", "BTC"]


def _fixed_circular_layout(nodes: list) -> dict:
    """Deterministic circular layout in a fixed asset order."""
    n = len(nodes)
    pos = {}
    for i, name in enumerate(nodes):
        theta = np.pi / 2 - 2 * np.pi * i / n   # start at top, go clockwise
        pos[name] = np.array([np.cos(theta), np.sin(theta)])
    return pos


def plot_network(lag_tab: pd.DataFrame, title: str, out_pdf: Path) -> None:
    """Draw the directed graph implied by lag_tab. Nodes are assets coloured
    by asset class; an edge a -> b is drawn whenever lag_tab.loc[a, b] is a
    positive integer, labelled with that lag. Node positions are fixed
    across the four figures so plots are visually comparable."""
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
        # Curved edges with prominent arrowheads
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

    # Legend for asset classes
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
             OUTPUT_DIR / "network_returns_weekly.pdf")
plot_network(lag_rM, "Returns causality network (monthly)",
             OUTPUT_DIR / "network_returns_monthly.pdf")
plot_network(lag_vW, "Volatility spillover network (weekly, EWMA)",
             OUTPUT_DIR / "network_vol_weekly.pdf")
plot_network(lag_vM, "Volatility spillover network (monthly, EWMA)",
             OUTPUT_DIR / "network_vol_monthly.pdf")


# === Part 9: Contemporaneous correlations ===================================
# Helpful for the narrative: we report the contemporaneous correlation matrix
# at each frequency so the report can compare the lead-lag picture against the
# instantaneous co-movement picture.
ret_W.corr().round(2).to_csv(OUTPUT_DIR / "contemp_corr_returns_weekly.csv")
ret_M.corr().round(2).to_csv(OUTPUT_DIR / "contemp_corr_returns_monthly.csv")
vol_W.corr().round(2).to_csv(OUTPUT_DIR / "contemp_corr_vol_weekly.csv")
vol_M.corr().round(2).to_csv(OUTPUT_DIR / "contemp_corr_vol_monthly.csv")


# === Wrap-up: print the four tables to stdout for quick inspection ==========
def show(name: str, t: pd.DataFrame) -> None:
    print(f"\n--- {name} ---")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(t)


show("Returns causality lags (weekly)",  lag_rW)
show("Returns causality lags (monthly)", lag_rM)
show("Volatility causality lags (weekly)",  lag_vW)
show("Volatility causality lags (monthly)", lag_vM)

# Count edges per frequency for the narrative
print("\nEdge counts (significant causalities):")
for name, tab in [("ret_W", lag_rW), ("ret_M", lag_rM),
                  ("vol_W", lag_vW), ("vol_M", lag_vM)]:
    n_edges = sum(1 for a in tab.index for b in tab.columns
                  if isinstance(tab.loc[a, b], (int, np.integer)) and tab.loc[a, b] > 0)
    print(f"  {name}: {n_edges} edges out of 90 ordered pairs")

print("\nDone. Artefacts written to ./output/")
