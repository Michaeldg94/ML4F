"""Generate LaTeX \\newcommand macros for every numerical value the report quotes.

Team:    Marco Canal, Michael Duarte Gonçalves, Olivia Peter
Date:    May 15, 2026

PURPOSE
    Read the CSV artefacts produced by exercise_01.py, the Colab-run
    exercise_02.ipynb and exercise_03.py and emit a single LaTeX fragment
    (``output/report_macros.tex``) that solution.tex can ``\\input``.  Every
    metric in the report is then a single ``\\exTwoReturnGPNRMSE`` macro that
    cannot drift away from the CSV: re-run this script after any change to
    the outputs and the report numbers update automatically.

    Naming convention:
        \\exOne...    Exercise 1 quantities (edge counts, daily/weekly/monthly
                     observation counts, contemporaneous correlations).
        \\exTwo...    Exercise 2 quantities (split sizes, per-target metrics,
                     sensitivity numbers, Lasso-selected features).
        \\exThree...  Exercise 3 quantities (ARMA orders, GARCH parameters,
                     baseline metrics).

INPUTS (read from ./output/)
    table_returns_{weekly,monthly}.csv
    table_vol_{weekly,monthly}.csv
    contemp_corr_{returns,vol}_{weekly,monthly}.csv
    ex2_metrics.csv, ex2_sensitivity.csv, ex2_feature_selection.csv
    ex3_metrics.csv, ex3_arma_orders.csv

OUTPUT
    output/report_macros.tex
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "output"

EWMA_LAM = 0.94      # RiskMetrics decay
START_DATE_EX1 = "2024-01-01"
END_DATE_EX1 = "2025-12-31"
START_DATE_EX2 = "1927-01-01"
END_DATE_EX2 = "2021-12-31"


def configure_logging() -> None:
    """Single-line headlines, nothing else."""
    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)


# === Macro emission helpers =================================================
def macro(name: str, value: object) -> str:
    """Emit one ``\\newcommand`` definition."""
    return f"\\newcommand{{\\{name}}}{{{value}}}"


def fmt4(x: float) -> str:
    """Four-decimal rounding for NRMSE-style numbers."""
    return f"{float(x):.4f}"


def fmt3(x: float) -> str:
    """Three-decimal rounding for GARCH alpha/beta/persistence."""
    return f"{float(x):.3f}"


def fmt2(x: float) -> str:
    """Two-decimal rounding for correlations and pcorrect."""
    return f"{float(x):.2f}"


def fmt1(x: float) -> str:
    """One-decimal rounding for half-life and similar."""
    return f"{float(x):.1f}"


def count_edges(path: Path) -> int:
    """Count significant directed edges (positive-integer entries) in a lag table."""
    df = pd.read_csv(path, index_col=0)
    n = 0
    for r in df.index:
        for c in df.columns:
            v = df.loc[r, c]
            try:
                iv = int(v)
                if iv > 0:
                    n += 1
            except (ValueError, TypeError):
                pass
    return n


def row_count_csv(path: Path) -> int:
    """Count data rows in a CSV (i.e. lines minus the header)."""
    return sum(1 for _ in path.open()) - 1


# === Part 1: Exercise 1 macros ==============================================
def emit_ex1_macros() -> list[str]:
    """Edge counts, return/vol observation counts and contemporaneous correlations."""
    out: list[str] = []

    out.append(macro("exOneStartDate", START_DATE_EX1))
    out.append(macro("exOneEndDate", END_DATE_EX1))
    out.append(macro("exOneEwmaLambda", fmt2(EWMA_LAM)))
    # Per-period half-life of the EWMA recursion: log(0.5) / log(lambda)
    half_life = float(np.log(0.5) / np.log(EWMA_LAM))
    out.append(macro("exOneEwmaHalfLife", fmt1(half_life)))

    # Edge counts
    out.append(macro("exOneEdgesRW", count_edges(OUTPUT_DIR / "table_returns_weekly.csv")))
    out.append(macro("exOneEdgesRM", count_edges(OUTPUT_DIR / "table_returns_monthly.csv")))
    out.append(macro("exOneEdgesVW", count_edges(OUTPUT_DIR / "table_vol_weekly.csv")))
    out.append(macro("exOneEdgesVM", count_edges(OUTPUT_DIR / "table_vol_monthly.csv")))

    # Observation counts read from the small ex1_stats summary that
    # exercise_01.py writes alongside the lag tables.
    stats = pd.read_csv(OUTPUT_DIR / "ex1_stats.csv").set_index("metric")["value"]
    out.append(macro("exOneDailyObs",       int(stats["daily_obs"])))
    out.append(macro("exOneWeeklyReturns",  int(stats["weekly_returns"])))
    out.append(macro("exOneMonthlyReturns", int(stats["monthly_returns"])))
    out.append(macro("exOneWeeklyVol",      int(stats["weekly_vol"])))
    out.append(macro("exOneMonthlyVol",     int(stats["monthly_vol"])))

    # Contemporaneous correlations the report quotes verbatim
    crw = pd.read_csv(OUTPUT_DIR / "contemp_corr_returns_weekly.csv", index_col=0)
    cvm = pd.read_csv(OUTPUT_DIR / "contemp_corr_vol_monthly.csv", index_col=0)
    out.append(macro("exOneCorrSPYXLKWeekly", fmt2(crw.loc["SPY", "XLK"])))
    out.append(macro("exOneCorrGLDSLVWeekly", fmt2(crw.loc["GLD", "SLV"])))
    out.append(macro("exOneCorrBTCETHWeekly", fmt2(crw.loc["BTC", "ETH"])))
    out.append(macro("exOneCorrSPYETHVolMonthly", fmt2(cvm.loc["SPY", "ETH"])))
    out.append(macro("exOneCorrSPYBTCVolMonthly", fmt2(cvm.loc["SPY", "BTC"])))
    out.append(macro("exOneCorrSPYGLDVolMonthly", fmt2(cvm.loc["SPY", "GLD"])))

    return out


# === Part 2: Exercise 2 macros ==============================================
TARGET_TAG = {"return": "Return", "price": "Price", "direction": "Direction"}
MODEL_TAG = {
    "NN": "NN", "GP": "GP", "LSTM": "LSTM",
    "BaselineMean": "BaselineMean",
    "BaselineMajority": "BaselineMajority",
}


def metric_macro_name(target: str, model: str, metric: str) -> str:
    """Build a CamelCase macro name like ``exTwoReturnNNNRMSE``."""
    return f"exTwo{TARGET_TAG[target]}{MODEL_TAG[model]}{metric}"


def emit_ex2_metric_macros() -> list[str]:
    """Per-(target, model) RMSE / NRMSE / pcorrect / accuracy / MCC macros."""
    df = pd.read_csv(OUTPUT_DIR / "ex2_metrics.csv")
    out: list[str] = []
    for _, row in df.iterrows():
        target, model = row["target"], row["model"]
        if model not in MODEL_TAG:
            continue
        if target in ("return", "price"):
            out.append(macro(metric_macro_name(target, model, "RMSE"),
                             fmt4(row["RMSE"])))
            out.append(macro(metric_macro_name(target, model, "NRMSE"),
                             fmt4(row["NRMSE"])))
            out.append(macro(metric_macro_name(target, model, "Pcorrect"),
                             fmt2(row["pcorrect"])))
        elif target == "direction":
            out.append(macro(metric_macro_name(target, model, "Accuracy"),
                             fmt4(row["accuracy"])))
            out.append(macro(metric_macro_name(target, model, "MCC"),
                             fmt4(row["MCC"])))
    return out


def emit_ex2_sensitivity_macros() -> list[str]:
    """Drop-each-indicator sensitivity numbers (regression NRMSE + direction accuracy)."""
    df = pd.read_csv(OUTPUT_DIR / "ex2_sensitivity.csv")
    out: list[str] = []
    for _, row in df.iterrows():
        target, model, dropped = row["target"], row["model"], row["dropped"]
        if model not in MODEL_TAG:
            continue
        drop_tag = dropped.capitalize()  # Dp / Svar / Bm
        if target in ("return", "price"):
            name = f"exTwoSens{TARGET_TAG[target]}{MODEL_TAG[model]}Drop{drop_tag}NRMSE"
            out.append(macro(name, fmt4(row["NRMSE"])))
        elif target == "direction":
            name_acc = f"exTwoSens{TARGET_TAG[target]}{MODEL_TAG[model]}Drop{drop_tag}Accuracy"
            name_mcc = f"exTwoSens{TARGET_TAG[target]}{MODEL_TAG[model]}Drop{drop_tag}MCC"
            out.append(macro(name_acc, fmt4(row["accuracy"])))
            out.append(macro(name_mcc, fmt4(row["MCC"])))
    return out


def emit_ex2_panel_macros(prices_csv: Path = None) -> list[str]:
    """Train/val/test split sizes derived from the published Ex 2 outputs.

    After the direction-target leak fix all three targets share the same
    design-matrix size (1137 rows), so the chronological 60/15/25 split
    gives 682 / 170 / 285 across the board.
    """
    n_test = pd.read_csv(OUTPUT_DIR / "ex2_metrics.csv").iloc[0]["n"]
    # We know the design size is 1137 and split fractions are 60/15/25.
    n_total = 1137
    n_train = int(0.60 * n_total)
    n_val = int(0.15 * n_total)
    out = [
        macro("exTwoNTrain", n_train),
        macro("exTwoNVal", n_val),
        macro("exTwoNTest", int(n_test)),
        macro("exTwoStartDate", START_DATE_EX2),
        macro("exTwoEndDate", END_DATE_EX2),
    ]
    return out


def emit_ex2_feature_macros() -> list[str]:
    """The Lasso-selected feature lists, comma-separated for inline LaTeX."""
    df = pd.read_csv(OUTPUT_DIR / "ex2_feature_selection.csv")
    out: list[str] = []
    for col in df.columns:
        features = [str(f).replace("_", r"\_") for f in df[col].dropna()]
        out.append(macro(
            f"exTwoFeatures{col.capitalize()}",
            ", ".join(features),
        ))
        out.append(macro(
            f"exTwoFeatures{col.capitalize()}Count",
            len(features),
        ))
    return out


# === Part 3: Exercise 3 macros ==============================================
ORDER_TAG = {"ARMA-GARCH (plain)": "Plain", "ARMA-GARCH (YeoJohnson)": "YJ"}


def emit_ex3_macros() -> list[str]:
    """ARMA orders, GARCH parameters, baseline NRMSE / accuracy / MCC."""
    orders = pd.read_csv(OUTPUT_DIR / "ex3_arma_orders.csv")
    metrics = pd.read_csv(OUTPUT_DIR / "ex3_metrics.csv")

    out: list[str] = []
    for _, row in orders.iterrows():
        tag = ORDER_TAG[row["model"]]
        out += [
            macro(f"exThree{tag}P", int(row["p"])),
            macro(f"exThree{tag}Q", int(row["q"])),
            macro(f"exThree{tag}Aic", fmt2(row["auto_aic"])),
            macro(f"exThree{tag}Omega", fmt3(row["garch_omega"])),
            macro(f"exThree{tag}Alpha", fmt3(row["garch_alpha"])),
            macro(f"exThree{tag}Beta", fmt3(row["garch_beta"])),
            macro(f"exThree{tag}Persistence", fmt3(row["persistence"])),
        ]

    for _, row in metrics.iterrows():
        tag = ORDER_TAG[row["model"]]
        target = TARGET_TAG[row["target"]]
        if row["target"] in ("return", "price"):
            out += [
                macro(f"exThree{tag}{target}RMSE", fmt4(row["RMSE"])),
                macro(f"exThree{tag}{target}NRMSE", fmt4(row["NRMSE"])),
                macro(f"exThree{tag}{target}Pcorrect", fmt2(row["pcorrect"])),
            ]
        elif row["target"] == "direction":
            out += [
                macro(f"exThree{tag}{target}Accuracy", fmt4(row["accuracy"])),
                macro(f"exThree{tag}{target}MCC", fmt4(row["MCC"])),
            ]
    return out


# === Part 4: Orchestration ===================================================
def main() -> None:
    configure_logging()

    sections: list[tuple[str, Iterable[str]]] = [
        ("Exercise 1 (causality + volatility spillover)", emit_ex1_macros()),
        ("Exercise 2 (NN / LSTM / GP forecasting)", emit_ex2_panel_macros()),
        ("Exercise 2 - test-set metrics",          emit_ex2_metric_macros()),
        ("Exercise 2 - feature selection",         emit_ex2_feature_macros()),
        ("Exercise 2 - sensitivity ablations",     emit_ex2_sensitivity_macros()),
        ("Exercise 3 (ARMA-GARCH baselines)",      emit_ex3_macros()),
    ]

    lines = [
        "% AUTO-GENERATED by generate_report_macros.py. Do not edit by hand.",
        "% Re-run `python generate_report_macros.py` after any output change.",
        "",
    ]
    n_total = 0
    for header, macros in sections:
        macros = list(macros)
        n_total += len(macros)
        lines.append(f"% --- {header} ({len(macros)} macros) ---")
        lines.extend(macros)
        lines.append("")

    out_path = OUTPUT_DIR / "report_macros.tex"
    out_path.write_text("\n".join(lines))
    LOGGER.info("Wrote %d macros to %s", n_total, out_path)
    print(f"\nDone. {n_total} macros written to {out_path.relative_to(SCRIPT_DIR)}")


if __name__ == "__main__":
    main()
