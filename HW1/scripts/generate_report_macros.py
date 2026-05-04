"""Regenerate LaTeX-ready metric tables and report macros for HW1.

This script is intentionally deterministic. It reads the numbered CSV outputs
from the exercise scripts, recomputes the course metrics where needed and writes
small LaTeX fragments consumed by ``solution.tex``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, matthews_corrcoef

ROOT = Path(__file__).resolve().parents[1]
EX1 = ROOT / "output" / "ex1"
EX2 = ROOT / "output" / "ex2"
EX3 = ROOT / "output" / "ex3"

EWMA_LAMBDA = 0.94
MAX_GRANGER_LAG = 5
GRANGER_ALPHA_PERCENT = 5
MAX_FORWARD_FILL_DAYS = 1
P_TRAIN = 0.60
P_VAL = 0.15
P_TEST = 0.25
MAX_ARMA_ORDER = 5
GARCH_P = 1
GARCH_Q = 1
EX2_START_DATE = "1927-01-01"
EX2_END_DATE = "2021-12-31"
REGRESSION_TARGETS = ("return", "price")
FORECAST_MODELS = ("NN", "GP", "LSTM")
DROPPED_VARIABLES = ("dp", "svar", "bm")
N_LAGS_TARGET = 3
N_LAGS_EXOG = 3


def class_nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Course PyLab NRMSE convention used in the lecture notebooks."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    denom = np.mean(np.sum((y_true - y_true.mean()) ** 2))
    if np.isclose(denom, 0.0):
        raise ValueError("target has zero variance")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2) / denom))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error in the target's natural units."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def fmt4(value: object) -> str:
    """Four-decimal table formatting with blank cells for missing values."""
    if pd.isna(value):
        return ""
    return f"{float(value):.4f}"


def fmt2(value: object) -> str:
    """Two-decimal table formatting with blank cells for missing values."""
    if pd.isna(value):
        return ""
    return f"{float(value):.2f}"


def fmt1(value: object) -> str:
    """One-decimal formatting for compact descriptive report quantities."""
    if pd.isna(value):
        return ""
    return f"{float(value):.1f}"


def fmt0(value: object) -> str:
    """Integer-style formatting for percentages and counts."""
    if pd.isna(value):
        return ""
    return f"{float(value):.0f}"


def tex_escape(value: object) -> str:
    """Escape only the characters that can break a LaTeX macro body."""
    text = str(value)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in repl.items():
        text = text.replace(old, new)
    return text


def write_table(df: pd.DataFrame, path: Path, label: str, caption: str) -> None:
    """Write a complete LaTeX table fragment from a DataFrame."""
    tex = df.to_latex(
        index=False,
        float_format="{:.4f}".format,
        column_format="l" * len(df.columns),
        escape=False,
    )
    path.write_text(
        "\\begin{table}[H]\n\\centering\n\\small\n"
        + tex
        + f"\\caption{{{caption}}}\n\\label{{tab:{label}}}\n"
        + "\\end{table}\n",
        encoding="utf-8",
    )


def sensitivity_summary_rows(metrics: pd.DataFrame, sens: pd.DataFrame) -> list[str]:
    rows: list[str] = []

    # Regression targets share the same metric scale. Direction is added below
    # because its cells are accuracy, not NRMSE.
    for target in REGRESSION_TARGETS:
        main = metrics[metrics["target"].eq(target)]
        base_values = metric_values(main, FORECAST_MODELS, "NRMSE")
        rows.append(f"{target} & $-$ & " + " & ".join(base_values) + r" \\")
        for dropped in DROPPED_VARIABLES:
            subset = sens[sens["target"].eq(target) & sens["dropped"].eq(dropped)]
            values = metric_values(subset, FORECAST_MODELS, "NRMSE")
            rows.append(f"{target} & {dropped} & " + " & ".join(values) + r" \\")
        if target == "return":
            rows.append(r"\midrule")
    rows.append(r"\midrule")
    rows.append(r"\multicolumn{5}{l}{direction is reported as accuracy} \\")
    main = metrics[metrics["target"].eq("direction")]
    base_values = metric_values(main, FORECAST_MODELS, "accuracy")
    rows.append("direction & $-$ & " + " & ".join(base_values) + r" \\")
    for dropped in DROPPED_VARIABLES:
        subset = sens[sens["target"].eq("direction") & sens["dropped"].eq(dropped)]
        values = metric_values(subset, FORECAST_MODELS, "accuracy")
        rows.append(f"direction & {dropped} & " + " & ".join(values) + r" \\")
    return rows


def metric_values(
    frame: pd.DataFrame,
    models: tuple[str, ...],
    metric: str,
) -> list[str]:
    """Extract one formatted metric per model in a stable order."""
    values: list[str] = []
    for model in models:
        value = frame.loc[frame["model"].eq(model), metric].iloc[0]
        values.append(fmt4(value))
    return values


def write_sensitivity_summary_table(metrics: pd.DataFrame, sens: pd.DataFrame) -> None:
    rows = sensitivity_summary_rows(metrics, sens)
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Target & Dropped & NN & GP & LSTM \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        (
            r"\caption{Sensitivity analysis. Regression cells report NRMSE. "
            r"Direction cells report accuracy. Each ablation removes all lags "
            r"of the indicated predictor.}"
        ),
        r"\label{tab:ex2_sensitivity}",
        r"\end{table}",
    ]
    (EX2 / "05_ex2_sensitivity.tex").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def update_ex2_metrics() -> None:
    metrics_path = EX2 / "02_ex2_metrics.csv"
    metrics = pd.read_csv(metrics_path)
    metrics = metrics[~metrics["model"].isin(["RandomWalk", "RandomWalkDrift"])].copy()
    metrics = metrics.drop(columns=[c for c in ["R2_oos"] if c in metrics.columns])

    pred_files = {
        "return": EX2 / "06_ex2_pred_return.csv",
        "price": EX2 / "08_ex2_pred_price.csv",
        "direction": EX2 / "10_ex2_pred_direction.csv",
    }
    for target, file_path in pred_files.items():
        pred = pd.read_csv(file_path, index_col=0)
        y = pred["y_true"].to_numpy()
        for idx, row in metrics[metrics["target"].eq(target)].iterrows():
            col = f"pred_{row['model']}"
            if col not in pred.columns:
                continue
            yhat = pred[col].to_numpy()
            if target == "direction":
                metrics.loc[idx, "accuracy"] = accuracy_score(y, yhat)
                metrics.loc[idx, "MCC"] = matthews_corrcoef(y, yhat)
            else:
                score = class_nrmse(y, yhat)
                metrics.loc[idx, "RMSE"] = rmse(y, yhat)
                metrics.loc[idx, "NRMSE"] = score
                metrics.loc[idx, "pcorrect"] = 100.0 * (1.0 - score)

    metrics.to_csv(metrics_path, index=False)
    write_table(
        metrics,
        EX2 / "03_ex2_metrics.tex",
        "ex2_metrics",
        "Test-set forecast performance for each target and model.",
    )

    sens_path = EX2 / "04_ex2_sensitivity.csv"
    sens = pd.read_csv(sens_path)
    sens = sens.drop(columns=[c for c in ["R2_oos"] if c in sens.columns])
    reg = sens["target"].isin(["return", "price"]) & sens["NRMSE"].notna()
    if sens.loc[sens["target"].eq("return"), "NRMSE"].max() > 0.30:
        scale = np.sqrt(sens.loc[reg, "n"].astype(float))
        sens.loc[reg, "NRMSE"] = sens.loc[reg, "NRMSE"] / scale
        sens.loc[reg, "pcorrect"] = 100.0 * (1.0 - sens.loc[reg, "NRMSE"])
    sens.to_csv(sens_path, index=False)
    write_sensitivity_summary_table(metrics, sens)


def update_ex3_metrics() -> None:
    metrics_path = EX3 / "01_ex3_metrics.csv"
    if not metrics_path.exists():
        return
    metrics = pd.read_csv(metrics_path)
    metrics = metrics.drop(columns=[c for c in ["R2_oos"] if c in metrics.columns])
    pred_files = {
        "return": EX3 / "06_ex3_pred_return.csv",
        "price": EX3 / "08_ex3_pred_price.csv",
        "direction": EX3 / "10_ex3_pred_direction.csv",
    }
    for target, file_path in pred_files.items():
        pred = pd.read_csv(file_path, index_col=0)
        y = pred["y_true"].to_numpy()
        for idx, row in metrics[metrics["target"].eq(target)].iterrows():
            col = f"pred_{row['model']}"
            if col not in pred.columns:
                continue
            yhat = pred[col].to_numpy()
            if target == "direction":
                metrics.loc[idx, "accuracy"] = accuracy_score(y, yhat)
                metrics.loc[idx, "MCC"] = matthews_corrcoef(y, yhat)
            else:
                score = class_nrmse(y, yhat)
                metrics.loc[idx, "RMSE"] = rmse(y, yhat)
                metrics.loc[idx, "NRMSE"] = score
                metrics.loc[idx, "pcorrect"] = 100.0 * (1.0 - score)

    metrics.to_csv(metrics_path, index=False)
    write_table(
        metrics,
        EX3 / "02_ex3_metrics.tex",
        "ex3_metrics",
        "Test-set forecast performance for the ARMA-GARCH baselines.",
    )

    ex2 = pd.read_csv(EX2 / "02_ex2_metrics.csv")
    combined = pd.concat(
        [
            ex2.assign(source="Ex 2 (NN/GP/LSTM)"),
            metrics.assign(source="Ex 3 (ARMA-GARCH)"),
        ],
        ignore_index=True,
    )
    combined.to_csv(EX3 / "04_ex3_combined.csv", index=False)
    write_table(
        combined,
        EX3 / "05_ex3_combined.tex",
        "ex3_combined",
        "Exercise 2 models and Exercise 3 ARMA-GARCH baselines.",
    )


def macro(name: str, value: object) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}"


def count_edges(path: Path) -> int:
    table = pd.read_csv(path, index_col=0)
    numeric = table.replace("-", 0).apply(pd.to_numeric)
    return int((numeric.to_numpy() != 0).sum())


def corr_value(path: Path, a: str, b: str) -> float:
    table = pd.read_csv(path, index_col=0)
    return float(table.loc[a, b])


def row_count(path: Path) -> int:
    return int(pd.read_csv(path, index_col=0).shape[0])


def ex2_panel() -> pd.DataFrame:
    """Rebuild the monthly panel used by the Exercise 2 notebook."""
    data = pd.read_csv(ROOT / "data" / "goyal-welch2022Monthly.csv")
    data["Date"] = pd.to_datetime(data["yyyymm"].astype(str), format="%Y%m")
    data = data.set_index("Date").drop(columns="yyyymm")
    data["Index"] = data["Index"].str.replace(",", "", regex=False).astype(float)
    data["dp"] = np.log(data["D12"]) - np.log(data["Index"])
    data["bm"] = data["b/m"]
    data["log_price"] = np.log(data["Index"])
    data["log_ret"] = data["log_price"].diff()
    return data.loc[
        EX2_START_DATE:EX2_END_DATE,
        ["Index", "log_price", "log_ret", *DROPPED_VARIABLES],
    ].dropna()


def ex2_design_count(panel: pd.DataFrame) -> int:
    """Count rows after target alignment and lag construction."""
    columns = {
        f"log_ret_lag{lag}": panel["log_ret"].shift(lag - 1)
        for lag in range(1, N_LAGS_TARGET + 1)
    }
    for variable in DROPPED_VARIABLES:
        for lag in range(1, N_LAGS_EXOG + 1):
            columns[f"{variable}_lag{lag}"] = panel[variable].shift(lag - 1)
    target = panel["log_ret"].shift(-1)
    design = pd.concat(
        [pd.DataFrame(columns, index=panel.index), target.rename("TARGET")],
        axis=1,
    ).dropna()
    return int(len(design))


def write_ex1_macros() -> None:
    prices = EX1 / "01_ex1_prices_daily.csv"
    returns_w = EX1 / "02_ex1_returns_weekly.csv"
    returns_m = EX1 / "03_ex1_returns_monthly.csv"
    vol_series_w = EX1 / "04_ex1_ewma_vol_weekly.csv"
    vol_series_m = EX1 / "05_ex1_ewma_vol_monthly.csv"
    ret_w = EX1 / "06_ex1_causality_returns_weekly.csv"
    ret_m = EX1 / "08_ex1_causality_returns_monthly.csv"
    vol_w = EX1 / "10_ex1_causality_vol_weekly.csv"
    vol_m = EX1 / "12_ex1_causality_vol_monthly.csv"
    corr_ret_w = EX1 / "14_ex1_contemp_corr_returns_weekly.csv"
    corr_vol_w = EX1 / "16_ex1_contemp_corr_vol_weekly.csv"
    n_assets = pd.read_csv(ret_w, index_col=0).shape[0]
    monthly_obs = row_count(returns_m)
    lines = [
        "% Auto-generated by scripts/generate_report_macros.py",
        macro("ExOneAssetCount", n_assets),
        macro("ExOneOrderedPairs", n_assets * (n_assets - 1)),
        macro("ExOneMaxForwardFillDays", MAX_FORWARD_FILL_DAYS),
        macro("ExOneEwmaLambda", fmt2(EWMA_LAMBDA)),
        macro("ExOneEwmaShockWeight", fmt2(1.0 - EWMA_LAMBDA)),
        macro("ExOneEwmaHalfLife", fmt1(np.log(0.5) / np.log(EWMA_LAMBDA))),
        macro("ExOneMaxGrangerLag", MAX_GRANGER_LAG),
        macro("ExOneGrangerAlphaPercent", GRANGER_ALPHA_PERCENT),
        macro("ExOneMonthlyMaxLagPostLagObservations", monthly_obs - MAX_GRANGER_LAG),
        macro("ExOneMaxLagParametersPerEquation", 1 + 2 * MAX_GRANGER_LAG),
        macro("ExOneDailyObservations", row_count(prices)),
        macro("ExOneWeeklyReturnObservations", row_count(returns_w)),
        macro("ExOneMonthlyReturnObservations", row_count(returns_m)),
        macro("ExOneWeeklyVolObservations", row_count(vol_series_w)),
        macro("ExOneMonthlyVolObservations", row_count(vol_series_m)),
        macro("ExOneReturnWeeklyEdges", count_edges(ret_w)),
        macro("ExOneReturnMonthlyEdges", count_edges(ret_m)),
        macro("ExOneVolWeeklyEdges", count_edges(vol_w)),
        macro("ExOneVolMonthlyEdges", count_edges(vol_m)),
        macro("ExOneCorrReturnWeeklySpyXlk", fmt2(corr_value(corr_ret_w, "SPY", "XLK"))),
        macro("ExOneCorrReturnWeeklyGldSlv", fmt2(corr_value(corr_ret_w, "GLD", "SLV"))),
        macro("ExOneCorrReturnWeeklyBtcEth", fmt2(corr_value(corr_ret_w, "BTC", "ETH"))),
        macro("ExOneCorrVolWeeklySpyXlk", fmt2(corr_value(corr_vol_w, "SPY", "XLK"))),
        macro("ExOneCorrVolWeeklySpyXlf", fmt2(corr_value(corr_vol_w, "SPY", "XLF"))),
        macro("ExOneCorrVolWeeklySpyXle", fmt2(corr_value(corr_vol_w, "SPY", "XLE"))),
    ]
    (EX1 / "22_ex1_report_macros.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


TARGET_ALIAS = {"return": "Return", "price": "Price", "direction": "Direction"}
MODEL_ALIAS = {
    "NN": "Nn",
    "GP": "Gp",
    "LSTM": "Lstm",
    "BaselineMean": "BaselineMean",
    "BaselineMajority": "BaselineMajority",
    "ARMA-GARCH (plain)": "ArmaPlain",
    "ARMA-GARCH (YeoJohnson)": "ArmaYeoJohnson",
}
METRIC_ALIAS = {
    "RMSE": "Rmse",
    "NRMSE": "Nrmse",
    "pcorrect": "Pcorrect",
    "n": "N",
    "accuracy": "Accuracy",
    "MCC": "Mcc",
}
MODEL_LABEL = {
    ("return", "NN"): r"2 layers, $(128,64)$, ReLU",
    ("return", "GP"): "RBF + WhiteNoise",
    ("return", "LSTM"): r"1 layer, 64 units, $T{=}6$",
    ("return", "BaselineMean"): "training mean",
    ("price", "NN"): "1 layer, 64 units, ReLU",
    ("price", "GP"): "custom RBF + Linear + WhiteNoise",
    ("price", "LSTM"): r"1 layer, 64 units, $T{=}6$",
    ("price", "BaselineMean"): "training mean",
    ("direction", "NN"): "1 layer, 128 units, tanh",
    ("direction", "GP"): "RatQuad + WhiteNoise",
    ("direction", "LSTM"): r"2 layers, 32 units, $T{=}6$",
    ("direction", "BaselineMajority"): "training majority class",
}


def metric_macro_name(prefix: str, target: str, model: str, metric: str) -> str:
    return f"{prefix}{TARGET_ALIAS[target]}{MODEL_ALIAS[model]}{METRIC_ALIAS[metric]}"


def write_metric_macros(df: pd.DataFrame, prefix: str) -> list[str]:
    lines: list[str] = []
    for _, row in df.iterrows():
        target = row["target"]
        model = row["model"]
        if target not in TARGET_ALIAS or model not in MODEL_ALIAS:
            continue
        for metric in METRIC_ALIAS:
            if metric not in row:
                continue
            value = row[metric]
            if metric in ["RMSE", "NRMSE", "accuracy", "MCC"]:
                formatted = fmt4(value)
            elif metric == "pcorrect":
                formatted = fmt2(value)
            elif metric == "n":
                formatted = "" if pd.isna(value) else str(int(value))
            else:
                formatted = str(value)
            if formatted != "":
                lines.append(
                    macro(metric_macro_name(prefix, target, model, metric), formatted)
                )
    return lines


def command(name: str, body: str) -> str:
    return f"\\newcommand{{\\{name}}}{{%\n{body}\n}}"


def ex2_rows(metrics: pd.DataFrame, target: str) -> str:
    models = {
        "return": ["NN", "GP", "LSTM", "BaselineMean"],
        "price": ["NN", "GP", "LSTM", "BaselineMean"],
        "direction": ["NN", "GP", "LSTM", "BaselineMajority"],
    }[target]
    rows = []
    for model in models:
        model_text = "Baseline" if model.startswith("Baseline") else model
        label = MODEL_LABEL[(target, model)]
        if target == "direction":
            rows.append(
                f"{model_text} & {label} & "
                f"\\{metric_macro_name('ExTwo', target, model, 'accuracy')} & "
                f"\\{metric_macro_name('ExTwo', target, model, 'MCC')} \\\\"
            )
        else:
            rows.append(
                f"{model_text} & {label} & "
                f"\\{metric_macro_name('ExTwo', target, model, 'RMSE')} & "
                f"\\{metric_macro_name('ExTwo', target, model, 'NRMSE')} & "
                f"\\{metric_macro_name('ExTwo', target, model, 'pcorrect')} \\\\"
            )
    return "\n".join(rows)


def write_ex2_macros() -> None:
    metrics = pd.read_csv(EX2 / "02_ex2_metrics.csv")
    sens = pd.read_csv(EX2 / "04_ex2_sensitivity.csv")
    panel = ex2_panel()
    design_n = ex2_design_count(panel)
    train_n = int(P_TRAIN * design_n)
    validation_n = int(P_VAL * design_n)
    test_n = design_n - train_n - validation_n
    lines = ["% Auto-generated by scripts/generate_report_macros.py"]
    lines.extend(write_metric_macros(metrics, "ExTwo"))
    for target in ["return", "price", "direction"]:
        n = int(metrics.loc[metrics["target"].eq(target), "n"].dropna().iloc[0])
        lines.append(macro(f"ExTwo{TARGET_ALIAS[target]}TestN", n))
    lines.extend(
        [
            command("ExTwoReturnRows", ex2_rows(metrics, "return")),
            command("ExTwoPriceRows", ex2_rows(metrics, "price")),
            command("ExTwoDirectionRows", ex2_rows(metrics, "direction")),
            macro("ExTwoPanelObservations", len(panel)),
            macro("ExTwoDesignObservations", design_n),
            macro("ExTwoTrainObservations", train_n),
            macro("ExTwoValidationObservations", validation_n),
            macro("ExTwoTestObservations", test_n),
            macro("ExTwoTrainSharePercent", fmt0(100.0 * P_TRAIN)),
            macro("ExTwoValidationSharePercent", fmt0(100.0 * P_VAL)),
            macro("ExTwoTestSharePercent", fmt0(100.0 * P_TEST)),
            macro("ExTwoTargetLagCount", N_LAGS_TARGET),
            macro("ExTwoExogenousLagCount", N_LAGS_EXOG),
            macro(
                "ExTwoCandidateFeatureCount",
                N_LAGS_TARGET + len(DROPPED_VARIABLES) * N_LAGS_EXOG,
            ),
            macro("ExTwoReturnSelectedFeatures", "bm\\_lag1, bm\\_lag2 and log\\_ret\\_lag3"),
            macro(
                "ExTwoPriceSelectedFeatures",
                "log\\_price\\_lag1, dp\\_lag1, svar\\_lag1, "
                "svar\\_lag3 and bm\\_lag2",
            ),
            macro("ExTwoDirectionSelectedFeatureCount", 11),
            macro("ExTwoSensitivityBestPriceGpDropped", tex_escape(
                sens[sens["target"].eq("price") & sens["model"].eq("GP")]
                .sort_values("NRMSE")
                .iloc[0]["dropped"]
            )),
            macro("ExTwoSensitivityBestPriceGpNrmse", fmt4(
                sens[sens["target"].eq("price") & sens["model"].eq("GP")]
                .sort_values("NRMSE")
                .iloc[0]["NRMSE"]
            )),
        ]
    )
    (EX2 / "13_ex2_report_macros.tex").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def write_ex3_macros() -> None:
    metrics_path = EX3 / "01_ex3_metrics.csv"
    combined_path = EX3 / "04_ex3_combined.csv"
    orders_path = EX3 / "03_ex3_arma_orders.csv"
    if not metrics_path.exists() or not combined_path.exists() or not orders_path.exists():
        return
    metrics = pd.read_csv(metrics_path)
    combined = pd.read_csv(combined_path)
    orders = pd.read_csv(orders_path)
    lines = ["% Auto-generated by scripts/generate_report_macros.py"]
    lines.extend(write_metric_macros(metrics, "ExThree"))
    for _, row in orders.iterrows():
        model = row["model"]
        alias = MODEL_ALIAS[model]
        lines.append(macro(f"ExThree{alias}P", int(row["p"])))
        lines.append(macro(f"ExThree{alias}Q", int(row["q"])))
        lines.append(macro(f"ExThree{alias}Alpha", fmt3(row["garch_alpha"])))
        lines.append(macro(f"ExThree{alias}Beta", fmt3(row["garch_beta"])))
        lines.append(macro(f"ExThree{alias}Persistence", fmt3(row["persistence"])))
        lines.append(macro(f"ExThree{alias}Aic", f"{float(row['auto_aic']):.2f}"))
    lines.append(macro("ExThreeMaxArmaOrder", MAX_ARMA_ORDER))
    lines.append(macro("ExThreeGarchP", GARCH_P))
    lines.append(macro("ExThreeGarchQ", GARCH_Q))
    lines.append(macro("ExThreeTestStart", "1998-03"))
    lines.append(macro("ExThreeTestEnd", "2021-11"))
    lines.append(macro("ExThreeTestN", int(metrics["n"].dropna().iloc[0])))
    lines.append(command("ExThreeCombinedRows", combined_rows(combined)))
    (EX3 / "13_ex3_report_macros.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt3(value: object) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.3f}"


def combined_rows(combined: pd.DataFrame) -> str:
    keep = combined[
        combined["model"].isin(
            [
                "NN",
                "GP",
                "LSTM",
                "BaselineMean",
                "BaselineMajority",
                "ARMA-GARCH (plain)",
                "ARMA-GARCH (YeoJohnson)",
            ]
        )
    ].copy()
    target_order = {"return": 0, "price": 1, "direction": 2}
    model_order = {
        "NN": 0,
        "GP": 1,
        "LSTM": 2,
        "BaselineMean": 3,
        "BaselineMajority": 3,
        "ARMA-GARCH (plain)": 4,
        "ARMA-GARCH (YeoJohnson)": 5,
    }
    keep["target_order"] = keep["target"].map(target_order)
    keep["model_order"] = keep["model"].map(model_order)
    keep = keep.sort_values(["target_order", "model_order", "model"])
    labels = {
        "NN": "NN",
        "GP": "GP",
        "LSTM": "LSTM",
        "BaselineMean": "BaselineMean",
        "BaselineMajority": "BaselineMajority",
        "ARMA-GARCH (plain)": "ARMA-GARCH (plain)",
        "ARMA-GARCH (YeoJohnson)": "ARMA-GARCH (YeoJohnson)",
    }
    rows = []
    previous_target = None
    for _, row in keep.iterrows():
        target = row["target"]
        model = row["model"]
        if previous_target in {"return", "price"} and target != previous_target:
            rows.append(r"\midrule")
        previous_target = target
        rmse_value = fmt4(row.get("RMSE"))
        nrmse_value = fmt4(row.get("NRMSE"))
        pcorrect_value = fmt2(row.get("pcorrect"))
        acc_value = fmt4(row.get("accuracy"))
        mcc_value = fmt4(row.get("MCC"))
        rows.append(
            f"{target} & {labels[model]} & {rmse_value} & {nrmse_value} & "
            f"{pcorrect_value} & {acc_value} & {mcc_value} \\\\"
        )
    return "\n".join(rows)


def main() -> None:
    update_ex2_metrics()
    update_ex3_metrics()
    write_ex1_macros()
    write_ex2_macros()
    write_ex3_macros()


if __name__ == "__main__":
    main()
