"""Exercise 3: ARMA-GARCH baselines for the S&P 500 forecast.

Team:  Marco Canal, Michael Duarte Gonçalves, Olivia Peter
Date:  May 15, 2026

What this script does:
    Fits a plain ARMA-GARCH and a Yeo-Johnson-transformed ARMA-GARCH on
    the same chronological train+validation block used in Exercise 2, then
    scores both on the same test block. The point is a clean head-to-head
    against the neural-network / GP / LSTM line-up from Exercise 2.

    Numbered CSVs, LaTeX fragments and figures land in output/ex3/.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from arch import arch_model
from pmdarima.arima import auto_arima
from sklearn.metrics import accuracy_score, matthews_corrcoef, mean_squared_error
from sklearn.preprocessing import PowerTransformer
from statsmodels.tsa.arima.model import ARIMA

# auto_arima, arch, and statsmodels all complain about small samples or boundary fits. Real failures are caught further down.
warnings.filterwarnings("ignore")

LOGGER = logging.getLogger(__name__)


# --- Part 1: paths, knobs, and what we save

SEED = 7
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "scripts" else SCRIPT_DIR

DATA_FILE = PROJECT_ROOT / "data" / "goyal-welch2022Monthly.csv"
OUTPUT_DIR = PROJECT_ROOT / "output" / "ex3"
EX2_OUTPUT_DIR = PROJECT_ROOT / "output" / "ex2"

# Same window as Exercise 2.
START_DATE = "1927-01-01"
END_DATE   = "2021-12-31"
N_LAGS = 3

# Match the Exercise 2 split exactly, otherwise the head-to-head is not honest.
P_TRAIN = 0.60
P_VAL   = 0.15

# Numbered output files.
# The LaTeX side just \input{} them.
OUT = {
    "ex2_metrics": "02_ex2_metrics.csv",
    "metrics_csv": "01_ex3_metrics.csv",
    "metrics_tex": "02_ex3_metrics.tex",
    "orders": "03_ex3_arma_orders.csv",
    "combined_csv": "04_ex3_combined.csv",
    "combined_tex": "05_ex3_combined.tex",
    "pred_return": "06_ex3_pred_return.csv",
    "fig_return": "07_ex3_fig_return.pdf",
    "pred_price": "08_ex3_pred_price.csv",
    "fig_price": "09_ex3_fig_price.pdf",
    "pred_direction": "10_ex3_pred_direction.csv",
    "fig_direction": "11_ex3_fig_direction.pdf",
    "residual_diagnostics": "12_ex3_residual_diagnostics.csv",
}


def configure_logging() -> None:
    """Keep logs concise. Full tables go to disk, not to stdout."""
    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)


# ----- Part 2: load the Goyal-Welch panel

def load_monthly_panel() -> pd.DataFrame:
    """Pull Goyal-Welch from disk and build the two base series we need.

    The CSV stores the index level as a string with thousands separators,
    so we strip those and then take log price and log return on top.
    """
    LOGGER.info("Loading Goyal-Welch monthly data")
    data = pd.read_csv(DATA_FILE)
    data["Date"] = pd.to_datetime(data["yyyymm"].astype(str), format="%Y%m")
    data = data.set_index("Date").drop(columns="yyyymm")
    data["Index"] = data["Index"].str.replace(",", "", regex=False).astype(float)
    data["log_price"] = np.log(data["Index"])
    data["log_ret"] = data["log_price"].diff()

    panel = data.loc[START_DATE:END_DATE, ["Index", "log_price", "log_ret"]]
    panel = panel.dropna()
    LOGGER.info(
        "Panel has %d rows from %s to %s",
        len(panel),
        panel.index.min().date(),
        panel.index.max().date(),
    )
    return panel


# -- Part 3 -- design and chronological split

def return_design(panel: pd.DataFrame) -> pd.DataFrame:
    """Rebuild the Exercise 2 design so the split lines up date for date.

    Note: we don't actually feed these lagged columns into the ARMA fit
    (the state-space model carries its own history). We only build them so
    the indexing logic matches Exercise 2 exactly.
    """
    target = panel["log_ret"].shift(-1)
    lagged_returns = {
        f"log_ret_lag{lag}": panel["log_ret"].shift(lag - 1)
        for lag in range(1, N_LAGS + 1)
    }
    features = pd.DataFrame(lagged_returns, index=panel.index)
    return pd.concat([features, target.rename("TARGET")], axis=1).dropna()


def chronological_split_index(design: pd.DataFrame) -> tuple[pd.Index, pd.Index]:
    """Train+validation indices and test indices, in calendar order."""
    n_obs = len(design)
    n_train = int(P_TRAIN * n_obs)
    n_val = int(P_VAL * n_obs)
    split = n_train + n_val
    fit_idx = design.index[:split]
    test_idx = design.index[split:]
    LOGGER.info(
        "Fit period %s to %s with %d observations",
        fit_idx[0].date(),
        fit_idx[-1].date(),
        len(fit_idx),
    )
    LOGGER.info(
        "Test period %s to %s with %d observations",
        test_idx[0].date(),
        test_idx[-1].date(),
        len(test_idx),
    )
    return fit_idx, test_idx


# Part 4 --- the course scoring conventions

def nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """PyLab NRMSE used throughout the course.

    The variance in the denominator is taken on the realised series itself
    (not on the residuals). That is the PyLab convention, so we keep it.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    denominator = np.mean(np.sum((y_true - y_true.mean()) ** 2))
    if np.isclose(denominator, 0.0):
        raise ValueError("target has zero variance")
    mse = mean_squared_error(y_true, y_pred)
    return float(np.sqrt(mse / denominator))


def pcorrect(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """100 * (1 - NRMSE). Heads up: this is not a sign accuracy."""
    return 100.0 * (1.0 - nrmse(y_true, y_pred))


# Part 5 ----- ARMA + GARCH on the residuals -----

def transform_returns(
    fit_returns: np.ndarray,
    test_returns: np.ndarray,
    transformer: PowerTransformer | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Optional Yeo-Johnson transform.

    Fit on the estimation block,
    apply the same map to the test block.
    """
    if transformer is None:
        return fit_returns, test_returns
    fit_transformed = transformer.fit_transform(fit_returns.reshape(-1, 1)).ravel()
    test_transformed = transformer.transform(test_returns.reshape(-1, 1)).ravel()
    return fit_transformed, test_transformed


def fit_baseline(
    fit_returns: np.ndarray,
    test_returns: np.ndarray,
    transformer: PowerTransformer | None = None,
    max_pq: int = 5,
) -> dict:
    """One full pass: ARMA by AIC, GARCH on residuals, forecast test block.

    Returns a small dict with predictions, fitted orders, and GARCH params.
    """
    fit_t, test_t = transform_returns(fit_returns, test_returns, transformer)

    # Returns are already stationary for this benchmark, so we pin d = 0.
    order_search = auto_arima(
        fit_t,
        seasonal=False,
        d=0,
        max_p=max_pq,
        max_q=max_pq,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
        trace=False,
        information_criterion="aic",
    )
    p_order, _, q_order = order_search.order
    LOGGER.info(
        "Selected ARMA(%d,%d) with AIC %.2f",
        p_order,
        q_order,
        order_search.aic(),
    )

    arima = ARIMA(
        fit_t,
        order=(p_order, 0, q_order),
        enforce_stationarity=True,
        enforce_invertibility=True,
    ).fit()

    # ARMA handles the conditional mean; GARCH only sees residuals. Scaling by 100 helps numerical conditioning for the arch package and does not change alpha or beta (the persistence params we actually report).
    garch = arch_model(
        arima.resid * 100.0,
        mean="Zero",
        vol="GARCH",
        p=1,
        q=1,
        dist="normal",
    ).fit(disp="off")
    alpha = garch.params["alpha[1]"]
    beta = garch.params["beta[1]"]

    # Apply the fitted ARMA filter to the full series, so we can read off
    # the test-block one-step-ahead predictions without refitting.
    full_series = np.concatenate([fit_t, test_t])
    extended = arima.apply(full_series, refit=False)
    fitted_full = np.asarray(extended.fittedvalues)
    test_pred_t = fitted_full[len(fit_t) :]

    if transformer is None:
        test_pred = test_pred_t
    else:
        test_pred = transformer.inverse_transform(test_pred_t.reshape(-1, 1)).ravel()

    return {
        "pred": test_pred,
        "order": (p_order, q_order),
        "arima": arima,
        "garch": garch,
        "garch_omega": garch.params["omega"],
        "garch_alpha": alpha,
        "garch_beta": beta,
        "persistence": alpha + beta,
        "aic": float(order_search.aic()),
    }


# -------- Part 6: metric rows for the report

def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target: str,
    model: str,
    params: str,
) -> dict:
    """Row of metrics for return and price (the continuous targets)."""
    score = nrmse(y_true, y_pred)
    return {
        "target": target,
        "model": model,
        "params": params,
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "NRMSE": score,
        "pcorrect": 100.0 * (1.0 - score),
        "n": int(len(y_true)),
        "accuracy": np.nan,
        "MCC": np.nan,
    }


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target: str,
    model: str,
    params: str,
) -> dict:
    """Row of metrics for the binary direction target."""
    return {
        "target": target,
        "model": model,
        "params": params,
        "RMSE": np.nan,
        "NRMSE": np.nan,
        "pcorrect": np.nan,
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
    }


def describe_parameters(result: dict) -> str:
    """One-line summary of the fitted ARMA + GARCH parameters."""
    p_order, q_order = result["order"]
    return (
        f"ARMA({p_order},{q_order}); GARCH(1,1) "
        f"alpha={result['garch_alpha']:.3f}, "
        f"beta={result['garch_beta']:.3f}, "
        f"alpha+beta={result['persistence']:.3f}"
    )


def run_baselines(fit_returns: np.ndarray, test_returns: np.ndarray) -> dict[str, dict]:
    """Run both flavours of ARMA-GARCH."""
    # The two specifications we report on:
    # - plain log returns
    # - Yeo-Johnson-normalised log returns
    variants = {
        "ARMA-GARCH (plain)": None,
        "ARMA-GARCH (YeoJohnson)": PowerTransformer(method="yeo-johnson"),
    }
    results = {}
    for name, transformer in variants.items():
        LOGGER.info("Fitting %s", name)
        results[name] = fit_baseline(
            fit_returns,
            test_returns,
            transformer=transformer,
        )
    return results


def build_result_tables(
    results: dict[str, dict],
    y_return: np.ndarray,
    y_price: np.ndarray,
    y_direction: np.ndarray,
    log_price_t: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    """Turn the dict of fitted models into the artifacts the report needs.

    We get back: a metrics DataFrame, an orders DataFrame, and a nested
    dict of test-block predictions keyed by target then by model.
    """
    metric_rows = []
    order_rows = []
    predictions: dict[str, dict[str, np.ndarray]] = {
        "return": {},
        "price": {},
        "direction": {},
    }

    for model_name, result in results.items():
        return_pred = result["pred"]
        # Price prediction = today's log price + predicted log return.
        price_pred = log_price_t.values + return_pred
        direction_pred = (return_pred > 0).astype(int)
        params = describe_parameters(result)

        metric_rows.append(
            regression_metrics(y_return, return_pred, "return", model_name, params)
        )
        metric_rows.append(
            regression_metrics(y_price, price_pred, "price", model_name, params)
        )
        metric_rows.append(
            classification_metrics(
                y_direction,
                direction_pred,
                "direction",
                model_name,
                params,
            )
        )

        predictions["return"][model_name] = return_pred
        predictions["price"][model_name] = price_pred
        predictions["direction"][model_name] = direction_pred

        order_rows.append(
            {
                "model": model_name,
                "p": result["order"][0],
                "q": result["order"][1],
                "auto_aic": result["aic"],
                "garch_omega": result["garch_omega"],
                "garch_alpha": result["garch_alpha"],
                "garch_beta": result["garch_beta"],
                "persistence": result["persistence"],
            }
        )

    return pd.DataFrame(metric_rows), pd.DataFrame(order_rows), predictions


# Part 7 -- combine with Exercise 2 and write LaTeX --

def combine_with_ex2_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """Stack our metrics on top of Ex2's so the report has one comparison table.

    If Ex2 has not been run yet, we just return our own metrics and log a
    warning so the user notices.
    """
    ex2_path = EX2_OUTPUT_DIR / OUT["ex2_metrics"]
    if not ex2_path.exists():
        LOGGER.warning("Skipping combined table because %s is missing", ex2_path)
        return metrics

    ex2_metrics = pd.read_csv(ex2_path)
    return pd.concat(
        [
            ex2_metrics.assign(source="Ex 2 (NN/GP/LSTM)"),
            metrics.assign(source="Ex 3 (ARMA-GARCH)"),
        ],
        ignore_index=True,
    )


def write_latex_table(
    table: pd.DataFrame,
    file_name: str,
    label: str,
    caption: str,
) -> None:
    """Wrap a pandas tabular in a booktabs table environment."""
    tex = table.to_latex(
        index=False,
        float_format="{:.4f}".format,
        column_format="l" * len(table.columns),
        escape=False,
    )
    output = (
        "\\begin{table}[H]\n\\centering\n\\small\n"
        + tex
        + "\\caption{"
        + caption
        + "}\n\\label{tab:"
        + label
        + "}\n\\end{table}\n"
    )
    (OUTPUT_DIR / file_name).write_text(output, encoding="utf-8")


def write_tables(metrics: pd.DataFrame, orders: pd.DataFrame, combined: pd.DataFrame) -> None:
    """Save the three table files (CSV + LaTeX) the report reads back."""
    metrics.to_csv(OUTPUT_DIR / OUT["metrics_csv"], index=False)
    orders.to_csv(OUTPUT_DIR / OUT["orders"], index=False)
    combined.to_csv(OUTPUT_DIR / OUT["combined_csv"], index=False)

    write_latex_table(
        metrics,
        OUT["metrics_tex"],
        "ex3_metrics",
        "Test-set forecast performance for the ARMA(p,q)+GARCH(1,1) "
        "baselines (plain and Yeo-Johnson-normalised).",
    )
    write_latex_table(
        combined,
        OUT["combined_tex"],
        "ex3_combined",
        "Head-to-head comparison: Exercise 2 ML models versus Exercise 3 "
        "ARMA-GARCH baselines on the same chronological test split.",
    )


# ----- Part 8 ----- residual diagnostics

def residual_diagnostics(
    y_truths: dict[str, np.ndarray],
    predictions: dict[str, dict[str, np.ndarray]],
) -> pd.DataFrame:
    """Mean, std and first-lag autocorrelation of the residuals.

    Direction is skipped on purpose. Residuals on a 0/1 target are not really informative.
    """
    rows = []
    for target, model_predictions in predictions.items():
        if target == "direction":
            continue
        y_true = np.asarray(y_truths[target])
        for model_name, y_pred in model_predictions.items():
            residual = y_true - np.asarray(y_pred)
            rows.append(
                {
                    "target": target,
                    "model": model_name,
                    "residual_mean": float(np.mean(residual)),
                    "residual_std": float(np.std(residual, ddof=1)),
                    "residual_acf1": float(pd.Series(residual).autocorr(lag=1)),
                }
            )
    return pd.DataFrame(rows)


# Part 9 ---- prediction CSVs and the forecast figures ----

def plot_predictions(
    target: str,
    dates: pd.Index,
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    """One figure per target. Realised series in black, each model on top in a dashed style."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(dates, y_true, label="actual", color="black", linewidth=1.4)
    # We only have two ARMA-GARCH variants, so a tiny fixed style iterator does the job.
    style_iter = iter([("--", "tab:purple"), (":", "tab:red")])
    for model_name, y_pred in predictions.items():
        linestyle, colour = next(style_iter)
        ax.plot(
            dates,
            np.asarray(y_pred),
            label=model_name,
            linestyle=linestyle,
            color=colour,
            alpha=0.85,
            linewidth=1.2,
        )
    ax.set_title(f"Test-set predictions: {target} (Ex 3 baselines)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def write_predictions_and_figures(
    test_idx: pd.Index,
    y_truths: dict[str, np.ndarray],
    predictions: dict[str, dict[str, np.ndarray]],
) -> None:
    """Write the per-target prediction CSV, the figure, and the residual diagnostics."""
    pred_files = {
        "return": OUT["pred_return"],
        "price": OUT["pred_price"],
        "direction": OUT["pred_direction"],
    }
    fig_files = {
        "return": OUT["fig_return"],
        "price": OUT["fig_price"],
        "direction": OUT["fig_direction"],
    }

    for target, model_predictions in predictions.items():
        y_true = y_truths[target]
        output = pd.DataFrame({"y_true": y_true}, index=test_idx)
        for model_name, y_pred in model_predictions.items():
            output[f"pred_{model_name}"] = np.asarray(y_pred)
        output.to_csv(OUTPUT_DIR / pred_files[target])
        plot_predictions(
            target,
            test_idx,
            y_true,
            model_predictions,
            OUTPUT_DIR / fig_files[target],
        )

    residual_diagnostics(y_truths, predictions).to_csv(
        OUTPUT_DIR / OUT["residual_diagnostics"],
        index=False,
    )


# --- Wrap-up: glue everything together

def main() -> None:
    configure_logging()
    np.random.seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    panel = load_monthly_panel()
    design = return_design(panel)
    fit_idx, test_idx = chronological_split_index(design)

    # Estimation block (train + validation) and test block, as separate arrays.
    returns_full = design["TARGET"]
    returns_fit = returns_full.loc[fit_idx]
    returns_test = returns_full.loc[test_idx]
    # log_price_t is the log price at time t; the one-step price target is the value at t + 1.
    log_price_t = panel["log_price"].reindex(test_idx)
    log_price_t1 = panel["log_price"].shift(-1).reindex(test_idx)

    results = run_baselines(returns_fit.values, returns_test.values)
    y_truths = {
        "return": returns_test.values,
        "price": log_price_t1.values,
        "direction": (returns_test.values > 0).astype(int),
    }
    metrics, orders, predictions = build_result_tables(
        results,
        y_truths["return"],
        y_truths["price"],
        y_truths["direction"],
        log_price_t,
    )
    combined = combine_with_ex2_metrics(metrics)

    write_tables(metrics, orders, combined)
    write_predictions_and_figures(test_idx, y_truths, predictions)
    LOGGER.info("Artifacts written to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
