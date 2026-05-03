# =============================================================================
# exercise_03.py
# Baseline forecasts: ARMA(p,q) + GARCH(1,1), with and without a
# normalising transformation. Compared against Exercise 2.
#
# Team:    [TEAM MEMBER NAMES TO BE FILLED IN]
# Date:    2026-05-02
#
# PURPOSE:
#   For the same Goyal-Welch monthly SP500 series and the same chronological
#   60/15/25 split as Exercise 2, fit a univariate ARMA(p,q) + GARCH(1,1)
#   baseline on the log-return series (with order p,q selected by AIC via
#   pmdarima.auto_arima). Two variants:
#     (a) Plain  : ARMA(p,q) directly on log returns.
#     (b) YeoJohnson : ARMA(p,q) on Yeo-Johnson-transformed returns; the
#         resulting forecasts are inverse-transformed back to the original
#         return scale before scoring. Yeo-Johnson is one of the invertible
#         normalising transformations defined in Lab 1 (and is the natural
#         drop-in replacement for the R `bestNormalize` workflow when the
#         data are real-valued and can include negatives).
#   Point forecasts are taken from the ARMA mean component; the GARCH(1,1)
#   piece supplies a conditional-variance series that we use only to verify
#   the existence of an ARCH effect and to quote the GARCH persistence.
#   For the price target we add the predicted return to the realised lagged
#   log price; for the direction target we threshold the predicted return at
#   zero. We then merge the four resulting (target, model) rows into the same
#   metrics table as Exercise 2 to enable a head-to-head comparison.
#
# INPUTS:
#   - goyal-welch2022Monthly.csv
#   - output/ex2_metrics.csv  (loaded for the head-to-head table)
#
# OUTPUTS:
#   - output/ex3_metrics.csv
#   - output/ex3_metrics.tex
#   - output/ex3_combined.csv  (Ex 2 + Ex 3 baselines, target by model)
#   - output/ex3_combined.tex
#   - output/ex3_arma_orders.csv  (which (p,d,q) auto_arima picked + GARCH params)
#   - output/ex3_pred_<target>.csv
#   - output/ex3_fig_<target>.pdf
# =============================================================================

# === Part 1: Setup ===========================================================
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import mean_squared_error, accuracy_score, matthews_corrcoef

from pmdarima.arima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model

SEED = 7
np.random.seed(SEED)

DATA_FILE = Path("goyal-welch2022Monthly.csv")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = "1927-01-01"
END_DATE   = "2021-12-31"

# Same chronological fractions as Exercise 2 so the test sets line up exactly.
P_TRAIN = 0.60
P_VAL   = 0.15
P_TEST  = 0.25


# === Part 2: Load Goyal-Welch and replicate the Exercise 2 split ============
print("Loading Goyal-Welch monthly data.")
gw = pd.read_csv(DATA_FILE)
gw["Date"] = pd.to_datetime(gw["yyyymm"].astype(str), format="%Y%m")
gw.set_index("Date", inplace=True)
gw.drop("yyyymm", axis=1, inplace=True)
gw["Index"] = gw["Index"].str.replace(",", "", regex=False).astype(float)
gw["log_price"] = np.log(gw["Index"])
gw["log_ret"]   = gw["log_price"].diff()

panel = gw.loc[START_DATE:END_DATE, ["Index", "log_price", "log_ret"]].copy()
panel = panel.dropna()
print(f"Panel rows: {len(panel)} ({panel.index.min().date()} to {panel.index.max().date()})")

# Build the same design (return target) as exercise_02.py so the test indices
# line up exactly. We only need the index alignment, not the regressors.
N_LAGS = 3
y = panel["log_ret"].shift(-1)
cols = {}
for lag in range(1, N_LAGS + 1):
    cols[f"log_ret_lag{lag}"] = panel["log_ret"].shift(lag - 1)
X_design = pd.DataFrame(cols, index=panel.index)
df_design = pd.concat([X_design, y.rename("TARGET")], axis=1).dropna()
n = len(df_design)
n_train = int(P_TRAIN * n)
n_val   = int(P_VAL   * n)
fit_idx  = df_design.index[:n_train + n_val]      # train + val
test_idx = df_design.index[n_train + n_val:]      # test
print(f"Fit period:  {fit_idx[0].date()} to {fit_idx[-1].date()}  ({len(fit_idx)} obs)")
print(f"Test period: {test_idx[0].date()} to {test_idx[-1].date()}  ({len(test_idx)} obs)")

# Endogenous series for the ARMA: at design row t the target is log_ret_{t+1};
# equivalently we model the log-return series with one chronological alignment
# step. The sequence we feed to ARMA is the realised log_ret values at t+1.
returns_full = df_design["TARGET"].copy()
returns_fit  = returns_full.loc[fit_idx]
returns_te   = returns_full.loc[test_idx]

# Realised log price at time t for every test row, used to roll the predicted
# returns into a predicted log price log_p_{t+1} = log_p_t + r_pred_{t+1}.
log_price_t   = panel["log_price"].reindex(test_idx)
log_price_t1  = panel["log_price"].shift(-1).reindex(test_idx)


# === Part 3: ARMA(p,q) + GARCH(1,1) one-shot forecasting helper =============
def fit_baseline(returns_fit_arr: np.ndarray,
                 returns_te_arr:  np.ndarray,
                 transformer=None,
                 max_pq: int = 5) -> dict:
    """
    Fit an ARMA(p,q) on returns_fit_arr (optionally Yeo-Johnson-transformed),
    pick (p,q) by AIC via pmdarima.auto_arima, then layer a GARCH(1,1) on the
    ARMA residuals via the arch library. Return one-step-ahead conditional-mean
    forecasts on the test period (in the original scale if a transformer was
    given, via the inverse Yeo-Johnson).
    """
    # Apply transformer: fit on the same train+validation block used by the
    # ARMA estimator (the test set is never touched).
    if transformer is not None:
        returns_fit_t = transformer.fit_transform(returns_fit_arr.reshape(-1, 1)).ravel()
        returns_te_t  = transformer.transform(returns_te_arr.reshape(-1, 1)).ravel()
    else:
        returns_fit_t = returns_fit_arr
        returns_te_t  = returns_te_arr

    # Order selection by AIC; d=0 because returns are stationary, no seasonal
    # component for monthly equity returns at this frequency.
    auto = auto_arima(returns_fit_t, seasonal=False, d=0,
                      max_p=max_pq, max_q=max_pq,
                      stepwise=True, suppress_warnings=True,
                      error_action="ignore", trace=False,
                      information_criterion="aic")
    p, d_, q = auto.order
    print(f"  auto_arima picked ARMA({p},{q})  (AIC={auto.aic():.2f})")

    # Fit the ARMA via statsmodels for direct access to .apply() (which lets
    # us extend the fitted model with new data without re-estimating the
    # parameters; this gives genuine one-step-ahead forecasts for the test
    # block conditional on the realised history).
    arima = ARIMA(returns_fit_t, order=(p, 0, q),
                  enforce_stationarity=True,
                  enforce_invertibility=True).fit()

    # GARCH(1,1) on ARMA residuals, mean='Zero' since the ARMA already
    # absorbs the conditional mean. Returns are in the lab's natural scale
    # (decimal log-returns), so we rescale by 100 for numerical stability of
    # the GARCH MLE; this only affects the scale of omega, alpha, beta does
    # not change.
    garch = arch_model(arima.resid * 100.0, mean="Zero",
                       vol="GARCH", p=1, q=1, dist="normal").fit(disp="off")
    omega = garch.params["omega"]
    alpha = garch.params["alpha[1]"]
    beta  = garch.params["beta[1]"]
    persistence = alpha + beta

    # One-step-ahead test forecasts (static parameters, expanding history).
    # statsmodels SARIMAX has .apply(new_endog) which builds a model with the
    # same fitted parameters on a longer endogenous series; .fittedvalues
    # then contains the in-sample one-step-ahead conditional means for every
    # row, including those past the original training cut.
    full_t = np.concatenate([returns_fit_t, returns_te_t])
    extended = arima.apply(full_t, refit=False)
    fitted_full = np.asarray(extended.fittedvalues)
    test_pred_t = fitted_full[len(returns_fit_t):]

    # Inverse-transform back to the original return scale
    if transformer is not None:
        test_pred = transformer.inverse_transform(test_pred_t.reshape(-1, 1)).ravel()
    else:
        test_pred = test_pred_t

    return {
        "pred":         test_pred,
        "order":        (p, q),
        "arima":        arima,
        "garch":        garch,
        "garch_omega":  omega,
        "garch_alpha":  alpha,
        "garch_beta":   beta,
        "persistence":  persistence,
        "aic":          float(auto.aic()),
    }


# === Part 4: Metrics =========================================================
def nrmse(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    denom = np.mean(np.sum((y_true - y_true.mean()) ** 2))
    return float(np.sqrt(mean_squared_error(y_true, y_pred) / denom))


def pcorrect(y_true, y_pred):
    return (1.0 - nrmse(y_true, y_pred)) * 100.0


def metrics_regression(y_true, y_pred, label_target, label_model, params):
    return {
        "target":   label_target,
        "model":    label_model,
        "params":   params,
        "RMSE":     float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "NRMSE":    nrmse(y_true, y_pred),
        "pcorrect": pcorrect(y_true, y_pred),
        "n":        int(len(y_true)),
        "accuracy": np.nan,
        "MCC":      np.nan,
    }


def metrics_classification(y_true, y_pred, label_target, label_model, params):
    return {
        "target":   label_target,
        "model":    label_model,
        "params":   params,
        "RMSE":     np.nan,
        "NRMSE":    np.nan,
        "pcorrect": np.nan,
        "n":        int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "MCC":      float(matthews_corrcoef(y_true, y_pred)),
    }


# === Part 5: Run the two ARMA-GARCH variants ================================
returns_fit_arr = returns_fit.values
returns_te_arr  = returns_te.values

results = {}
for tag, transformer in [
    ("ARMA-GARCH (plain)",        None),
    ("ARMA-GARCH (YeoJohnson)",   PowerTransformer(method="yeo-johnson")),
]:
    print(f"\n--- {tag} ---")
    out = fit_baseline(returns_fit_arr, returns_te_arr, transformer=transformer)
    results[tag] = out


# === Part 6: Build the metrics rows ==========================================
all_rows = []
arma_orders = []
all_preds = {"return": {}, "price": {}, "direction": {}}

# Realised test targets
y_return    = returns_te_arr
y_price     = log_price_t1.values             # log_price_{t+1}
y_direction = (returns_te_arr > 0).astype(int)

for tag, out in results.items():
    return_pred = out["pred"]
    price_pred  = log_price_t.values + return_pred       # log_p_t + r_pred
    direction_pred = (return_pred > 0).astype(int)

    pq = f"({out['order'][0]},{out['order'][1]})"
    params_str = (
        f"ARMA{pq}; GARCH(1,1) "
        f"alpha={out['garch_alpha']:.3f}, beta={out['garch_beta']:.3f}, "
        f"alpha+beta={out['persistence']:.3f}"
    )

    all_rows.append(metrics_regression(y_return, return_pred,
                                       "return", tag, params_str))
    all_rows.append(metrics_regression(y_price, price_pred,
                                       "price",  tag, params_str))
    all_rows.append(metrics_classification(y_direction, direction_pred,
                                           "direction", tag, params_str))

    all_preds["return"][tag]    = return_pred
    all_preds["price"][tag]     = price_pred
    all_preds["direction"][tag] = direction_pred

    arma_orders.append({
        "model":       tag,
        "p":           out["order"][0],
        "q":           out["order"][1],
        "auto_aic":    out["aic"],
        "garch_omega": out["garch_omega"],
        "garch_alpha": out["garch_alpha"],
        "garch_beta":  out["garch_beta"],
        "persistence": out["persistence"],
    })


metrics_df = pd.DataFrame(all_rows)
print("\n--- Ex 3 baseline metrics ---")
with pd.option_context("display.max_columns", None, "display.width", 220):
    print(metrics_df.to_string(index=False))

orders_df = pd.DataFrame(arma_orders)
print("\n--- Auto-selected orders / GARCH parameters ---")
print(orders_df.to_string(index=False))


# === Part 7: Combine with Ex 2 metrics for a head-to-head table =============
ex2_path = OUTPUT_DIR / "ex2_metrics.csv"
if ex2_path.exists():
    ex2 = pd.read_csv(ex2_path)
    combined = pd.concat([ex2.assign(source="Ex 2 (NN/GP/LSTM)"),
                          metrics_df.assign(source="Ex 3 (ARMA-GARCH)")],
                         ignore_index=True)
    combined.to_csv(OUTPUT_DIR / "ex3_combined.csv", index=False)
    print("\n--- Combined Ex 2 + Ex 3 metrics ---")
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(combined.to_string(index=False))
else:
    print(f"WARNING: {ex2_path} not found; skipping combined table.")
    combined = metrics_df


# === Part 8: Save artefacts ==================================================
metrics_df.to_csv(OUTPUT_DIR / "ex3_metrics.csv", index=False)
orders_df.to_csv(OUTPUT_DIR / "ex3_arma_orders.csv", index=False)


def df_to_latex(df: pd.DataFrame, name: str, caption: str):
    """Wrap a pandas DataFrame to_latex output in a [H] table for placement."""
    tex = df.to_latex(index=False, float_format="{:.4f}".format,
                      column_format="l" * len(df.columns), escape=False)
    out = (
        "\\begin{table}[H]\n\\centering\n\\small\n"
        + tex
        + "\\caption{" + caption + "}\n\\label{tab:" + name + "}\n"
        + "\\end{table}\n"
    )
    (OUTPUT_DIR / f"{name}.tex").write_text(out)


df_to_latex(metrics_df, "ex3_metrics",
            "Test-set forecast performance for the ARMA(p,q)+GARCH(1,1) "
            "baselines (plain and Yeo-Johnson-normalised).")
df_to_latex(combined, "ex3_combined",
            "Head-to-head comparison: Exercise 2 ML models versus Exercise 3 "
            "ARMA-GARCH baselines on the same chronological test split.")


# Per-target predictions and figures
y_truths = {"return": y_return, "price": y_price, "direction": y_direction}
for target_kind, preds in all_preds.items():
    y_te_arr = y_truths[target_kind]
    out = pd.DataFrame({"y_true": y_te_arr}, index=test_idx)
    for m_name, p in preds.items():
        out[f"pred_{m_name}"] = np.asarray(p)
    out.to_csv(OUTPUT_DIR / f"ex3_pred_{target_kind}.csv")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(test_idx, y_te_arr, label="actual", color="black", linewidth=1.4)
    style_iter = iter([("--", "tab:purple"), (":", "tab:red")])
    for m_name, p in preds.items():
        ls, c = next(style_iter)
        ax.plot(test_idx, np.asarray(p), label=m_name,
                linestyle=ls, color=c, alpha=0.85, linewidth=1.2)
    ax.set_title(f"Test-set predictions: {target_kind} (Ex 3 baselines)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"ex3_fig_{target_kind}.pdf", bbox_inches="tight")
    plt.close(fig)


print("\nDone. Artefacts written to ./output/")
