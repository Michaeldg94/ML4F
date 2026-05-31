# =============================================================================
# exercise_02.py
# HW2 ex 2: Machine Learning for Option Valuation
#
# Team:    Marco Canal, Michael Duarte Goncalves, Olivia Peter
# Date:    2026-05-22
#
# PURPOSE:
#   Learn the pricing function of two option contracts with machine learning:
#   (a) American Put and Call options, with ground truth from a
#       Cox-Ross-Rubinstein binomial tree, and
#   (b) Barrier down-and-out Call options, with ground truth from a Monte
#       Carlo simulation of the Heston model. The Heston simulator uses the
#       improved Euler scheme from exercise_03 (vectorised, full-truncation
#       discretisation of the variance process, and antithetic variates) so
#       that the ground-truth dataset can be generated in reasonable time.
#
#   For each option contract we
#     1. sample n random parameter combinations,
#     2. compute the true price with the slow numerical method,
#     3. train Gaussian Process Regression and a feed-forward neural
#        network on the (X, y) tuples,
#     4. evaluate prediction accuracy on a held-out test set,
#     5. report computational speedup between the numerical method and the
#        trained ML model at prediction time.
#
# INPUTS:
#   No external data files. All ground-truth prices are generated locally
#   from random parameter draws.
#
# OUTPUTS:
#   output/ex02_dataset_american.csv  (parameter combinations and prices)
#   output/ex02_dataset_barrier.csv
#   output/ex02_metrics_american.csv  (per-model accuracy and timing)
#   output/ex02_metrics_barrier.csv
#   output/ex02_pred_vs_true_<type>_<model>.pdf
#   output/ex02_residuals_<type>_<model>.pdf
#
# STRUCTURE:
#   Part 1: Imports, paths, constants
#   Part 2: Ground-truth pricers
#           2a: Cox-Ross-Rubinstein binomial tree for American Put and Call
#           2b: Heston Monte Carlo for Barrier down-and-out Call
#   Part 3: Synthetic-dataset generation
#   Part 4: ML pipeline (GPR and feed-forward NN)
#   Part 5: Evaluation and reporting
# =============================================================================

# =============================================================================
# Part 1: Imports, paths, constants
# =============================================================================

import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SEED = 42
np.random.seed(SEED)

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Dataset sizes (assignment allows 500 to 2000; we pick 1000 for both).
N_AMERICAN = 1000
N_BARRIER = 1000

# Numerical-method discretisations
BINOMIAL_STEPS = 500     # CRR tree steps; 500 gives ~1e-3 accuracy
HESTON_STEPS = 100       # time steps per path for Heston MC
HESTON_PATHS = 10_000    # number of MC paths per option (antithetic, so 5000 base + 5000 reflected)

# =============================================================================
# Part 2: Ground-truth pricers
# =============================================================================

# -----------------------------------------------------------------------------
# 2a: Cox-Ross-Rubinstein binomial tree for American options
# -----------------------------------------------------------------------------

def crr_american(S0, K, T, r, q, sigma, n_steps=BINOMIAL_STEPS, opt_type="put"):
    """Price an American option by the Cox-Ross-Rubinstein binomial tree.

    Uses the standard up/down factors u = exp(sigma sqrt(dt)), d = 1/u, the
    risk-neutral probability p = (exp((r-q) dt) - d) / (u - d), and
    backward induction with an early-exercise check at every node. Returns
    the option value at the root.
    """
    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    disc = np.exp(-r * dt)
    p = (np.exp((r - q) * dt) - d) / (u - d)
    if not (0 < p < 1):
        # parameter combination outside the no-arbitrage region for this dt
        return np.nan

    j = np.arange(n_steps + 1)
    ST = S0 * (u ** (n_steps - j)) * (d ** j)
    if opt_type == "call":
        V = np.maximum(ST - K, 0.0)
    else:
        V = np.maximum(K - ST, 0.0)

    for step in range(n_steps - 1, -1, -1):
        S_step = S0 * (u ** np.arange(step, -1, -1)) * (d ** np.arange(0, step + 1, 1))
        cont = disc * (p * V[:-1] + (1.0 - p) * V[1:])
        if opt_type == "call":
            exercise = np.maximum(S_step - K, 0.0)
        else:
            exercise = np.maximum(K - S_step, 0.0)
        V = np.maximum(cont, exercise)
    return float(V[0])


# -----------------------------------------------------------------------------
# 2b: Heston Monte Carlo for Barrier down-and-out Call
# -----------------------------------------------------------------------------

def heston_mc_barrier_call(S0, K, T, r, q, v0, theta, kappa, eta, rho, B,
                           n_steps=HESTON_STEPS, n_paths=HESTON_PATHS,
                           seed=SEED):
    """Price a down-and-out European call under the Heston model by Monte
    Carlo with full-truncation Euler for the variance and an antithetic
    variates pair. The path is killed if S ever falls at or below B.

    The improved Euler scheme that drives this function is the deliverable
    of exercise 3; here we just use it. Vectorisation over paths makes the
    inner Python loop length only n_steps. The seed defaults to the global
    SEED so an unseeded call still produces deterministic prices."""
    rng = np.random.default_rng(seed)

    half = n_paths // 2
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)

    S = np.full(2 * half, S0, dtype=np.float64)
    v = np.full(2 * half, v0, dtype=np.float64)
    alive = np.ones(2 * half, dtype=bool)

    for _ in range(n_steps):
        # Antithetic pair: first half uses Z, second half uses -Z
        Z1 = rng.standard_normal(half)
        Z2 = rng.standard_normal(half)
        Z1_full = np.concatenate([Z1, -Z1])
        Z2_full = np.concatenate([Z2, -Z2])
        W1 = Z1_full
        W2 = rho * Z1_full + np.sqrt(1.0 - rho * rho) * Z2_full

        # Full-truncation Euler for variance: use max(v, 0) in diffusion and
        # drift; v itself is allowed to go slightly negative but is floored
        # to zero where it appears under a square root
        v_pos = np.maximum(v, 0.0)
        v = (v + kappa * (theta - v_pos) * dt
             + eta * np.sqrt(v_pos) * sqrt_dt * W2)

        # Log-Euler for the price (variance-reducing, exactly positive)
        S = S * np.exp((r - q - 0.5 * v_pos) * dt
                       + np.sqrt(v_pos) * sqrt_dt * W1)
        alive = alive & (S > B)

    payoff = np.where(alive, np.maximum(S - K, 0.0), 0.0)
    return float(np.exp(-r * T) * payoff.mean())


# =============================================================================
# Part 3: Synthetic-dataset generation
# =============================================================================

# Parameter ranges chosen to span a realistic but tractable region of equity
# options markets. Bounds avoid pathological cases (T very small, sigma very
# large) while remaining diverse enough for the ML model to be useful.

S0_FIXED = 100.0   # spot price held at 100 so that K varies relative to it


def sample_american_params(rng):
    """Random parameter draw for an American option. The contract type is
    sampled uniformly so that the dataset spans both put and call, in line
    with the assignment phrasing 'American Put/Call options'. The spot is
    fixed at 100 because the assignment lists K, T, r, q and sigma as the
    parameters to vary; varying both S0 and K would just over-cover the
    moneyness dimension."""
    is_call = int(rng.integers(0, 2))   # 0 = put, 1 = call
    return {
        "K": rng.uniform(80.0, 120.0),
        "T": rng.uniform(0.1, 2.0),
        "r": rng.uniform(0.0, 0.05),
        "q": rng.uniform(0.0, 0.03),
        "sigma": rng.uniform(0.1, 0.6),
        "is_call": is_call,
    }


def sample_barrier_params(rng):
    """Random parameter draw for a Barrier down-and-out Call option. The
    spot is held fixed at 100 (see sample_american_params for the rationale)
    and the barrier B is sampled as a fraction of $S_0$ so that $B < S_0$
    is guaranteed by construction. The remaining nine parameters are the
    ones listed by the assignment for the Heston-MC ground truth."""
    return {
        "K": rng.uniform(80.0, 120.0),
        "T": rng.uniform(0.25, 1.0),
        "r": rng.uniform(0.0, 0.05),
        "q": rng.uniform(0.0, 0.03),
        "v0": rng.uniform(0.01, 0.10),
        "theta": rng.uniform(0.01, 0.15),
        "kappa": rng.uniform(0.5, 5.0),
        "eta": rng.uniform(0.1, 0.6),
        "rho": rng.uniform(-0.9, -0.1),
        "B": S0_FIXED * rng.uniform(0.60, 0.95),
    }


def generate_american_dataset(n, rng):
    """Sample n American option parameter combinations (uniform mix of put
    and call), price each by binomial tree, return a DataFrame with one row
    per option and a column of true prices, plus the total wall-clock
    pricing time."""
    rows = []
    t0 = time.perf_counter()
    for i in range(n):
        p = sample_american_params(rng)
        opt_type = "call" if p["is_call"] else "put"
        price = crr_american(S0_FIXED, p["K"], p["T"], p["r"], p["q"],
                             p["sigma"], n_steps=BINOMIAL_STEPS,
                             opt_type=opt_type)
        p["price"] = price
        rows.append(p)
        if (i + 1) % 100 == 0:
            print(f"  american {i + 1}/{n}, elapsed {time.perf_counter() - t0:.1f}s")
    elapsed = time.perf_counter() - t0
    df = pd.DataFrame(rows).dropna(subset=["price"]).reset_index(drop=True)
    n_calls = int(df["is_call"].sum())
    print(f"  american dataset done: n={len(df)} "
          f"({len(df) - n_calls} puts, {n_calls} calls), "
          f"total time {elapsed:.1f}s, per option "
          f"{elapsed / max(len(df), 1) * 1000:.1f} ms")
    return df, elapsed


def generate_barrier_dataset(n, rng):
    """Sample n Heston barrier parameter combinations, price each by the
    improved Heston Monte Carlo, return the dataset DataFrame and total
    pricing time."""
    rows = []
    t0 = time.perf_counter()
    for i in range(n):
        p = sample_barrier_params(rng)
        # Use a per-option seed for reproducibility while still varying paths
        seed = int(rng.integers(0, 2 ** 31 - 1))
        price = heston_mc_barrier_call(
            S0_FIXED, p["K"], p["T"], p["r"], p["q"], p["v0"], p["theta"],
            p["kappa"], p["eta"], p["rho"], p["B"],
            n_steps=HESTON_STEPS, n_paths=HESTON_PATHS, seed=seed,
        )
        p["price"] = price
        rows.append(p)
        if (i + 1) % 100 == 0:
            print(f"  barrier {i + 1}/{n}, elapsed {time.perf_counter() - t0:.1f}s")
    elapsed = time.perf_counter() - t0
    df = pd.DataFrame(rows).dropna(subset=["price"]).reset_index(drop=True)
    print(f"  barrier dataset done: n={len(df)}, total time {elapsed:.1f}s, "
          f"per option {elapsed / max(len(df), 1) * 1000:.1f} ms")
    return df, elapsed


# =============================================================================
# Part 4: ML pipeline (GPR and feed-forward NN)
# =============================================================================

def fit_gpr(X_train, y_train):
    """Fit a Gaussian Process Regressor with a Matern 5/2 kernel plus a
    white-noise term to absorb Monte Carlo noise on the barrier side. The
    kernel hyperparameters are optimised by sklearn during fit."""
    kernel = (ConstantKernel(1.0, (1e-2, 1e3))
              * Matern(length_scale=np.ones(X_train.shape[1]),
                       length_scale_bounds=(1e-2, 1e3), nu=2.5)
              + WhiteKernel(noise_level=1e-3,
                            noise_level_bounds=(1e-8, 1e1)))
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2,
                                   normalize_y=True, random_state=SEED)
    gpr.fit(X_train, y_train)
    return gpr


def fit_nn(X_train, y_train):
    """Fit a feed-forward neural network. Two hidden layers, tanh
    activation, Adam optimiser. Hyperparameters chosen to be reasonable for
    a smooth low-dimensional regression problem rather than tuned."""
    nn = MLPRegressor(hidden_layer_sizes=(64, 64), activation="tanh",
                      solver="adam", alpha=1e-4, learning_rate_init=1e-3,
                      max_iter=5000, tol=1e-6, random_state=SEED,
                      early_stopping=True, validation_fraction=0.1,
                      n_iter_no_change=30)
    nn.fit(X_train, y_train)
    return nn


def train_eval(df, feature_cols, label, num_time):
    """Train GPR and NN on the supplied dataset and report accuracy on a
    20 percent held-out test set. Reports the prediction time per query as
    the inference speedup metric.

    num_time is the total pricing time on the full dataset for the slow
    numerical method, so per-option ground-truth time = num_time / n.
    """
    X = df[feature_cols].values
    y = df["price"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED)
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    rows = []
    # Numerical baseline per-option time
    per_option_num_ms = (num_time / len(df)) * 1000

    for name, fitter in [("GPR", fit_gpr), ("NN", fit_nn)]:
        print(f"  fitting {label} {name}.")
        t0 = time.perf_counter()
        model = fitter(X_train_s, y_train)
        train_time = time.perf_counter() - t0

        # Inference time: predict the full test set once and divide
        t0 = time.perf_counter()
        y_pred = model.predict(X_test_s)
        pred_time_total = time.perf_counter() - t0
        per_pred_ms = (pred_time_total / len(X_test_s)) * 1000

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        # Mean absolute percentage error on options worth at least 1 percent
        # of S0_FIXED (one currency unit when S0 = 100), so the metric is
        # not dominated by a handful of near-zero deep-OTM or near-knockout
        # contracts where a few cents of absolute error give meaningless
        # relative errors.
        mape_floor = 0.01 * S0_FIXED
        mask = np.abs(y_test) > mape_floor
        mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask]))

        rows.append({
            "model": name, "label": label,
            "n_train": len(X_train_s), "n_test": len(X_test_s),
            "MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape,
            "fit_time_s": train_time,
            "pred_time_per_option_ms": per_pred_ms,
            "num_time_per_option_ms": per_option_num_ms,
            "speedup_per_query": per_option_num_ms / per_pred_ms
            if per_pred_ms > 0 else np.nan,
        })
        plot_pred_vs_true(label, name, y_test, y_pred)
        plot_residuals(label, name, y_test, y_pred)
    return pd.DataFrame(rows)


def plot_pred_vs_true(label, name, y_true, y_pred):
    fig, ax = plt.subplots(figsize=(6, 6))
    lim_min = min(y_true.min(), y_pred.min())
    lim_max = max(y_true.max(), y_pred.max())
    ax.plot([lim_min, lim_max], [lim_min, lim_max],
            "k--", linewidth=0.8, label="y = x")
    ax.scatter(y_true, y_pred, s=10, alpha=0.7,
               color="tab:blue")
    ax.set_xlabel("true price (numerical)")
    ax.set_ylabel("predicted price (ML)")
    ax.set_title(f"{label}: predicted vs true, model {name}")
    ax.legend()
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"ex02_pred_vs_true_{label}_{name}.pdf",
                bbox_inches="tight")
    plt.close(fig)


def plot_residuals(label, name, y_true, y_pred):
    residuals = y_pred - y_true
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_true, residuals, s=10, alpha=0.7, color="tab:orange")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("true price")
    ax.set_ylabel("residual (predicted - true)")
    ax.set_title(f"{label}: residuals vs true, model {name}")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"ex02_residuals_{label}_{name}.pdf",
                bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Part 5: Driver
# =============================================================================

def main():
    rng = np.random.default_rng(SEED)

    print("Generating American Put dataset.")
    df_am, t_am = generate_american_dataset(N_AMERICAN, rng)
    df_am.to_csv(OUT_DIR / "ex02_dataset_american.csv", index=False)

    print("Generating Barrier down-and-out Call dataset.")
    df_ba, t_ba = generate_barrier_dataset(N_BARRIER, rng)
    df_ba.to_csv(OUT_DIR / "ex02_dataset_barrier.csv", index=False)

    am_features = ["K", "T", "r", "q", "sigma", "is_call"]
    ba_features = ["K", "T", "r", "q", "v0", "theta", "kappa",
                   "eta", "rho", "B"]

    print("\nTraining ML models on American Put dataset.")
    res_am = train_eval(df_am, am_features, "american", t_am)
    res_am.to_csv(OUT_DIR / "ex02_metrics_american.csv", index=False)
    print(res_am[["model", "MAE", "RMSE", "R2", "MAPE",
                  "fit_time_s", "pred_time_per_option_ms",
                  "num_time_per_option_ms", "speedup_per_query"]]
          .to_string(index=False))

    print("\nTraining ML models on Barrier Call dataset.")
    res_ba = train_eval(df_ba, ba_features, "barrier", t_ba)
    res_ba.to_csv(OUT_DIR / "ex02_metrics_barrier.csv", index=False)
    print(res_ba[["model", "MAE", "RMSE", "R2", "MAPE",
                  "fit_time_s", "pred_time_per_option_ms",
                  "num_time_per_option_ms", "speedup_per_query"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
