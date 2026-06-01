"""HW2 Exercise 2: Machine learning for option valuation.

Learns the pricing map of two contracts from synthetic data: American
Put/Call options labelled by a Cox-Ross-Rubinstein binomial tree, and
down-and-out Barrier calls labelled by Heston Monte Carlo (the improved
vectorised engine of exercise_03). For each contract we sample 1000
parameter draws, price them, fit a Gaussian process and a feed-forward
network, then report accuracy, no-arbitrage sanity checks and the speedup
of one ML prediction over one numerical pricing call. Outputs land in
output/ex02_*.

Authors: Marco Canal, Michael Duarte Gonçalves, Olivia Peter.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel, Matern, WhiteKernel)
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SEED = 42
np.random.seed(SEED)

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "output" / "ex2"        # exercise-2 outputs live here
FIG_DIR = OUT_DIR / "figures"            # plots, numbered within this folder
TAB_DIR = OUT_DIR / "tables"             # CSVs, numbered within this folder
for _d in (FIG_DIR, TAB_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Dataset sizes (assignment allows 500 to 2000; we pick 1000 for both).
N_AMERICAN = 1000
N_BARRIER = 1000

# Numerical-method discretisations.
BINOMIAL_STEPS = 500     # CRR tree steps; 500 gives ~1e-3 accuracy
HESTON_STEPS = 100       # time steps per Heston path
HESTON_PATHS = 10_000    # antithetic: 5000 base paths + 5000 reflected

# --- Ground-truth pricers ----------------------------------------------------
# (a) CRR binomial tree for American Put/Call.


def crr_american(S0, K, T, r, q, sigma,
                 n_steps=BINOMIAL_STEPS, opt_type="put"):
    """Price an American option by a Cox-Ross-Rubinstein binomial tree,
    with an early-exercise check at every node."""
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
        up = u ** np.arange(step, -1, -1)
        down = d ** np.arange(0, step + 1)
        S_step = S0 * up * down
        cont = disc * (p * V[:-1] + (1.0 - p) * V[1:])
        if opt_type == "call":
            exercise = np.maximum(S_step - K, 0.0)
        else:
            exercise = np.maximum(K - S_step, 0.0)
        V = np.maximum(cont, exercise)
    return float(V[0])


# (b) Heston Monte Carlo for the down-and-out Barrier call.


def heston_mc_barrier_call(S0, K, T, r, q, v0, theta, kappa, eta, rho, B,
                           n_steps=HESTON_STEPS, n_paths=HESTON_PATHS,
                           seed=SEED, return_se=False):
    """Price a down-and-out call under Heston by Monte Carlo.

    Full-truncation variance, log-Euler price, antithetic variates. A path
    knocks out the moment S falls to or below B. With return_se=True also
    returns the antithetic standard error, the label noise floor.
    """
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

        # full truncation: floor v at zero wherever it sits under a sqrt
        v_pos = np.maximum(v, 0.0)
        v = (v + kappa * (theta - v_pos) * dt
             + eta * np.sqrt(v_pos) * sqrt_dt * W2)

        # Log-Euler for the price (variance-reducing, exactly positive)
        S = S * np.exp((r - q - 0.5 * v_pos) * dt
                       + np.sqrt(v_pos) * sqrt_dt * W1)
        alive = alive & (S > B)

    payoff = np.where(alive, np.maximum(S - K, 0.0), 0.0)
    disc = np.exp(-r * T) * payoff
    price = float(disc.mean())
    if return_se:
        # antithetic SE: average each Z / -Z pair, then SE across pairs
        pair = 0.5 * (disc[:half] + disc[half:])
        se = float(pair.std(ddof=1) / np.sqrt(half)) if half > 1 else np.nan
        return price, se
    return price


# --- Synthetic-dataset generation --------------------------------------------

# Parameter ranges span a realistic but tractable region of equity options.

S0_FIXED = 100.0   # spot price held at 100 so that K varies relative to it


def sample_american_params(rng):
    """Random American-option draw. Contract type is uniform over put/call.
    Spot is fixed at 100, so only K, T, r, q and sigma vary."""
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
    """Random barrier-option draw. Spot is fixed at 100 and B is a fraction
    of it, so B < S0 by construction."""
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
    """Sample n American options, price each by binomial tree, and return the
    dataset with its total pricing time."""
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
            print(f"  american {i + 1}/{n}, "
                  f"elapsed {time.perf_counter() - t0:.1f}s")
    elapsed = time.perf_counter() - t0
    df = pd.DataFrame(rows).dropna(subset=["price"]).reset_index(drop=True)
    n_calls = int(df["is_call"].sum())
    print(f"  american dataset done: n={len(df)} "
          f"({len(df) - n_calls} puts, {n_calls} calls), "
          f"total time {elapsed:.1f}s, per option "
          f"{elapsed / max(len(df), 1) * 1000:.1f} ms")
    return df, elapsed


def generate_barrier_dataset(n, rng):
    """Sample n barrier options, price each by Heston Monte Carlo, and return
    the dataset (with per-option mc_se) and its total pricing time."""
    rows = []
    t0 = time.perf_counter()
    for i in range(n):
        p = sample_barrier_params(rng)
        # Use a per-option seed for reproducibility while still varying paths
        seed = int(rng.integers(0, 2 ** 31 - 1))
        price, se = heston_mc_barrier_call(
            S0_FIXED, p["K"], p["T"], p["r"], p["q"], p["v0"], p["theta"],
            p["kappa"], p["eta"], p["rho"], p["B"],
            n_steps=HESTON_STEPS, n_paths=HESTON_PATHS, seed=seed,
            return_se=True,
        )
        p["price"] = price
        p["mc_se"] = se          # Monte Carlo noise floor for this option
        rows.append(p)
        if (i + 1) % 100 == 0:
            print(f"  barrier {i + 1}/{n}, "
                  f"elapsed {time.perf_counter() - t0:.1f}s")
    elapsed = time.perf_counter() - t0
    df = pd.DataFrame(rows).dropna(subset=["price"]).reset_index(drop=True)
    print(f"  barrier dataset done: n={len(df)}, total time {elapsed:.1f}s, "
          f"per option {elapsed / max(len(df), 1) * 1000:.1f} ms")
    print(f"  barrier MC standard error: mean {df['mc_se'].mean():.4f}, "
          f"90th pct {df['mc_se'].quantile(0.90):.4f}")
    return df, elapsed


# --- ML pipeline: GPR and feed-forward NN ------------------------------------

def fit_gpr(X_train, y_train):
    """Fit a GPR with a Matern 5/2 kernel plus a white-noise term for the
    barrier label noise. Hyperparameters are fitted by sklearn."""
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
    """Fit a two-layer tanh feed-forward network (Adam, early stopping).
    Hyperparameters are reasonable defaults, not tuned."""
    nn = MLPRegressor(hidden_layer_sizes=(64, 64), activation="tanh",
                      solver="adam", alpha=1e-4, learning_rate_init=1e-3,
                      max_iter=5000, tol=1e-6, random_state=SEED,
                      early_stopping=True, validation_fraction=0.1,
                      n_iter_no_change=30)
    nn.fit(X_train, y_train)
    return nn


def train_eval(df, feature_cols, label, num_time):
    """Train GPR and NN and report accuracy, timing and break-even.

    The 80/20 split is on row indices (fixed seed), so the same rows feed
    both models and trace back to df. Features are standardised on the train
    split. Break-even is T_fit / (t_num - t_pred). Returns (metrics, no-arb
    report, put/call split).
    """
    idx = np.arange(len(df))
    idx_tr, idx_te = train_test_split(idx, test_size=0.2, random_state=SEED)
    X = df[feature_cols].values
    y = df["price"].values
    X_train, X_test = X[idx_tr], X[idx_te]
    y_train, y_test = y[idx_tr], y[idx_te]
    df_test = df.iloc[idx_te].reset_index(drop=True)
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    per_option_num_ms = (num_time / len(df)) * 1000
    rows, preds = [], {}
    for name, fitter in [("GPR", fit_gpr), ("NN", fit_nn)]:
        print(f"  fitting {label} {name}.")
        t0 = time.perf_counter()
        model = fitter(X_train_s, y_train)
        train_time = time.perf_counter() - t0

        # Time one full prediction pass, then divide by the test size.
        t0 = time.perf_counter()
        y_pred = model.predict(X_test_s)
        per_pred_ms = (time.perf_counter() - t0) / len(X_test_s) * 1000
        preds[name] = y_pred

        # Barrier no-arbitrage: raising B toward spot makes knock-out likelier,
        # so a down-and-out call must weakly lose value as B rises.
        mono_viol = np.nan
        if label == "barrier" and "B" in feature_cols:
            bi = feature_cols.index("B")
            x_up = X_test.copy()
            x_up[:, bi] = np.minimum(x_up[:, bi] + 2.0, S0_FIXED - 0.5)
            pred_up = model.predict(scaler.transform(x_up))
            mono_viol = float(np.mean(pred_up > y_pred + 1e-3))

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        # MAPE on options worth at least 1% of S0, so near-zero contracts do
        # not blow up the ratio
        mask = np.abs(y_test) > 0.01 * S0_FIXED
        mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask]))
        saved_ms = per_option_num_ms - per_pred_ms
        breakeven = train_time * 1000 / saved_ms if saved_ms > 0 else np.nan

        rows.append({
            "model": name, "label": label,
            "n_train": len(X_train_s), "n_test": len(X_test_s),
            "MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape,
            "fit_time_s": train_time,
            "pred_time_per_option_ms": per_pred_ms,
            "num_time_per_option_ms": per_option_num_ms,
            "speedup_per_query": per_option_num_ms / per_pred_ms
            if per_pred_ms > 0 else np.nan,
            "breakeven_queries": breakeven,
            "barrier_mono_viol": mono_viol,
        })
        plot_pred_vs_true(label, name, y_test, y_pred)
        plot_residuals(label, name, y_test, y_pred)

    noarb, split = evaluate_predictions(label, df_test, y_test, preds)
    return pd.DataFrame(rows), noarb, split


def evaluate_predictions(label, df_test, y_test, preds):
    """No-arbitrage checks and, for the American mix, the put-versus-call
    error split. These catch what the aggregate metrics hide."""
    strike = df_test["K"].values
    noarb_rows, split_rows = [], []
    for name, y_pred in preds.items():
        neg = float(np.mean(y_pred < -1e-6))
        if label == "american":
            is_call = df_test["is_call"].values.astype(bool)
            intrinsic = np.where(is_call,
                                 np.maximum(S0_FIXED - strike, 0.0),
                                 np.maximum(strike - S0_FIXED, 0.0))
            upper = np.where(is_call, S0_FIXED, strike)
            below_intrinsic = float(np.mean(y_pred < intrinsic - 1e-3))
            above_upper = float(np.mean(y_pred > upper + 1e-6))
            for subset, mask in [("put", ~is_call), ("call", is_call)]:
                split_rows.append({
                    "label": label, "model": name, "subset": subset,
                    "n": int(mask.sum()),
                    "MAE": mean_absolute_error(y_test[mask], y_pred[mask]),
                    "RMSE": np.sqrt(mean_squared_error(
                        y_test[mask], y_pred[mask])),
                    "R2": r2_score(y_test[mask], y_pred[mask]),
                })
        else:
            # A down-and-out call can sit below its vanilla intrinsic, so the
            # only clean bounds are non-negativity and the spot ceiling.
            below_intrinsic = np.nan
            above_upper = float(np.mean(y_pred > S0_FIXED + 1e-6))
        noarb_rows.append({
            "label": label, "model": name,
            "frac_negative": neg,
            "frac_below_intrinsic": below_intrinsic,
            "frac_above_upper_bound": above_upper,
        })
    return pd.DataFrame(noarb_rows), pd.DataFrame(split_rows)


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
    fig.savefig(FIG_DIR / f"01_pred_vs_true_{label}_{name}.pdf",
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
    fig.savefig(FIG_DIR / f"02_residuals_{label}_{name}.pdf",
                bbox_inches="tight")
    plt.close(fig)


# --- Driver ------------------------------------------------------------------

def main():
    rng = np.random.default_rng(SEED)

    print("Generating American Put/Call dataset.")
    df_am, t_am = generate_american_dataset(N_AMERICAN, rng)
    df_am.to_csv(TAB_DIR / "01_dataset_american.csv", index=False)

    print("Generating Barrier down-and-out Call dataset.")
    df_ba, t_ba = generate_barrier_dataset(N_BARRIER, rng)
    df_ba.to_csv(TAB_DIR / "01_dataset_barrier.csv", index=False)

    am_features = ["K", "T", "r", "q", "sigma", "is_call"]
    ba_features = ["K", "T", "r", "q", "v0", "theta", "kappa",
                   "eta", "rho", "B"]
    report_cols = ["model", "MAE", "RMSE", "R2", "MAPE", "fit_time_s",
                   "pred_time_per_option_ms", "speedup_per_query",
                   "breakeven_queries"]

    print("\nTraining ML models on the American dataset.")
    res_am, noarb_am, split_am = train_eval(df_am, am_features,
                                            "american", t_am)
    res_am.to_csv(TAB_DIR / "02_metrics_american.csv", index=False)
    noarb_am.to_csv(TAB_DIR / "04_noarb_american.csv", index=False)
    split_am.to_csv(TAB_DIR / "03_putcall_american.csv", index=False)
    print(res_am[report_cols].to_string(index=False))
    print(split_am.to_string(index=False))
    print(noarb_am.to_string(index=False))

    print("\nTraining ML models on the Barrier dataset.")
    res_ba, noarb_ba, _ = train_eval(df_ba, ba_features, "barrier", t_ba)
    res_ba.to_csv(TAB_DIR / "02_metrics_barrier.csv", index=False)
    noarb_ba.to_csv(TAB_DIR / "04_noarb_barrier.csv", index=False)
    print(res_ba[report_cols].to_string(index=False))
    print(noarb_ba.to_string(index=False))


if __name__ == "__main__":
    main()
