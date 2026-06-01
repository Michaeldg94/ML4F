"""HW2 Exercise 3 — Faster Euler Monte Carlo.

Four cumulative speedups over the lab's loop-based Euler MC, benchmarked on a
vanilla European call against the Black-Scholes truth:
    V0  scalar Python loop (baseline),
    V1  vectorised across paths,
    V2  + antithetic variates,
    V3  + log-Euler (exact for GBM),
    V4  + scrambled Sobol quasi-Monte-Carlo.
We also pit the vectorised full-truncation Heston engine of exercise_02
against a naive scalar Heston. Outputs land in output/ex03_*.

Authors: Marco Canal, Michael Duarte Gonçalves, Olivia Peter.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as si
import matplotlib.pyplot as plt
from scipy.stats import qmc

SEED = 42
np.random.seed(SEED)

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "output" / "ex3"        # exercise-3 outputs live here
FIG_DIR = OUT_DIR / "figures"            # plots, numbered within this folder
TAB_DIR = OUT_DIR / "tables"             # CSVs, numbered within this folder
for _d in (FIG_DIR, TAB_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# --- Black-Scholes-Merton reference ------------------------------------------

def bsm_call(S0, K, T, r, sigma):
    """Black-Scholes-Merton closed-form European Call price."""
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * si.norm.cdf(d1) - K * np.exp(-r * T) * si.norm.cdf(d2)


# --- Vanilla GBM Euler MC variants -------------------------------------------
# All variants price a European call on a non-dividend asset under geometric
# Brownian motion with constant volatility. They differ only in how the walk
# is discretised and in which random source feeds it.

def euler_v0_scalar(S0, K, T, r, sigma, n_steps, n_paths, seed):
    """V0: faithful translation of the lab's loop-based Euler MC. Two
    nested Python loops, scalar draws, no variance reduction. Uses the
    arithmetic Euler scheme dS = r S dt + sigma S sqrt(dt) Z."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    payoff_sum = 0.0
    for _ in range(n_paths):
        S = S0
        for _ in range(n_steps):
            Z = rng.normal()
            S = S + r * S * dt + sigma * S * sqrt_dt * Z
        payoff_sum += max(S - K, 0.0)
    return np.exp(-r * T) * payoff_sum / n_paths


def euler_v1_vec(S0, K, T, r, sigma, n_steps, n_paths, seed):
    """V1: vectorised Euler. Paths run in parallel through a NumPy array;
    only the time loop remains in Python. Same discretisation as V0."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    S = np.full(n_paths, S0, dtype=np.float64)
    for _ in range(n_steps):
        Z = rng.standard_normal(n_paths)
        S = S + r * S * dt + sigma * S * sqrt_dt * Z
    payoff = np.maximum(S - K, 0.0)
    return np.exp(-r * T) * payoff.mean()


def euler_v2_vec_anti(S0, K, T, r, sigma, n_steps, n_paths, seed):
    """V2: vectorised + antithetic variates. Half the paths use Z, the
    other half use -Z, so the symmetric component of the noise is
    integrated exactly and only the antisymmetric residual contributes
    Monte-Carlo variance."""
    rng = np.random.default_rng(seed)
    half = n_paths // 2
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    S = np.full(2 * half, S0, dtype=np.float64)
    for _ in range(n_steps):
        Z = rng.standard_normal(half)
        Z_full = np.concatenate([Z, -Z])
        S = S + r * S * dt + sigma * S * sqrt_dt * Z_full
    payoff = np.maximum(S - K, 0.0)
    return np.exp(-r * T) * payoff.mean()


def euler_v3_log_anti(S0, K, T, r, sigma, n_steps, n_paths, seed):
    """V3: log-Euler + antithetic. Steps log S directly, which is exact for
    GBM and keeps S positive."""
    rng = np.random.default_rng(seed)
    half = n_paths // 2
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    drift = (r - 0.5 * sigma ** 2) * dt
    diffusion = sigma * sqrt_dt
    log_S = np.full(2 * half, np.log(S0), dtype=np.float64)
    for _ in range(n_steps):
        Z = rng.standard_normal(half)
        Z_full = np.concatenate([Z, -Z])
        log_S = log_S + drift + diffusion * Z_full
    S = np.exp(log_S)
    payoff = np.maximum(S - K, 0.0)
    return np.exp(-r * T) * payoff.mean()


def euler_v4_qmc(S0, K, T, r, sigma, n_steps, n_paths, seed):
    """V4: log-Euler + antithetic + scrambled Sobol QMC, for faster than
    pseudo-random convergence at a small generation cost."""
    half = n_paths // 2
    sampler = qmc.Sobol(d=n_steps, scramble=True, seed=seed)
    u = sampler.random(half)
    # avoid u = 0 or 1 which map to +/- infinity under the inverse normal
    u = np.clip(u, 1e-12, 1 - 1e-12)
    Z = si.norm.ppf(u)                                # shape (half, n_steps)
    Z_full = np.concatenate([Z, -Z], axis=0)          # antithetic pair

    dt = T / n_steps
    drift = (r - 0.5 * sigma ** 2) * dt
    diffusion = sigma * np.sqrt(dt)
    increments = drift + diffusion * Z_full           # shape (2*half, n_steps)
    log_S = np.log(S0) + np.cumsum(increments, axis=1)
    S_T = np.exp(log_S[:, -1])
    payoff = np.maximum(S_T - K, 0.0)
    return np.exp(-r * T) * payoff.mean()


# --- Heston variants: naive scalar vs vectorised full-truncation -------------

def heston_scalar_naive(S0, K, T, r, q, v0, theta, kappa, eta, rho, B,
                        n_steps, n_paths, seed):
    """Naive scalar Heston MC (nested loops, no antithetic). The slow
    baseline for the Heston benchmark only."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    payoff_sum = 0.0
    for _ in range(n_paths):
        S = S0
        v = v0
        alive = True
        for _ in range(n_steps):
            Z1 = rng.normal()
            Z2 = rng.normal()
            W1 = Z1
            W2 = rho * Z1 + np.sqrt(1.0 - rho * rho) * Z2
            v_eff = max(v, 0.0)
            v = (v + kappa * (theta - v_eff) * dt
                 + eta * np.sqrt(v_eff) * sqrt_dt * W2)
            v = max(v, 0.0)
            S = S + r * S * dt + np.sqrt(v_eff) * S * sqrt_dt * W1 - q * S * dt
            if S <= B:
                alive = False
        if alive:
            payoff_sum += max(S - K, 0.0)
    return np.exp(-r * T) * payoff_sum / n_paths


def heston_vec_fulltrunc(S0, K, T, r, q, v0, theta, kappa, eta, rho, B,
                         n_steps, n_paths, seed):
    """Vectorised Heston used by exercise_02: full-truncation variance,
    antithetic variates, log-Euler price, with a path-wise knockout mask."""
    rng = np.random.default_rng(seed)
    half = n_paths // 2
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    S = np.full(2 * half, S0, dtype=np.float64)
    v = np.full(2 * half, v0, dtype=np.float64)
    alive = np.ones(2 * half, dtype=bool)
    for _ in range(n_steps):
        Z1 = rng.standard_normal(half)
        Z2 = rng.standard_normal(half)
        W1 = np.concatenate([Z1, -Z1])
        Z2_full = np.concatenate([Z2, -Z2])
        W2 = rho * W1 + np.sqrt(1.0 - rho * rho) * Z2_full
        v_pos = np.maximum(v, 0.0)
        v = (v + kappa * (theta - v_pos) * dt
             + eta * np.sqrt(v_pos) * sqrt_dt * W2)
        S = S * np.exp((r - q - 0.5 * v_pos) * dt
                       + np.sqrt(v_pos) * sqrt_dt * W1)
        alive = alive & (S > B)
    payoff = np.where(alive, np.maximum(S - K, 0.0), 0.0)
    return np.exp(-r * T) * payoff.mean()


# --- Benchmark driver --------------------------------------------------------

def bench_one(name, fn, kwargs, true_value=None):
    """Time fn on kwargs and return (name, price, time_ms, abs_err)."""
    t0 = time.perf_counter()
    price = fn(**kwargs)
    t = (time.perf_counter() - t0) * 1000
    err = (price - true_value) if true_value is not None else np.nan
    return {"variant": name, "price": price, "time_ms": t, "abs_err": err}


def bench_euler():
    """Benchmark the five vanilla-Euler variants on a single, mildly
    challenging European Call parameter set, and check pricing error
    against the BSM closed form."""
    S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.03, 0.25
    n_steps = 200
    n_paths = 20_000
    kwargs = dict(S0=S0, K=K, T=T, r=r, sigma=sigma,
                  n_steps=n_steps, n_paths=n_paths, seed=SEED)
    true_value = bsm_call(S0, K, T, r, sigma)

    variants = [
        # V0 runs at fewer paths because the scalar loop is slow.
        ("V0 scalar loop", euler_v0_scalar, dict(kwargs, n_paths=2000)),
        ("V1 vectorised", euler_v1_vec, kwargs),
        ("V2 + antithetic", euler_v2_vec_anti, kwargs),
        ("V3 + log-Euler", euler_v3_log_anti, kwargs),
        ("V4 + Sobol QMC", euler_v4_qmc, kwargs),
    ]
    rows = []
    for name, fn, kw in variants:
        row = bench_one(name, fn, kw, true_value=true_value)
        row["n_paths"] = kw["n_paths"]
        rows.append(row)
        print(f"  {name}: price={row['price']:.4f} BSM={true_value:.4f} "
              f"err={row['abs_err']:+.4f} time={row['time_ms']:.1f} ms "
              f"(n_paths={kw['n_paths']})")
    return pd.DataFrame(rows), true_value


def bench_heston():
    """Benchmark the Heston scalar-naive against the improved vectorised
    full-truncation version on one barrier option."""
    p = dict(S0=100.0, K=100.0, T=1.0, r=0.03, q=0.0, v0=0.04, theta=0.04,
             kappa=2.0, eta=0.3, rho=-0.5, B=80.0)
    base_kwargs = dict(**p, n_steps=100, n_paths=10_000, seed=SEED)
    rows = []
    # The naive scalar Heston is too slow at 10_000 paths, so benchmark at
    # 1000 paths and scale the time linearly when reporting per-option cost
    naive_kwargs = dict(base_kwargs, n_paths=1000)
    rows.append({"variant": "scalar naive Heston (1000 paths)",
                 **{k: v for k, v in
                    bench_one("scalar", heston_scalar_naive,
                              naive_kwargs).items() if k != "variant"},
                 "n_paths": 1000})
    rows.append({"variant": "vectorised full-truncation Heston (10000 paths)",
                 **{k: v for k, v in
                    bench_one("vec", heston_vec_fulltrunc,
                              base_kwargs).items() if k != "variant"},
                 "n_paths": 10_000})
    # Now scale the scalar time to a fair 10000-paths-equivalent estimate
    scaled = rows[0]["time_ms"] * (10_000 / 1000)
    rows.append({"variant": "scalar naive Heston (scaled to 10000 paths)",
                 "price": rows[0]["price"], "time_ms": scaled,
                 "abs_err": np.nan, "n_paths": 10_000})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df


def bench_convergence(n_seeds=20):
    """Plot RMS error vs n_paths for V1 (vec), V2 (anti), V4 (Sobol).
    The error at each n_paths is averaged across n_seeds independent
    seeds so the convergence trend is not dominated by single-realisation
    fluctuations."""
    S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.03, 0.25
    n_steps = 200
    true_value = bsm_call(S0, K, T, r, sigma)
    grid = [256, 512, 1024, 2048, 4096, 8192, 16384]
    rows = []
    for n_paths in grid:
        for name, fn in [("V1 vectorised", euler_v1_vec),
                         ("V2 antithetic", euler_v2_vec_anti),
                         ("V4 Sobol QMC", euler_v4_qmc)]:
            errors = []
            for s in range(n_seeds):
                price = fn(S0=S0, K=K, T=T, r=r, sigma=sigma,
                           n_steps=n_steps, n_paths=n_paths,
                           seed=SEED + 1000 * s)
                errors.append(price - true_value)
            errors = np.asarray(errors)
            rms = float(np.sqrt(np.mean(errors ** 2)))
            rows.append({"variant": name, "n_paths": n_paths,
                         "rms_err": rms,
                         "mean_err": float(errors.mean())})
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8, 6))
    for v in df["variant"].unique():
        sub = df[df["variant"] == v]
        ax.loglog(sub["n_paths"], sub["rms_err"], marker="o", label=v)
    # reference O(1/sqrt N) and O(1/N) lines anchored at the first V1 point
    n0 = grid[0]
    e_ref = df[(df["variant"] == "V1 vectorised") &
               (df["n_paths"] == n0)]["rms_err"].iloc[0]
    ax.loglog(grid, [e_ref * np.sqrt(n0 / n) for n in grid],
              "k--", linewidth=0.6, label="ref. 1/sqrt(N)")
    ax.loglog(grid, [e_ref * (n0 / n) for n in grid],
              "k:", linewidth=0.6, label="ref. 1/N")
    ax.set_xlabel("number of paths N")
    ax.set_ylabel(f"RMS |MC - BSM| over {n_seeds} seeds")
    ax.set_title(f"Euler MC convergence ({n_seeds} seeds averaged)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_convergence.pdf", bbox_inches="tight")
    plt.close(fig)
    return df


def main():
    print("=== Vanilla Euler MC benchmark ===")
    euler_df, _ = bench_euler()
    euler_df.to_csv(TAB_DIR / "01_euler_bench.csv", index=False)

    print("\n=== Heston MC benchmark ===")
    heston_df = bench_heston()
    heston_df.to_csv(TAB_DIR / "02_heston_bench.csv", index=False)

    print("\n=== Convergence study ===")
    conv_df = bench_convergence()
    conv_df.to_csv(TAB_DIR / "03_convergence.csv", index=False)
    print("  saved convergence plot")


if __name__ == "__main__":
    main()
