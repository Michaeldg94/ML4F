# ML4F

Coursework repository.

Each homework lives in its own subdirectory.

## HW1

The full problem set is inside [`HW1/`](HW1/). The compiled report is
[`HW1/solution.pdf`](HW1/solution.pdf) (17 pages).

### Layout

```
HW1/
├── HW1_Arratia26bse.pdf          assignment statement
├── goyal-welch2022Monthly.csv    Goyal-Welch (2022) monthly dataset
├── HW1_getassets.ipynb           helper notebook (data download)
├── RHW1_helper_GWcausalSP500.R   R helper script
├── exercise_01.py                Ex 1: causality + volatility spillover
├── exercise_02.ipynb             Ex 2: NN vs LSTM vs GP (clean source)
├── exercise_02_EXECUTED.ipynb    Ex 2: same notebook with Colab run outputs
├── exercise_03.py                Ex 3: ARMA-GARCH baseline + Yeo-Johnson
├── requirements.txt              exact-pin dependencies (Python 3.11)
├── solution.tex                  LaTeX source of the report
├── solution.pdf                  compiled report
└── output/                       all generated artefacts
    ├── table_*.{csv,tex}         Ex 1: 4 directed lag tables
    ├── network_*.pdf             Ex 1: 4 causality network plots
    ├── contemp_corr_*.csv        Ex 1: 4 contemporaneous correlation matrices
    ├── ex2_metrics.{csv,tex}     Ex 2: main test metrics
    ├── ex2_sensitivity.{csv,tex} Ex 2: drop-each-indicator sensitivity
    ├── ex2_pred_*.csv            Ex 2: aligned actual vs predicted
    ├── ex2_fig_*.pdf             Ex 2: three prediction plots
    ├── ex2_feature_selection.csv Ex 2: Lasso / L1-logistic survivors
    ├── ex3_metrics.{csv,tex}     Ex 3: ARMA-GARCH baseline metrics
    ├── ex3_combined.{csv,tex}    Ex 3: head-to-head with Ex 2
    ├── ex3_arma_orders.csv       Ex 3: auto-selected (p,q) and GARCH params
    ├── ex3_pred_*.csv            Ex 3: aligned actual vs predicted
    └── ex3_fig_*.pdf             Ex 3: three prediction plots
```

### Just want to inspect the results?

Open [`HW1/solution.pdf`](HW1/solution.pdf). Every table and figure in the
report is also available as a standalone CSV / PDF inside `HW1/output/`.

### Want to re-run end-to-end?

```bash
cd HW1
pip install -r requirements.txt    # tested on Python 3.11.4

# Exercise 1 (causality + volatility spillover; ~30 s on CPU)
python3 exercise_01.py

# Exercise 2 (NN vs LSTM vs GP; minutes on a GPU, hours on CPU)
# Open exercise_02.ipynb in Google Colab with a GPU runtime and run all cells.
# It writes its outputs into HW1/output/ if mounted at /content/drive/MyDrive/ML4F_HW1.

# Exercise 3 (ARMA-GARCH baseline; ~30 s on CPU; reads ex2_metrics.csv from output/)
python3 exercise_03.py

# Recompile the report (TeX Live; pdflatex twice for cross-references)
pdflatex solution.tex && pdflatex solution.tex
```

### What the report concludes

- **Ex 1.** At weekly frequency the returns network is sparse (six edges, all
  at lag 1) while the EWMA volatility network is dense (twenty-one edges); the
  equity block (SPY, XLK, XLF, XLE) is the dominant source of cross-asset
  volatility spillover, with TLT/GLD/GSG and BTC as receivers.
- **Ex 2.** With TimeSeriesSplit Lasso feature selection, no model beats the
  unconditional-mean baseline on next-month returns; on price the GP with a
  custom RBF + Linear + WhiteNoise kernel wins; on direction every model
  either degenerates to the majority class or produces non-trivial but
  slightly worse-than-random predictions.
- **Ex 3.** Plain and Yeo-Johnson-normalised ARMA(p,q)+GARCH(1,1) baselines
  beat or tie every Exercise 2 model on every target, including direction,
  where they are the only models with a positive Matthews correlation.

The full data-scientist commentary lives in `HW1/solution.pdf`.

### Authors

See `HW1/solution.pdf` cover page.

## HW2

The full problem set is inside [`HW2/`](HW2/). The compiled report is
[`HW2/solution.pdf`](HW2/solution.pdf) (9 pages).

### Layout

```
HW2/
├── HW2_Arratia26BSE.pdf          assignment statement
├── dataSent/                     provided price + sentiment data (19 tickers)
├── exercise_01.py                Ex 1: sentiment-slope + RVT-confirmation strategy (DB, HSBC)
├── exercise_01_extended.py       Ex 1: extended search over all full-history tickers
├── exercise_02.py                Ex 2: GPR + feed-forward NN option pricing
├── exercise_03.py                Ex 3: Euler MC speed-ups V0-V4 + Heston full-truncation
├── requirements.txt              exact-pin dependencies (Python 3.11)
├── solution.tex                  LaTeX source of the report
├── solution.pdf                  compiled report (9 pages)
└── output/                       all generated artefacts
    ├── ex01_whole_*.csv          Ex 1: whole-period metrics per ticker
    ├── ex01_rolling_*.csv        Ex 1: rolling-window summaries (w = 254 and 127)
    ├── ex01_tuning_*.csv         Ex 1: grid-search trace per strategy family
    ├── ex01_equity_*.pdf         Ex 1: equity curve, strategy vs buy-and-hold
    ├── ex01_signals_*.pdf        Ex 1: signal vs price and ev
    ├── ex01_rolling_dist_*.pdf   Ex 1: per-window excess distribution at w = 254
    ├── ex02_dataset_*.csv        Ex 2: synthetic (X, y) option-price datasets
    ├── ex02_metrics_*.csv        Ex 2: per-model accuracy and inference speedup
    ├── ex02_pred_vs_true_*.pdf   Ex 2: predicted vs true price scatter
    ├── ex02_residuals_*.pdf      Ex 2: residual plots
    ├── ex03_euler_bench.csv      Ex 3: V0-V4 vanilla European-call benchmark
    ├── ex03_heston_bench.csv     Ex 3: scalar vs vectorised full-truncation Heston
    └── ex03_convergence.{csv,pdf} Ex 3: RMS error vs path count, 20 seeds averaged
```

### Just want to inspect the results?

Open [`HW2/solution.pdf`](HW2/solution.pdf). Every table and figure in the
report is also available as a standalone CSV / PDF inside `HW2/output/`.

### Want to re-run end-to-end?

```bash
cd HW2
pip install -r requirements.txt    # tested on Python 3.11.4

# Exercise 1 (sentiment trading strategy; about 1 min on CPU)
python3 exercise_01.py

# Exercise 2 (GPR + NN option pricing; about 1 min on CPU)
python3 exercise_02.py

# Exercise 3 (Euler MC improvements + convergence study; about 30 s on CPU)
python3 exercise_03.py

# Recompile the report (TeX Live; pdflatex twice for cross-references)
pdflatex solution.tex && pdflatex solution.tex
```

### What the report concludes

- **Ex 1.** A sentiment-slope signal confirmed by an RVT news-volume z-score,
  tuned by grid search and evaluated with a one-percent round-trip cost,
  extracts genuine alpha on Deutsche Bank and HSBC over 2018 to 2020. DB
  (ev = BBr) averages a per-window excess of +0.82 at the one-year horizon and
  beats buy-and-hold in 100 percent of 347 rolling windows; HSBC
  (ev = positiveP) averages +0.29 with a 74 percent win rate. Both stocks fell
  60 to 70 percent over the sample, so the long-short version dominates because
  the short leg captures the March 2020 drawdown.
- **Ex 2.** Gaussian Process Regression and a feed-forward neural network learn
  the pricing functions of American Put and Call options (Cox-Ross-Rubinstein
  binomial ground truth) and Barrier down-and-out Call options (Heston
  Monte-Carlo ground truth) to an R-squared of 0.999, with the GPR more
  accurate and the NN one to two orders of magnitude faster to fit and predict.
  Inference runs two to four orders of magnitude quicker than the numerical
  pricers.
- **Ex 3.** Four cumulative improvements to the Euler Monte-Carlo method
  (vectorisation across paths, antithetic variates, the log-Euler scheme and a
  scrambled Sobol low-discrepancy sequence), plus a full-truncation Euler scheme
  for the Heston variance, cut barrier-option pricing time about 95-fold and let
  the Sobol variant beat the 1/sqrt(N) error rate for large path counts.

The full data-scientist commentary lives in `HW2/solution.pdf`.

### Authors

See `HW2/solution.pdf` cover page.
