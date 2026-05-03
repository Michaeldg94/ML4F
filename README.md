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
