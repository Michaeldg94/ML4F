# Machine Learning for Finance: HW1

This folder contains the cleaned HW1 report, code, data and checking package.
The main file to read is `solution.pdf`. The main file to edit is
`solution.tex`.

## Quick Map

- `solution.pdf`: compiled homework report.
- `solution.tex`: LaTeX source for the report.
- `references.bib`: papers, lecture-linked sources and the SEC web reference.
- `requirements.txt`: Python dependencies used by the scripts.
- `data/goyal-welch2022Monthly.csv`: Goyal-Welch monthly data for Exercises 2
  and 3.
- `docs/HW1_Arratia26bse.pdf`: assignment statement.
- `solution_old.tex`: older draft used only as a comparison source.

## Code

- `scripts/exercise_01.py`: downloads and cleans the asset data, builds return
  and EWMA-volatility Granger tables, writes Exercise 1 figures and macros.
- `scripts/exercise_02.ipynb`: notebook for NN, GP and LSTM forecasts.
- `scripts/exercise_03.py`: ARMA-GARCH and Yeo-Johnson benchmark.
- `scripts/generate_report_macros.py`: gathers generated metrics into LaTeX
  macros so the report does not hard-code result numbers.

## Output Folders

- `output/ex1/`: prices, returns, EWMA volatility, Granger tables, correlation
  matrices, network figures and Exercise 1 macros.
- `output/ex2/`: feature selection, forecast metrics, sensitivity results,
  prediction CSVs, figures and Exercise 2 macros.
- `output/ex3/`: ARMA-GARCH metrics, model orders, residual diagnostics,
  prediction CSVs, figures and Exercise 3 macros.

## Build The Report

From this folder:

```bash
python -m pip install -r requirements.txt
python scripts/exercise_01.py
python scripts/exercise_03.py
python scripts/generate_report_macros.py
latexmk -pdf solution.tex
```

Exercise 2 is a notebook workflow. Open `scripts/exercise_02.ipynb`, run it
with the same environment, then rerun `scripts/generate_report_macros.py` and
`latexmk -pdf solution.tex`.

## Checks

The most useful direct checks are:

```bash
python HW1_prompt_engineering_package/scripts/check_lab_code_references.py
python HW1_prompt_engineering_package/scripts/check_source_references.py
python HW1_prompt_engineering_package/scripts/check_citation_grounding.py
```

The full prompt-engineering test suite expects the project to be visible as a
folder named `HW1`. One simple way to run it without renaming this cleaned
folder is:

```bash
tmp=/tmp/hw1_check
mkdir -p "$tmp/ProblemSets"
cp -a HW1_prompt_engineering_package "$tmp/ProblemSets/"
ln -s "$PWD" "$tmp/ProblemSets/HW1"
cd "$tmp/ProblemSets/HW1_prompt_engineering_package"
python -m pytest tests
```

## Reference And Evidence Files

- `HW1_prompt_engineering_package/docs/lab_code_reference_check.md`: confirms
  the report's formulas and lab mentions are tied to the correct class labs.
- `HW1_prompt_engineering_package/docs/source_reference_check.md`: source
  grounding audit.
- `HW1_prompt_engineering_package/docs/table_figure_audit.md`: table and
  figure commentary audit.
- `HW1_prompt_engineering_package/docs/economic_writing_style_patterns.md`:
  writing-style rules used for the final prose pass.

## Practical Notes

- Do not edit generated result numbers directly in `solution.tex`. Add or
  update macros through the scripts.
- The table captions are kept stable. Explanatory table notes appear
  immediately after the captions.
- The only recent-event citation used in the report is the SEC spot Bitcoin
  ETP approval. Other recent-event sources are documented as checked context.
