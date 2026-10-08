# Milagro Regression Visualizer

An interactive OLS regression tool for analyzing Milagro store profitability. Select predictors, fit a model, inspect diagnostics, and export a business-ready report package — all from a browser.

---

## Setup & Deployment

### 1. Install dependencies

```bash
pip install streamlit pandas numpy statsmodels matplotlib scipy scikit-learn openpyxl
```

### 2. Required data files

Place all three CSV files **in the same directory** as `regression_quantifier.py`:

| File                    | Contents                                                       |
| ----------------------- | -------------------------------------------------------------- |
| `train_data.csv`      | Training stores with`annual.profit`                          |
| `test_data.csv`       | Hold-out stores with`annual.profit`                          |
| `site_const_data.csv` | Under-construction stores with`Kathleen.Previous.Prediction` |

### 3. Run the app

```bash
streamlit run regression_quantifier.py
```

The app opens in your browser at `http://localhost:8501`. No other configuration is needed.

---

## Using the App

1. **Select variables** in the left sidebar — features are grouped by category (store characteristics, demographics, economic, education, commute). Individual groups have All/None shortcuts.
2. Click **▶ Run Regression** to fit the OLS model.
3. Review metrics, diagnostic plots, and per-store comparisons in the main panel.
4. Click **📦 Download Report Package** to export the full PDF + Excel pack (see below).

---

## Report Package

Clicking the download button generates `milagro_report_package.zip` with 8 files:

| File                                 | What it shows                                                                                          | Why it's included                                                                |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------- |
| `01_test_actual_vs_predicted.pdf`  | Scatter of actual vs predicted profit on the 9 held-out test stores                                    | Primary model validation — shows how well predictions track reality             |
| `02_model_vs_original_scatter.pdf` | OLS model vs Original predictions for the 12 under-construction stores; top-5 divergent stores labeled | Surfaces where the two forecasts most disagree — actionable for site decisions  |
| `03_metrics_comparison.pdf`        | Side-by-side bar chart: R², OSR², RMSE, MAPE for OLS vs Original model                               | Quantifies the improvement (or trade-off) in forecast accuracy                   |
| `04_correlation_matrix.pdf`        | Pearson correlation heatmap of selected predictors + target                                            | Reveals multicollinearity and strongest drivers before interpreting coefficients |
| `05_per_store_comparison.pdf`      | Table of all under-construction stores sorted by                                                       | difference                                                                       |
| `05_per_store_comparison.xlsx`     | Same table, Excel-formatted with conditional coloring and dollar/percent formats                       | Ready to paste into a memo or share with ops teams                               |
| `06_coefficient_table.pdf`         | OLS coefficients sorted by magnitude; insignificant features (p > 0.05) grayed                         | Communicates which levers matter and by how much                                 |
| `06_coefficient_table.xlsx`        | Same table with 95% CIs, t-stats, p-values                                                             | Allows further analysis or inclusion in technical appendices                     |

All PDFs use a white, print-ready theme. All figures include titles and labeled axes.

### Modifying the report outputs

The export functions are defined between the `# Export figures` and `# Excel export helpers` section headers (~lines 220–450). Each is self-contained:

- **Change a plot's title or axis labels:** edit the `ax.set_title(...)`, `ax.set_xlabel(...)`, `ax.set_ylabel(...)` calls inside the relevant `export_fig_*` function.
- **Add or remove a file from the zip:** edit the `named_figs` or `named_xlsx` lists inside `main()` (look for the `# Download Report Package` comment).
- **Change the baseline model** used in the metrics comparison chart: update `ORIGINAL_MODEL_FEATURES` near the top of the file (currently `["agg.inc", "sqft", "col.grad", "com60"]`).
- **Adjust Excel formatting** (colors, column widths, number formats): edit `export_excel_per_store()` or `export_excel_coefficients()`.

---

## Notes

- The app caches data loading (`@st.cache_data`), so CSV files are only read once per session. If you swap in new data files, restart the app.
- The report package is regenerated on every page render after running a regression — there is no separate "generate" step. For large feature sets with many dummy variables, this may take a few seconds.
- `openpyxl` is required for Excel export. It ships with most pandas installations but can be installed separately with `pip install openpyxl` if needed.
