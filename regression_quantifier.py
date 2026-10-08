import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.stats as stats
from sklearn.metrics import mean_absolute_error, mean_squared_error
import io
import zipfile

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

TARGET = "annual.profit"

NUMERIC_FEATURES = [
    "sqft", "pop", "agemed", "non.us.citizen",
    "agg.inc", "med.inc", "noHS", "HS", "some.col", "col.grad",
    "post.grad", "com0", "com15", "com30", "com60", "drive",
    "public", "walk", "home", "other", "lci", "nearcomp",
    "nearmil", "gini", "housemed",
]

CATEGORICAL_FEATURES = ["state", "intersect", "freestand"]

# Original (baseline) model feature set for metrics comparison
ORIGINAL_MODEL_FEATURES = ["agg.inc", "sqft", "col.grad", "com60"]

# Feature groups for sidebar checkboxes
FEATURE_GROUPS = {
    "🏪 Store & Location": ["sqft", "lci", "nearcomp", "nearmil"],
    "👥 Demographics":     ["pop", "agemed", "non.us.citizen", "gini"],
    "💰 Economic":         ["agg.inc", "med.inc", "housemed"],
    "🎓 Education":        ["noHS", "HS", "some.col", "col.grad", "post.grad"],
    "🕐 Commute Time":     ["com0", "com15", "com30", "com60"],
    "🚗 Transport Mode":   ["drive", "public", "walk", "home", "other"],
}

DEFAULT_CHECKED = {"agg.inc", "sqft", "col.grad", "com60"}

FEATURE_LABELS = {
    "sqft":            "sqft — Restaurant size (sq ft)",
    "pop":             "pop — Census tract population",
    "agemed":          "agemed — Median age of census tract (yrs)",
    "non.us.citizen":  "non.us.citizen — % non-US citizens",
    "agg.inc":         "agg.inc — Total annual tract income ($)",
    "med.inc":         "med.inc — Median household income ($)",
    "noHS":            "noHS — % without high school diploma",
    "HS":              "HS — % high school only",
    "some.col":        "some.col — % some college (no degree)",
    "col.grad":        "col.grad — % 4-year college graduates",
    "post.grad":       "post.grad — % post-graduate degree",
    "com0":            "com0 — % commute < 15 min",
    "com15":           "com15 — % commute 15–30 min",
    "com30":           "com30 — % commute 30–60 min",
    "com60":           "com60 — % commute > 60 min",
    "drive":           "drive — % drive to work",
    "public":          "public — % take public transit",
    "walk":            "walk — % walk to work",
    "home":            "home — % work from home",
    "other":           "other — % other commute method",
    "lci":             "lci — Labor cost index (3rd party)",
    "nearcomp":        "nearcomp — # competing fast-casual stores nearby",
    "nearmil":         "nearmil — Distance to nearest Milagro (miles)",
    "gini":            "gini — Income inequality (Gini coefficient)",
    "housemed":        "housemed — Median monthly housing spend ($)",
    "state":           "state — US State (categorical)",
    "intersect":       "intersect — Located at street intersection",
    "freestand":       "freestand — Free-standing building",
}

# ─────────────────────────────────────────────
# Data loading & preprocessing
# ─────────────────────────────────────────────

@st.cache_data
def load_data():
    train = pd.read_csv("train_data.csv")
    test  = pd.read_csv("test_data.csv")
    site  = pd.read_csv("site_const_data.csv")

    # One-hot encode categoricals
    train_enc = pd.get_dummies(train, columns=CATEGORICAL_FEATURES, drop_first=True)
    test_enc  = pd.get_dummies(test,  columns=CATEGORICAL_FEATURES, drop_first=True)
    site_enc  = pd.get_dummies(site,  columns=CATEGORICAL_FEATURES, drop_first=True)

    # Align test and site to train's column schema
    train_enc, test_enc  = train_enc.align(test_enc,  join="left", axis=1, fill_value=0)
    train_enc, site_enc  = train_enc.align(site_enc,  join="left", axis=1, fill_value=0)

    # Convert bool dummy columns to int
    for enc in (train_enc, test_enc, site_enc):
        bool_cols = enc.select_dtypes(include="bool").columns
        enc[bool_cols] = enc[bool_cols].astype(int)

    y_train = train_enc[TARGET]
    y_test  = test_enc[TARGET]
    X_train_full = train_enc.drop(columns=[TARGET, "store.number"], errors="ignore")
    X_test_full  = test_enc.drop(columns=[TARGET, "store.number"], errors="ignore")

    # Site data: keep store numbers and original predictions
    site_store_numbers = site_enc["store.number"].values
    orig_preds         = site["Kathleen.Previous.Prediction"].values
    X_site_full = site_enc.drop(
        columns=["store.number", "Kathleen.Previous.Prediction", TARGET], errors="ignore"
    )

    y_bar = float(y_train.mean())

    cat_dummies = {}
    for cat in CATEGORICAL_FEATURES:
        cat_dummies[cat] = [c for c in X_train_full.columns if c.startswith(cat + "_")]

    return X_train_full, X_test_full, X_site_full, y_train, y_test, \
           y_bar, cat_dummies, site_store_numbers, orig_preds


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────

def osr2(y_true, y_pred, y_bar_train):
    ssr = np.sum((y_true - y_pred) ** 2)
    sst = np.sum((y_true - y_bar_train) ** 2)
    return 1.0 - ssr / sst

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None))) * 100.0

def compute_metrics(y_true_train, y_pred_train, y_true_test, y_pred_test, y_bar_train):
    return {
        "R²  (in-sample)":      float(1 - np.sum((y_true_train - y_pred_train)**2) /
                                           np.sum((y_true_train - y_true_train.mean())**2)),
        "OSR² (out-of-sample)": float(osr2(y_true_test, y_pred_test, y_bar_train)),
        "RMSE":                 float(np.sqrt(mean_squared_error(y_true_test, y_pred_test))),
        "Max Error":            float(np.max(np.abs(y_true_test - y_pred_test))),
        "MAPE (%)":             float(mape(y_true_test, y_pred_test)),
        "MAE":                  float(mean_absolute_error(y_true_test, y_pred_test)),
    }

# ─────────────────────────────────────────────
# App plots  (dark theme)
# ─────────────────────────────────────────────

def make_plots(y_train, y_pred_train, y_test, y_pred_test, result):
    fig = plt.figure(figsize=(14, 11))
    fig.patch.set_facecolor("#0f1117")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)

    ax_style = dict(facecolor="#1a1d27", labelcolor="white",
                    titlecolor="white", titlesize=11, titleweight="bold")

    def style_ax(ax, title, xlabel, ylabel):
        ax.set_facecolor(ax_style["facecolor"])
        ax.set_title(title, color=ax_style["titlecolor"],
                     fontsize=ax_style["titlesize"], fontweight=ax_style["titleweight"], pad=8)
        ax.set_xlabel(xlabel, color="lightgray", fontsize=9)
        ax.set_ylabel(ylabel, color="lightgray", fontsize=9)
        ax.tick_params(colors="lightgray", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#3a3d4a")

    # 1. Actual vs Predicted (test set)
    ax1 = fig.add_subplot(gs[0, 0])
    style_ax(ax1, "Actual vs Predicted (test)", "Predicted ($)", "Actual ($)")
    combined_min = min(y_test.min(), y_pred_test.min())
    combined_max = max(y_test.max(), y_pred_test.max())
    ax1.plot([combined_min, combined_max], [combined_min, combined_max],
             color="#ff6b6b", linewidth=1.5, linestyle="--", label="Ideal")
    ax1.scatter(y_pred_test, y_test, color="#4fc3f7", alpha=0.85, s=60, zorder=3, label="Test obs")
    ax1.legend(fontsize=8, labelcolor="lightgray", facecolor="#1a1d27", edgecolor="#3a3d4a")
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))

    # 2. Residuals vs Fitted (training set)
    ax2 = fig.add_subplot(gs[0, 1])
    style_ax(ax2, "Residuals vs Fitted (train)", "Fitted value ($)", "Residual ($)")
    residuals_train = y_train.values - y_pred_train
    ax2.axhline(0, color="#ff6b6b", linewidth=1.5, linestyle="--")
    ax2.scatter(y_pred_train, residuals_train, color="#a5d6a7", alpha=0.4, s=18, zorder=3)
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e3:.0f}K"))

    # 3. QQ Plot (training residuals)
    ax3 = fig.add_subplot(gs[1, 0])
    style_ax(ax3, "QQ Plot (train residuals)", "Theoretical quantiles", "Sample quantiles")
    (osm, osr_vals), (slope, intercept, _) = stats.probplot(residuals_train, dist="norm")
    ax3.scatter(osm, osr_vals, color="#ce93d8", alpha=0.5, s=18, zorder=3)
    line_x = np.array([osm[0], osm[-1]])
    ax3.plot(line_x, slope * line_x + intercept, color="#ff6b6b", linewidth=1.5, linestyle="--")

    # 4. Coefficient bar chart
    ax4 = fig.add_subplot(gs[1, 1])
    style_ax(ax4, "Coefficients (excl. const)", "Feature", "Coefficient value")
    coef_df = (pd.DataFrame({"coef": result.params, "pval": result.pvalues})
               .drop(index="const", errors="ignore")
               .sort_values("coef", key=abs, ascending=True))
    colors = ["#ef9a9a" if v < 0 else "#80cbc4" for v in coef_df["coef"]]
    alphas = [0.5 if p > 0.05 else 1.0 for p in coef_df["pval"]]
    bars = ax4.barh(coef_df.index, coef_df["coef"], color=colors, alpha=0.9)
    for bar, alpha in zip(bars, alphas):
        bar.set_alpha(alpha)
    ax4.axvline(0, color="white", linewidth=0.8, linestyle="--", alpha=0.5)
    ax4.tick_params(axis="y", labelsize=7)
    ax4.tick_params(axis="x", labelsize=8)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#80cbc4", label="Positive (p≤0.05)"),
        Patch(facecolor="#ef9a9a", label="Negative (p≤0.05)"),
        Patch(facecolor="gray",    alpha=0.5, label="Insig. (p>0.05)"),
    ]
    ax4.legend(handles=legend_elements, fontsize=7, labelcolor="lightgray",
               facecolor="#1a1d27", edgecolor="#3a3d4a", loc="lower right")

    return fig


# ─────────────────────────────────────────────
# Export figures  (light / business theme)
# ─────────────────────────────────────────────

_OLS_COLOR  = "#1a5276"   # dark blue for OLS model
_ORIG_COLOR = "#d35400"   # burnt orange for original model
_REF_COLOR  = "#c0392b"   # red dashed reference lines

def _light_ax(ax):
    ax.set_facecolor("#f7f7f7")
    ax.grid(True, alpha=0.35, linestyle="--", color="#cccccc", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_edgecolor("#aaaaaa")
    ax.tick_params(labelsize=10, colors="#333333")


def export_fig_actual_vs_predicted(y_test, y_pred_test):
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("white")
    _light_ax(ax)

    lo = min(y_test.min(), y_pred_test.min()) * 0.93
    hi = max(y_test.max(), y_pred_test.max()) * 1.07
    ax.plot([lo, hi], [lo, hi], color=_REF_COLOR, lw=1.5, ls="--", label="Perfect fit", zorder=2)
    ax.scatter(y_pred_test, y_test, color=_OLS_COLOR, s=90, alpha=0.85, zorder=3, label="Test stores (n=9)")

    ax.set_title("OLS Model: Actual vs Predicted Annual Profit\n(Out-of-Sample Test Set)",
                 fontsize=13, fontweight="bold", color="#1a1a1a", pad=12)
    ax.set_xlabel("Predicted Annual Profit", fontsize=11, color="#333333")
    ax.set_ylabel("Actual Annual Profit", fontsize=11, color="#333333")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))
    ax.legend(fontsize=10, framealpha=0.9)
    fig.tight_layout()
    return fig


def export_fig_model_vs_original(y_pred_site, orig_preds, store_numbers):
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("white")
    _light_ax(ax)

    lo = min(y_pred_site.min(), orig_preds.min()) * 0.93
    hi = max(y_pred_site.max(), orig_preds.max()) * 1.07
    ax.plot([lo, hi], [lo, hi], color=_REF_COLOR, lw=1.5, ls="--", label="Agreement line", zorder=2)
    ax.scatter(orig_preds, y_pred_site, color=_OLS_COLOR, s=80, alpha=0.85, zorder=3, label="Under-construction stores")

    abs_diffs = np.abs(y_pred_site - orig_preds)
    for i in np.argsort(abs_diffs)[::-1][:5]:
        ax.annotate(f"#{int(store_numbers[i])}",
                    (orig_preds[i], y_pred_site[i]),
                    textcoords="offset points", xytext=(7, 4),
                    fontsize=8, color="#555555",
                    arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.8))

    ax.set_title("Under-Construction Stores:\nOLS Model vs Original Prediction",
                 fontsize=13, fontweight="bold", color="#1a1a1a", pad=12)
    ax.set_xlabel("Original Prediction ($)", fontsize=11, color="#333333")
    ax.set_ylabel("OLS Model Prediction ($)", fontsize=11, color="#333333")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))
    ax.legend(fontsize=10, framealpha=0.9)
    fig.tight_layout()
    return fig


def export_fig_metrics_comparison(metrics_ols, metrics_orig):
    specs = [
        ("R²\n(In-Sample)",      "R²  (in-sample)",      lambda v: f"{v:.4f}", "higher"),
        ("OSR²\n(Out-of-Sample)","OSR² (out-of-sample)",  lambda v: f"{v:.4f}", "higher"),
        ("RMSE",                 "RMSE",                  lambda v: f"${v:,.0f}", "lower"),
        ("MAPE (%)",             "MAPE (%)",              lambda v: f"{v:.2f}%", "lower"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(14, 5))
    fig.patch.set_facecolor("white")
    fig.suptitle("Model Performance: OLS vs Original Model (Test Set)",
                 fontsize=14, fontweight="bold", color="#1a1a1a", y=1.01)

    labels = ["OLS\nModel", "Original\nModel"]
    colors = [_OLS_COLOR, _ORIG_COLOR]

    for ax, (title, key, fmt, direction) in zip(axes, specs):
        v_ols  = metrics_ols[key]
        v_orig = metrics_orig[key]
        ols_wins = (v_ols >= v_orig) if direction == "higher" else (v_ols <= v_orig)

        alphas = [1.0 if ols_wins else 0.45, 0.45 if ols_wins else 1.0]
        bars = ax.bar(labels, [v_ols, v_orig], color=colors, alpha=0.9, width=0.5)
        for bar, alpha in zip(bars, alphas):
            bar.set_alpha(alpha)

        y_max = max(v_ols, v_orig)
        for bar, v in zip(bars, [v_ols, v_orig]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + y_max * 0.02,
                    fmt(v), ha="center", va="bottom", fontsize=9,
                    fontweight="bold", color="#1a1a1a")

        direction_symbol = "↑ higher is better" if direction == "higher" else "↓ lower is better"
        winner = "OLS wins" if ols_wins else "Original wins"
        ax.set_title(title, fontsize=11, fontweight="bold", color="#1a1a1a", pad=8)
        ax.set_xlabel(f"{direction_symbol}\n{winner}", fontsize=8, color="#666666")
        _light_ax(ax)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(0, y_max * 1.22)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=_OLS_COLOR,  label="OLS Model"),
        Patch(facecolor=_ORIG_COLOR, label="Original Model"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2,
               fontsize=10, framealpha=0.9, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    return fig


def export_fig_correlation(X_train_full, selected_cols, y_train_vals):
    corr_data = X_train_full[selected_cols].copy()
    corr_data[TARGET] = y_train_vals
    corr = corr_data.corr()
    n = len(corr)

    size = max(7, n * 0.65)
    fig, ax = plt.subplots(figsize=(size, size * 0.88))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Pearson correlation coefficient", fontsize=10, color="#333333")
    cbar.ax.tick_params(labelsize=9, colors="#333333")
    cbar.outline.set_edgecolor("#cccccc")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=9, color="#333333")
    ax.set_yticklabels(corr.columns, fontsize=9, color="#333333")
    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc")

    if n <= 16:
        for i in range(n):
            for j in range(n):
                v = corr.values[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(v) > 0.6 else "#1a1a1a")

    ax.set_title("Feature Correlation Matrix — Selected Predictors + Target Annual Profit",
                 fontsize=13, fontweight="bold", color="#1a1a1a", pad=14)
    fig.tight_layout()
    return fig


def export_pdf_table_per_store(store_numbers, y_pred_site, orig_preds):
    diffs     = y_pred_site - orig_preds
    pct_diffs = diffs / np.abs(orig_preds)       # decimal proportion
    abs_diffs = np.abs(diffs)
    order     = np.argsort(abs_diffs)[::-1]

    col_labels = ["Store #", "OLS Model ($)", "Original ($)", "Difference ($)", "Diff (%)"]
    rows = []
    row_diffs = []
    for i in order:
        rows.append([
            f"#{int(store_numbers[i])}",
            f"${y_pred_site[i]:,.0f}",
            f"${orig_preds[i]:,.0f}",
            f"${diffs[i]:+,.0f}",
            f"{pct_diffs[i]:+.1%}",
        ])
        row_diffs.append(diffs[i])

    n_rows = len(rows)
    fig_h = max(4.5, 0.38 * n_rows + 2.0)
    fig, ax = plt.subplots(figsize=(11, fig_h))
    fig.patch.set_facecolor("white")
    ax.set_axis_off()

    tbl = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.5)

    # Header
    HDR_BG, HDR_FG = "#1F4E79", "white"
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor(HDR_BG)
        tbl[0, j].set_text_props(color=HDR_FG, fontweight="bold")

    GREEN, RED, STRIPE = "#C6EFCE", "#FFC7CE", "#f0f4f8"
    for row_i, d in enumerate(row_diffs, start=1):
        diff_color = GREEN if d >= 0 else RED
        for j in range(len(col_labels)):
            if j in (3, 4):
                tbl[row_i, j].set_facecolor(diff_color)
            else:
                tbl[row_i, j].set_facecolor(STRIPE if row_i % 2 == 0 else "white")

    ax.set_title(
        "Under-Construction Stores: OLS Model vs Original Prediction\n"
        "(sorted by absolute difference; green = model higher, red = model lower)",
        fontsize=12, fontweight="bold", color="#1a1a1a", pad=16,
    )
    fig.tight_layout()
    return fig


def export_pdf_table_coefficients(result):
    ci = result.conf_int()
    df = pd.DataFrame({
        "Feature":     result.params.index,
        "Coefficient": result.params.values,
        "Std Error":   result.bse.values,
        "t-stat":      result.tvalues.values,
        "p-value":     result.pvalues.values,
        "CI (95%)":    [f"[{lo:.4f}, {hi:.4f}]" for lo, hi in zip(ci[0], ci[1])],
        "_sig":        result.pvalues.values <= 0.05,
    })
    # Sort: most influential first, const last
    const_row  = df[df["Feature"] == "const"]
    other_rows = (df[df["Feature"] != "const"]
                  .sort_values("Coefficient", key=abs, ascending=False))
    df = pd.concat([other_rows, const_row]).reset_index(drop=True)

    col_labels = ["Feature", "Coefficient", "Std Error", "t-stat", "p-value", "95% CI"]
    rows = []
    sig_flags = []
    for _, row in df.iterrows():
        p = row["p-value"]
        star = "*" if p <= 0.05 else ""
        rows.append([
            row["Feature"],
            f"{row['Coefficient']:.4f}",
            f"{row['Std Error']:.4f}",
            f"{row['t-stat']:.3f}",
            f"{p:.4f}{star}",
            row["CI (95%)"],
        ])
        sig_flags.append(bool(row["_sig"]))

    n_rows = len(rows)
    fig_h = max(4.5, 0.38 * n_rows + 2.5)
    fig, ax = plt.subplots(figsize=(12, fig_h))
    fig.patch.set_facecolor("white")
    ax.set_axis_off()

    tbl = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.5)

    HDR_BG, HDR_FG = "#1F4E79", "white"
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor(HDR_BG)
        tbl[0, j].set_text_props(color=HDR_FG, fontweight="bold")

    STRIPE = "#f0f4f8"
    for row_i, sig in enumerate(sig_flags, start=1):
        for j in range(len(col_labels)):
            tbl[row_i, j].set_facecolor(STRIPE if row_i % 2 == 0 else "white")
            if not sig:
                tbl[row_i, j].set_text_props(color="#999999")

    ax.set_title(
        "OLS Regression Coefficients\n(* p ≤ 0.05 — insignificant features shown in gray, sorted by |coefficient|)",
        fontsize=12, fontweight="bold", color="#1a1a1a", pad=16,
    )
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────
# Excel export helpers
# ─────────────────────────────────────────────

def _style_excel_header(ws, n_cols):
    from openpyxl.styles import Font, PatternFill, Alignment
    from openpyxl.utils import get_column_letter
    hdr_fill = PatternFill(start_color="1F4E79", end_color="1F4E79", fill_type="solid")
    hdr_font = Font(bold=True, color="FFFFFF", size=11)
    for j in range(1, n_cols + 1):
        cell = ws.cell(row=1, column=j)
        cell.fill = hdr_fill
        cell.font = hdr_font
        cell.alignment = Alignment(horizontal="center", vertical="center")
    ws.row_dimensions[1].height = 22


def export_excel_per_store(store_numbers, y_pred_site, orig_preds):
    from openpyxl.styles import PatternFill, Alignment, Font
    from openpyxl.utils import get_column_letter

    diffs     = y_pred_site - orig_preds
    pct_diffs = diffs / np.abs(orig_preds)   # decimal for Excel % format
    abs_diffs = np.abs(diffs)
    order     = np.argsort(abs_diffs)[::-1]

    df = pd.DataFrame({
        "Store #":                  store_numbers[order].astype(int),
        "OLS Model Prediction ($)": y_pred_site[order],
        "Original Prediction ($)":  orig_preds[order],
        "Difference ($)":           diffs[order],
        "Difference (%)":           pct_diffs[order],
    })

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Per-Store Comparison")
        ws = writer.sheets["Per-Store Comparison"]

        _style_excel_header(ws, len(df.columns))

        col_widths = [10, 26, 26, 18, 16]
        for j, w in enumerate(col_widths, 1):
            ws.column_dimensions[get_column_letter(j)].width = w

        green_fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
        red_fill   = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
        stripe     = PatternFill(start_color="EBF3FB", end_color="EBF3FB", fill_type="solid")

        for i, (row_idx, d) in enumerate(zip(range(2, len(df) + 2), diffs[order])):
            diff_fill = green_fill if d >= 0 else red_fill
            for col_j in range(1, 6):
                cell = ws.cell(row=row_idx, column=col_j)
                cell.alignment = Alignment(horizontal="center")
                if i % 2 == 1 and col_j not in (4, 5):
                    cell.fill = stripe
            ws.cell(row=row_idx, column=2).number_format = '$#,##0'
            ws.cell(row=row_idx, column=3).number_format = '$#,##0'
            ws.cell(row=row_idx, column=4).number_format = '$#,##0;[Red]-$#,##0'
            ws.cell(row=row_idx, column=4).fill = diff_fill
            ws.cell(row=row_idx, column=5).number_format = '0.0%'
            ws.cell(row=row_idx, column=5).fill = diff_fill

    buf.seek(0)
    return buf.getvalue()


def export_excel_coefficients(result):
    from openpyxl.styles import PatternFill, Alignment, Font
    from openpyxl.utils import get_column_letter

    ci = result.conf_int()
    df = pd.DataFrame({
        "Feature":          result.params.index,
        "Coefficient":      result.params.values,
        "Std Error":        result.bse.values,
        "t-statistic":      result.tvalues.values,
        "p-value":          result.pvalues.values,
        "CI Lower (2.5%)":  ci[0].values,
        "CI Upper (97.5%)": ci[1].values,
        "Significant":      (result.pvalues.values <= 0.05),
    })
    const_row  = df[df["Feature"] == "const"]
    other_rows = df[df["Feature"] != "const"].sort_values("Coefficient", key=abs, ascending=False)
    df = pd.concat([other_rows, const_row]).reset_index(drop=True)

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.drop(columns=["Significant"]).to_excel(writer, index=False, sheet_name="Coefficients")
        ws = writer.sheets["Coefficients"]

        _style_excel_header(ws, 7)

        col_widths = [22, 14, 12, 13, 10, 18, 18]
        for j, w in enumerate(col_widths, 1):
            ws.column_dimensions[get_column_letter(j)].width = w

        sig_vals = df["Significant"].values
        stripe   = PatternFill(start_color="EBF3FB", end_color="EBF3FB", fill_type="solid")
        dim_font = Font(color="999999")

        for i, (row_idx, sig) in enumerate(zip(range(2, len(df) + 2), sig_vals)):
            for col_j in range(1, 8):
                cell = ws.cell(row=row_idx, column=col_j)
                cell.alignment = Alignment(horizontal="center")
                if i % 2 == 1:
                    cell.fill = stripe
                if not sig:
                    cell.font = dim_font
            for col_j in (2, 3, 6, 7):
                ws.cell(row=row_idx, column=col_j).number_format = '0.0000'
            ws.cell(row=row_idx, column=4).number_format = '0.000'
            ws.cell(row=row_idx, column=5).number_format = '0.0000'

    buf.seek(0)
    return buf.getvalue()


# ─────────────────────────────────────────────
# Zip builder
# ─────────────────────────────────────────────

def build_report_zip(named_figs, named_xlsx):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fname, fig in named_figs:
            fig_buf = io.BytesIO()
            fig.savefig(fig_buf, format="pdf", bbox_inches="tight",
                        dpi=150, facecolor="white")
            plt.close(fig)
            fig_buf.seek(0)
            zf.writestr(fname, fig_buf.getvalue())
        for fname, xlsx_bytes in named_xlsx:
            zf.writestr(fname, xlsx_bytes)
    buf.seek(0)
    return buf.getvalue()


# ─────────────────────────────────────────────
# Streamlit app
# ─────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Milagro Regression Visualizer",
        page_icon="📊",
        layout="wide",
    )

    st.title("📊 Milagro Regression Visualizer")
    st.caption("OLS regression on Milagro store data  ·  target: **annual.profit**")

    X_train_full, X_test_full, X_site_full, y_train, y_test, \
        y_bar, cat_dummies, site_store_numbers, orig_preds = load_data()

    # ── Initialise session-state defaults once ─────────────────────────
    for feat in NUMERIC_FEATURES:
        if f"cb_{feat}" not in st.session_state:
            st.session_state[f"cb_{feat}"] = feat in DEFAULT_CHECKED
    for cat in CATEGORICAL_FEATURES:
        if f"cb_{cat}" not in st.session_state:
            st.session_state[f"cb_{cat}"] = False

    # ── Sidebar ──────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Variable Selection")

        g_col1, g_col2 = st.columns(2)
        if g_col1.button("✅ All", use_container_width=True):
            for feat in NUMERIC_FEATURES:
                st.session_state[f"cb_{feat}"] = True
            for cat in CATEGORICAL_FEATURES:
                st.session_state[f"cb_{cat}"] = True
        if g_col2.button("🗑 None", use_container_width=True):
            for feat in NUMERIC_FEATURES:
                st.session_state[f"cb_{feat}"] = False
            for cat in CATEGORICAL_FEATURES:
                st.session_state[f"cb_{cat}"] = False

        st.divider()

        for group_name, group_feats in FEATURE_GROUPS.items():
            with st.expander(group_name, expanded=any(
                st.session_state.get(f"cb_{f}", False) for f in group_feats
            )):
                c1, c2 = st.columns(2)
                if c1.button("All", key=f"all_{group_name}", use_container_width=True):
                    for f in group_feats:
                        st.session_state[f"cb_{f}"] = True
                if c2.button("None", key=f"none_{group_name}", use_container_width=True):
                    for f in group_feats:
                        st.session_state[f"cb_{f}"] = False
                for feat in group_feats:
                    label     = FEATURE_LABELS.get(feat, feat).split(" — ")[0]
                    help_text = FEATURE_LABELS.get(feat, feat).split(" — ", 1)[-1] \
                                if " — " in FEATURE_LABELS.get(feat, feat) else None
                    st.checkbox(label, key=f"cb_{feat}", help=help_text)

        with st.expander("🗂 Categorical", expanded=any(
            st.session_state.get(f"cb_{c}", False) for c in CATEGORICAL_FEATURES
        )):
            for cat in CATEGORICAL_FEATURES:
                label     = FEATURE_LABELS.get(cat, cat).split(" — ")[0]
                help_text = FEATURE_LABELS.get(cat, cat).split(" — ", 1)[-1] \
                            if " — " in FEATURE_LABELS.get(cat, cat) else None
                st.checkbox(label, key=f"cb_{cat}", help=help_text)

        st.divider()
        run = st.button("▶  Run Regression", use_container_width=True, type="primary")

    # ── Collect checked features ──────────────────────────────────────────
    selected_numeric = [f for f in NUMERIC_FEATURES if st.session_state.get(f"cb_{f}", False)]
    selected_cat     = [c for c in CATEGORICAL_FEATURES if st.session_state.get(f"cb_{c}", False)]

    # ── Main panel ───────────────────────────────────────────────────────
    if not run:
        st.info("Select variables in the sidebar and click **▶ Run Regression** to begin.")
        return

    selected_cols = list(selected_numeric)
    for cat in selected_cat:
        selected_cols.extend(cat_dummies.get(cat, []))

    if not selected_cols:
        st.warning("Please select at least one feature before running.")
        return

    # Subset & add constant
    X_tr   = sm.add_constant(X_train_full[selected_cols], has_constant="add")
    X_te   = sm.add_constant(X_test_full[selected_cols],  has_constant="add")
    X_site = sm.add_constant(X_site_full[selected_cols],  has_constant="add")

    # Fit main model
    result       = sm.OLS(y_train, X_tr).fit()
    y_pred_train = result.predict(X_tr).values
    y_pred_test  = result.predict(X_te).values
    y_pred_site  = result.predict(X_site).values

    # Fit original (baseline) model for comparison
    X_tr_orig   = sm.add_constant(X_train_full[ORIGINAL_MODEL_FEATURES], has_constant="add")
    X_te_orig   = sm.add_constant(X_test_full[ORIGINAL_MODEL_FEATURES],  has_constant="add")
    result_orig  = sm.OLS(y_train, X_tr_orig).fit()
    orig_pred_train = result_orig.predict(X_tr_orig).values
    orig_pred_test  = result_orig.predict(X_te_orig).values

    # ── Regression formula ────────────────────────────────────────────────
    rhs = " + ".join(selected_cols) if selected_cols else "—"
    st.markdown(f"**Running:** `annual.profit ~ {rhs}`")

    # ── Metrics row ──────────────────────────────────────────────────────
    metrics      = compute_metrics(y_train, y_pred_train, y_test, y_pred_test, y_bar)
    orig_metrics = compute_metrics(y_train, orig_pred_train, y_test, orig_pred_test, y_bar)

    st.subheader("Model Metrics")
    m_cols = st.columns(6)
    formats = {
        "R²  (in-sample)":      lambda v: f"{v:.4f}",
        "OSR² (out-of-sample)": lambda v: f"{v:.4f}",
        "RMSE":                 lambda v: f"${v:,.0f}",
        "Max Error":            lambda v: f"${v:,.0f}",
        "MAPE (%)":             lambda v: f"{v:.2f}%",
        "MAE":                  lambda v: f"${v:,.0f}",
    }
    for col, (name, value) in zip(m_cols, metrics.items()):
        col.metric(name, formats[name](value))

    # ── Revenue summary ──────────────────────────────────────────────────
    st.divider()
    rev_cols = st.columns([1, 1, 2])
    with rev_cols[0]:
        st.metric("Σ Predicted test revenue", f"${y_pred_test.sum():,.0f}",
                  help="Sum of OLS predictions over the 9 test stores")
    with rev_cols[1]:
        st.metric("Σ Actual test revenue", f"${y_test.sum():,.0f}",
                  delta=f"${y_pred_test.sum() - y_test.sum():,.0f} difference",
                  delta_color="off")

    # ── Download Report Package ───────────────────────────────────────────
    st.divider()
    with st.spinner("Generating report package…"):
        named_figs = [
            ("01_test_actual_vs_predicted.pdf",
             export_fig_actual_vs_predicted(y_test, y_pred_test)),
            ("02_model_vs_original_scatter.pdf",
             export_fig_model_vs_original(y_pred_site, orig_preds, site_store_numbers)),
            ("03_metrics_comparison.pdf",
             export_fig_metrics_comparison(metrics, orig_metrics)),
            ("04_correlation_matrix.pdf",
             export_fig_correlation(X_train_full, selected_cols, y_train.values)),
            ("05_per_store_comparison.pdf",
             export_pdf_table_per_store(site_store_numbers, y_pred_site, orig_preds)),
            ("06_coefficient_table.pdf",
             export_pdf_table_coefficients(result)),
        ]
        named_xlsx = [
            ("05_per_store_comparison.xlsx",
             export_excel_per_store(site_store_numbers, y_pred_site, orig_preds)),
            ("06_coefficient_table.xlsx",
             export_excel_coefficients(result)),
        ]
        zip_bytes = build_report_zip(named_figs, named_xlsx)

    st.download_button(
        label="📦 Download Report Package (.zip)",
        data=zip_bytes,
        file_name="milagro_report_package.zip",
        mime="application/zip",
        use_container_width=True,
    )

    # ── Under-construction store predictions ──────────────────────────────
    st.divider()
    st.subheader("🚧 Under-Construction Store Predictions")

    site_col1, site_col2, site_col3 = st.columns(3)
    site_col1.metric(
        "Σ Predicted revenue (model)",
        f"${y_pred_site.sum():,.0f}",
        help=f"Sum of OLS model predictions for {len(y_pred_site)} under-construction stores",
    )
    site_col2.metric(
        "Σ Original predictions",
        f"${orig_preds.sum():,.0f}",
        delta=f"${y_pred_site.sum() - orig_preds.sum():,.0f} vs Original",
        delta_color="off",
        help="Original Prediction column from site_const_data.csv",
    )
    site_col3.metric("# stores", len(y_pred_site))

    # ── Difference metrics ────────────────────────────────────────────────
    diffs_all = y_pred_site - orig_preds
    diff_cols = st.columns(6)
    diff_cols[0].metric("Avg difference", f"${diffs_all.mean():,.0f}",
                        help="Mean of (OLS Model − Original) across all stores")
    diff_cols[1].metric("Median difference", f"${np.median(diffs_all):,.0f}")
    diff_cols[2].metric("Model > Original", int((diffs_all > 0).sum()),
                        help="Stores where OLS model predicts higher profit than Original")
    diff_cols[3].metric("Model < Original", int((diffs_all < 0).sum()),
                        help="Stores where OLS model predicts lower profit than Original")
    diff_cols[4].metric("Max overestimate\n(model vs Original)", f"${diffs_all.max():,.0f}",
                        help="Largest single-store gap where model exceeds Original prediction")
    diff_cols[5].metric("Max underestimate\n(model vs Original)", f"${diffs_all.min():,.0f}",
                        help="Largest single-store gap where model falls below Original prediction")

    # ── Model vs Original comparison plots ───────────────────────────────
    fig_site, axes_site = plt.subplots(1, 2, figsize=(14, 5))
    fig_site.patch.set_facecolor("#0f1117")

    diffs    = diffs_all
    abs_diffs = np.abs(diffs)
    sort_idx  = np.argsort(abs_diffs)[::-1]

    ax_s = axes_site[0]
    ax_s.set_facecolor("#1a1d27")
    combined_min = min(y_pred_site.min(), orig_preds.min())
    combined_max = max(y_pred_site.max(), orig_preds.max())
    ax_s.plot([combined_min, combined_max], [combined_min, combined_max],
              color="#ff6b6b", linewidth=1.5, linestyle="--", label="Equal")
    ax_s.scatter(orig_preds, y_pred_site, color="#4fc3f7", alpha=0.85, s=60, zorder=3)
    for i in sort_idx[:5]:
        ax_s.annotate(f"#{site_store_numbers[i]}",
                      (orig_preds[i], y_pred_site[i]),
                      textcoords="offset points", xytext=(6, 4),
                      color="white", fontsize=7)
    ax_s.set_title("Model vs Original (under-construction)", color="white",
                   fontsize=11, fontweight="bold")
    ax_s.set_xlabel("Original Prediction ($)", color="lightgray", fontsize=9)
    ax_s.set_ylabel("Model Prediction ($)", color="lightgray", fontsize=9)
    ax_s.tick_params(colors="lightgray", labelsize=8)
    for spine in ax_s.spines.values():
        spine.set_edgecolor("#3a3d4a")
    ax_s.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))
    ax_s.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))
    ax_s.legend(fontsize=8, labelcolor="lightgray", facecolor="#1a1d27", edgecolor="#3a3d4a")

    ax_b = axes_site[1]
    ax_b.set_facecolor("#1a1d27")
    top10_idx    = sort_idx[:10]
    top10_labels = [f"#{site_store_numbers[i]}" for i in top10_idx]
    top10_diffs  = diffs[top10_idx]
    bar_colors   = ["#80cbc4" if d >= 0 else "#ef9a9a" for d in top10_diffs]
    ax_b.barh(top10_labels[::-1], top10_diffs[::-1], color=bar_colors[::-1])
    ax_b.axvline(0, color="white", linewidth=0.8, linestyle="--", alpha=0.5)
    ax_b.set_title("Top-10 most different stores (Model − Original)",
                   color="white", fontsize=11, fontweight="bold")
    ax_b.set_xlabel("Difference ($)", color="lightgray", fontsize=9)
    ax_b.tick_params(colors="lightgray", labelsize=8)
    for spine in ax_b.spines.values():
        spine.set_edgecolor("#3a3d4a")
    ax_b.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e3:.0f}K"))

    fig_site.tight_layout(pad=2)
    st.pyplot(fig_site, use_container_width=True)
    plt.close(fig_site)

    with st.expander("📄 Per-store predictions (all stores, sorted by |difference|)", expanded=False):
        site_df = pd.DataFrame({
            "Store #":                  site_store_numbers,
            "Model Prediction ($)":     y_pred_site.round(2),
            "Original Prediction ($)":  orig_preds.round(2),
            "Difference ($)":           diffs.round(2),
            "|Difference| ($)":         abs_diffs.round(2),
        }).sort_values("|Difference| ($)", ascending=False)
        st.dataframe(
            site_df.style.format({
                "Model Prediction ($)":    "${:,.0f}",
                "Original Prediction ($)": "${:,.0f}",
                "Difference ($)":          "${:,.0f}",
                "|Difference| ($)":        "${:,.0f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

    # ── Diagnostic Plots ─────────────────────────────────────────────────
    st.divider()
    st.subheader("Diagnostic Plots")
    fig = make_plots(y_train, y_pred_train, y_test, y_pred_test, result)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # ── Correlation matrix ────────────────────────────────────────────────
    st.divider()
    with st.expander("🔗 Feature correlation matrix", expanded=False):
        corr_data = X_train_full[selected_cols].copy()
        corr_data[TARGET] = y_train.values
        corr = corr_data.corr()

        n = len(corr)
        fig_corr, ax_corr = plt.subplots(figsize=(max(6, n * 0.55), max(5, n * 0.5)))
        fig_corr.patch.set_facecolor("#0f1117")
        ax_corr.set_facecolor("#1a1d27")

        im = ax_corr.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        cbar = fig_corr.colorbar(im, ax=ax_corr, fraction=0.03, pad=0.04)
        cbar.ax.tick_params(colors="lightgray", labelsize=8)
        cbar.outline.set_edgecolor("#3a3d4a")

        ax_corr.set_xticks(range(n))
        ax_corr.set_yticks(range(n))
        ax_corr.set_xticklabels(corr.columns, rotation=45, ha="right",
                                color="lightgray", fontsize=8)
        ax_corr.set_yticklabels(corr.columns, color="lightgray", fontsize=8)
        for spine in ax_corr.spines.values():
            spine.set_edgecolor("#3a3d4a")

        if n <= 15:
            for i in range(n):
                for j in range(n):
                    ax_corr.text(j, i, f"{corr.values[i, j]:.2f}",
                                 ha="center", va="center", fontsize=7,
                                 color="white" if abs(corr.values[i, j]) > 0.5 else "black")

        ax_corr.set_title("Pearson correlation — selected features + target",
                          color="white", fontsize=11, fontweight="bold", pad=10)
        fig_corr.tight_layout()
        st.pyplot(fig_corr, use_container_width=True)
        plt.close(fig_corr)

    # ── Model summary ─────────────────────────────────────────────────────
    st.divider()
    with st.expander("📋 Full model summary (statsmodels)", expanded=False):
        summary_df = pd.DataFrame({
            "Coefficient":    result.params,
            "Std Error":      result.bse,
            "t-stat":         result.tvalues,
            "p-value":        result.pvalues,
            "CI lower 2.5%":  result.conf_int()[0],
            "CI upper 97.5%": result.conf_int()[1],
        }).round(4)
        st.dataframe(summary_df, use_container_width=True)

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("R² (in-sample)",      f"{result.rsquared:.4f}")
        col_b.metric("Adj. R²",             f"{result.rsquared_adj:.4f}")
        col_c.metric("F-statistic p-value", f"{result.f_pvalue:.4e}")

        st.caption(
            f"N train = {int(result.nobs)}  ·  "
            f"k = {len(result.params) - 1} predictors  ·  "
            f"AIC = {result.aic:.1f}  ·  BIC = {result.bic:.1f}"
        )


if __name__ == "__main__":
    main()
