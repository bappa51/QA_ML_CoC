# ============================================================
# START: The process begins
# ============================================================

import os
import re
import importlib
import numpy as np  # type: ignore
import pandas as pd # type: ignore

# Optional plotting support (only if matplotlib is available)
plt = None
try:
    import matplotlib
    if os.name != "nt" and not os.environ.get("DISPLAY"):
        matplotlib.use("Agg", force=True)
    else:
        for backend in ("TkAgg", "Qt5Agg", "QtAgg", "WXAgg", "Agg"):
            try:
                matplotlib.use(backend, force=True)
                break
            except Exception:
                continue
    plt = importlib.import_module("matplotlib.pyplot")
except Exception:
    plt = None

# Optional random forest support (only if scikit-learn is available)
RandomForestRegressor = None
try:
    from sklearn.ensemble import RandomForestRegressor
except Exception:
    RandomForestRegressor = None


# ============================================================
# CONFIGURATION SECTION
# ============================================================
# Defines key parameters used across the program

SHEET_NAME = "Sheet1"
TARGET_AXIS = "X"

SHOW_CONTRIBUTIONS = True
TOP_N_RADAR = 12


# ============================================================
# FIND DATA FILE:
# The system looks for the expected CoQ sample data file in the current folder.
# It will prefer COQ_FullSampleData.csv or COQ_FullSampleData.xlsx,
# and fall back to any supported CSV/XLSX file if needed.
# ============================================================
def find_input_file():
    script_dir = os.getcwd()
    print(f"[INFO] Working Directory: {script_dir}")

    files = os.listdir(script_dir)
    lower_files = {f.lower(): f for f in files}

    preferred_names = [
        "coq_fullsampledata.csv",
        "coq_fullsampledata.xlsx",
        "coq_fullsampledata.xls",
    ]

    for name in preferred_names:
        if name in lower_files:
            selected = lower_files[name]
            print(f"[INFO] Using data file: {selected}")
            return os.path.join(script_dir, selected)

    supported = sorted(
        [
            f for f in files
            if f.lower().endswith((".csv", ".xlsx", ".xls"))
        ]
    )
    if supported:
        selected = supported[0]
        print(f"[INFO] Using data file: {selected}")
        return os.path.join(script_dir, selected)

    raise FileNotFoundError("No CSV or Excel data file found in directory")


# ============================================================
# LOAD DATA:
# The file is opened and data is read into the system.
# ============================================================
def load_data(path):
    print(f"[INFO] Loading file: {path}")
    ext = os.path.splitext(path)[1].lower()

    if ext == ".csv":
        return pd.read_csv(path)
    if ext in {".xlsx", ".xls"}:
        engine = "openpyxl" if ext == ".xlsx" else None
        return pd.read_excel(path, engine=engine)

    raise ValueError(f"Unsupported data file type: {ext}")


# Helper to correctly sort release columns
def release_num(col):
    m = re.search(r"(?i)release\s*(\d+)", str(col))
    return int(m.group(1)) if m else 10**9


# Helper to save plots
def save_plot(filename):
    """Save current plot if matplotlib is available and show it."""
    if plt is not None:
        try:
            plt.savefig(filename, dpi=100, bbox_inches='tight')
            print(f"[INFO] Plot saved: {filename}")
            if os.name == "nt":
                try:
                    os.startfile(filename)
                    print(f"[INFO] Opened plot file: {filename}")
                except Exception:
                    pass
        except Exception as e:
            print(f"[WARNING] Could not save plot: {e}")
        try:
            plt.show(block=True)
        except Exception as e:
            print(f"[WARNING] Could not display plot: {e}")


# ============================================================
# MAIN PROCESS
# ============================================================
def main():

    # -----------------------------
    # START EXECUTION FLOW
    # -----------------------------

    path = find_input_file()        # Find Data File
    df = load_data(path)            # Load Data

    # ============================================================
    # BASIC CHECK:
    # Check if required column "Axis" exists
    # If not → stop execution
    # ============================================================
    df.columns = [str(c).strip() for c in df.columns]

    if "Axis" not in df.columns:
        raise ValueError("Column 'Axis' missing")

    # ============================================================
    # CLEAN & ORGANIZE DATA:
    # - Fix column names
    # - Identify Release columns
    # - Standardize text values
    # ============================================================
    release_cols = [c for c in df.columns if re.match(r"(?i)^release\s*\d+", str(c))]
    release_cols = sorted(release_cols, key=release_num)

    df["Axis"] = df["Axis"].str.strip().str.upper()

    axis_matrix = df.set_index("Axis")[release_cols].apply(pd.to_numeric, errors="coerce")

    # ============================================================
    # PREPARE DATA FOR ANALYSIS:
    # - Separate target variable
    # - Separate feature variables
    # - Remove incomplete data
    # ============================================================
    target = TARGET_AXIS.upper()

    feature_axes = [a for a in axis_matrix.index if a != target]

    train_axes = [target] + feature_axes
    train_table = axis_matrix.loc[train_axes]

    valid_mask = train_table.notna().all(axis=0)
    valid_releases = [c for c in release_cols if valid_mask[c]]

    train_table = train_table[valid_releases]

    y = train_table.loc[target].values
    X = train_table.loc[feature_axes].values.T

    # ============================================================
    # BUILD PREDICTION MODEL:
    # Learn relationship between inputs and target using regression
    # ============================================================
    X_design = np.column_stack([np.ones(len(y)), X])
    beta = np.linalg.lstsq(X_design, y, rcond=None)[0]

    intercept = beta[0]
    coefs = beta[1:]

    y_pred = X_design @ beta

    # ============================================================
    # CHECK MODEL ACCURACY:
    # Evaluate performance using R² score
    # ============================================================
    r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
    print(f"\n[MODEL] R² = {r2:.4f}")

    # ============================================================
    # MAKE PREDICTION:
    # Using latest available data
    # ============================================================
    last_rel = valid_releases[-1]
    scenario = train_table.loc[feature_axes, last_rel].to_dict()

    x0 = np.array([scenario[a] for a in feature_axes])
    y_hat = intercept + np.dot(x0, coefs)

    print(f"[PREDICTION] {target} = {y_hat:.2f}")

    # ============================================================
    # EXPLAIN CONTRIBUTIONS:
    # Show how each factor contributes to final prediction
    # Includes percentage contribution and correlation
    # ============================================================
    print("\n[CONTRIBUTION BREAKDOWN WITH % AND CORRELATION]")

    # Calculate correlations between features and target
    correlations = np.corrcoef(X.T, y)[:-1, -1]

    print(f"{'Axis':<10} {'Value':>10} {'Coef':>10} {'Corr':>10} {'Contr':>12} {'%':>10}")
    print("-" * 74)

    total_contribution = 0
    records = []

    for axis, val, coef, corr in zip(feature_axes, x0, coefs, correlations):
        contrib = val * coef
        records.append((axis, val, coef, corr, contrib))
        total_contribution += contrib

    total = intercept + total_contribution

    for axis, val, coef, corr, contrib in records:
        pct = (contrib / total) * 100 if total != 0 else 0
        print(f"{axis:<10} {val:>10.2f} {coef:>10.4f} {corr:>10.4f} {contrib:>12.4f} {pct:>9.2f}%")

    print("-" * 74)
    print(f"{'Intercept':<10} {'':>10} {intercept:>10.4f} {'':>10} {intercept:>12.4f} {(intercept/total)*100:>9.2f}%")
    print("-" * 74)
    print(f"{'TOTAL':<10} {'':>10} {'':>10} {'':>10} {total:>12.4f} {'100.00%':>10}")

    # ============================================================
    # CREATE VISUALS:
    # If charting is available, generate insights
    # ============================================================
    if plt is not None:

        soothing_blue = "#0657F9"
        # -----------------------------
        # VISUAL 1: Actual vs Predicted
        # -----------------------------
        plt.scatter(y, y_pred, color="red", label="Actual data")
        plt.plot([min(y), max(y)], [min(y), max(y)], '--', color="blue", label="Ideal fit")
        plt.title(f"Actual vs Predicted (R²={r2:.3f})", color="blue")
        plt.xlabel("Actual Values", color="blue")
        plt.ylabel("Predicted Values", color="blue")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(frameon=False)

        ax = plt.gca()
        ax.spines["bottom"].set_color("blue")
        ax.spines["left"].set_color("blue")
        ax.tick_params(axis="x", colors="blue")
        ax.tick_params(axis="y", colors="blue")

        save_plot("actual_vs_predicted.png")


        # -----------------------------
        # VISUAL 2: Contribution Ranking
        # -----------------------------
        contribs = x0 * coefs
        sorted_idx = np.argsort(contribs)[::-1]

        sorted_axes = [feature_axes[i] for i in sorted_idx]
        sorted_contribs = contribs[sorted_idx]

        x_pos = np.arange(len(sorted_axes))

       # soothing_blue = "#412B24"

        plt.figure(figsize=(12, 6))
        plt.bar(x_pos, sorted_contribs, color=soothing_blue)

        # Axis names clearly shown
        plt.xticks(x_pos, sorted_axes, rotation=60)

        plt.xlabel("Feature Axes")       # Explicit X-axis title
        plt.ylabel("Contribution")
        plt.title("Feature Contribution Ranking (Highest to Lowest)")

        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.show()


        # -----------------------------
        # VISUAL 3: Radar Chart (Top Contributors)
        # -----------------------------
        contribs = x0 * coefs
        order = np.argsort(np.abs(contribs))[::-1]

        idx = order[:min(TOP_N_RADAR, len(feature_axes))]

        radar_axes = [feature_axes[i] for i in idx]
        radar_vals = contribs[idx]

        vmin, vmax = radar_vals.min(), radar_vals.max()
        radar_norm = (radar_vals - vmin) / (vmax - vmin) if vmax != vmin else np.ones_like(radar_vals)

        labels = radar_axes + [radar_axes[0]]
        values = np.append(radar_norm, radar_norm[0])
        angles = np.linspace(0, 2*np.pi, len(labels))

        plt.figure(figsize=(7,7))
        ax = plt.subplot(111, polar=True)

        # Use consistent color
        ax.plot(angles, values, color=soothing_blue)
        ax.fill(angles, values, color=soothing_blue, alpha=0.3)

        # Axis names clearly visible
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(radar_axes, fontsize=10)

        plt.title("Top Contributors Radar")
        plt.show()


        # -----------------------------
        # VISUAL 4: Correlation Bar Chart
        # -----------------------------
        sorted_corr_idx = np.argsort(correlations)[::-1]
        sorted_corr_axes = [feature_axes[i] for i in sorted_corr_idx]
        sorted_corr_values = correlations[sorted_corr_idx]

        x_pos = np.arange(len(sorted_corr_axes))

        plt.figure(figsize=(12, 6))
        plt.bar(x_pos, sorted_corr_values, color=soothing_blue)

        # Axis names clearly shown
        plt.xticks(x_pos, sorted_corr_axes, rotation=60)

        plt.xlabel("Feature Axes")
        plt.ylabel("Correlation with Target")
        plt.title("Feature Correlation with Target (Highest to Lowest)")
        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        plt.grid(True, linestyle="--", alpha=0.6, axis='y')
        plt.tight_layout()
        plt.show()


        # -----------------------------
        # VISUAL 5: Learned Coefficients per Feature Axis
        # -----------------------------
        sorted_coef_idx = np.argsort(np.abs(coefs))[::-1]
        sorted_coef_axes = [feature_axes[i] for i in sorted_coef_idx]
        sorted_coef_values = coefs[sorted_coef_idx]

        x_pos = np.arange(len(sorted_coef_axes))

        plt.figure(figsize=(12, 6))
        colors = [soothing_blue if val >= 0 else '#FF6B6B' for val in sorted_coef_values]
        plt.bar(x_pos, sorted_coef_values, color=colors)

        # Axis names clearly shown
        plt.xticks(x_pos, sorted_coef_axes, rotation=60)

        plt.xlabel("Feature Axes")
        plt.ylabel("Coefficient Value")
        plt.title("Learned Coefficients per Feature Axis (Largest Magnitude First)")
        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        plt.grid(True, linestyle="--", alpha=0.6, axis='y')
        plt.tight_layout()
        plt.show()

    # ============================================================
    # RANDOM FOREST ANALYSIS:
    # Run after the main linear model and contribution analysis.
    # ============================================================
    if RandomForestRegressor is not None:
        print("\n[RF ANALYSIS] Training Random Forest Regressor...")
        rf_model = RandomForestRegressor(
            n_estimators=500,
            random_state=42,
            max_features="sqrt",
            min_samples_leaf=1
        )
        rf_model.fit(X, y)

        rf_pred = rf_model.predict(X)
        rf_r2 = 1 - np.sum((y - rf_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
        print(f"[RF MODEL] R² = {rf_r2:.4f}")

        rf_input = x0.reshape(1, -1)
        rf_y_hat = rf_model.predict(rf_input)[0]
        print(f"[RF PREDICTION] {target} = {rf_y_hat:.2f}")

        rf_importance = pd.Series(rf_model.feature_importances_, index=feature_axes)
        rf_importance = rf_importance.sort_values(ascending=False)
        print("\n[RF FEATURE IMPORTANCE]")
        print(rf_importance.to_string())

        if plt is not None:
            plt.figure(figsize=(12, 6))
            plt.scatter(y, rf_pred, color="forestgreen", label="Actual vs RF Predicted")
            plt.plot([min(y), max(y)], [min(y), max(y)], '--', color="darkgreen", label="Ideal fit")
            plt.title(f"Random Forest: Actual vs Predicted (R²={rf_r2:.3f})", color="darkgreen")
            plt.xlabel("Actual Values", color="darkgreen")
            plt.ylabel("Predicted Values", color="darkgreen")
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.legend(frameon=False)
            plt.tight_layout()
            save_plot("rf_actual_vs_predicted.png")
            plt.close()

            plt.figure(figsize=(12, 6))
            importance_plot = rf_importance.sort_values(ascending=True)
            plt.barh(importance_plot.index, importance_plot.values, color="#2E8B57")
            plt.xlabel("Importance")
            plt.ylabel("Feature Axis")
            plt.title("Random Forest Feature Importance")
            plt.grid(True, linestyle="--", alpha=0.6, axis='x')
            plt.tight_layout()
            save_plot("rf_feature_importance.png")
            plt.close()
    else:
        print("\n[RF ANALYSIS] sklearn is not installed; skipping Random Forest analysis.")

    # ============================================================
    # FINAL SECTION: Correlation matrix of principal CoQ drivers
    # Matches the reference figure layout and highlights the main driver relationships.
    # ============================================================
    if plt is not None and len(feature_axes) > 0:
        corr_labels = [target] + feature_axes
        corr_matrix = train_table.loc[corr_labels].T.corr()

        print("\n[CORRELATION MATRIX] Principal CoQ drivers")
        print(corr_matrix.round(3).to_string())

        fig, ax = plt.subplots(figsize=(10, 8))
        heatmap = ax.imshow(corr_matrix.values, cmap="coolwarm", vmin=-1, vmax=1)

        ax.set_title("Figure 7. Correlation matrix of principal CoQ drivers", fontsize=20, pad=14)
        ax.set_xticks(np.arange(len(corr_labels)))
        ax.set_yticks(np.arange(len(corr_labels)))
        ax.set_xticklabels(corr_labels, rotation=45, ha="right")
        ax.set_yticklabels(corr_labels)

        for i in range(len(corr_labels)):
            for j in range(len(corr_labels)):
                val = corr_matrix.iloc[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="black", fontsize=9)

        cbar = fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Pearson correlation", rotation=270, labelpad=20)

        plt.tight_layout()
        save_plot("correlation_matrix_principal_coq_drivers.png")
        plt.close(fig)

# ============================================================
# END: Process completes
# ============================================================

if __name__ == "__main__":
    main()