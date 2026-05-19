# ============================================================
# START: The process begins
# ============================================================

import os
import re
import sys
import importlib
import numpy as np  # type: ignore
import pandas as pd # type: ignore

# Optional plotting support (only if matplotlib is available)
plt = None
try:
    matplotlib = importlib.import_module("matplotlib")
    if os.environ.get("DISPLAY", "") == "":
        matplotlib.use("Agg")
    plt = importlib.import_module("matplotlib.pyplot")
except Exception:
    plt = None


def save_plot(filename):
    if plt is None:
        return
    try:
        plt.savefig(filename, bbox_inches="tight")
        plt.close()
        print(f"[INFO] Saved plot: {filename}")
    except Exception as exc:
        print(f"[WARN] Could not save plot {filename}: {exc}")

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
# The system looks for a CSV file in the current folder.
# If no file is found → it stops with an error.
# ============================================================
def find_input_file(path=None):
    if path:
        if os.path.isfile(path):
            print(f"[INFO] Using specified file: {path}")
            return path
        raise FileNotFoundError(f"Specified file does not exist: {path}")

    script_dir = os.getcwd()
    print(f"[INFO] Working Directory: {script_dir}")

    files = os.listdir(script_dir)
    csv_files = sorted([f for f in files if f.lower().endswith(".csv")])
    if csv_files:
        print(f"[INFO] Using CSV file: {csv_files[0]}")
        return os.path.join(script_dir, csv_files[0])

    raise FileNotFoundError("No CSV found in directory")


# ============================================================
# LOAD DATA:
# The file is opened and data is read into the system.
# ============================================================
def load_data(path):
    print(f"[INFO] Loading file: {path}")
    return pd.read_csv(path)


# Helper to correctly sort release columns
def release_num(col):
    m = re.search(r"(?i)release\s*(\d+)", str(col))
    return int(m.group(1)) if m else 10**9


# ============================================================
# MAIN PROCESS
# ============================================================
def main():

    # -----------------------------
    # START EXECUTION FLOW
    # -----------------------------

    path = find_input_file(sys.argv[1] if len(sys.argv) > 1 else None)  # Find Data File or accept path
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
    if not release_cols:
        raise ValueError("No release columns found. Expected columns like 'Release1', 'Release2', etc.")

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
    if not feature_axes:
        raise ValueError(f"No feature axes found besides target '{target}'.")

    train_axes = [target] + feature_axes
    train_table = axis_matrix.loc[train_axes]

    valid_mask = train_table.notna().all(axis=0)
    valid_releases = [c for c in release_cols if valid_mask[c]]
    if not valid_releases:
        raise ValueError("No complete releases available after filtering for valid numeric values.")

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
    # Includes percentage contribution
    # ============================================================
    print("\n[CONTRIBUTION BREAKDOWN WITH %]")

    print(f"{'Axis':<10} {'Value':>10} {'Coef':>10} {'Contr':>12} {'%':>10}")
    print("-" * 60)

    total_contribution = 0
    records = []

    for axis, val, coef in zip(feature_axes, x0, coefs):
        contrib = val * coef
        records.append((axis, val, coef, contrib))
        total_contribution += contrib

    total = intercept + total_contribution

    for axis, val, coef, contrib in records:
        pct = (contrib / total) * 100 if total != 0 else 0
        print(f"{axis:<10} {val:>10.2f} {coef:>10.4f} {contrib:>12.4f} {pct:>9.2f}%")

    print("-" * 60)
    print(f"{'Intercept':<10} {'':>10} {intercept:>10.4f} {intercept:>12.4f} {(intercept/total)*100:>9.2f}%")
    print("-" * 60)
    print(f"{'TOTAL':<10} {'':>10} {'':>10} {total:>12.4f} {'100.00%':>10}")

    # ============================================================
    # CREATE VISUALS:
    # If charting is available, generate insights
    # ============================================================
    if plt is not None:

        soothing_blue = "#963A05"
        # -----------------------------
        # VISUAL 1: Actual vs Predicted
        # -----------------------------
      
        plt.scatter(y, y_pred, color=soothing_blue)
        plt.plot([min(y), max(y)], [min(y), max(y)], '--', color=soothing_blue)
        plt.title(f"Actual vs Predicted (R²={r2:.3f})")
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.grid(True, linestyle="--", alpha=0.6)
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
        save_plot("contribution_ranking.png")


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
        save_plot("top_contributors_radar.png")

# ============================================================
# END: Process completes
# ============================================================

if __name__ == "__main__":
    main()
