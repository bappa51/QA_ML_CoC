import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# 0) CONFIGURATION SECTION
# -----------------------------
# This section defines all configurable parameters for the model.
# These can be adjusted without changing the core logic.

SHEET_NAME = "Sheet1"   # Only used when input is Excel
TARGET_AXIS = "X"       # Target variable to predict (e.g., defects or cost)

INCLUDE_AXES = None     # Include specific axes (None = use all except target)
EXCLUDE_AXES = set()    # Exclude unwanted axes (e.g., leakage metrics)

STANDARDIZE_FEATURES = False  # Enable if scaling required
future_scenario = None        # Optional override scenario

SHOW_CONTRIBUTIONS = True     # Enable radar visualization
TOP_N_RADAR = 12              # Top factors for radar chart
TOP_N_COEFS = None            # Limit coefficient chart if needed


# -----------------------------
# 1) AUTO FILE DETECTION
# -----------------------------
# Automatically detects CSV or Excel file in same directory as script.
def find_input_file():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    files = os.listdir(script_dir)

    csv_files = sorted([f for f in files if f.lower().endswith(".csv")])
    if csv_files:
        print(f"[INFO] Using CSV file: {csv_files[0]}")
        return os.path.join(script_dir, csv_files[0])

    excel_files = sorted([f for f in files if f.lower().endswith((".xlsx", ".xls"))])
    if excel_files:
        print(f"[INFO] Using Excel file: {excel_files[0]}")
        return os.path.join(script_dir, excel_files[0])

    raise FileNotFoundError("No CSV or Excel file found.")


# -----------------------------
# 2) LOAD DATA
# -----------------------------
# Loads data dynamically based on file type
def load_data(path):
    ext = os.path.splitext(path)[1].lower()
    print(f"[INFO] Loading file: {path}")

    if ext == ".csv":
        return pd.read_csv(path)
    elif ext in [".xlsx", ".xls"]:
        try:
            return pd.read_excel(path, sheet_name=SHEET_NAME, engine="openpyxl")
        except:
            print("[WARN] Sheet not found, loading first sheet")
            return pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


# -----------------------------
# HELPER FUNCTION
# -----------------------------
def release_num(col):
    """Extract numeric release number from column name"""
    m = re.search(r"(?i)release\s*(\d+)", str(col))
    return int(m.group(1)) if m else 10**9


# -----------------------------
# MAIN EXECUTION PIPELINE
# -----------------------------
def main():

    # -----------------------------
    # DATA INGESTION
    # -----------------------------
    INPUT_PATH = find_input_file()
    df = load_data(INPUT_PATH)

    df.columns = [str(c).strip() for c in df.columns]

    if "Axis" not in df.columns:
        raise ValueError("Column 'Axis' missing")

    # Identify release columns dynamically
    release_cols = [
        c for c in df.columns
        if re.match(r"(?i)^release\s*\d+", str(c))
    ]

    release_cols = sorted(release_cols, key=release_num)

    df["Axis"] = df["Axis"].str.strip().str.upper()

    # Create structured matrix (Axes vs Releases)
    axis_matrix = (
        df.set_index("Axis")[release_cols]
        .apply(pd.to_numeric, errors="coerce")
        .dropna(how="all")
    )

    target = TARGET_AXIS.upper()

    if target not in axis_matrix.index:
        raise ValueError("Target axis not found")

    # -----------------------------
    # FEATURE ENGINEERING
    # -----------------------------
    # Select predictors dynamically
    all_axes = list(axis_matrix.index)
    feature_axes = [a for a in all_axes if a != target]

    train_axes = [target] + feature_axes
    train_table = axis_matrix.loc[train_axes]

    # Keep only valid releases
    valid_mask = train_table.notna().all(axis=0)
    valid_releases = [c for c in release_cols if valid_mask[c]]

    train_table = train_table[valid_releases]

    y = train_table.loc[target].values
    X = train_table.loc[feature_axes].values.T

    print(f"[DATA] Samples: {len(y)}, Features: {len(feature_axes)}")

    # -----------------------------
    # MODEL TRAINING (LINEAR REGRESSION)
    # -----------------------------
    # Fits regression using least squares approach
    X_design = np.column_stack([np.ones(len(y)), X])
    beta = np.linalg.lstsq(X_design, y, rcond=None)[0]

    intercept = beta[0]
    coefs = beta[1:]

    y_pred = X_design @ beta

    # Model goodness-of-fit
    r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)

    print("\n[MODEL RESULTS]")
    print(f"Intercept: {intercept:.4f}")
    print(f"R² Score: {r2:.4f}")

    # -----------------------------
    # BASELINE SCENARIO (LAST RELEASE)
    # -----------------------------
    # Uses last available release as baseline
    last_rel = valid_releases[-1]
    scenario = train_table.loc[feature_axes, last_rel].to_dict()

    x0 = np.array([scenario[a] for a in feature_axes])
    y_hat = intercept + np.dot(x0, coefs)

    print(f"[PREDICTION] {target} = {y_hat:.2f}")

    # -----------------------------
    # PLOT 1: ACTUAL vs PREDICTED
    # -----------------------------
    # Shows model accuracy visually
    COLOR_SCATTER = "#1f77b4"   # Blue
    COLOR_LINE = "#d62728"      # Red

    plt.figure()
    plt.scatter(y, y_pred, color=COLOR_SCATTER)
    plt.plot([min(y), max(y)], [min(y), max(y)], linestyle="--", color=COLOR_LINE)

    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"Actual vs Predicted (R²={r2:.3f})")
    plt.grid()
    plt.show()

    # -----------------------------
    # PLOT 2: FEATURE COEFFICIENTS
    # -----------------------------
    # Shows influence of each feature on prediction
    COLOR_BAR = "#ff7f0e"   # Orange

    plt.figure(figsize=(10, 5))
    plt.bar(feature_axes, coefs, color=COLOR_BAR)
    plt.axhline(0, color='black')
    plt.xticks(rotation=60)
    plt.title("Feature Influence (Coefficients)")
    plt.grid(axis='y')
    plt.show()

    # -----------------------------
    # PLOT 3: RADAR CHART (TOP CONTRIBUTORS)
    # -----------------------------
    # Highlights most impactful factors in baseline scenario
    if SHOW_CONTRIBUTIONS:

        contribs = x0 * coefs
        abs_contrib = np.abs(contribs)
        order = np.argsort(abs_contrib)[::-1]

        top_n = min(TOP_N_RADAR, len(feature_axes))
        idx = order[:top_n]

        radar_axes = [feature_axes[i] for i in idx]
        radar_vals = contribs[idx]

        # Normalize values for radar visualization
        vmin, vmax = radar_vals.min(), radar_vals.max()
        radar_norm = (radar_vals - vmin) / (vmax - vmin) if vmax != vmin else np.ones_like(radar_vals)

        labels = radar_axes + [radar_axes[0]]
        values = np.append(radar_norm, radar_norm[0])

        angles = np.linspace(0, 2*np.pi, len(labels))

        # Radar colors
        COLOR_RADAR_LINE = "#2ca02c"   # Green
        COLOR_RADAR_FILL = "#98df8a"

        plt.figure(figsize=(7, 7))
        ax = plt.subplot(111, polar=True)

        ax.plot(angles, values, color=COLOR_RADAR_LINE, linewidth=2)
        ax.fill(angles, values, color=COLOR_RADAR_FILL, alpha=0.4)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(radar_axes)
        ax.set_yticklabels([])

        plt.title(
            f"Top {top_n} Contributors (Baseline = {last_rel})\nPredicted {TARGET_AXIS} = {y_hat:.2f}",
            pad=20
        )

        plt.tight_layout()
        plt.show()


# -----------------------------
# SCRIPT ENTRY POINT
# -----------------------------
if __name__ == "__main__":
    main()
