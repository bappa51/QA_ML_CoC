import os
import re
import importlib
import numpy as np  # type: ignore
import pandas as pd # type: ignore

plt = None
try:
    plt = importlib.import_module("matplotlib.pyplot")
except Exception:  # pragma: no cover - optional plotting
    plt = None

# -----------------------------
# CONFIG
# -----------------------------
SHEET_NAME = "Sheet1"
TARGET_AXIS = "X"

SHOW_CONTRIBUTIONS = True
TOP_N_RADAR = 12


# -----------------------------
# FILE DETECTION (CLOUD SAFE)
# -----------------------------
def find_input_file():
    script_dir = os.getcwd()
    print(f"[INFO] Working Directory: {script_dir}")

    files = os.listdir(script_dir)

    csv_files = sorted([f for f in files if f.lower().endswith(".csv")])
    if csv_files:
        print(f"[INFO] Using CSV file: {csv_files[0]}")
        return os.path.join(script_dir, csv_files[0])

    raise FileNotFoundError("No CSV found in directory")


def load_data(path):
    print(f"[INFO] Loading file: {path}")
    return pd.read_csv(path)


def release_num(col):
    m = re.search(r"(?i)release\s*(\d+)", str(col))
    return int(m.group(1)) if m else 10**9


# -----------------------------
# MAIN
# -----------------------------
def main():

    path = find_input_file()
    df = load_data(path)

    df.columns = [str(c).strip() for c in df.columns]

    if "Axis" not in df.columns:
        raise ValueError("Column 'Axis' missing")

    release_cols = [c for c in df.columns if re.match(r"(?i)^release\s*\d+", str(c))]
    release_cols = sorted(release_cols, key=release_num)

    df["Axis"] = df["Axis"].str.strip().str.upper()

    axis_matrix = df.set_index("Axis")[release_cols].apply(pd.to_numeric, errors="coerce")

    target = TARGET_AXIS.upper()

    feature_axes = [a for a in axis_matrix.index if a != target]

    train_axes = [target] + feature_axes
    train_table = axis_matrix.loc[train_axes]

    valid_mask = train_table.notna().all(axis=0)
    valid_releases = [c for c in release_cols if valid_mask[c]]

    train_table = train_table[valid_releases]

    y = train_table.loc[target].values
    X = train_table.loc[feature_axes].values.T

    # -----------------------------
    # REGRESSION
    # -----------------------------
    X_design = np.column_stack([np.ones(len(y)), X])
    beta = np.linalg.lstsq(X_design, y, rcond=None)[0]

    intercept = beta[0]
    coefs = beta[1:]

    y_pred = X_design @ beta

    r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)

    print(f"\n[MODEL] R² = {r2:.4f}")

    # -----------------------------
    # SCENARIO (LAST RELEASE)
    # -----------------------------
    last_rel = valid_releases[-1]
    scenario = train_table.loc[feature_axes, last_rel].to_dict()

    x0 = np.array([scenario[a] for a in feature_axes])
    y_hat = intercept + np.dot(x0, coefs)

    print(f"[PREDICTION] {target} = {y_hat:.2f}")

    # -----------------------------
    # CONTRIBUTION WITH %
    # -----------------------------
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

    # -----------------------------
    # PLOT 1
    # -----------------------------
    if plt is not None:
        plt.scatter(y, y_pred, color="blue")
        plt.plot([min(y), max(y)], [min(y), max(y)], '--', color="red")
        plt.title(f"Actual vs Predicted (R²={r2:.3f})")
        plt.grid()
        plt.show()

        # -----------------------------
        # PLOT 2
        # -----------------------------
        plt.figure(figsize=(10, 5))
        plt.bar(feature_axes, coefs, color="orange")
        plt.xticks(rotation=60)
        plt.title("Feature Coefficients")
        plt.grid()
        plt.show()

    # -----------------------------
    # PLOT 3 (RADAR)
    # -----------------------------
    if SHOW_CONTRIBUTIONS:

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

        ax.plot(angles, values, color="green")
        ax.fill(angles, values, color="lightgreen", alpha=0.4)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(radar_axes)

        plt.title("Top Contributors Radar")
        plt.show()


# -----------------------------
# ENTRY
# -----------------------------
if __name__ == "__main__":
    main()
