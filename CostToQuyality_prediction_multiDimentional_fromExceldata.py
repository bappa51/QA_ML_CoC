import numpy as np
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt
import re

# -----------------------------
# 0) CONFIG
# -----------------------------
EXCEL_PATH = "Book1_CoC_FullSampleData.xlsx"   # <-- update path/name as needed
SHEET_NAME = "Sheet1"

TARGET_AXIS = "X"          # e.g., "X" = defects (target). Can be "R" to predict cost, etc.
INCLUDE_AXES = None        # None => use ALL axes except TARGET_AXIS
EXCLUDE_AXES = set()       # e.g., {"AF"} if you want to drop leakage from predictors

STANDARDIZE_FEATURES = False

# Optional: scenario override (values keyed by Axis code).
# If None, uses the LAST usable release values for each predictor axis.
future_scenario = None
# Example:
# future_scenario = {"Y": 420, "Z": 235, "P": 6, "Q": 3600, "R": 38000}

SHOW_CONTRIBUTIONS = True

# For readability when many axes exist
TOP_N_RADAR = 12           # radar shows top contributors only
TOP_N_COEFS = None         # None => all coefs; or set to 20 to show top 20 by abs(coef)

# -----------------------------
# 1) READ INPUT DATA (DYNAMIC AXES) FROM EXCEL
# -----------------------------
# Expected Excel structure: Parameter | Axis | Release 1 | release 2 | Release 3 | ... [1](https://my.shell.com/personal/koushik_dutta_shell_com/_layouts/15/Doc.aspx?sourcedoc=%7B0ECB2284-F83F-442E-A3FE-59C8F51151CA%7D&file=Book1_CoC_Expanded_v2.xlsx&action=default&mobileredirect=true)
df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME, engine="openpyxl")

# Detect release columns like "Release 1", "release 2", ... and sort by release number [1](https://my.shell.com/personal/koushik_dutta_shell_com/_layouts/15/Doc.aspx?sourcedoc=%7B0ECB2284-F83F-442E-A3FE-59C8F51151CA%7D&file=Book1_CoC_Expanded_v2.xlsx&action=default&mobileredirect=true)
release_cols = [
    c for c in df.columns
    if isinstance(c, str) and re.match(r"(?i)^release\s*\d+", c.strip())
]
if not release_cols:
    raise ValueError("No release columns found. Expected columns like 'Release 1', 'release 2', etc.")

def release_num(col):
    m = re.search(r"(?i)release\s*(\d+)", str(col))
    return int(m.group(1)) if m else 10**9

release_cols = sorted(release_cols, key=release_num)

# Normalize axis codes (upper-case) [1](https://my.shell.com/personal/koushik_dutta_shell_com/_layouts/15/Doc.aspx?sourcedoc=%7B0ECB2284-F83F-442E-A3FE-59C8F51151CA%7D&file=Book1_CoC_Expanded_v2.xlsx&action=default&mobileredirect=true)
df["Axis"] = df["Axis"].astype(str).str.strip().str.upper()

# Build axis_matrix: index=Axis, columns=Release columns; values numeric
axis_matrix = (
    df.set_index("Axis")[release_cols]
      .apply(pd.to_numeric, errors="coerce")
)

# Drop axes that are completely empty
axis_matrix = axis_matrix.dropna(how="all", axis=0)

target = TARGET_AXIS.strip().upper()
if target not in axis_matrix.index:
    raise ValueError(f"TARGET_AXIS='{TARGET_AXIS}' not found in Excel Axis column.")

all_axes = list(axis_matrix.index)

# Decide predictor axes dynamically
if INCLUDE_AXES is None:
    feature_axes = [a for a in all_axes if a != target]
else:
    include_norm = [str(a).strip().upper() for a in INCLUDE_AXES]
    feature_axes = [a for a in include_norm if a in all_axes and a != target]

exclude_norm = {str(a).strip().upper() for a in EXCLUDE_AXES}
feature_axes = [a for a in feature_axes if a not in exclude_norm]

if len(feature_axes) == 0:
    raise ValueError("No feature axes left after applying INCLUDE_AXES / EXCLUDE_AXES.")

# -----------------------------
# 2) ALIGN DATA BY COMMON RELEASES (NO TRUNCATION)
# -----------------------------
# Create a training table with target + all features, aligned by release columns [1](https://my.shell.com/personal/koushik_dutta_shell_com/_layouts/15/Doc.aspx?sourcedoc=%7B0ECB2284-F83F-442E-A3FE-59C8F51151CA%7D&file=Book1_CoC_Expanded_v2.xlsx&action=default&mobileredirect=true)
train_axes = [target] + feature_axes
train_table = axis_matrix.loc[train_axes, release_cols]

# Keep only releases where ALL selected axes have values (avoids mismatched lengths)
valid_mask = train_table.notna().all(axis=0)
valid_releases = [c for c in release_cols if valid_mask[c]]

if len(valid_releases) < 2:
    raise ValueError(
        f"Not enough common releases with complete data across target + features. "
        f"Usable releases found: {len(valid_releases)}"
    )

train_table = train_table[valid_releases]

# Target vector (y) and feature matrix (X_feat)
y = train_table.loc[target].to_numpy(dtype=float)  # shape: (n_samples,)
X_feat = train_table.loc[feature_axes].to_numpy(dtype=float).T  # shape: (n_samples, n_features)

n_samples, n_features = X_feat.shape
print(f"[DATA] Target axis = {target}")
print(f"[DATA] #Samples (releases used) = {n_samples}")
print(f"[DATA] #Features (axes used) = {n_features}")
print(f"[DATA] Feature axes = {feature_axes}")

# -----------------------------
# 3) FIT LINEAR REGRESSION (LEAST SQUARES) WITH DYNAMIC FEATURES
# -----------------------------
if STANDARDIZE_FEATURES:
    feat_means = X_feat.mean(axis=0)
    feat_stds = X_feat.std(axis=0, ddof=0)
    feat_stds[feat_stds == 0.0] = 1.0
    X_scaled = (X_feat - feat_means) / feat_stds
    X_design = np.column_stack([np.ones(n_samples), X_scaled])
else:
    feat_means = None
    feat_stds = None
    X_design = np.column_stack([np.ones(n_samples), X_feat])

beta, *_ = np.linalg.lstsq(X_design, y, rcond=None)
intercept = float(beta[0])
coefs = beta[1:].astype(float)  # length = n_features

y_pred = X_design @ beta

# R^2
ss_res = float(np.sum((y - y_pred) ** 2))
ss_tot = float(np.sum((y - y.mean()) ** 2))
r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

print("\n[MODEL] y_hat = intercept + Σ(coef_i * feature_i)")
print(f"        intercept = {intercept:.6f}")
for ax, c in zip(feature_axes, coefs):
    print(f"        coef[{ax}] = {c:.6f}")
print(f"[METRIC] R^2 = {r2:.6f}")

# -----------------------------
# 4) SCENARIO SELECTION (DYNAMIC) + PREDICTION
# -----------------------------
# Default scenario uses the last valid release values for each predictor
last_rel = valid_releases[-1]
default_scenario = train_table.loc[feature_axes, last_rel].to_dict()

# Apply scenario overrides if provided (keys by Axis)
if future_scenario is None:
    scenario = default_scenario
else:
    scenario = default_scenario.copy()
    for k, v in future_scenario.items():
        scenario[str(k).strip().upper()] = float(v)

# Build feature vector x0 in the same order as feature_axes
x0 = np.array([float(scenario[a]) for a in feature_axes], dtype=float)

if STANDARDIZE_FEATURES:
    x0_scaled = (x0 - feat_means) / feat_stds
    y_hat = float(intercept + x0_scaled @ coefs)
    contribs = x0_scaled * coefs
else:
    y_hat = float(intercept + x0 @ coefs)
    contribs = x0 * coefs

print(f"\n[PRED] Predicted future {target} for scenario = {y_hat:.4f}")
print(f"[INFO] Scenario base = last usable release '{last_rel}' with overrides applied (if any).")

# -----------------------------
# 5) PLOT #1: ACTUAL vs PREDICTED (DYNAMIC TARGET)
# -----------------------------
plt.figure(figsize=(7.5, 5.5))
plt.scatter(y, y_pred, s=70, color="#b4761f", label="Predicted vs Actual")

xy_min = float(min(np.min(y), np.min(y_pred)))
xy_max = float(max(np.max(y), np.max(y_pred)))
plt.plot([xy_min, xy_max], [xy_min, xy_max], "k--", lw=1, label="y = x")

plt.xlabel(f"Actual {target}")
plt.ylabel(f"Predicted {target}")
plt.title(f"Actual vs Predicted {target} (R²={r2:.3f})")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------
# 6) PLOT #2: RADAR PLOT (TOP-N CONTRIBUTIONS, DYNAMIC FEATURES)
# -----------------------------
if SHOW_CONTRIBUTIONS:
    abs_contrib = np.abs(contribs)
    order = np.argsort(abs_contrib)[::-1]

    if TOP_N_RADAR is None or TOP_N_RADAR >= n_features:
        idx = order
    else:
        idx = order[:TOP_N_RADAR]

    radar_axes = [feature_axes[i] for i in idx]
    radar_vals = contribs[idx].astype(float)

    # Normalize for radar display (0..1)
    vmin, vmax = float(np.min(radar_vals)), float(np.max(radar_vals))
    if np.isclose(vmin, vmax):
        radar_norm = np.ones_like(radar_vals) * 0.5
    else:
        radar_norm = (radar_vals - vmin) / (vmax - vmin)

    # Close loop
    labels = radar_axes + [radar_axes[0]]
    values = np.concatenate([radar_norm, radar_norm[:1]])

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=True)

    fig = plt.figure(figsize=(7.5, 7.0))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, values, color="#ff7f0e", lw=2)
    ax.fill(angles, values, color="#ff0e0e", alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_axes)
    ax.set_yticklabels([])

    ax.set_title(
        f"Top-{len(radar_axes)} Feature Contributions (normalized)\nScenario prediction: {target}={y_hat:.2f}",
        pad=20
    )
    plt.tight_layout()
    plt.show()

# -----------------------------
# 7) PLOT #3: COEFFICIENT BAR CHART (DYNAMIC FEATURES)
# -----------------------------
plot_axes = feature_axes
plot_vals = coefs

if TOP_N_COEFS is not None and TOP_N_COEFS < len(plot_axes):
    order = np.argsort(np.abs(plot_vals))[::-1][:TOP_N_COEFS]
    plot_axes = [plot_axes[i] for i in order]
    plot_vals = plot_vals[order]

plt.figure(figsize=(max(9, 0.35 * len(plot_axes)), 5.2))
plt.bar(plot_axes, plot_vals, color="#f59206")
plt.axhline(0, color="black", lw=0.8)
plt.xlabel("Feature Axis")
plt.ylabel("Coefficient")
plt.title("Learned Coefficients per Feature Axis")
plt.grid(True, axis="y", alpha=0.25)
plt.xticks(rotation=60, ha="right")
plt.tight_layout()
plt.show()