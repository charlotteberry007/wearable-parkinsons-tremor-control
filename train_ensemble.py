"""
train_ensemble.py  –  Train XGBoost + RandomForest ensemble for tremor detection
==================================================================================
• Input  : Dataset.csv  (columns: aX, aY, aZ, gX, gY, gZ, mX, mY, mZ, Result)
• Output : scaler.pkl, xgb_model.pkl, rf_model.pkl  (workspace root)
• Method : 18 scale-invariant features  (see feature_extractor.py)
           StratifiedKFold(5) cross-validation
           Probability calibration (isotonic)
           Voluntary-motion augmentation at DATASET scale

Run:
    python train_ensemble.py
"""

import os, warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")   # headless
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, balanced_accuracy_score
)
from xgboost import XGBClassifier

from feature_extractor import extract_features, FEATURE_NAMES, WINDOW_N, FS

warnings.filterwarnings("ignore")
np.random.seed(42)

WORKSPACE = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(WORKSPACE, "Dataset.csv")

# ─── dataset loading ──────────────────────────────────────────────────────────

def load_dataset(path: str):
    df = pd.read_csv(path)
    print(f"[DATA] Loaded {len(df)} rows, columns: {list(df.columns)}")

    # keep only IMU columns + label
    imu_cols = ["aX", "aY", "aZ", "gX", "gY", "gZ"]
    for c in imu_cols:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' missing from dataset.")

    # strip magnetometer (always -1, useless)
    data = df[imu_cols + ["Result"]].dropna().reset_index(drop=True)
    print(f"[DATA] After drop NaN: {len(data)} rows")
    print(f"[DATA] Class balance -> 0:{(data.Result==0).sum()}  1:{(data.Result==1).sum()}")
    return data


# ─── windowing ────────────────────────────────────────────────────────────────

def make_windows(data: pd.DataFrame, step: int = 50):
    """
    Slide a window of WINDOW_N samples with 'step' stride.
    Each window inherits the majority label of its samples.
    """
    imu = data[["aX", "aY", "aZ", "gX", "gY", "gZ"]].values
    labels = data["Result"].values

    X_list, y_list = [], []
    i = 0
    while i + WINDOW_N <= len(imu):
        w = imu[i: i + WINDOW_N]
        l_majority = int(np.round(labels[i: i + WINDOW_N].mean()))
        feat = extract_features(w)
        X_list.append(feat)
        y_list.append(l_majority)
        i += step

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    print(f"[WINDOW] {len(X)} windows  ->  0:{(y==0).sum()}  1:{(y==1).sum()}")
    return X, y


# ─── data augmentation ────────────────────────────────────────────────────────
#
# All synthetic signals are generated at TRAINING DATASET scale:
#   accel: 16384 LSB/g (±2g), so 1g = 16384 counts
#   gyro:  16.4 LSB/dps  (±2000 dps mode)
#   gX DC bias = ~5002 LSB (observed in dataset)
#
# We add:
#   400 voluntary motion windows  (0.8–2.5 Hz, large amplitude) → label 0
#   200 resting / barely moving   (noise only)                   → label 0
#   100 high-amplitude tremor     (3.5–7.5 Hz, strong)          → label 1  (extra)

def _make_signal(n, freq, amp_acc, amp_gyro, phase=0.0, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    t = np.arange(n) / FS
    ax = amp_acc * np.sin(2*np.pi*freq*t + phase) + rng.normal(0, amp_acc*0.1, n)
    ay = amp_acc * np.sin(2*np.pi*freq*t + phase + 0.4) + rng.normal(0, amp_acc*0.1, n)
    az = rng.normal(-13338, 400, n)   # gravity on Z axis (dataset convention)
    gx = rng.normal(5002, 80, n)      # DC bias
    gy = amp_gyro * np.sin(2*np.pi*freq*t + phase) + rng.normal(-239, amp_gyro*0.1, n)
    gz = amp_gyro * np.sin(2*np.pi*freq*t + phase + 0.4) + rng.normal(275, amp_gyro*0.1, n)
    return np.stack([ax, ay, az, gx, gy, gz], axis=1)


def augment(X_real: np.ndarray, y_real: np.ndarray):
    rng = np.random.default_rng(7)
    aug_X, aug_y = [], []

    # --- voluntary motion (label 0) ---
    for i in range(400):
        freq  = rng.uniform(0.8, 2.5)
        amp_a = rng.uniform(3000, 12000)      # large arm swing
        amp_g = rng.uniform(2000, 12000)
        phase = rng.uniform(0, 2*np.pi)
        w = _make_signal(WINDOW_N, freq, amp_a, amp_g, phase, rng)
        aug_X.append(extract_features(w))
        aug_y.append(0)

    # --- resting (label 0) ---
    for i in range(200):
        ax = rng.normal(54, 150, WINDOW_N)
        ay = rng.normal(5756, 150, WINDOW_N)
        az = rng.normal(-13338, 150, WINDOW_N)
        gx = rng.normal(5002, 50, WINDOW_N)
        gy = rng.normal(-239, 50, WINDOW_N)
        gz = rng.normal(275, 50, WINDOW_N)
        w = np.stack([ax, ay, az, gx, gy, gz], axis=1)
        aug_X.append(extract_features(w))
        aug_y.append(0)

    # --- strong tremor (label 1) ---
    for i in range(100):
        freq  = rng.uniform(3.5, 7.5)
        amp_a = rng.uniform(500, 3000)         # moderate tremor amplitude
        amp_g = rng.uniform(300, 2000)
        phase = rng.uniform(0, 2*np.pi)
        w = _make_signal(WINDOW_N, freq, amp_a, amp_g, phase, rng)
        aug_X.append(extract_features(w))
        aug_y.append(1)

    aug_X = np.array(aug_X, dtype=np.float32)
    aug_y = np.array(aug_y, dtype=np.int32)
    print(f"[AUG] Added {len(aug_X)} synthetic windows  ->  0:{(aug_y==0).sum()}  1:{(aug_y==1).sum()}")

    X_all = np.vstack([X_real, aug_X])
    y_all = np.concatenate([y_real, aug_y])
    return X_all, y_all


# ─── model definitions ────────────────────────────────────────────────────────

def build_xgb():
    return XGBClassifier(
        max_depth=4,
        n_estimators=300,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.5,
        scale_pos_weight=1.0,   # balanced dataset; adjust if needed
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
        n_jobs=-1,
    )


def build_rf():
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        max_features="sqrt",
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )


# ─── cross-validation ─────────────────────────────────────────────────────────

def cross_validate(X: np.ndarray, y: np.ndarray):
    """5-fold stratified CV, print per-fold and mean ± std."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scaler = StandardScaler()

    xgb_accs, rf_accs, ens_accs = [], [], []
    print("\n---- 5-Fold Cross-Validation ------------------------------------")

    for fold, (tr, te) in enumerate(skf.split(X, y), 1):
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]

        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s  = sc.transform(X_te)

        xgb = build_xgb()
        xgb.fit(X_tr_s, y_tr)

        rf = build_rf()
        rf.fit(X_tr_s, y_tr)

        p_xgb = xgb.predict_proba(X_te_s)[:, 1]
        p_rf  = rf.predict_proba(X_te_s)[:, 1]
        p_ens = 0.6 * p_xgb + 0.4 * p_rf

        xgb_acc = balanced_accuracy_score(y_te, p_xgb >= 0.5)
        rf_acc  = balanced_accuracy_score(y_te, p_rf  >= 0.5)
        ens_acc = balanced_accuracy_score(y_te, p_ens >= 0.5)

        xgb_accs.append(xgb_acc)
        rf_accs.append(rf_acc)
        ens_accs.append(ens_acc)

        print(f"  Fold {fold}: XGB={xgb_acc:.3f}  RF={rf_acc:.3f}  Ensemble={ens_acc:.3f}")

    print(f"\n  XGB      mean={np.mean(xgb_accs):.3f}  std={np.std(xgb_accs):.3f}")
    print(f"  RF       mean={np.mean(rf_accs):.3f}  std={np.std(rf_accs):.3f}")
    print(f"  Ensemble mean={np.mean(ens_accs):.3f}  std={np.std(ens_accs):.3f}")
    print("-----------------------------------------------------------------\n")
    return np.mean(ens_accs)


# ─── final training ───────────────────────────────────────────────────────────

def train_final(X: np.ndarray, y: np.ndarray):
    """Train on full dataset, calibrate, save artefacts."""
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    print("[TRAIN] Fitting XGBoost on full dataset ...")
    xgb_base = build_xgb()
    xgb_base.fit(X_s, y)
    xgb_cal = CalibratedClassifierCV(xgb_base, method="isotonic", cv=3)
    xgb_cal.fit(X_s, y)

    print("[TRAIN] Fitting RandomForest on full dataset ...")
    rf_base = build_rf()
    rf_base.fit(X_s, y)
    rf_cal = CalibratedClassifierCV(rf_base, method="isotonic", cv=3)
    rf_cal.fit(X_s, y)

    # ── sanity check on training data ──
    p_xgb = xgb_cal.predict_proba(X_s)[:, 1]
    p_rf  = rf_cal.predict_proba(X_s)[:, 1]
    p_ens = 0.6 * p_xgb + 0.4 * p_rf
    y_pred = (p_ens >= 0.40).astype(int)

    print("\n---- Training-set report ----------------------------------------")
    print(classification_report(y, y_pred, target_names=["Normal/Voluntary", "Tremor"]))
    cm = confusion_matrix(y, y_pred)
    print(f"Confusion matrix:\n{cm}\n")

    # ── feature importances ──
    print("---- Feature Importances (XGBoost) ------------------------------")
    fi_xgb = xgb_base.feature_importances_
    for name, fi in sorted(zip(FEATURE_NAMES, fi_xgb), key=lambda x: -x[1]):
        bar = "#" * int(fi * 60)
        print(f"  {name:<30s} {fi:.4f}  {bar}")

    print("\n---- Feature Importances (RandomForest) -------------------------")
    fi_rf = rf_base.feature_importances_
    for name, fi in sorted(zip(FEATURE_NAMES, fi_rf), key=lambda x: -x[1]):
        bar = "#" * int(fi * 80)
        print(f"  {name:<30s} {fi:.4f}  {bar}")

    # ── save ──
    scaler_path  = os.path.join(WORKSPACE, "scaler.pkl")
    xgb_path     = os.path.join(WORKSPACE, "xgb_model.pkl")
    rf_path      = os.path.join(WORKSPACE, "rf_model.pkl")

    joblib.dump(scaler, scaler_path)
    joblib.dump(xgb_cal, xgb_path)
    joblib.dump(rf_cal,  rf_path)

    print(f"\n[SAVE] scaler.pkl  -> {scaler_path}")
    print(f"[SAVE] xgb_model.pkl -> {xgb_path}")
    print(f"[SAVE] rf_model.pkl  -> {rf_path}")

    # ── feature importance bar chart ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, fi, title in [
        (axes[0], fi_xgb, "XGBoost Feature Importances"),
        (axes[1], fi_rf,  "RandomForest Feature Importances"),
    ]:
        idx = np.argsort(fi)[::-1]
        ax.barh([FEATURE_NAMES[i] for i in idx][::-1], [fi[i] for i in idx][::-1])
        ax.set_title(title)
        ax.set_xlabel("Importance")
    plt.tight_layout()
    chart_path = os.path.join(WORKSPACE, "feature_importances.png")
    plt.savefig(chart_path, dpi=100)
    print(f"[SAVE] feature_importances.png -> {chart_path}")

    return scaler, xgb_cal, rf_cal


# ─── entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("  Tremor Detection - Training Pipeline")
    print("  Movement-only (18 scale-invariant features, no audio)")
    print("=" * 65)

    # 1. Load
    data = load_dataset(DATASET_PATH)

    # 2. Window
    X_real, y_real = make_windows(data, step=50)

    # 3. Augment
    X_all, y_all = augment(X_real, y_real)
    print(f"[TOTAL] {len(X_all)} windows  ->  0:{(y_all==0).sum()}  1:{(y_all==1).sum()}")

    # 4. Cross-validate
    mean_ens_acc = cross_validate(X_all, y_all)
    if mean_ens_acc < 0.80:
        print(f"[WARN] CV balanced-accuracy {mean_ens_acc:.3f} < 0.80 - check data quality")

    # 5. Train final model + save
    scaler, xgb_model, rf_model = train_final(X_all, y_all)

    print("\n[DONE] Training complete.  Models saved to workspace root.")
    print("       Next: python predict_serial.py --simulate")
