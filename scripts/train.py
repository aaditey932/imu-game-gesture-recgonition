"""
Load collected gesture data, extract features, train RandomForest, save model.
"""

import sys
from pathlib import Path

_repo_src = Path(__file__).resolve().parents[1] / "src"
if _repo_src.is_dir():
    sys.path.insert(0, str(_repo_src))

import csv
import os
import time
import argparse
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib

from imugesture.paths import DATA_CSV, MODEL_JOBLIB, ensure_data_dirs

WINDOW_SIZE = 15
N_FLEX_CHANNELS = 2
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Train/test split: "group" = GroupShuffleSplit by take_id (no leakage across takes) when
# multiple take_id values exist; "stratified" = always stratify by class and ignore groups.
SPLIT_MODE = "group"


def _row_to_window(row: dict) -> list:
    """Build window list from raw CSV row (IMU + flex per timestep)."""
    window = []
    for i in range(WINDOW_SIZE):
        ax = float(row.get(f"ax_{i}", 0))
        ay = float(row.get(f"ay_{i}", 0))
        az = float(row.get(f"az_{i}", 0))
        gx = float(row.get(f"gx_{i}", 0))
        gy = float(row.get(f"gy_{i}", 0))
        gz = float(row.get(f"gz_{i}", 0))
        flex = []
        for j in range(N_FLEX_CHANNELS):
            key = f"f{j}_{i}"
            v = row.get(key, "")
            flex.append(float(v) if (v is not None and str(v).strip() != "") else 0.0)
        window.append((ax, ay, az, gx, gy, gz) + tuple(flex))
    return window


def load_csv(data_path, feature_names, n_features, window_to_features_fn):
    """
    Load gesture CSV. Returns:
      (rows, source) where rows is list of (label_str, feature_vector ndarray shape (n_features,), take_id),
      and source is \"csv_columns\" if v2 feature columns were read from the file, else \"computed\".

    If all FEATURE_NAMES columns exist, reads them directly; otherwise builds the window and calls
    window_to_features_fn (same math as live inference).
    """
    if not os.path.exists(data_path):
        return None, None

    rows = []
    with open(data_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if not fieldnames or fieldnames[0] != "label":
            return None, None
        has_precomputed = all(name in fieldnames for name in feature_names)

        for row in reader:
            label = row.get("label", "").strip()
            if not label:
                continue
            take_id_val = row.get("take_id", "")
            take_id = int(take_id_val) if (take_id_val and str(take_id_val).strip()) else 0

            if has_precomputed:
                vec = []
                for name in feature_names:
                    v = row.get(name, "")
                    vec.append(float(v) if (v is not None and str(v).strip() != "") else 0.0)
                feats = np.array(vec, dtype=float)
                if feats.shape[0] != n_features:
                    return None, None
            else:
                window = _row_to_window(row)
                feats = window_to_features_fn(window)

            rows.append((label, feats, take_id))

    src = "csv_columns" if has_precomputed else "computed"
    return rows, src


def parse_args():
    p = argparse.ArgumentParser(
        description="Train gesture classifier from collected IMU + flex windows."
    )
    p.add_argument(
        "--csv",
        type=str,
        default=None,
        metavar="PATH",
        help=f"Training CSV (default: {DATA_CSV}). Use a file with v2 feature columns from append_feature_columns.py, or raw-only rows (features computed on the fly).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate data loading and class counts, then exit without training/saving model.",
    )
    return p.parse_args()


def main():
    from imugesture.features import (
        FEATURE_NAMES,
        FEATURE_SCHEMA,
        N_FEATURES,
        window_to_features,
    )

    args = parse_args()

    data_path = args.csv if args.csv else str(DATA_CSV)
    if not os.path.exists(data_path):
        print(
            f"{data_path} not found. Run scripts/collect_data.py first to record labeled gestures."
        )
        return

    data, feat_source = load_csv(data_path, FEATURE_NAMES, N_FEATURES, window_to_features)
    if not data:
        print("No valid rows in CSV or invalid format.")
        return
    if args.dry_run:
        labels = sorted({label for label, _, _ in data})
        print(f"Dry run OK: loaded {len(data)} rows from {data_path}")
        print(f"Feature source: {feat_source} (csv_columns = use precomputed v2 columns; computed = from IMU/flex)")
        print(f"Classes found ({len(labels)}): {labels}")
        print(f"Feature schema: {FEATURE_SCHEMA}, n_features={N_FEATURES}")
        return

    print(
        f"Training with features from {feat_source} ({data_path}) — schema {FEATURE_SCHEMA}, n={N_FEATURES}",
        flush=True,
    )

    X_list = []
    y_list = []
    groups = []
    for label, feats, take_id in data:
        X_list.append(feats)
        y_list.append(label)
        groups.append(take_id)

    X = np.vstack(X_list)
    y = np.array(y_list)
    groups = np.array(groups)

    n_classes = len(set(y_list))
    if n_classes < 2:
        print(f"Need at least 2 gesture classes. Found: {n_classes}. Collect more labels.")
        return

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    unique_groups = np.unique(groups)
    use_group_split = False
    if SPLIT_MODE == "group" and len(unique_groups) >= 2:
        gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        train_idx, test_idx = next(gss.split(X, y_enc, groups=groups))
        if len(test_idx) == 0:
            print(
                "Group split produced empty test set; falling back to stratified split.",
                flush=True,
            )
        else:
            use_group_split = True
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y_enc[train_idx], y_enc[test_idx]

    if not use_group_split:
        if SPLIT_MODE == "group" and len(unique_groups) < 2:
            print(
                "No take_id in CSV; using random split. Re-collect with new collector for group-based split.",
                flush=True,
            )
        elif SPLIT_MODE == "stratified":
            print("Using stratified row split (SPLIT_MODE=stratified).", flush=True)
        n_samples = len(X)
        test_count = max(n_classes, int(round(TEST_SIZE * n_samples)))
        test_count = min(test_count, n_samples - 1)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_enc, test_size=test_count, stratify=y_enc, random_state=RANDOM_STATE
        )

    clf = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        oob_score=True,
    )
    print("Training classifier...")
    t_fit = time.perf_counter()
    clf.fit(X_train, y_train)
    print(f"Training finished in {time.perf_counter() - t_fit:.1f}s.")

    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)
    print(f"Train accuracy: {train_acc:.3f}")
    print(f"Test accuracy:  {test_acc:.3f}")
    print(f"OOB score:      {clf.oob_score_:.3f}")

    y_pred = clf.predict(X_test)
    print("\nClassification report (test set):")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_test, y_pred))

    package = {
        "model": clf,
        "label_encoder": le,
        "feature_schema": FEATURE_SCHEMA,
        "feature_names": list(FEATURE_NAMES),
        "n_features": N_FEATURES,
    }
    ensure_data_dirs()
    joblib.dump(package, MODEL_JOBLIB)
    print(f"Model saved to {MODEL_JOBLIB}")


if __name__ == "__main__":
    main()
