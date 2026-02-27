"""
Load collected gesture data, extract features, train RandomForest, save model.
"""

import csv
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib

_DATA_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(_DATA_DIR, "gesture_data.csv")
MODEL_FILE = os.path.join(_DATA_DIR, "model.joblib")
WINDOW_SIZE = 15
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_csv(data_path):
    """
    Load gesture_data.csv. Each row: label + flattened window (ax_0, ..., gz_{N-1}) + optional timestamp_iso, take_id.
    Returns: list of (label_str, window_list, take_id). take_id is 0 when column is missing.
    """
    if not os.path.exists(data_path):
        return None

    rows = []
    with open(data_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if not fieldnames or fieldnames[0] != "label":
            return None
        for row in reader:
            label = row.get("label", "").strip()
            if not label:
                continue
            window = []
            for i in range(WINDOW_SIZE):
                ax = float(row.get(f"ax_{i}", 0))
                ay = float(row.get(f"ay_{i}", 0))
                az = float(row.get(f"az_{i}", 0))
                gx = float(row.get(f"gx_{i}", 0))
                gy = float(row.get(f"gy_{i}", 0))
                gz = float(row.get(f"gz_{i}", 0))
                window.append((ax, ay, az, gx, gy, gz))
            take_id_val = row.get("take_id", "")
            take_id = int(take_id_val) if (take_id_val and str(take_id_val).strip()) else 0
            rows.append((label, window, take_id))
    return rows


def main():
    from features import window_to_features

    if not os.path.exists(DATA_FILE):
        print(f"{DATA_FILE} not found. Run collect_data.py first to record labeled gestures.")
        return

    data = load_csv(DATA_FILE)
    if not data:
        print("No valid rows in CSV or invalid format.")
        return

    X_list = []
    y_list = []
    groups = []
    for label, window, take_id in data:
        X_list.append(window_to_features(window))
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
    use_group_split = len(unique_groups) >= 2

    if use_group_split:
        gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        train_idx, test_idx = next(gss.split(X, y_enc, groups=groups))
        if len(test_idx) == 0:
            use_group_split = False
        else:
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y_enc[train_idx], y_enc[test_idx]

    if not use_group_split:
        if len(unique_groups) < 2:
            print("No take_id in CSV; using random split. Re-collect with new collector for group-based split.")
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
    clf.fit(X_train, y_train)

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
    }
    joblib.dump(package, MODEL_FILE)
    print(f"Model saved to {MODEL_FILE}")


if __name__ == "__main__":
    main()
