"""
Quick evaluator for UFC XGBoost models.
Scans a directory of saved models (JSON) and prints their test set accuracy.
"""

import os
import re
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

# ===================== CONFIG ===================== #
MODEL_DIR = "../../../saved_models/xgboost/single_split/"   # your folder path
MODEL_PATTERN = r"ufc_xgb_single_(?:TRIAL\d{3}|FINAL).*\.json$"
TEST_DATA_PATH = "../../../data/train_test/test_data.csv"
DISPLAY_COLS = ["current_fight_date", "fighter_a", "fighter_b"]

# ================================================== #

def load_test_data():
    df = pd.read_csv(TEST_DATA_PATH)
    y = df["winner"]
    X = df.drop(["winner"] + DISPLAY_COLS, axis=1)
    # ensure numeric
    X = X.select_dtypes(include=[np.number]).fillna(0)
    return X, y


def main():
    pattern = re.compile(MODEL_PATTERN)
    files = [f for f in os.listdir(MODEL_DIR) if pattern.match(f)]

    if not files:
        print("No model files found.")
        return

    X_test, y_test = load_test_data()

    print(f"Found {len(files)} model(s) in {MODEL_DIR}\n")
    results = []

    for f in sorted(files):
        path = os.path.join(MODEL_DIR, f)
        model = xgb.XGBClassifier(enable_categorical=True)
        model.load_model(path)

        # Align test columns to model’s training feature order
        feats = model.get_booster().feature_names
        X_aligned = X_test.reindex(columns=feats).fillna(0)

        y_pred = model.predict(X_aligned)
        acc = accuracy_score(y_test, y_pred)
        results.append((f, acc))
        print(f"{f}: {acc:.4f}")

    avg_acc = np.mean([a for _, a in results])
    print("\n==========================")
    print(f"Average Accuracy: {avg_acc:.4f}")
    print("==========================")

if __name__ == "__main__":
    main()
