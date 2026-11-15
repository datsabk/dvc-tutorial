"""Stage 3: score the model on the test set; write metrics and plot data for DVC.

    python src/evaluate.py models/model.pkl data/prepared/test.csv
Writes metrics.json, plots/roc.csv and plots/confusion.csv.
"""
import json
import os
import pickle
import sys

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve


def main(model_path, test_path):
    with open(model_path, "rb") as fh:
        model = pickle.load(fh)
    df = pd.read_csv(test_path)
    X, y = df.drop(columns=["churned"]), df["churned"]
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)

    metrics = {
        "accuracy": round(accuracy_score(y, pred), 4),
        "precision": round(precision_score(y, pred), 4),
        "recall": round(recall_score(y, pred), 4),
        "f1": round(f1_score(y, pred), 4),
        "roc_auc": round(roc_auc_score(y, proba), 4),
    }
    with open("metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)
        fh.write("\n")

    os.makedirs("plots", exist_ok=True)
    fpr, tpr, _ = roc_curve(y, proba)
    pd.DataFrame({"fpr": fpr.round(4), "tpr": tpr.round(4)}).to_csv("plots/roc.csv", index=False)
    pd.DataFrame({"actual": y, "predicted": pred}).to_csv("plots/confusion.csv", index=False)
    print(json.dumps(metrics))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
