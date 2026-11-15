"""Stage 2: train a random-forest churn model.

    python src/train.py data/prepared/train.csv models/model.pkl
"""
import os
import pickle
import sys

import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier


def main(train_path, model_path):
    params = yaml.safe_load(open("params.yaml"))["train"]
    df = pd.read_csv(train_path)
    X, y = df.drop(columns=["churned"]), df["churned"]

    model = RandomForestClassifier(n_estimators=params["n_estimators"], max_depth=params["max_depth"],
                                   class_weight=params.get("class_weight"),  # null or "balanced"
                                   random_state=params["seed"], n_jobs=-1)
    model.fit(X, y)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as fh:
        pickle.dump(model, fh)
    print(f"trained {params['n_estimators']} trees (max_depth={params['max_depth']}, "
          f"class_weight={params.get('class_weight')}) on {len(df)} rows")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
