"""Stage 1: clean the raw CSV and split it into train/test sets.

    python src/prepare.py data/raw/customers.csv data/prepared
"""
import os
import sys

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


def main(raw_path, out_dir):
    params = yaml.safe_load(open("params.yaml"))["prepare"]
    df = pd.read_csv(raw_path)

    df["monthly_charges"] = df["monthly_charges"].fillna(df["monthly_charges"].median())
    df["paperless_billing"] = (df["paperless_billing"] == "yes").astype(int)
    df = pd.get_dummies(df, columns=["contract", "internet_service"], dtype=int)
    df = df.drop(columns=["customer_id", "signup_month"])

    train, test = train_test_split(df, test_size=params["test_size"], random_state=params["seed"],
                                   stratify=df["churned"])
    os.makedirs(out_dir, exist_ok=True)
    train.to_csv(os.path.join(out_dir, "train.csv"), index=False)
    test.to_csv(os.path.join(out_dir, "test.csv"), index=False)
    print(f"prepared {len(train)} train / {len(test)} test rows from {raw_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
