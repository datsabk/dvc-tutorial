#!/usr/bin/env python3
"""Generate the dummy customer-churn datasets used in the DVC tutorial.

Standard library only and fully deterministic (fixed seed), so everyone gets
byte-identical files and therefore the same DVC hashes.

    python scripts/generate_data.py            # writes starter-data/customers_v1.csv and _v2.csv

v1: 2,000 customers who signed up in the first half of the year.
v2: the same 2,000 rows plus 1,000 newer customers ("a new month of data
    arrived"), which is what we use to demonstrate data versioning.
"""
import csv
import math
import os
import random

SEED = 42
V1_ROWS, V2_EXTRA_ROWS = 2000, 1000
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "starter-data")
FIELDS = ["customer_id", "signup_month", "tenure_months", "monthly_charges", "support_tickets",
          "contract", "internet_service", "paperless_billing", "churned"]
CONTRACTS = [("month-to-month", 0.55), ("one-year", 0.25), ("two-year", 0.20)]
INTERNET = [("fiber", 0.45), ("dsl", 0.35), ("none", 0.20)]


def pick(rng, options):
    r, acc = rng.random(), 0.0
    for value, weight in options:
        acc += weight
        if r <= acc:
            return value
    return options[-1][0]


def customer(rng, idx, month_range):
    contract = pick(rng, CONTRACTS)
    internet = pick(rng, INTERNET)
    tenure = max(1, int(rng.gammavariate(2.0, 9.0 if contract == "month-to-month" else 16.0)))
    base = {"fiber": 85.0, "dsl": 55.0, "none": 25.0}[internet]
    charges = round(max(18.0, rng.gauss(base, 12.0)), 2)
    tickets = min(9, int(rng.expovariate(0.9)))
    paperless = rng.random() < 0.6

    # Churn depends on the features in a believable way, plus noise.
    z = (-1.2
         + {"month-to-month": 1.4, "one-year": -0.3, "two-year": -1.6}[contract]
         - 0.045 * tenure
         + 0.018 * (charges - 60)
         + 0.35 * tickets
         + (0.3 if paperless else 0.0)
         + rng.gauss(0, 0.6))
    churned = rng.random() < 1 / (1 + math.exp(-z))

    # A few missing charges, so prepare.py has something to clean.
    charges_out = "" if rng.random() < 0.015 else f"{charges:.2f}"
    return {
        "customer_id": f"C{idx:05d}",
        "signup_month": f"2026-{rng.randint(*month_range):02d}",
        "tenure_months": tenure,
        "monthly_charges": charges_out,
        "support_tickets": tickets,
        "contract": contract,
        "internet_service": internet,
        "paperless_billing": "yes" if paperless else "no",
        "churned": int(churned),
    }


def write(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def main():
    rng = random.Random(SEED)
    v1 = [customer(rng, i, (1, 6)) for i in range(1, V1_ROWS + 1)]
    extra = [customer(rng, i, (7, 7)) for i in range(V1_ROWS + 1, V1_ROWS + V2_EXTRA_ROWS + 1)]
    os.makedirs(OUT_DIR, exist_ok=True)
    write(os.path.join(OUT_DIR, "customers_v1.csv"), v1)
    write(os.path.join(OUT_DIR, "customers_v2.csv"), v1 + extra)


if __name__ == "__main__":
    main()
