# DVC tutorial: a hands-on playground for data version control

A small, self-contained machine-learning project for learning [DVC](https://dvc.org) end to end:
tracking data, sharing it through a remote, time-travelling between dataset versions, building a
reproducible pipeline, comparing metrics and running experiments.

📖 **Step-by-step article:** [DVC tutorial: version data, pipelines and ML experiments](https://abhishekkothari.in/blog/dvc-tutorial-data-version-control-for-data-scientists.html)

## What's inside

```
.
├── starter-data/
│   ├── customers_v1.csv    # 2,000 synthetic telecom customers (Jan-Jun signups)
│   └── customers_v2.csv    # v1 + 1,000 July signups, to practise data versioning
├── src/
│   ├── prepare.py          # clean + one-hot encode + train/test split
│   ├── train.py            # random-forest churn model
│   └── evaluate.py         # metrics.json + plot data (ROC, confusion matrix)
├── scripts/
│   └── generate_data.py    # regenerates the CSVs (stdlib only, fixed seed)
├── params.yaml             # hyperparameters DVC tracks
└── requirements.txt
```

The data is **dummy data**: synthetic customers with a churn label that depends on contract type,
tenure, monthly charges and support tickets, plus a few missing values to clean. It is generated
deterministically, so your DVC hashes will match the ones in the article.

| column | example | notes |
|---|---|---|
| `customer_id` | `C00001` | dropped before training |
| `signup_month` | `2026-02` | dropped before training |
| `tenure_months` | `18` | |
| `monthly_charges` | `83.47` | ~1.5% missing |
| `support_tickets` | `2` | |
| `contract` | `month-to-month` / `one-year` / `two-year` | one-hot encoded |
| `internet_service` | `fiber` / `dsl` / `none` | one-hot encoded |
| `paperless_billing` | `yes` / `no` | |
| `churned` | `0` / `1` | target, ~30% positive |

## Branches

| branch | what it is |
|---|---|
| `main` | **Start here.** Plain Git project, DVC not initialised yet. Follow the article from step 1. |
| `solution` | The finished state after the article: `.dvc` files, `dvc.yaml`, `dvc.lock`, metrics. Its default DVC remote is the public, read-only `storage` branch, so `dvc pull` works with no credentials. |
| `storage` | Content-addressed DVC remote served over HTTPS from `raw.githubusercontent.com`. Don't check it out; DVC reads it for you. |

## Quick start

```sh
git clone https://github.com/datsabk/dvc-tutorial.git
cd dvc-tutorial
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
dvc version
```

Then follow the article. Want to peek at the end result first?

```sh
git checkout solution
dvc pull          # downloads data, prepared splits and the model from the public storage branch
dvc repro         # "Data and pipelines are up to date."
dvc metrics show
```

Or grab just the dataset from any other project, no clone needed:

```sh
dvc get https://github.com/datsabk/dvc-tutorial data/raw/customers.csv --rev solution
```

## Regenerating the data

```sh
python scripts/generate_data.py
```

## License

MIT, see [LICENSE](LICENSE). The data is synthetic; use it for anything.
