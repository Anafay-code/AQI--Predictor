# AQI Predictor — Serverless MLOps

End-to-end air-quality forecasting system. Predicts AQI 72 hours ahead for 7 cities. Cloud-first: Hopsworks feature store + model registry, GitHub Actions CI/CD, Streamlit dashboard.

**Cities:** London, New York, Beijing, Delhi, Paris, Karachi, Lahore
**Forecast horizon:** 72 hours (t+1h … t+72h, direct multi-output — not autoregressive)
**Models:** Ridge · Random Forest · LSTM (best per city auto-selected by RMSE)

---

## Architecture

```
Data sources (AQICN + OpenWeather)
        │
        ▼
Feature pipeline ──► Hopsworks Feature Store (online FG v2, RonDB)
        │                       │
        │                       ▼
        │             Training pipeline (Ridge/RF/LSTM, 5-fold TimeSeriesSplit)
        │                       │
        │                       ▼
        │             Hopsworks Model Registry (versioned)
        ▼                       │
Inference pipeline ◄────────────┘
        │
        ▼
Dashboard (Streamlit) + REST API (Flask)
```

Cloud is single source of truth — no local data files. CSV output deliberately disallowed (rubric).

---

> **Friend/teammate setting this up?** Follow `INSTRUCTIONS.txt` — linear, copy-paste steps with every error+fix. This README is the reference.

## Prerequisites

- **Python 3.10 ONLY** (REQUIRED — 3.11/3.12/3.14 break TensorFlow + hopsworks wheels)
- The **`.env` file** (gitignored) — get it from the project owner. Holds the API keys and the Hopsworks project name (`tenpearls`). Data + models live in the owner's cloud project, so you must use the owner's keys.
- API keys (only if creating your own project, all free tier):
  - [AQICN token](https://aqicn.org/data-platform/token/)
  - [OpenWeather key](https://openweathermap.org/api)
  - [Hopsworks key + project](https://app.hopsworks.ai)

---

## Setup

```bash
cd AQI_Predictor

# 1. Create Python 3.10 venv
#   Windows (py launcher):
py -3.10 -m venv venv
#   Linux / macOS:
python3.10 -m venv venv

# 2. Activate
#   Windows PowerShell:
venv\Scripts\Activate.ps1
#   Windows CMD:
venv\Scripts\activate.bat
#   Linux / macOS:
source venv/bin/activate

# 3. Install deps (TensorFlow ~350MB, allow time)
pip install --upgrade pip
pip install --timeout 300 --retries 20 -r requirements.txt
```

### Configure keys

```bash
cp .env.example .env
```

Edit `.env`:

```
AQICN_API_TOKEN=your_aqicn_token
OPENWEATHER_API_KEY=your_openweather_key
HOPSWORKS_API_KEY=your_hopsworks_key
HOPSWORKS_PROJECT=your_project_name
```

`.env` is gitignored — keys never committed.

---

## Run

### Fastest — full pipeline for all 7 cities (one shot)

```bash
python scripts/full_run.py
```

Backfills 90 days → engineers features → pushes to cloud FG → trains Ridge/RF/LSTM → pushes to registry. ~20-30 min (LSTM is slow).

### Launch dashboard

```bash
# Always launch via the venv python (guarantees correct interpreter):
python -m streamlit run app/streamlit_app.py
```

Opens at **http://localhost:8501**. Select city → history + 72h forecast charts.

> First city load is slow (downloads ~144MB model from cloud). Cached after — instant on repeat. Login cached per session.
>
> Use `python -m streamlit ...` (not bare `streamlit`) — the bare shim can resolve to a non-venv Python that lacks `hopsworks`.

### Launch REST API (optional)

```bash
python app/flask_api.py
```

Serves at **http://localhost:5000**. Endpoints: current AQI + 72h forecast JSON per city.

---

## Pipelines (run individually)

```bash
# Backfill synthetic/historical features → cloud
python scripts/backfill.py --city london --days 90
python scripts/backfill.py                       # all cities, 90 days

# Hourly feature fetch (latest reading → cloud FG)
python scripts/run_feature_pipeline.py --city london
python scripts/run_feature_pipeline.py           # all cities

# Train models → cloud registry
python scripts/run_training_pipeline.py --city london --days 90
python scripts/run_training_pipeline.py --days 90   # all cities
```

All scripts exit 0 on API gaps / thin store (won't fail CI on transient issues).

---

## Tests

```bash
pytest tests/ -v
```

Covers feature engineering, AQI alert classification, regression metrics. No API keys needed (synthetic fixtures).

---

## Notebook (EDA)

```bash
jupyter notebook notebooks/EDA.ipynb
```

`EDA_executed.ipynb` = pre-run version with output. Seasonality, autocorrelation, feature correlations.

---

## CI/CD (GitHub Actions)

Three workflows in `.github/workflows/`:

| Workflow | Trigger | Action |
|----------|---------|--------|
| `ci.yml` | push / PR | pytest (no secrets, always green) |
| `feature_pipeline.yml` | hourly cron | fetch AQI → cloud FG |
| `training_pipeline.yml` | daily 02:00 UTC | retrain → cloud registry |

**Activate:**

```bash
git init && git add . && git commit -m "AQI predictor"
gh repo create aqi-predictor --private --source=. --push
gh secret set -f .env      # sets all 4 secrets at once
```

Then GitHub → Actions tab → enable. Manual run: Actions → workflow → "Run workflow".

---

## Project layout

```
AQI_Predictor/
├── src/
│   ├── config.py                    # central config: cities, keys, horizons
│   ├── feature_pipeline/
│   │   ├── data_fetcher.py          # AQICN + OpenWeather (synthetic fallback)
│   │   ├── feature_engineer.py      # time/lag/rolling/cyclical + 72h targets
│   │   └── feature_store.py         # Hopsworks online FG v2 (cloud-only)
│   ├── training_pipeline/
│   │   ├── train.py                 # CV-train Ridge/RF/LSTM
│   │   ├── model_registry.py        # Hopsworks registry + disk cache
│   │   ├── evaluate.py              # RMSE/MAE/R²/MAPE
│   │   └── models/                  # ridge / random_forest / lstm builders
│   ├── inference_pipeline/
│   │   └── predict.py               # 72h forecast + CI bands
│   └── utils/shap_explainer.py      # feature importance
├── app/
│   ├── streamlit_app.py             # dashboard (port 8501)
│   └── flask_api.py                 # REST API (port 5000)
├── scripts/                         # backfill, feature, training, full_run
├── tests/                           # pytest suite
├── notebooks/EDA.ipynb              # exploratory analysis
├── .github/workflows/               # CI/CD
├── requirements.txt
└── .env.example
```

---

## Config knobs (`src/config.py`)

| Var | Default | Meaning |
|-----|---------|---------|
| `FORECAST_HOURS` | 72 | prediction horizon |
| `LOOKBACK_HOURS` | 48 | LSTM input window |
| `ALERT_THRESHOLD` | 150 | AQI ≥ this → hazard alert |
| `MIN_TRAIN_ROWS` | 200 | min rows to train |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `python3.10 is not recognized` (Windows) | Use `py -3.10 -m venv venv` |
| `No module named 'hopsworks'` | Wrong Python. Activate venv (`(venv)` in prompt), launch via `python -m streamlit ...` |
| `Query must be a string unless using sqlalchemy` | `pip install "sqlalchemy>=2.0,<2.1"` — pandas 2.3 can't detect sqlalchemy 1.4 conns (pinned in requirements) |
| `Feature load failed` + login error | `.env` missing/wrong keys. Must sit next to `requirements.txt`, `HOPSWORKS_PROJECT=tenpearls` |
| `ModuleNotFoundError: imp` on hopsworks install | Use Python 3.10, not 3.12/3.14 |
| TensorFlow won't install | `pip install --timeout 300 --retries 20 tensorflow==2.21.0` |
| `numpy.dtype size changed` | `pip install --force-reinstall -r requirements.txt` |
| PowerShell "running scripts is disabled" | `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` |
| `RemoteDisconnected` on cloud login | Auto-retried (4 attempts) — free-tier serving API flakiness |
| Dashboard slow first load | Expected — 144MB model download, cached after |
| `No cloud features for <city>` | Run `python scripts/run_feature_pipeline.py --city <city>` (london always works) |
| protobuf version conflict warning | Harmless — ignore, runs fine |
