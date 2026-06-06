# Pearls AQI Predictor — Technical Report

**3-day Air Quality Index forecasting · end-to-end serverless MLOps**
Cities: london, new york, beijing, delhi, paris, karachi, lahore · Horizon: 72 h

---

## 1. Problem statement
Forecast the Air Quality Index (AQI) for the next **72 hours** per city, served through an automated, scalable, **100% serverless** pipeline with an interactive dashboard. AQI is a public-health signal; an accurate 3-day outlook lets sensitive groups plan exposure.

## 2. Data sources
| Source | Role | Notes |
|--------|------|-------|
| **AQICN** (`api.waqi.info`) | Real-time AQI + pollutants (pm2.5, pm10, o3, no2, so2, co) | token via `AQICN_API_TOKEN` |
| **OpenWeather** | Weather (temp, humidity, pressure, wind) | key via `OPENWEATHER_API_KEY` |
| **Synthetic generator** | Offline / CI / backfill | physically-plausible hourly series with daily+weekly seasonality + drift |

History APIs are paid, so backfill synthesises a plausible series ending now; with live keys the same code path consumes real observations.

## 3. EDA findings (see `notebooks/EDA.ipynb`)
- **Strong daily seasonality** — AQI peaks mid-afternoon, troughs pre-dawn (traffic + photochemistry).
- **Weekly component** — weekday > weekend in traffic-heavy cities.
- **Autocorrelation** — AQI is highly autocorrelated at lags 1–24 h → lag/rolling features dominate.
- **City scale differs** — delhi/lahore baselines ~3–4× london → models trained per city.
- **PM2.5 ↔ AQI** — near-monotonic; pm2.5 is the primary AQI driver (EPA breakpoints).

## 4. Feature engineering
Inputs (`src/feature_pipeline/feature_engineer.py`), all **leak-free** (only past info):
- **Time:** hour, day, month, dayofweek, is_weekend
- **Cyclical:** hour_sin/cos, month_sin/cos
- **Lags:** aqi_lag_{1,3,6,12,24}
- **Rolling:** mean/std over 6 & 24 h, max over 24 h (all shifted 1)
- **Derived:** aqi_change_rate, aqi_trend_3h
- **Interactions:** pm25_pm10_ratio, temp_humidity, wind_dispersion

**Targets (the key fix):** `create_forecast_targets()` builds `aqi_t+1h … aqi_t+72h` (AQI shifted −h). Models learn a **direct multi-horizon mapping** — one forward pass yields 72 distinct values, so the forecast genuinely varies (no flat nowcast, no autoregressive rollout).

## 5. Models & experiments
| Model | Type | Multi-output | Role |
|-------|------|--------------|------|
| **Ridge** (scaled, `MultiOutputRegressor`) | Statistical | wrapped | baseline |
| **Random Forest** (200 trees) | Ensemble | native | workhorse |
| **LSTM** (`Dense(72)` head) | Deep learning | native | seq→vector |

Validation: **5-fold `TimeSeriesSplit`** (no shuffling — respects time order). Metrics: **RMSE, MAE, R², MAPE**. Best model auto-selected by lowest RMSE and tracked in `latest.json["_best"]`.

### Measured run (london, 90-day synthetic backfill, offline)
| Model | RMSE | MAE | R² |
|-------|------|-----|-----|
| Ridge | 19.13 | 14.80 | −0.35 |
| **Random Forest (best)** | **17.26** | **13.78** | **0.07** |
| LSTM | — | — | (skipped: TensorFlow wheel absent on Py 3.14) |

R² is modest because the offline series is largely a seasonal random walk at a 72-h horizon; with live AQICN/OpenWeather data and longer history the signal-to-noise improves. The forecast variation check (σ ≈ 12 AQI over the 72-h curve, range 108–157) confirms a **real, non-flat** forecast.

## 6. Architecture (serverless)
```
AQICN + OpenWeather ─► Feature pipeline ─► Feature Store ─► Training pipeline ─► Model Registry
                                              (Hopsworks)                          (Hopsworks)
                                                   └──────► Inference 72h ──► Streamlit + Flask
```
- **Feature store + model registry:** Hopsworks free tier (cloud, source of truth); local SQLite/joblib fallback for dev/CI. **No CSV ever written.**
- **Compute:** GitHub Actions (no owned server).
- **Dashboard:** Streamlit Community Cloud.

## 7. CI/CD design
| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `ci.yml` | push / PR | tests-only, **no secrets → always green** |
| `feature_pipeline.yml` | hourly cron + dispatch | fetch → engineer → store |
| `training_pipeline.yml` | daily cron + dispatch | train → evaluate → registry |

Reliability: scheduled jobs **exit 0** on data gaps (warning, not failure); cloud store persists data across runs so training never hits "Not enough feature data" on a cold cache.

## 8. Dashboard
`app/streamlit_app.py`: current AQI + health band, 72-h forecast chart with ±10% interval and hazardous threshold line, 3-day min/mean/max summary, recent history, pollutant bars, SHAP feature importance, hazardous-hour alerts. `app/flask_api.py` exposes 7 REST endpoints.

## 9. Explainability & alerts
- **SHAP** (`src/utils/shap_explainer.py`) — top drivers: `aqi_change_rate`, `aqi_roll_max_24`, `aqi_trend_3h`, `wind_dispersion`.
- **Alerts** (`src/utils/alerts.py`) — EPA 6-level classification, health text, hazardous flag at AQI ≥ 150.

## 10. Limitations & future work
- Live history is synthetic without paid APIs; integrate a real historical air-quality archive (e.g. OpenAQ) for stronger training data.
- TensorFlow LSTM needs a supported Python (≤3.12 wheels); pin runtime to enable the deep model.
- Add per-horizon quantile models for calibrated intervals instead of a fixed ±10% band.
- Hyperparameter search (Optuna) and per-city model selection.
- Monitoring: data-drift + prediction-drift dashboards on the feature store.

## 11. Deliverables checklist
- ✅ End-to-end AQI prediction system
- ✅ Scalable automated pipeline (hourly feature + daily training)
- ✅ Interactive dashboard (real-time + 72-h forecast)
- ✅ This detailed report
- ✅ No CSV output · cloud-ready store + registry · real 3-day forecast · green CI
