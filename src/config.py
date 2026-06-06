"""Central config: env keys, cities, paths, horizons, feature windows, AQI thresholds."""
from __future__ import annotations

import os
from pathlib import Path

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent

# Load .env by ABSOLUTE path — cwd-independent (streamlit child procs differ).
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models" / "registry"
SQLITE_PATH = DATA_DIR / "feature_store.db"

for _d in (DATA_DIR, RAW_DIR, PROCESSED_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# API keys (from env; absent -> graceful synthetic fallback)
# ----------------------------------------------------------------------------
AQICN_API_TOKEN = os.getenv("AQICN_API_TOKEN", "").strip()
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "").strip()
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY", "").strip()
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT", "").strip()

USE_HOPSWORKS = bool(HOPSWORKS_API_KEY)

# ----------------------------------------------------------------------------
# Cities (name -> lat/lon for OpenWeather + AQICN station slug)
# ----------------------------------------------------------------------------
CITIES = {
    "london":   {"lat": 51.5074, "lon": -0.1278,  "aqicn": "london"},
    "new york": {"lat": 40.7128, "lon": -74.0060, "aqicn": "newyork"},
    "beijing":  {"lat": 39.9042, "lon": 116.4074, "aqicn": "beijing"},
    "delhi":    {"lat": 28.6139, "lon": 77.2090,  "aqicn": "delhi"},
    "paris":    {"lat": 48.8566, "lon": 2.3522,   "aqicn": "paris"},
    "karachi":  {"lat": 24.8607, "lon": 67.0011,  "aqicn": "karachi"},
    "lahore":   {"lat": 31.5204, "lon": 74.3587,  "aqicn": "lahore"},
}
CITY_NAMES = list(CITIES.keys())

# ----------------------------------------------------------------------------
# Forecast horizon + feature windows
# ----------------------------------------------------------------------------
FORECAST_HOURS = 72          # 3-day forecast
LOOKBACK_HOURS = 48          # LSTM input window
LAG_HOURS = [1, 3, 6, 12, 24]
ROLLING_WINDOWS = [6, 24]

# ----------------------------------------------------------------------------
# AQI levels (US EPA breakpoints) + alerting
# ----------------------------------------------------------------------------
AQI_LEVELS = [
    (0,   50,  "Good",                           "#00e400", "🟢"),
    (51,  100, "Moderate",                        "#ffff00", "🟡"),
    (101, 150, "Unhealthy for Sensitive Groups",  "#ff7e00", "🟠"),
    (151, 200, "Unhealthy",                        "#ff0000", "🔴"),
    (201, 300, "Very Unhealthy",                   "#8f3f97", "🟣"),
    (301, 500, "Hazardous",                        "#7e0023", "🟤"),
]
ALERT_THRESHOLD = 150        # AQI >= this -> hazardous alert

# Minimum rows required before a training run is meaningful.
MIN_TRAIN_ROWS = 200

# PM2.5 (ug/m3) -> AQI EPA breakpoints: (Clow, Chigh, Ilow, Ihigh)
PM25_BREAKPOINTS = [
    (0.0,   12.0,  0,   50),
    (12.1,  35.4,  51,  100),
    (35.5,  55.4,  101, 150),
    (55.5,  150.4, 151, 200),
    (150.5, 250.4, 201, 300),
    (250.5, 500.4, 301, 500),
]
