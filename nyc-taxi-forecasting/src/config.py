"""Shared paths, immutable data splits, and the feature contract."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
REFERENCE_DIR = ROOT / "data" / "reference"
PROCESSED_DIR = ROOT / "data" / "processed"
PREDICTIONS_DIR = ROOT / "data" / "predictions"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
YEAR = 2024

HOURLY_PATH = PROCESSED_DIR / "hourly_pickups"
FEATURE_PATH = PROCESSED_DIR / "features"
ZONE_LOOKUP_PATH = REFERENCE_DIR / "taxi_zone_lookup.csv"
ZONE_GEOJSON_PATH = REFERENCE_DIR / "taxi_zones.geojson"

# Each row forecasts pickups in [pickup_hour, pickup_hour + 1h).
# Temporal split is by *forecast target time*, not by trip file/month name.
TRAIN_END = "2024-09-01 00:00:00"
VAL_END = "2024-11-01 00:00:00"
TEST_END = "2025-01-01 00:00:00"

NUMERIC_FEATURES = [
    "hour_of_day", "day_of_week", "month", "is_weekend",
    "lag_1", "lag_24", "lag_168", "avg_last_24", "avg_last_168",
]
FORECAST_COLUMNS = [
    "PULocationID", "pickup_hour", "pickup_count", "baseline_pred",
    "linear_pred", "gbt_pred",
]
