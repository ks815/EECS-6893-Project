"""Historical replay dashboard; consumes saved Spark predictions, not raw trips."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import PREDICTIONS_DIR, REPORT_DIR, ZONE_GEOJSON_PATH, ZONE_LOOKUP_PATH

st.set_page_config(page_title="NYC Yellow Taxi Pickup Forecasts", page_icon="🚕", layout="wide")
st.title("🚕 NYC Yellow Taxi Pickup Forecasts")
st.caption("One-hour-ahead prediction · 2024 held-out historical replay · observed yellow taxi pickups")

@st.cache_data(show_spinner="Loading forecast results...")
def read_predictions(split: str) -> pd.DataFrame:
    directory = PREDICTIONS_DIR / split
    if not directory.exists():
        raise FileNotFoundError(f"{directory} missing. Run python -m src.train first.")
    frame = pd.read_parquet(directory)
    frame["pickup_hour"] = pd.to_datetime(frame["pickup_hour"])
    return frame


@st.cache_data
def read_lookup() -> pd.DataFrame:
    frame = pd.read_csv(ZONE_LOOKUP_PATH)
    return frame.loc[frame["LocationID"].between(1, 263), ["LocationID", "Borough", "Zone"]]


@st.cache_data
def read_metrics() -> pd.DataFrame:
    path = REPORT_DIR / "metrics.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def color_for_value(value: float, maximum: float) -> list[int]:
    if not np.isfinite(value) or value <= 0:
        return [220, 224, 229, 150]
    scale = np.log1p(value) / max(np.log1p(maximum), 0.00001)
    # A single-hue palette: low = pale blue, high = saturated navy.
    return [int(210 - 175 * scale), int(224 - 137 * scale),
            int(242 - 90 * scale), 195]


def map_geojson(map_frame: pd.DataFrame, value_column: str) -> dict:
    with ZONE_GEOJSON_PATH.open(encoding="utf-8") as f:
        data = json.load(f)
    values = dict(zip(map_frame["PULocationID"].astype(int), map_frame[value_column].astype(float)))
    max_value = max(values.values(), default=1.0)
    for feature in data.get("features", []):
        props = feature.setdefault("properties", {})
        raw_id = props.get("locationid", props.get("LocationID", props.get("location_id")))
        try:
            location_id = int(raw_id)
        except (TypeError, ValueError):
            location_id = -1
        number = values.get(location_id, 0.0)
        props["fill_color"] = color_for_value(number, max_value)
        props["pickup_value"] = round(number, 1)
        props["zone_label"] = props.get("zone", props.get("Zone", str(location_id)))
    return data


try:
    lookup = read_lookup()
except (FileNotFoundError, ValueError) as exc:
    st.error(f"Missing or invalid zone lookup: {exc}")
    st.stop()

split = st.sidebar.radio("Held-out period", options=["test", "validation"],
                         format_func=lambda s: "Nov–Dec 2024 (Test)" if s == "test" else "Sep–Oct 2024 (Validation)")
try:
    predictions = read_predictions(split)
except (FileNotFoundError, OSError, ValueError) as exc:
    st.error(str(exc))
    st.code("python scripts/download_data.py\npython -m src.etl\npython -m src.features\npython -m src.train\npython -m src.evaluate")
    st.stop()

predictions = predictions.merge(lookup, left_on="PULocationID", right_on="LocationID", how="left")
predictions["zone_display"] = (
    predictions["Borough"].fillna("Unknown") + " — " +
    predictions["Zone"].fillna("Zone " + predictions["PULocationID"].astype(str))
)

unique_zones = predictions[["PULocationID", "zone_display"]].drop_duplicates().sort_values("zone_display")
options = unique_zones["PULocationID"].astype(int).tolist()
labels = dict(zip(unique_zones["PULocationID"].astype(int), unique_zones["zone_display"]))
selected_zone = st.sidebar.selectbox("Taxi zone", options=options,
                                      format_func=lambda key: f"{labels[key]} (ID {key})")
all_dates = sorted(predictions["pickup_hour"].dt.date.unique().tolist())
selected_date = st.sidebar.selectbox("Historical forecast date", options=all_dates, index=len(all_dates) - 1)
selected_hour = st.sidebar.slider("Forecast hour (NYC wall-clock)", 0, 23, 17)
forecast_at = pd.Timestamp(selected_date) + pd.Timedelta(hours=selected_hour)
model_name = st.sidebar.selectbox("Map overlay", ["pickup_count", "baseline_pred", "linear_pred", "gbt_pred"],
                                  format_func=lambda name: {
                                      "pickup_count": "Actual pickups", "baseline_pred": "Weekly baseline",
                                      "linear_pred": "Linear Regression", "gbt_pred": "Gradient Boosted Trees",
                                  }[name])

selected_record = predictions.loc[
    (predictions["PULocationID"] == selected_zone) &
    (predictions["pickup_hour"] == forecast_at)
]
if selected_record.empty:
    st.warning("No observation exists for the selected hour in this replay dataset.")
    st.stop()
selected_record = selected_record.iloc[0]
st.subheader(f"{labels[selected_zone]} · {forecast_at:%b %d, %Y at %I:%M %p}")
columns = st.columns(4)
for col, label, name in zip(columns,
                            ["Actual pickups", "Last week", "Linear regression", "GBT"],
                            ["pickup_count", "baseline_pred", "linear_pred", "gbt_pred"]):
    col.metric(label, f"{selected_record[name]:,.0f}")

col_chart, col_map = st.columns([1.2, 1.0])
with col_chart:
    st.markdown("#### Past 7 days: actual vs model forecasts")
    history = predictions.loc[
        (predictions["PULocationID"] == selected_zone) &
        (predictions["pickup_hour"] <= forecast_at) &
        (predictions["pickup_hour"] > forecast_at - pd.Timedelta(days=7))
    ].sort_values("pickup_hour")
    chart = history.set_index("pickup_hour")[["pickup_count", "baseline_pred", "linear_pred", "gbt_pred"]]
    chart = chart.rename(columns={"pickup_count": "Actual", "baseline_pred": "Weekly baseline",
                                  "linear_pred": "Linear", "gbt_pred": "GBT"})
    st.line_chart(chart, height=420)
    st.caption("Forecasts at each historical hour use actual observations from earlier hours only.")

with col_map:
    st.markdown("#### NYC taxi-zone pickup map")
    map_frame = predictions.loc[predictions["pickup_hour"] == forecast_at]
    if ZONE_GEOJSON_PATH.exists():
        geometry = map_geojson(map_frame, model_name)
        deck = pdk.Deck(
            layers=[pdk.Layer("GeoJsonLayer", data=geometry, pickable=True,
                              stroked=True, filled=True, get_fill_color="properties.fill_color",
                              get_line_color=[105, 120, 130, 120], line_width_min_pixels=0.5)],
            initial_view_state=pdk.ViewState(latitude=40.71, longitude=-73.94, zoom=10, pitch=0),
            tooltip={"html": "<b>{zone_label}</b><br/>Pickups: {pickup_value}"},
            map_style=None,
        )
        st.pydeck_chart(deck, use_container_width=True)
        st.caption("Color intensity uses log scaling within the selected hour; pale gray = zero/no record.")
    else:
        st.info("Download taxi_zones.geojson to enable the choropleth map.")

metrics = read_metrics()
if not metrics.empty:
    st.markdown("#### Overall hold-out performance")
    summary = metrics.loc[(metrics["split"] == split) & (metrics["segment"] == "overall"),
                          ["model", "mae", "rmse", "n"]].copy()
    st.dataframe(summary.rename(columns={"model": "Model", "mae": "MAE", "rmse": "RMSE",
                                         "n": "Zone-hour observations"}), hide_index=True,
                 use_container_width=True)

st.caption("Source: NYC TLC 2024 Yellow Taxi Trip Records. These are reported pickups, not unmet travel demand.")
