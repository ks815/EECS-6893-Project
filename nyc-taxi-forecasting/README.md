# NYC Taxi Pickup Forecasting and Visualization with Apache Spark

Columbia University — EECS E6893 Big Data Analytics | Project scaffold

## 1. Research objective

Predict **observed yellow taxi pickups** per NYC taxi zone for the next hour,
using past pickup counts and known calendar features. Determine whether models
beat the **same hour last week** baseline (seasonal naive, `lag_168`).

This is a **rolling one-hour-ahead historical replay**. For a prediction at time
`t`, counts for `t-1`, `t-24`, `t-168` and averages up to `t-1` are available.
Future hours and the target `pickup_count[t]` never enter `features[t]`.

## 2. System architecture

```text
2024 TLC yellow_tripdata_YYYY-MM.parquet (12 files)    TLC zone lookup + GeoJSON
           |                                                    |
           v                                                    v
       PySpark ETL  ------------------------------> Zone/hour lookup
           |   validate 2024 pickup time & mapped pickup zone
           |   count pickups per zone-hour; complete zero hours
           v
  data/processed/hourly_pickups/
           |
           v
       Spark Window Features (lag_1, lag_24, lag_168, 24/168h means)
           |
           v
  data/processed/features/
           |
           v
  Temporal splits: Train Jan-Aug | Val Sep-Oct | Test Nov-Dec
           |
           v
  Baseline lag168 | Spark ML Linear Regression | Spark ML GBT
           |
           +----------> saved models/ and validation/test predictions
                                  |
                                  +--> MAE/RMSE / peak / zone-volume report
                                  +--> Streamlit map + selected-hour replay

 Separate benchmark: 1/3/6/12 months Parquet read, filter and aggregation.
```

## 3. Project folders

```text
src/config.py             Paths, temporal boundaries and feature set
src/spark_utils.py        Spark session configuration
src/etl.py                Read, validate, count, complete zone × hour panel
src/features.py           Past-only lag/rolling features
src/train.py              MLlib preprocessing, Linear Regression, GBT, inference
src/evaluate.py           Distributed MAE and RMSE, peak and zone groups
src/scalability.py        Repeatable timed read/filter/aggregation runs
scripts/download_data.py  Official TLC downloads

dashboard/app.py          Interactive held-out replay, metrics and map
tests/                   Synthetic tests (no TLC files required)
docs/report_outline.md    Report chapters and figures

data/raw/                Downloaded 2024 yellow trip Parquets (gitignored)
data/reference/          Taxi zone lookup and GeoJSON (gitignored)
data/processed/          Derived Spark Parquets (gitignored)
data/predictions/        Validation and test predictions (gitignored)
models/                  Trained Spark ML models (gitignored)
reports/                 Metrics and benchmark results (gitignored)
```

## 4. Installation (macOS/Linux)

Use Python 3.10–3.12 and a compatible Java runtime (Java 17 recommended
for Spark 3.5). A MacBook with 16+ GB RAM is preferable for the full year;
large training jobs may be better on a Spark cluster or Dataproc.

```bash
cd nyc-taxi-forecasting
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
java -version
```

**Small-start data download** (January first, to validate ingestion):

```bash
python scripts/download_data.py --months 1
python -m src.etl --months 1
python -m src.features
pytest -q
```

This one-month run **does not produce trained/test models**: the research split
requires the full year.

**Full run** (several GB of input and resource-intensive model fitting):

```bash
python scripts/download_data.py
python -m src.etl --shuffle-partitions 16
python -m src.features --shuffle-partitions 16
python -m src.train --shuffle-partitions 16 --gbt-iterations 30
python -m src.evaluate
streamlit run dashboard/app.py
```

**Independent Spark scalability experiment:**

```bash
python -m src.scalability --months 1 3 6 12 --repeats 3 --warmup 1 --shuffle-partitions 16
```

The output artifacts `reports/metrics.csv` and `reports/scalability.csv` are
prepared for report tables/plots. The dashboard reads locally saved predictions;
no live ingestion or external API inference is required.

## 5. Feature table / data contract

| Field | Meaning | Known at time `t`? |
|---|---|---|
| `PULocationID` | Taxi zone for the target hour | Yes |
| `pickup_hour` | Beginning of target clock hour | Yes |
| `pickup_count` | **Observed target**, trips picked up in `t` | **No** (label only) |
| `hour_of_day`, `day_of_week`, `month`, `is_weekend` | Calendar | Yes |
| `lag_1` | Observed pickups at `t-1 hour` | Yes, after previous hour closed |
| `lag_24` | Observed pickups 24 hours before | Yes |
| `lag_168` | Observed pickups 168 hours before; baseline | Yes |
| `avg_last_24` | Mean pickups over `t-24 ... t-1` | Yes |
| `avg_last_168` | Mean pickups over `t-168 ... t-1` | Yes |

Training/validation/test boundaries are by **target hour**. The first 168
hours of January are omitted because a full week of preceding 2023 observations
was not downloaded. Calendar and historical features are built across the
entire panel *before* splitting. This does **not** leak future information because
all window frames end strictly at `t-1`.

The Linear and GBT models use `OneHotEncoder` for the zone, so a numeric zone ID
is not incorrectly treated as geographically ordered. Feature encoders fit on
**Jan–Aug training data only**. Model predictions are clipped to zero if negative;
results remain real-valued for MAE/RMSE, not rounded to integers. No tuning uses
test data; any hyperparameter selection should be based on validation only.

## 6. Metrics and experiment design

- Baseline: `y_hat(zone,t) = y(zone,t - 168h)`.
- `MAE = mean(|observed - predicted|)`.
- `RMSE = sqrt(mean((observed - predicted)^2))`.
- Peak-hour subgroup in scaffold: **07:00–10:59 and 17:00–20:59** NYC wall-clock.
- Low/medium/high zone groups use **train-only mean zone counts** and training
  tercile thresholds, never test-volume labels.
- Report validation and test separately; emphasize holdout test performance.
- If showing baseline-relative improvement, calculate `(baseline_MAE - model_MAE) / baseline_MAE`.
- The scalability benchmark measures **read + filter + groupBy/count** over
  different time horizons (not dense-panel fill or end-to-end training).
  It includes warmup and 3 recorded repeats; write the same machine, cores,
  RAM, partitions and Spark version in the final report. Results can vary with
  OS page cache, JVM warmup, disk, and other processes.

## 7. Data quality, edge cases, and scope

- NYC TLC reports trip records submitted by providers. They are an observable
  pickup proxy, **not unmet passenger demand**.
- Only pickup time and `PULocationID` are used to define the label; this avoids
  accidentally excluding valid pickups because payment/fare fields are unusual.
- The ETL restricts pickup **time** to the selected 2024 window, rather than
  trusting that every row of a 2024 monthly file is timestamped within 2024.
- Only mapped taxi zones 1–263 are scored. TLC IDs 264/265 (unknown/N/A) are
  excluded because they cannot be shown on the standard zone map.
- Zero records for a zone-hour become zero counts. This assumes missing
  reporting does not create false zeros; discuss this source limitation.
- **Daylight saving time (DST):** TLC timestamps are naive NYC-local times.
  The starter pipeline aggregates by wall-clock hour without disambiguating
  duplicate 01:00–01:59 on Nov 3, 2024, or the missing spring-forward hour on
  Mar 10, 2024. Therefore this is a **local-wall-clock approximation** and
  `lag_168` is a 168-row clock-hour lag, not an exact UTC elapsed-hour shift
  across DST. Flag/exclude DST boundary dates in a sensitivity analysis or
  explicitly resolve ambiguity using an additional authoritative timezone or
  timestamp source before claiming exact elapsed-hour correctness.
- **Forecast timing:** We assume completed, observed hourly counts are
  instantly available when forecasting the next hour. Delayed provider feeds
  would require longer lags in a deployed system.
- Dashboard is a **replay**, not streaming or a real-time forecast API.
- The scaffold does not include automatic hyperparameter search, caching
  optimizations for clusters, or data-ingestion anomaly dashboards yet.

## 8. Official input sources

- NYC TLC trip data portal: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
- 2024 January example: https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet
- Taxi zone lookup CSV: https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv
- NYC Open Data taxi zones: https://data.cityofnewyork.us/Transportation/NYC-Taxi-Zones/8meu-9t5y
- Geospatial JSON export: https://data.cityofnewyork.us/api/geospatial/8meu-9t5y?method=export&format=GeoJSON

## 9. Status of this starter

The repository provides **source code and tests**, not a claim of completed
experiments. At scaffold creation, syntax checking passed and the non-Spark
configuration test passed; PySpark integration tests could not execute in the
creation environment because PySpark was not installed. The 2024 TLC files, trained models, metric tables, and screenshots
are not included. Run the tests and pipeline on your own machine/cluster and
inspect actual results before submitting a report.
