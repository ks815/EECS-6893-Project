# Implementation Plan / Suggested milestones

## Milestone 1 — Data pipeline sanity check

- Download January 2024 + zone reference; verify Parquet fields and file schema.
- Validate timestamps, pickup location IDs, and daily/hourly pickup distributions.
- Run `python -m src.etl --months 1` then `python -m src.features`.
- Acceptance: approximately 263 × number of calendar hours in loaded months; raw spot-checks match Spark aggregation; no future values in the lag features.

## Milestone 2 — Full-year aggregation and features

- Download remaining 2024 months and run the full ETL.
- Verify full 2024 has 8,784 local wall-clock calendar hours before lag truncation (366 leap-year days × 24); examine DST days separately.
- Verify missing combinations are filled zero, not dropped; confirm each zone/hour has one target.
- Acceptance: one row per zone-hour with no missing lag values after 168-hour warmup.

## Milestone 3 — Training, validation, holdout test

- Weekly seasonal naive baseline: last week's observed count.
- Train one-hot zonal Linear Regression and Spark GBT on Jan–Aug only.
- Compare on Sep–Oct; adjust GBT iterations/regularization only based on validation, not test.
- Evaluate untouched Nov–Dec holdout; compute MAE/RMSE, peak/off-peak, high/medium/low volume zones.
- Acceptance: report factual scores and whether models beat last-week baseline (do not assume they do).

## Milestone 4 — Scalability and visualization

- Benchmark the SAME Spark read/filter/groupBy operation using 1/3/6/12 months, 3 repeats, a recorded machine, fixed partitions.
- Record CPU, RAM, Spark version, partitions, SSD vs HDD, Spark application/runtime.
- Launch Streamlit and replay a test-day hour with mapped actuals / model forecasts and a zone-specific chart.
- Acceptance: benchmark CSV with repeats and working dashboard screenshot.

## Milestone 5 — Report + demo

- Draw pipeline graph, show high-level feature & data contract, compare validation and test.
- Illustrate both a high-volume and a low-volume zone, and at least one failure mode.
- Discuss observed pickups vs unmet demand, DST ambiguity, feed latency assumptions, missing reporting vs true zero.
- Prepare a 3–5 minute demo: choose zone -> choose historical time -> inspect forecasts -> map -> show model metrics -> show scalability.

## Possible optional extensions

1. Holiday features (known calendar only) and event calendar (timestamped as-of forecasts).
2. Additional lags like 336 hours, and matched local hour/day-of-week mean history.
3. Training with a shorter in-sample horizon to measure trend drift.
4. DST-sensitive evaluation that excludes/reconciles March 10 & November 3, 2024, and checks effect on baseline comparisons.

*Priority*: finish correct baseline + ETL + metrics before adding complex GBT tuning or elaborate map animations.
