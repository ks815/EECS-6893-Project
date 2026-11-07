# EECS E6893 — Suggested Final Report Structure

1. **Introduction & research question**: observed pickup counts, not latent demand; compare to last-week baseline.
2. **Dataset & data quality**: 2024 TLC yellow taxi records; zone lookup; timestamp and location validity; missing/outlier reporting; DST limitations; zone-hour coverage.
3. **Distributed ETL in Spark**: monthly ingestion; selective schema; aggregation; zero-hour completion; data footprint; partition decisions; lineage.
4. **Problem setup & leakage controls**: one-hour-ahead rolling replay; training Jan–Aug, validation Sep–Oct, final holdout Nov–Dec; exact feature timing.
5. **Models**: seasonal naive baseline, Linear Regression, Gradient-Boosted Trees. Document hyperparameters and training budget.
6. **Results**: validation/test MAE & RMSE; error differences and relative improvement from baseline; peak vs off-peak; volume groups defined using train only.
7. **Spark scalability**: one/three/six/twelve months under fixed CPU/RAM/Spark settings, per-run timing, mean ± SD, input bytes; explain non-linear speedups and I/O cache caveat.
8. **Dashboard & replay demonstration**: screenshot zone selector, date/hour, historical forecast comparison chart and map.
9. **Discussion**: weak zones, outliers, calendar effects, DST, data-revision uncertainty and forecast-availability assumptions.
10. **Conclusions & future work**: weather, events, better models, live ingestion (outside scope).

## Suggested visual figures

- End-to-end architecture diagram.
- 24-hour pickup pattern across weekdays/weekends, grouped by borough.
- Holdout MAE/RMSE by model (bars, one for each metric).
- Actual vs predicted time series for Midtown and a low-volume zone.
- Peak/off-peak model error comparison.
- Scalability: runtime vs number of months with repeat variance.
- Dashboard screenshot with zone-specific choropleth.

**Do not claim that the more complex model wins until validated on the untouched test period.**
