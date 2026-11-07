import pytest
pytest.importorskip("pyspark")

from datetime import datetime

from src.etl import aggregate_trip_counts, dense_hourly_panel


def test_invalid_records_filtered_and_zeros_filled(spark):
    trips = spark.createDataFrame([
        (datetime(2024, 1, 1, 0, 3), 1),
        (datetime(2024, 1, 1, 0, 55), 1),
        (datetime(2024, 1, 1, 1, 10), 999),
        (datetime(2023, 12, 31, 23, 59), 1),
    ], ["tpep_pickup_datetime", "PULocationID"])
    zones = spark.createDataFrame([(1,), (2,)], ["PULocationID"])
    start, end = "2024-01-01 00:00:00", "2024-01-01 03:00:00"
    observed = aggregate_trip_counts(trips, zones, start, end)
    panel = dense_hourly_panel(spark, observed, zones, start, end)
    assert panel.count() == 6
    counts = {(r["PULocationID"], r["pickup_hour"].hour): r["pickup_count"]
              for r in panel.collect()}
    assert counts[(1, 0)] == 2
    assert counts[(1, 1)] == 0
    assert counts[(2, 0)] == 0
