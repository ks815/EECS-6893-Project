import pytest
pytest.importorskip("pyspark")

from datetime import datetime, timedelta

from src.features import add_time_features


def test_lags_use_only_prior_hours(spark):
    start = datetime(2024, 1, 1)
    rows = [(1, start + timedelta(hours=h), h) for h in range(200)]
    rows += [(2, start + timedelta(hours=h), h * 10) for h in range(200)]
    panel = spark.createDataFrame(rows, ["PULocationID", "pickup_hour", "pickup_count"])
    result = add_time_features(panel)
    assert result.count() == 2 * (200 - 168)
    row = result.where("PULocationID = 1 and pickup_hour = '2024-01-08 00:00:00'").first()
    assert row["pickup_count"] == 168
    assert row["lag_1"] == 167
    assert row["lag_24"] == 144
    assert row["lag_168"] == 0
    assert abs(row["avg_last_24"] - 155.5) < 1e-9
    assert abs(row["avg_last_168"] - 83.5) < 1e-9
    other = result.where("PULocationID = 2 and pickup_hour = '2024-01-08 00:00:00'").first()
    assert other["lag_1"] == 1670
