import pytest
pytest.importorskip("pyspark")

from datetime import datetime

from src.train import split_by_time


def test_time_splits_are_non_overlapping(spark):
    timestamps = [
        datetime(2024, 8, 31, 23), datetime(2024, 9, 1, 0),
        datetime(2024, 10, 31, 23), datetime(2024, 11, 1, 0),
        datetime(2024, 12, 31, 23),
    ]
    frame = spark.createDataFrame([(1, stamp) for stamp in timestamps],
                                 ["PULocationID", "pickup_hour"])
    splits = split_by_time(frame)
    assert {name: df.count() for name, df in splits.items()} == {
        "train": 1, "validation": 2, "test": 2,
    }
