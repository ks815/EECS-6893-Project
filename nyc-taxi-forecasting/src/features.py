"""Leakage-free, past-only Spark SQL window features."""
import argparse

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from src.config import FEATURE_PATH, HOURLY_PATH
from src.spark_utils import make_spark


def add_time_features(panel: DataFrame) -> DataFrame:
    window = Window.partitionBy("PULocationID").orderBy("pickup_hour")
    result = (
        panel
        .withColumn("hour_of_day", F.hour("pickup_hour").cast("double"))
        .withColumn("day_of_week", F.dayofweek("pickup_hour").cast("double"))
        .withColumn("month", F.month("pickup_hour").cast("double"))
        .withColumn("is_weekend", F.when(F.dayofweek("pickup_hour").isin(1, 7), 1.0).otherwise(0.0))
        .withColumn("lag_1", F.lag("pickup_count", 1).over(window).cast("double"))
        .withColumn("lag_24", F.lag("pickup_count", 24).over(window).cast("double"))
        .withColumn("lag_168", F.lag("pickup_count", 168).over(window).cast("double"))
        .withColumn("avg_last_24", F.avg("pickup_count").over(window.rowsBetween(-24, -1)))
        .withColumn("avg_last_168", F.avg("pickup_count").over(window.rowsBetween(-168, -1)))
    )
    # Jan 1-7 2024 have no 168h history; drop those rows in every split.
    return result.where(F.col("lag_168").isNotNull())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shuffle-partitions", type=int, default=16)
    args = parser.parse_args()
    spark = make_spark("NYC Taxi Feature Engineering", args.shuffle_partitions)
    try:
        panel = spark.read.parquet(str(HOURLY_PATH))
        features = add_time_features(panel)
        features.write.mode("overwrite").parquet(str(FEATURE_PATH))
        print(f"Wrote {spark.read.parquet(str(FEATURE_PATH)).count():,} feature rows to {FEATURE_PATH}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
