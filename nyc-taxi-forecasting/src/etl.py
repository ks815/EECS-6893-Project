"""Raw Parquet -> observed hourly counts -> dense zone-hour panel.

TLC publishes naive NYC-local pickup timestamps. This scaffold represents them
as wall-clock timestamps with a UTC Spark session for reproducibility; see the
README for the DST collision limitation in March/November 2024.
"""
import argparse
from datetime import datetime
from pathlib import Path

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src.config import HOURLY_PATH, RAW_DIR, ZONE_LOOKUP_PATH, YEAR
from src.spark_utils import make_spark


def month_paths(months: list[int]) -> list[str]:
    paths = [RAW_DIR / f"yellow_tripdata_{YEAR}-{month:02d}.parquet" for month in months]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Parquet files: {missing}. Run scripts/download_data.py first.")
    return [str(p) for p in paths]


def load_trips(spark, months: list[int]) -> DataFrame:
    # Read each file separately: other columns may have schema changes.
    dataframes = [
        spark.read.parquet(path).select("tpep_pickup_datetime", "PULocationID")
        for path in month_paths(months)
    ]
    trips = dataframes[0]
    for frame in dataframes[1:]:
        trips = trips.unionByName(frame)
    return trips


def taxi_zones(spark) -> DataFrame:
    if not ZONE_LOOKUP_PATH.is_file():
        raise FileNotFoundError(f"Missing lookup: {ZONE_LOOKUP_PATH}")
    return (
        spark.read.option("header", True).csv(str(ZONE_LOOKUP_PATH))
        .select(F.col("LocationID").cast("int").alias("PULocationID"),
                F.col("Borough"), F.col("Zone"))
        .where(F.col("PULocationID").between(1, 263))
        .dropDuplicates(["PULocationID"])
    )


def aggregate_trip_counts(trips: DataFrame, zone_ids: DataFrame,
                          start: str, end: str) -> DataFrame:
    """Count only valid timestamp/zone records in [start, end)."""
    valid = (
        trips.select(F.col("PULocationID").cast("int").alias("PULocationID"),
                     F.col("tpep_pickup_datetime").cast("timestamp").alias("pickup_time"))
        .where(F.col("pickup_time").isNotNull())
        .where((F.col("pickup_time") >= F.lit(start).cast("timestamp")) &
               (F.col("pickup_time") < F.lit(end).cast("timestamp")))
        .join(F.broadcast(zone_ids.select("PULocationID")), "PULocationID", "inner")
    )
    return (
        valid.groupBy("PULocationID", F.date_trunc("hour", "pickup_time").alias("pickup_hour"))
        .agg(F.count(F.lit(1)).cast("long").alias("pickup_count"))
    )


def dense_hourly_panel(spark, observed: DataFrame, zone_ids: DataFrame,
                       start: str, end: str) -> DataFrame:
    """Materialize zero-pickup hours for each official mapped taxi zone."""
    # A dense panel is necessary for lag_168 to mean exactly 168 clock-hour rows.
    hours = spark.range(1).select(
        F.explode(F.sequence(F.to_timestamp(F.lit(start)),
                             F.to_timestamp(F.lit(end)) - F.expr("INTERVAL 1 HOUR"),
                             F.expr("INTERVAL 1 HOUR"))).alias("pickup_hour")
    )
    return (
        zone_ids.select("PULocationID").crossJoin(hours)
        .join(observed, ["PULocationID", "pickup_hour"], "left")
        .fillna({"pickup_count": 0})
        .select("PULocationID", "pickup_hour", F.col("pickup_count").cast("long"))
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build hourly zone-level pickup counts")
    parser.add_argument("--months", nargs="+", type=int, default=list(range(1, 13)))
    parser.add_argument("--shuffle-partitions", type=int, default=16)
    args = parser.parse_args()
    months = sorted(set(args.months))
    if not months or months != list(range(1, max(months) + 1)):
        parser.error("ETL requires consecutive months starting at January: 1 2 ... N")
    end = datetime(YEAR + 1, 1, 1) if max(months) == 12 else datetime(YEAR, max(months) + 1, 1)
    start_str = f"{YEAR}-01-01 00:00:00"
    end_str = end.strftime("%Y-%m-%d %H:%M:%S")
    spark = make_spark("NYC Taxi Hourly ETL", args.shuffle_partitions)
    try:
        zones = taxi_zones(spark)
        trips = load_trips(spark, months)
        observed = aggregate_trip_counts(trips, zones, start_str, end_str)
        panel = dense_hourly_panel(spark, observed, zones, start_str, end_str)
        panel.write.mode("overwrite").parquet(str(HOURLY_PATH))
        print(f"Wrote {spark.read.parquet(str(HOURLY_PATH)).count():,} zone-hours to {HOURLY_PATH}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
