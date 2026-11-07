"""One consistent local Spark session for the entire pipeline."""
from pyspark.sql import SparkSession


def make_spark(app_name: str, shuffle_partitions: int = 16) -> SparkSession:
    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")  # Remove/override on an actual cluster.
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark
