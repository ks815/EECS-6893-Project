"""Fit Spark ML linear regression and GBT; replay one-hour-ahead forecasts.

For each target hour, lag/rolling features use ACTUAL counts observed in earlier
hours, including earlier validation/test hours. This is rolling one-step-ahead
evaluation, not recursive multi-step forecasting.
"""
import argparse

from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.regression import GBTRegressor, LinearRegression
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src.config import (FEATURE_PATH, MODEL_DIR, NUMERIC_FEATURES,
                        PREDICTIONS_DIR, TEST_END, TRAIN_END, VAL_END)
from src.spark_utils import make_spark


def split_by_time(frame: DataFrame) -> dict[str, DataFrame]:
    return {
        "train": frame.where(F.col("pickup_hour") < F.lit(TRAIN_END).cast("timestamp")),
        "validation": frame.where((F.col("pickup_hour") >= F.lit(TRAIN_END).cast("timestamp")) &
                                  (F.col("pickup_hour") < F.lit(VAL_END).cast("timestamp"))),
        "test": frame.where((F.col("pickup_hour") >= F.lit(VAL_END).cast("timestamp")) &
                            (F.col("pickup_hour") < F.lit(TEST_END).cast("timestamp"))),
    }


def preprocessor() -> Pipeline:
    indexer = StringIndexer(inputCol="zone_string", outputCol="zone_index", handleInvalid="keep")
    one_hot = OneHotEncoder(inputCols=["zone_index"], outputCols=["zone_vector"], handleInvalid="keep")
    assembler = VectorAssembler(inputCols=["zone_vector"] + NUMERIC_FEATURES,
                                outputCol="features", handleInvalid="error")
    return Pipeline(stages=[indexer, one_hot, assembler])


def add_zone_string(frame: DataFrame) -> DataFrame:
    return frame.withColumn("zone_string", F.col("PULocationID").cast("string"))


def forecast(frame: DataFrame, linear_model, gbt_model) -> DataFrame:
    keys = ["PULocationID", "pickup_hour"]
    base = frame.select(*keys, "pickup_count", F.col("lag_168").alias("baseline_pred"))
    linear = linear_model.transform(frame).select(
        *keys, F.greatest(F.lit(0.0), F.col("prediction")).alias("linear_pred"))
    gbt = gbt_model.transform(frame).select(
        *keys, F.greatest(F.lit(0.0), F.col("prediction")).alias("gbt_pred"))
    return base.join(linear, keys).join(gbt, keys)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shuffle-partitions", type=int, default=16)
    parser.add_argument("--gbt-iterations", type=int, default=30)
    args = parser.parse_args()
    spark = make_spark("NYC Taxi Spark ML Training", args.shuffle_partitions)
    try:
        raw = add_zone_string(spark.read.parquet(str(FEATURE_PATH)))
        splits = split_by_time(raw)
        counts = {key: data.count() for key, data in splits.items()}
        print("Rows by temporal split:", counts)
        if min(counts.values()) == 0:
            raise ValueError("All Jan-Dec 2024 data required for the defined train/validation/test split")
        trained_preprocessor = preprocessor().fit(splits["train"])
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        trained_preprocessor.write().overwrite().save(str(MODEL_DIR / "preprocessor"))

        training = trained_preprocessor.transform(splits["train"]).cache()
        print("Caching training feature vectors:", training.count())
        linear = LinearRegression(
            featuresCol="features", labelCol="pickup_count",
            maxIter=60, regParam=0.1, elasticNetParam=0.0,
        ).fit(training)
        linear.write().overwrite().save(str(MODEL_DIR / "linear"))
        print("Saved Linear Regression model")

        gbt = GBTRegressor(
            featuresCol="features", labelCol="pickup_count", maxIter=args.gbt_iterations,
            maxDepth=5, maxBins=64, stepSize=0.1, subsamplingRate=0.7, seed=42,
        ).fit(training)
        gbt.write().overwrite().save(str(MODEL_DIR / "gbt"))
        print("Saved GBT model")
        training.unpersist()

        for name in ("validation", "test"):
            prepared = trained_preprocessor.transform(splits[name])
            predictions = forecast(prepared, linear, gbt)
            output = PREDICTIONS_DIR / name
            predictions.write.mode("overwrite").parquet(str(output))
            print(f"Wrote {name} predictions -> {output}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
