"""Measure comparable cold, uncached Parquet read + filter + aggregation jobs.

One sample is Spark action wall time. This is not a pure ETL CPU-time metric.
"""
import argparse
import csv
import os
import platform
import time
from datetime import datetime

from src.config import RAW_DIR, REPORT_DIR, YEAR
from src.etl import aggregate_trip_counts, load_trips, taxi_zones
from src.spark_utils import make_spark


def benchmark(spark, n_months: int, zones) -> tuple[float, int]:
    spark.catalog.clearCache()
    start = time.perf_counter()
    trips = load_trips(spark, list(range(1, n_months + 1)))
    end = (f"{YEAR + 1}-01-01 00:00:00" if n_months == 12 else
           f"{YEAR}-{n_months + 1:02d}-01 00:00:00")
    grouped = aggregate_trip_counts(trips, zones, f"{YEAR}-01-01 00:00:00", end)
    n_rows = grouped.count()  # Trigger materialization; exclude zero fill/feature step.
    return time.perf_counter() - start, n_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--months", nargs="+", type=int, default=[1, 3, 6, 12])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--shuffle-partitions", type=int, default=16)
    args = parser.parse_args()
    if any(m not in range(1, 13) for m in args.months):
        parser.error("Month horizons must be in 1..12")
    if args.repeats < 1 or args.warmup < 0:
        parser.error("repeats >= 1 and warmup >= 0 required")
    spark = make_spark("NYC Taxi Scalability Benchmark", args.shuffle_partitions)
    try:
        zones = taxi_zones(spark).cache()
        zones.count()
        rows = []
        for n in args.months:
            input_bytes = sum((RAW_DIR / f"yellow_tripdata_{YEAR}-{m:02d}.parquet").stat().st_size
                              for m in range(1, n + 1))
            for rep in range(-args.warmup, args.repeats):
                seconds, output_rows = benchmark(spark, n, zones)
                print(f"{n} months / repeat {rep}: {seconds:.2f}s / {output_rows} rows", flush=True)
                if rep < 0:
                    continue
                rows.append({
                    "utc_run_time": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    "months": n, "repeat": rep + 1, "seconds": round(seconds, 3),
                    "output_rows": output_rows, "input_bytes": input_bytes,
                    "spark_version": spark.version,
                    "shuffle_partitions": args.shuffle_partitions,
                    "spark_master": spark.sparkContext.master,
                    "cpu_count": os.cpu_count(), "platform": platform.platform(),
                    "python_version": platform.python_version(),
                })
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        output = REPORT_DIR / "scalability.csv"
        with output.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote scalability measurements to {output}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
