"""Small, self-contained Spark tests; no large TLC download required."""
import pytest

@pytest.fixture(scope="session")
def spark():
    pytest.importorskip("pyspark")
    from pyspark.sql import SparkSession
    session = (
        SparkSession.builder.master("local[2]").appName("TaxiProjectTests")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    yield session
    session.stop()
