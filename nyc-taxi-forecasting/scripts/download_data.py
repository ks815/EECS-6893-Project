"""Download official TLC data; downloads are opt-in because files are large."""
import argparse
import sys
import time
import urllib.request
from pathlib import Path

# Allow direct execution via `python scripts/download_data.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import RAW_DIR, REFERENCE_DIR, YEAR

DATA_ROOT = "https://d37ci6vzurychx.cloudfront.net"
GEOJSON_URL = (
    "https://data.cityofnewyork.us/api/geospatial/8meu-9t5y"
    "?method=export&format=GeoJSON"
)


def download(url: str, destination: Path, retries: int = 3) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        print(f"Already exists: {destination}")
        return
    temporary = destination.with_name(destination.name + ".part")
    for attempt in range(retries):
        try:
            print(f"Downloading {url} -> {destination}", flush=True)
            request = urllib.request.Request(url, headers={"User-Agent": "NYC-Taxi-Course-Project/1.0"})
            with urllib.request.urlopen(request, timeout=90) as response, temporary.open("wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
            if temporary.stat().st_size == 0:
                raise ValueError("Downloaded file is empty")
            temporary.replace(destination)
            return
        except Exception as exc:
            temporary.unlink(missing_ok=True)
            if attempt == retries - 1:
                raise RuntimeError(f"Failed downloading {url}: {exc}") from exc
            time.sleep(2 ** attempt)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--months", nargs="+", type=int, default=list(range(1, 13)))
    parser.add_argument("--skip-geometry", action="store_true")
    args = parser.parse_args()
    if any(month not in range(1, 13) for month in args.months):
        parser.error("Months must be 1-12")

    for month in sorted(set(args.months)):
        filename = f"yellow_tripdata_{YEAR}-{month:02d}.parquet"
        download(f"{DATA_ROOT}/trip-data/{filename}", RAW_DIR / filename)
    download(f"{DATA_ROOT}/misc/taxi_zone_lookup.csv", REFERENCE_DIR / "taxi_zone_lookup.csv")
    if not args.skip_geometry:
        try:
            download(GEOJSON_URL, REFERENCE_DIR / "taxi_zones.geojson")
        except RuntimeError as exc:
            print(f"Warning: GeoJSON optional; map will be disabled: {exc}", file=sys.stderr)
    print("Download phase complete.")


if __name__ == "__main__":
    main()
