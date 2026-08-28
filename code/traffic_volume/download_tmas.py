"""Download FHWA TMAS station-hour traffic volume data.

Source: https://www.fhwa.dot.gov/policyinformation/tables/tmasdata/
Each year has one annual station-description zip and 12 monthly
continuous-count-station (CCS) volume zips. Files before 2020 are a fixed-
width legacy format; 2020 onward is pipe-delimited. Both are handled by
code/traffic_volume/parse_tmas.py.

Usage: python3 download_tmas.py [--years 2013-2024]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import get_logger

log = get_logger("download_tmas")

BASE_URL = "https://www.fhwa.dot.gov/policyinformation/tables/tmasdata"
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; research)"}
MONTHS = [
    "jan", "feb", "mar", "apr", "may", "jun",
    "jul", "aug", "sep", "oct", "nov", "dec",
]

RAW_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "tmas"


def _download(session: requests.Session, url: str, dest: Path, *, max_attempts: int = 4) -> bool:
    if dest.is_file() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(max_attempts):
        try:
            with session.get(url, headers=HEADERS, timeout=180, stream=True) as r:
                if r.status_code == 404:
                    log.warning("404 (not published): %s", url)
                    return False
                r.raise_for_status()
                tmp = dest.with_suffix(dest.suffix + ".part")
                with open(tmp, "wb") as fh:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        fh.write(chunk)
                tmp.rename(dest)
            return True
        except Exception as exc:
            wait = (5, 20, 60, 120)[min(attempt, 3)]
            log.warning("  attempt %d failed for %s: %s (retry in %ds)", attempt + 1, url, exc, wait)
            time.sleep(wait)
    log.error("  giving up on %s after %d attempts", url, max_attempts)
    return False


def download_year(session: requests.Session, year: int) -> dict[str, bool]:
    results: dict[str, bool] = {}
    station_url = f"{BASE_URL}/{year}/{year}_station_data.zip"
    station_dest = RAW_DIR / str(year) / f"{year}_station_data.zip"
    results["station"] = _download(session, station_url, station_dest)

    for month in MONTHS:
        vol_url = f"{BASE_URL}/{year}/{month}_{year}_ccs_data.zip"
        vol_dest = RAW_DIR / str(year) / f"{month}_{year}_ccs_data.zip"
        results[month] = _download(session, vol_url, vol_dest)
        time.sleep(0.5)
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", default="2013-2024",
                         help="Year range, e.g. 2013-2024")
    args = parser.parse_args(argv)
    start, end = (int(x) for x in args.years.split("-"))
    years = list(range(start, end + 1))

    session = requests.Session()
    for year in years:
        log.info("=== %d ===", year)
        results = download_year(session, year)
        ok = sum(results.values())
        log.info("  %d/%d files obtained for %d", ok, len(results), year)
    session.close()


if __name__ == "__main__":
    main()
