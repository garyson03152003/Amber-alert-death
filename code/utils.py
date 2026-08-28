"""Shared utilities used across pipeline scripts."""

import logging
import time
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd
import requests


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
                                datefmt="%H:%M:%S")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def download_file(
    url: str,
    dest: Path,
    session: Optional[requests.Session] = None,
    retries: int = 5,
    backoff: float = 2.0,
    chunk_size: int = 1 << 20,  # 1 MB
) -> Path:
    """
    Download *url* to *dest*, skipping if the file already exists.
    Retries with exponential back-off on transient errors.
    """
    log = get_logger("utils.download")
    if dest.exists():
        log.info("Already downloaded: %s", dest.name)
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    sess = session or requests.Session()

    for attempt in range(retries):
        try:
            log.info("Downloading %s  →  %s", url, dest)
            resp = sess.get(url, stream=True, timeout=120)
            resp.raise_for_status()
            with open(dest, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    fh.write(chunk)
            log.info("Saved %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
            return dest
        except (requests.RequestException, OSError) as exc:
            wait = backoff ** attempt
            log.warning("Attempt %d failed (%s); retrying in %.0fs", attempt + 1, exc, wait)
            time.sleep(wait)

    raise RuntimeError(f"Failed to download {url} after {retries} attempts")


def fips5(state: int | str, county: int | str) -> str:
    """Return zero-padded 5-digit FIPS code from state + county components."""
    return f"{int(state):02d}{int(county):03d}"


def build_county_timezone_map(gaz_zip: Path) -> dict[str, str]:
    """
    Return a dict mapping 5-digit county FIPS → IANA timezone name.

    Uses county population-weighted centroids from the Census gazetteer and
    timezonefinder to assign the correct IANA tz to each county.  Counties
    that span a tz boundary get the timezone of their centroid.

    Parameters
    ----------
    gaz_zip : path to 2023_Gaz_counties_national.zip (already downloaded)
    """
    from timezonefinder import TimezoneFinder

    log = get_logger("utils.tz")

    # Load county centroids from Census gazetteer
    with zipfile.ZipFile(gaz_zip) as zf:
        fname = [n for n in zf.namelist() if n.endswith(".txt")][0]
        with zf.open(fname) as f:
            counties = pd.read_csv(f, sep="\t", dtype=str)
    counties.columns = [c.strip() for c in counties.columns]
    counties["fips"] = counties["GEOID"].str.zfill(5)
    counties["lat"]  = pd.to_numeric(counties["INTPTLAT"],  errors="coerce")
    counties["lon"]  = pd.to_numeric(counties["INTPTLONG"], errors="coerce")
    counties = counties[["fips", "lat", "lon"]].dropna()

    tf = TimezoneFinder()
    log.info("Looking up IANA timezone for %d county centroids...", len(counties))

    tz_map = {}
    for _, row in counties.iterrows():
        tz = tf.timezone_at(lat=row["lat"], lng=row["lon"])
        if tz:
            tz_map[row["fips"]] = tz

    log.info("Timezone map built: %d counties, %d unique zones",
             len(tz_map), len(set(tz_map.values())))
    return tz_map
