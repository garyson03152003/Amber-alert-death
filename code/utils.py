"""Shared utilities used across pipeline scripts."""

import logging
import time
from pathlib import Path
from typing import Optional

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
