"""
build_acs_tract_car_share.py
=============================================================
Real ACS car-mode-share at the CENSUS TRACT level (B08301), for use as
the car-share input to the commuting dosage instead of the NHTS
national/MSASIZE-bucketed distance curve -- this is genuine local data
tied to actual geography rather than a modeled curve.

Why this supersedes the NHTS-curve approach: NHTS's public microdata
masks the respondent's specific metro area (CBSA), so the best we could
do there was bucket by a 6-category MSASIZE proxy. ACS B08301 is
published directly at the tract level (~85,000 tracts nationally), and
our LODES pipeline already tells us the exact home-tract for every
commuting flow -- so we can join REAL tract-level car share directly
onto REAL commuting pairs, no distance-curve inference needed.

Caveat: ACS B08301 tabulates commuting by RESIDENCE tract, not
workplace tract -- there is no standard "workplace area" car-mode-share
product from the Census Bureau (their workplace-side products, e.g.
LODES's WAC files, carry job counts and characteristics but not
transportation mode). So this is necessarily a HOME-tract car share,
which is also the behaviorally correct choice: mode choice (car
ownership, whether transit is available near home) is primarily a
property of where a commuter lives, not where they work.

Method: same Census ACS 2020 5-year Summary File bulk download already
used for county-level car share (01e_fetch_car_commuters.py) and county
cellular connectivity (01f_fetch_cell_connectivity.py), just requesting
the "*_Tracts_Block_Groups_Only.zip" per-state file (which the county
scripts deliberately skip, using "*_Not_Tracts_Block_Groups.zip"
instead) and filtering the geo header to summary level 140 (census
tract) rather than 050 (county). No Census API key needed (the bulk
summary files are public static downloads); the api.census.gov API
route was tried first and requires a registered key we don't have.

Output: data/processed/tract_car_share.parquet
  Columns: tract (str11, state+county+tract FIPS), total_workers,
  car_total, car_share
"""
import io
import sys
import warnings
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_PROC
from utils import get_logger

warnings.filterwarnings("ignore")
log = get_logger("acs_tract_car_share")

ACS_YEAR = 2020
OUT_PATH = DATA_PROC / "tract_car_share.parquet"
BASE_URL = f"https://www2.census.gov/programs-surveys/acs/summary_file/{ACS_YEAR}/data/5_year_by_state"
SEQ_NUM = "0027"  # B08301, confirmed in 01e_fetch_car_commuters.py for this vintage
GEO_SUMLEVEL_TRACT = "140"

STATES = {
    "Alabama": "al", "Alaska": "ak", "Arizona": "az", "Arkansas": "ar",
    "California": "ca", "Colorado": "co", "Connecticut": "ct", "Delaware": "de",
    "DistrictOfColumbia": "dc", "Florida": "fl", "Georgia": "ga", "Hawaii": "hi",
    "Idaho": "id", "Illinois": "il", "Indiana": "in", "Iowa": "ia",
    "Kansas": "ks", "Kentucky": "ky", "Louisiana": "la", "Maine": "me",
    "Maryland": "md", "Massachusetts": "ma", "Michigan": "mi", "Minnesota": "mn",
    "Mississippi": "ms", "Missouri": "mo", "Montana": "mt", "Nebraska": "ne",
    "Nevada": "nv", "NewHampshire": "nh", "NewJersey": "nj", "NewMexico": "nm",
    "NewYork": "ny", "NorthCarolina": "nc", "NorthDakota": "nd", "Ohio": "oh",
    "Oklahoma": "ok", "Oregon": "or", "Pennsylvania": "pa", "RhodeIsland": "ri",
    "SouthCarolina": "sc", "SouthDakota": "sd", "Tennessee": "tn", "Texas": "tx",
    "Utah": "ut", "Vermont": "vt", "Virginia": "va", "Washington": "wa",
    "WestVirginia": "wv", "Wisconsin": "wi", "Wyoming": "wy",
}


def parse_acs_int(val):
    try:
        f = float(str(val).replace(",", "").strip())
        return int(f) if f >= 0 else np.nan
    except Exception:
        return np.nan


def _download_with_retry(url: str, attempts: int = 3) -> bytes:
    # Plain urllib intermittently truncates large (100MB+) state files
    # through this environment's proxy (IncompleteRead, not fixed by
    # retrying urllib itself) -- curl has been reliable all session for
    # similarly large downloads, so shell out to it instead.
    import subprocess
    import tempfile
    last_exc = None
    for i in range(attempts):
        with tempfile.NamedTemporaryFile(suffix=".zip") as tmp:
            result = subprocess.run(
                ["curl", "-sS", "--fail", "--max-time", "240",
                 "--retry", "2", "--retry-delay", "2", "-o", tmp.name, url],
                capture_output=True, text=True)
            if result.returncode == 0:
                with open(tmp.name, "rb") as f:
                    return f.read()
            last_exc = RuntimeError(f"curl exit {result.returncode}: {result.stderr[:300]}")
    raise last_exc


def parse_legacy_tract_archive(zdata: bytes, *, sequence_number: str = SEQ_NUM) -> pd.DataFrame | None:
    """Parse one legacy ACS tract archive into the B08301 worker columns.

    This intentionally preserves the 2020 pilot's file layout assumptions so
    newer builders can reuse its parsing without changing its output.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(zdata)) as z:
            files = z.namelist()
            geo_file = next((f for f in files if f.startswith("g") and f.endswith(".csv")), None)
            seq_file = next((f for f in files if f.startswith("e") and sequence_number in f), None)
            if geo_file is None or seq_file is None:
                log.warning("missing geo=%s or seq%s=%s", geo_file, sequence_number, seq_file)
                return None

            with z.open(geo_file) as f:
                geo = pd.read_csv(f, encoding="latin-1", header=None, dtype=str)
            with z.open(seq_file) as f:
                seq = pd.read_csv(f, header=None, dtype=str, low_memory=False)

        tract_geo = geo[geo.iloc[:, 2] == GEO_SUMLEVEL_TRACT].copy()
        if tract_geo.empty:
            return None

        logrecno = tract_geo.iloc[:, 4].astype(str).str.zfill(7)
        # col 48 geoid looks like "14000US56001962700" -> last 11 chars = tract FIPS
        tract_fips = tract_geo.iloc[:, 48].str[-11:]
        geo_idx = pd.DataFrame({"logrecno": logrecno.values, "tract": tract_fips.values})

        seq["logrecno"] = seq.iloc[:, 5].astype(str).str.zfill(7)
        data_offset = 156  # SEQ_START=157, 1-indexed -> 0-indexed 156 (same as 01e, same vintage)
        ncols_needed = 2   # B08301_001 (total workers), B08301_002 (car/truck/van)

        if seq.shape[1] < data_offset + ncols_needed:
            log.warning("seq%s too narrow (%d cols)", sequence_number, seq.shape[1])
            return None

        data_cols = list(seq.columns[data_offset: data_offset + ncols_needed])
        merged = geo_idx.merge(seq[["logrecno"] + data_cols], on="logrecno", how="inner")
        if merged.empty:
            return None

        result = pd.DataFrame()
        result["tract"] = merged["tract"].values
        result["total_workers"] = merged.iloc[:, 2].apply(parse_acs_int)
        result["car_total"] = merged.iloc[:, 3].apply(parse_acs_int)
        return result

    except Exception as exc:
        log.warning("FAILED legacy tract archive: %s", exc)
        return None


def fetch_state(url_name: str) -> pd.DataFrame | None:
    url = f"{BASE_URL}/{url_name}_Tracts_Block_Groups_Only.zip"
    try:
        return parse_legacy_tract_archive(_download_with_retry(url))
    except Exception as exc:
        log.warning("FAILED %s: %s", url_name, exc)
        return None


def main():
    log.info("Downloading ACS %d B08301 tract-level car-commuters (%d states)...", ACS_YEAR, len(STATES))
    all_results = []
    errors = 0
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(fetch_state, name): name for name in STATES}
        for i, fut in enumerate(as_completed(futures), 1):
            name = futures[fut]
            try:
                df = fut.result()
            except Exception as exc:
                log.warning("Exception %s: %s", name, exc)
                df = None
            if df is not None and not df.empty:
                all_results.append(df)
            else:
                errors += 1
            if i % 10 == 0 or i == len(STATES):
                log.info("  %d/%d done (%d errors)", i, len(STATES), errors)

    if not all_results:
        log.error("No data fetched!")
        return

    final = pd.concat(all_results, ignore_index=True)
    final["car_share"] = final["car_total"] / final["total_workers"].replace(0, np.nan)
    final = final[~final["tract"].str.startswith("72")].copy()  # drop PR
    final = final.sort_values("tract").reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    final.to_parquet(OUT_PATH, index=False)
    log.info("Saved %d tracts -> %s", len(final), OUT_PATH)
    log.info("Tract car share: mean=%.1f%%, median=%.1f%%, min=%.1f%%, max=%.1f%%",
             final["car_share"].mean() * 100, final["car_share"].median() * 100,
             final["car_share"].min() * 100, final["car_share"].max() * 100)


if __name__ == "__main__":
    main()
