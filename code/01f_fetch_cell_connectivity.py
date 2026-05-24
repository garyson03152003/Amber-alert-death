"""
01f_fetch_cell_connectivity.py
Downloads ACS 2020 5-year B28002 (Presence and Types of Internet
Subscriptions in Household) at the county level for all 50 states + DC
via the Census Summary File.  Extracts the "cellular data plan" cell as
a proxy for WEA-reachable households.

B28002 is in sequence 141.  Start position = 18 (absolute 1-indexed,
including the 6 header cols), so data_offset = 17 (0-indexed).

B28002 column layout (13 cells, 0-indexed from data_offset = 17):
  +0 = B28002_001  Total households
  +1 = B28002_002  With an Internet subscription
  +2 = B28002_003  Dial-up with no other type
  +3 = B28002_004  Broadband of any type
  +4 = B28002_005  Cellular data plan          ← KEY VARIABLE
  +5 = B28002_006  Cellular data plan with no other type
  +6 = B28002_007  Cable/fiber/DSL
  ...
  +12 = B28002_013  No Internet access

Rationale: WEA (Wireless Emergency Alerts) reach only people with active
cellular service.  B28002_005 / B28002_001 gives the fraction of
households with a cellular data plan — our best ACS proxy for WEA
receptivity in a county.  National mean is ~70–75%; rural counties are
lower (~65%), urban/suburban counties are higher (~75–80%).

Note: B28001_005 (smartphone HH share, also seq 141) is an alternative
proxy (~81% for Autauga AL) if a higher estimate is preferred.

Output: data/processed/county_cell_connectivity.parquet
  Columns: fips(str5), hh_total, hh_cell_plan, cell_share
"""
import sys, warnings, urllib.request, zipfile, io
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from utils import get_logger
warnings.filterwarnings("ignore")
log = get_logger("cell_connectivity")

OUT_PATH = Path(__file__).parent.parent / "data" / "processed" / "county_cell_connectivity.parquet"
BASE_URL = ("https://www2.census.gov/programs-surveys/acs/summary_file/"
            "2020/data/5_year_by_state")

# B28002: absolute start position 18 (1-indexed) → 0-indexed data_offset = 17
DATA_OFFSET   = 17    # B28002_001 is at seq col 17 (0-indexed)
B28002_001    = 0     # relative: total households
B28002_005    = 4     # relative: cellular data plan households

GEO_SUMLEVEL = "050"  # county

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


def fetch_state(url_name: str, stab: str) -> pd.DataFrame | None:
    """Download one state zip, parse geo + seq141, return county B28002 rows."""
    url = f"{BASE_URL}/{url_name}_All_Geographies_Not_Tracts_Block_Groups.zip"
    try:
        with urllib.request.urlopen(url, timeout=120) as r:
            zdata = r.read()

        with zipfile.ZipFile(io.BytesIO(zdata)) as z:
            files = z.namelist()
            geo_file = next((f for f in files if f.startswith("g") and f.endswith(".csv")), None)
            seq_file = next((f for f in files if f.startswith("e") and "0141" in f), None)

            if geo_file is None or seq_file is None:
                log.warning("%s: missing geo=%s or seq141=%s", url_name, geo_file, seq_file)
                return None

            with z.open(geo_file) as f:
                geo = pd.read_csv(f, encoding="latin-1", header=None, dtype=str)
            with z.open(seq_file) as f:
                seq = pd.read_csv(f, header=None, dtype=str, low_memory=False)

        county_geo = geo[geo.iloc[:, 2] == GEO_SUMLEVEL].copy()
        if county_geo.empty:
            return None

        logrecno = county_geo.iloc[:, 4].astype(str).str.zfill(7)
        fips     = county_geo.iloc[:, 48].str[-5:]
        geo_idx  = pd.DataFrame({"logrecno": logrecno.values, "fips": fips.values})

        seq["logrecno"] = seq.iloc[:, 5].astype(str).str.zfill(7)
        ncols_needed = B28002_005 + 1   # need 5 columns: B28002_001 through B28002_005

        if seq.shape[1] < DATA_OFFSET + ncols_needed:
            log.warning("%s: seq141 too narrow (%d cols)", url_name, seq.shape[1])
            return None

        # Grab just the columns we need (logrecno + the 5 B28002 cells)
        data_cols = list(seq.columns[DATA_OFFSET : DATA_OFFSET + ncols_needed])
        merged = geo_idx.merge(seq[["logrecno"] + data_cols], on="logrecno", how="inner")
        if merged.empty:
            return None

        # merged positional layout:
        #   col 0 = logrecno, col 1 = fips
        #   col 2 = B28002_001 (total HH)  [DATA_OFFSET + B28002_001]
        #   col 6 = B28002_005 (cell plan HH) [DATA_OFFSET + B28002_005]
        # BUT: we extracted ncols_needed=5 data cols, so:
        #   col 2 = data_cols[0] = B28002_001
        #   col 6 = data_cols[4] = B28002_005
        result = pd.DataFrame()
        result["fips"]         = merged["fips"].values
        result["hh_total"]     = merged.iloc[:, 2].apply(parse_acs_int)   # B28002_001
        result["hh_cell_plan"] = merged.iloc[:, 6].apply(parse_acs_int)   # B28002_005

        return result

    except Exception as exc:
        log.warning("FAILED %s: %s", url_name, exc)
        return None


def main():
    log.info("Downloading ACS 2020 B28002 county cellular connectivity (%d states) …", len(STATES))

    all_results = []
    errors = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(fetch_state, name, stab): name
                   for name, stab in STATES.items()}
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
    final["cell_share"] = (final["hh_cell_plan"] /
                           final["hh_total"].replace(0, np.nan))

    # Drop Puerto Rico
    final = final[~final["fips"].str.startswith("72")].copy()
    final = final.sort_values("fips").reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    final.to_parquet(OUT_PATH, index=False)

    log.info("Saved %d counties → %s", len(final), OUT_PATH)
    log.info("Cellular plan share: mean=%.1f%%, median=%.1f%%, min=%.1f%%, max=%.1f%%",
             final["cell_share"].mean() * 100,
             final["cell_share"].median() * 100,
             final["cell_share"].min() * 100,
             final["cell_share"].max() * 100)
    log.info("Sample:\n%s", final.head(8).to_string())


if __name__ == "__main__":
    main()
