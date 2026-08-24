"""
01e_fetch_car_commuters.py
Downloads ACS 2020 5-year B08301 (Means of Transportation to Work) at the
county level for all 50 states + DC via the Census Summary File.

Strategy: for each state, download the full state-level zip (≈100MB),
extract only g*csv (geo header) and e*0027*.txt (sequence 27 = B08301),
then discard the rest.  Uses 8 parallel workers; ≈5–7 min total.

B08301 column layout (all 21 cells, 1-indexed from SEQ_START=157):
  1  = B08301_001  Total workers 16+
  2  = B08301_002  Car, truck, or van (drove alone + carpool)
  3  = B08301_003  Drove alone
  4–9              Carpool sub-categories
  10 = B08301_010  Carpooled (total)
  11–21            Other modes

Output: data/processed/county_car_commuters.parquet
  Columns: fips(str5), total_workers, car_total, drove_alone, carpooled, car_share
"""
import sys, os, time, urllib.request, zipfile, io, warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from utils import get_logger
warnings.filterwarnings("ignore")
log = get_logger("car_commuters")

ACS_YEAR = int(os.environ.get("ACS_YEAR", "2020"))
OUT_PATH = (Path(__file__).parent.parent / "data" / "processed" /
            ("county_car_commuters.parquet" if ACS_YEAR == 2020
             else f"county_car_commuters_{ACS_YEAR}.parquet"))
BASE_URL = ("https://www2.census.gov/programs-surveys/acs/summary_file/"
            f"{ACS_YEAR}/data/5_year_by_state")
LOOKUP_URL = ("https://www2.census.gov/programs-surveys/acs/summary_file/"
              f"{ACS_YEAR}/documentation/user_tools/"
              "ACS_5yr_Seq_Table_Number_Lookup.txt")


def resolve_table_location(table_id: str = "B08301") -> tuple[str, int]:
    """Look up this vintage's sequence number and start position for a table.

    The sequence is NOT stable across ACS vintages -- B08301 is sequence 0027
    in 2020 but 0028 in 2015, while the start position is 157 in both. Reading
    the wrong sequence yields a same-shaped 21-cell table of an entirely
    different variable, which would parse cleanly and be silently wrong, so
    this is derived from Census's own lookup rather than hardcoded.
    """
    import io as _io
    import urllib.request as _u
    with _u.urlopen(LOOKUP_URL, timeout=120) as r:
        text = r.read().decode("latin-1")
    lk = pd.read_csv(_io.StringIO(text), dtype=str)
    lk.columns = [c.strip() for c in lk.columns]
    sub = lk[lk["Table ID"].astype(str).str.strip() == table_id]
    if sub.empty:
        raise ValueError(f"{table_id} not present in the {ACS_YEAR} ACS lookup")
    seq = str(sub["Sequence Number"].dropna().iloc[0]).strip().zfill(4)
    start = int(str(sub["Start Position"].dropna().iloc[0]).strip())
    log.info("ACS %d: %s -> sequence %s, start position %d",
             ACS_YEAR, table_id, seq, start)
    return seq, start

# → 0-indexed data_offset = SEQ_START - 1 = 156
# B08301 layout (0-indexed from data_offset):
#  +0 = B08301_001 Total workers
#  +1 = B08301_002 Car, truck, or van (total)
#  +2 = B08301_003 Drove alone
#  +3 = B08301_004 Carpooled (total; sub-cats at +4..+8)
#  +9 = B08301_010 Public transportation  ← NOT carpooled
SEQ_NUM, SEQ_START = resolve_table_location("B08301")
GEO_SUMLEVEL = "050" # county summary level

# State name → URL slug (Census uses spaces replaced by underscores in URLs)
STATES = {
    "Alabama": "al", "Alaska": "ak", "Arizona": "az", "Arkansas": "ar",
    "California": "ca", "Colorado": "co", "Connecticut": "ct", "Delaware": "de",
    # Census uses CamelCase with no separators for multi-word states;
    # "District_of_Columbia" 404s in every vintage.
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
        f = float(str(val).replace(",","").strip())
        return int(f) if not (f < 0) else np.nan
    except Exception:
        return np.nan

def fetch_state(url_name: str, stab: str) -> pd.DataFrame | None:
    """Download one state zip, parse geo + seq27, return county B08301 rows."""
    url = f"{BASE_URL}/{url_name}_All_Geographies_Not_Tracts_Block_Groups.zip"
    try:
        with urllib.request.urlopen(url, timeout=120) as r:
            zdata = r.read()

        with zipfile.ZipFile(io.BytesIO(zdata)) as z:
            files = z.namelist()

            # Geo header: g20205xx.csv
            geo_file = next((f for f in files if f.startswith("g") and f.endswith(".csv")), None)
            # Seq27: e20205xx0027000.txt
            seq_file = next((f for f in files if f.startswith("e") and SEQ_NUM in f), None)
            if geo_file is None or seq_file is None:
                log.warning("%s: missing geo=%s or seq%s=%s", url_name, geo_file, SEQ_NUM, seq_file)
                return None

            with z.open(geo_file) as f:
                geo = pd.read_csv(f, encoding="latin-1", header=None, dtype=str)

            with z.open(seq_file) as f:
                seq = pd.read_csv(f, header=None, dtype=str, low_memory=False)

        # Filter geo to county rows (SUMLEVEL col 2 == "050")
        county_geo = geo[geo.iloc[:, 2] == GEO_SUMLEVEL].copy()
        if county_geo.empty:
            return None

        logrecno = county_geo.iloc[:, 4].astype(str).str.zfill(7)
        fips     = county_geo.iloc[:, 48].str[-5:]  # "05000US01001" → "01001"
        geo_idx  = pd.DataFrame({"logrecno": logrecno.values, "fips": fips.values})

        # Seq27: SEQ_START is ABSOLUTE 1-indexed position (including 6 header cols)
        # → 0-indexed: data_offset = SEQ_START - 1
        data_offset = SEQ_START - 1   # = 156
        seq["logrecno"] = seq.iloc[:, 5].astype(str).str.zfill(7)
        ncols_needed = 5  # B08301_001..004 + one spare

        if seq.shape[1] < data_offset + ncols_needed:
            log.warning("%s: seq%s too narrow (%d cols)", url_name, SEQ_NUM, seq.shape[1])
            return None

        data_cols = list(seq.columns[data_offset : data_offset + ncols_needed])
        merged = geo_idx.merge(seq[["logrecno"] + data_cols], on="logrecno", how="inner")
        if merged.empty:
            return None

        # merged cols: fips(0), logrecno(1), B08301_001(2), _002(3), _003(4), _004(5)
        result = pd.DataFrame()
        result["fips"]          = merged["fips"].values
        result["total_workers"] = merged.iloc[:, 2].apply(parse_acs_int)  # B08301_001
        result["car_total"]     = merged.iloc[:, 3].apply(parse_acs_int)  # B08301_002 (drove+carpool)
        result["drove_alone"]   = merged.iloc[:, 4].apply(parse_acs_int)  # B08301_003
        result["carpooled"]     = merged.iloc[:, 5].apply(parse_acs_int)  # B08301_004

        return result

    except Exception as exc:
        log.warning("FAILED %s: %s", url_name, exc)
        return None


def main():
    log.info("Downloading ACS 2020 B08301 county car-commuters (%d states) …", len(STATES))

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

    # Car share
    final["car_share"] = (final["car_total"] / final["total_workers"].replace(0, np.nan))

    # Drop PR if present
    final = final[~final["fips"].str.startswith("72")].copy()
    final = final.sort_values("fips").reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    final.to_parquet(OUT_PATH, index=False)

    log.info("Saved %d counties → %s", len(final), OUT_PATH)
    log.info("Car commute share: mean=%.1f%%, median=%.1f%%, min=%.1f%%, max=%.1f%%",
             final["car_share"].mean()*100,
             final["car_share"].median()*100,
             final["car_share"].min()*100,
             final["car_share"].max()*100)
    log.info("Sample:\n%s", final.head(8).to_string())


if __name__ == "__main__":
    main()
