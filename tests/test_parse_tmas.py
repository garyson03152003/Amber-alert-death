import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code" / "traffic_volume"))

from parse_tmas import parse_legacy_vol_line, parse_modern_vol_bytes, parse_modern_sta_bytes, _decode_coord


# A real record pulled from FHWA's jan_2015_ccs_data.zip (DC, station
# 000050). Confirmed byte-for-byte: DOW=5 (Thursday) for Jan 1, 2015, which
# is independently verifiable ground truth -- this is what pins down the
# legacy format's 2-digit year (vs. the 2013 TMG's documented 4-digit year).
_REAL_LEGACY_LINE = (
    "3112U0000503115010150117200886007550066100495002750026100381"
    "003350037800527007300081500878010110115901100010530091300859"
    "007680065300652005200"
)


def test_legacy_vol_line_matches_known_day_of_week():
    parsed = parse_legacy_vol_line(_REAL_LEGACY_LINE)
    assert parsed is not None
    assert parsed["state_fips"] == "11"
    assert parsed["station_id"] == "000050"
    assert parsed["year"] == 2015
    assert parsed["month"] == 1
    assert parsed["day"] == 1
    # January 1, 2015 was a Thursday -- FHWA's day-of-week code 5.
    assert parsed["day_of_week"] == 5
    assert len(parsed["hours"]) == 24
    assert parsed["hours"][0] == 1172
    assert parsed["hours"][-1] == 520


def test_legacy_vol_line_rejects_non_volume_record_type():
    non_volume = "S" + _REAL_LEGACY_LINE[1:]
    assert parse_legacy_vol_line(non_volume) is None


def test_legacy_vol_line_rejects_too_short_line():
    assert parse_legacy_vol_line("311") is None


def test_modern_vol_bytes_parses_pipe_delimited_header_and_hours():
    header = (
        "record_type|state_code|f_system|station_id|travel_dir|travel_lane|"
        "year_record|month_record|day_record|day_of_week|" +
        "|".join(f"hour_{h:02d}" for h in range(24)) + "|restrictions"
    )
    row = (
        "V|11|2U|000050|3|1|2023|1|2|2|" +
        "|".join(str(v) for v in range(24)) + "|"
    )
    raw = (header + "\n" + row + "\n").encode("latin-1")
    df = parse_modern_vol_bytes(raw)
    assert len(df) == 1
    assert df.iloc[0]["state_fips"] == "11"
    assert df.iloc[0]["year"] == 2023
    assert df.iloc[0]["hours"] == list(range(24))


def test_modern_vol_bytes_requires_hour_columns():
    raw = b"record_type|state_code\nV|11\n"
    try:
        parse_modern_vol_bytes(raw)
        assert False, "expected ValueError for missing hour columns"
    except ValueError:
        pass


def test_decode_coord_matches_known_alabama_station():
    # From a real 2015 station record: Washington/Mobile county line, AL.
    lat = _decode_coord("31076964", negate=False)
    lon = _decode_coord("88022908", negate=True)
    assert abs(lat - 31.076964) < 1e-9
    assert abs(lon - (-88.022908)) < 1e-9
    # Sanity: this is real south Alabama, not somewhere absurd.
    assert 30 < lat < 32
    assert -89 < lon < -87


def test_decode_coord_blank_is_none():
    assert _decode_coord("", negate=False) is None
    assert _decode_coord("   ", negate=True) is None


def test_modern_sta_bytes_decodes_county_and_coordinates():
    header = "record_type|state_code|station_id|county_code|latitude|longitude"
    row = "S|01|000010|097|31076964| 88022908"
    raw = (header + "\n" + row + "\n").encode("latin-1")
    df = parse_modern_sta_bytes(raw)
    assert len(df) == 1
    assert df.iloc[0]["county_fips"] == "097"
    assert abs(df.iloc[0]["latitude"] - 31.076964) < 1e-9
    assert abs(df.iloc[0]["longitude"] - (-88.022908)) < 1e-9
