import sys
from pathlib import Path

import pandas as pd
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))


def test_build_year_reads_paired_cached_sources_and_emits_coverage(tmp_path, monkeypatch):
    import ca_ccrs_severity as builder

    crash = pd.DataFrame(
        {
            "Collision Id": ["1", "2"],
            "Report Version": [1, 1],
            "Crash Date Time": ["2016-01-01 22:15:00", "2016-01-01 23:15:00"],
            "County Code": [1, 1],
            "NumberKilled": [1, 0],
        }
    )
    injury = pd.DataFrame(
        {
            "CollisionId": ["2"],
            "IsWitnessOnly": [False],
            "ExtentOfInjuryCode": ["SevereInactive"],
        }
    )
    files = {"crashes": tmp_path / "crashes.csv", "injuries": tmp_path / "injuries.csv"}
    crash.to_csv(files["crashes"], index=False)
    injury.to_csv(files["injuries"], index=False)

    def fake_download(_session, _url, path):
        source = files["crashes"] if "crashes" in path.name else files["injuries"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(source.read_bytes())
        return path, "a" * 64

    monkeypatch.setattr(builder, "download_resource", fake_download)
    day, hour, coverage = builder.build_year(
        2016,
        session=object(),
        raw_dir=tmp_path / "raw",
        resources={2016: {"crashes": "crash-url", "injuries": "injury-url"}},
    )

    assert day["ca_crashes"].sum() == 2
    assert day["ca_fatal_crashes"].sum() == 1
    assert day["ca_serious_crashes"].sum() == 1
    assert hour["hour"].tolist() == [22, 23]
    assert coverage["coverage_valid"] is True
    assert coverage["crash_sha256"] == "a" * 64


def test_build_year_fails_closed_when_injury_schema_is_missing(tmp_path, monkeypatch):
    import ca_ccrs_severity as builder

    crash = pd.DataFrame(
        {
            "Collision Id": ["1"],
            "Report Version": [1],
            "Crash Date Time": ["2016-01-01 22:15:00"],
            "County Code": [1],
            "NumberKilled": [0],
        }
    )
    injury = pd.DataFrame({"CollisionId": ["1"], "IsWitnessOnly": [False]})
    files = {"crashes": tmp_path / "crashes.csv", "injuries": tmp_path / "injuries.csv"}
    crash.to_csv(files["crashes"], index=False)
    injury.to_csv(files["injuries"], index=False)

    def fake_download(_session, _url, path):
        source = files["crashes"] if "crashes" in path.name else files["injuries"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(source.read_bytes())
        return path, "b" * 64

    monkeypatch.setattr(builder, "download_resource", fake_download)
    with pytest.raises((ValueError, KeyError), match="ExtentOfInjuryCode|usecols"):
        builder.build_year(
            2016,
            session=object(),
            raw_dir=tmp_path / "raw",
            resources={2016: {"crashes": "crash-url", "injuries": "injury-url"}},
        )
