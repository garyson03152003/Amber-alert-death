import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))


def test_ca_severity_runner_uses_collision_level_outcomes():
    import run_same_hour_ca_severity as runner

    assert runner.CA_HOURLY_PATH.name == "california_ccrs_severity_county_hour.parquet"
    assert runner.OUTCOMES == [
        "ca_fatal_crashes",
        "ca_serious_crashes",
        "ca_nonserious_crashes",
    ]
    assert "ca_serious_inj" not in runner.OUTCOMES


def test_hourly_event_study_exposes_collision_level_ca_outcomes():
    import run_hourly_alert_event_study as runner

    assert runner.CA_HOURLY.name == "california_ccrs_severity_county_hour.parquet"
    assert runner.OUTCOMES == [
        "ca_fatal_crashes",
        "ca_serious_crashes",
        "ca_nonserious_crashes",
    ]
