import sys
from pathlib import Path

import pandas as pd
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))


def _crashes():
    return pd.DataFrame(
        {
            "Collision Id": [1, 2, 3, 4, 4],
            "Report Version": [1, 1, 1, 1, 2],
            "Crash Date Time": [
                "2016-01-01 22:15:00",
                "2016-01-02 07:05:00",
                "2016-01-02 07:05:00",
                "2016-01-03 12:00:00",
                "2016-01-03 12:00:00",
            ],
            "County Code": [1, 1, 1, 1, 1],
            "NumberKilled": [1, 0, 0, 0, 0],
        }
    )


def _injuries():
    return pd.DataFrame(
        {
            "CollisionId": [1, 2, 3, 3, 4, 999],
            "IsWitnessOnly": [False, False, False, True, False, False],
            "ExtentOfInjuryCode": [None, "SevereInactive", "SuspectMinor", "SevereInactive", "UnknownCode", "Fatal"],
        }
    )


def test_classify_collisions_prioritizes_fatal_then_serious_and_excludes_witnesses():
    from ca_ccrs_severity import classify_collisions

    out, diagnostics = classify_collisions(_crashes(), _injuries())

    got = out.set_index("collision_id")["severity"].to_dict()
    assert got == {"1": "fatal", "2": "serious", "3": "nonserious", "4": "nonserious"}
    assert diagnostics["duplicate_crash_ids"] == 1
    assert diagnostics["orphan_injury_rows"] == 1
    assert diagnostics["witness_injury_rows_excluded"] == 1
    assert diagnostics["unknown_injury_code_rows"] == 1


def test_aggregate_severity_conserves_one_class_per_collision():
    from ca_ccrs_severity import aggregate_severity, classify_collisions

    classified, _ = classify_collisions(_crashes(), _injuries())
    day, hour = aggregate_severity(classified)

    assert day[["ca_fatal_crashes", "ca_serious_crashes", "ca_nonserious_crashes"]].sum().sum() == 4
    assert day["ca_crashes"].sum() == 4
    assert hour["ca_crashes"].sum() == 4
    row = hour.loc[(hour["fips"] == "06001") & (hour["hour"] == 22)].iloc[0]
    assert row["ca_fatal_crashes"] == 1
    assert row["ca_serious_crashes"] == 0


def test_classifier_requires_collision_date_county_and_killed_columns():
    from ca_ccrs_severity import classify_collisions

    with pytest.raises(ValueError, match="crashes missing required columns"):
        classify_collisions(pd.DataFrame({"Collision Id": [1]}), pd.DataFrame())
