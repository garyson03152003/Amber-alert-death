import importlib.util
from pathlib import Path

import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "code" / "fetch_other_wea_controls.py"
SPEC = importlib.util.spec_from_file_location("fetch_other_wea_controls", MODULE_PATH)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


def _record(**overrides):
    base = {
        "id": "id-1",
        "sent": "2024-01-02T04:00:00Z",
        "status": "Actual",
        "msgType": "Alert",
        "scope": "Public",
        "originalMessage": (
            "<alert><msgType>Alert</msgType><eventCode><valueName>SAME</valueName>"
            "<value>006001</value></eventCode><eventCode><valueName>WEA</valueName>"
            "<value>1</value></eventCode><area><areaDesc>Some County</areaDesc>"
            "</area></alert>"
        ),
    }
    base.update(overrides)
    return base


def test_candidate_requires_wea_delivery_and_rejects_amber_cmas_and_tests():
    assert module.classify_record(_record())["reason"] == "keep"
    assert module.classify_record(
        _record(originalMessage="<alert><msgType>Alert</msgType></alert>")
    )["reason"] == "no_wea_payload"
    assert module.classify_record(
        _record(originalMessage=(
            "<alert><msgType>Alert</msgType><parameter>"
            "<valueName>WEAHandling</valueName><value>Imminent Threat</value>"
            "</parameter></alert>"
        ))
    )["reason"] == "keep"
    assert module.classify_record(_record(originalMessage="<eventCode><valueName>SAME</valueName><value>006001</value></eventCode><eventCode><valueName>SAME</valueName><value>006000</value></eventCode><event>Child Abduction Emergency</event>"))["reason"] == "amber"
    assert module.classify_record(_record(originalMessage="<parameter><valueName>BLOCKCHANNEL</valueName><value>CMAS</value></parameter>"))["reason"] == "cmas_blocked"
    assert module.classify_record(_record(msgType="Cancel"))["reason"] == "keep"
    assert module.classify_record(_record(msgType="Cancel", originalMessage="<alert><msgType>Cancel</msgType><parameter><valueName>CMAMtext</valueName><value>Resolved</value></parameter></alert>"))["reason"] == "keep"
    assert module.classify_record(_record(status="Test"))["reason"] == "non_actual"


def test_missing_person_and_silver_alerts_are_treatment_not_controls():
    missing = _record(
        originalMessage=(
            "<alert><msgType>Alert</msgType><event>Missing Person</event>"
            "<eventCode><valueName>SAME</valueName><value>006001</value></eventCode>"
            "<parameter><valueName>CMAMtext</valueName><value>Missing person</value>"
            "</parameter></alert>"
        )
    )
    silver = _record(
        originalMessage=(
            "<alert><msgType>Alert</msgType><event>Silver Alert</event>"
            "<eventCode><valueName>SAME</valueName><value>006001</value></eventCode>"
            "<parameter><valueName>CMAMtext</valueName><value>Silver Alert</value>"
            "</parameter></alert>"
        )
    )
    assert module.person_alert_family(missing) == "missing_person"
    assert module.classify_record(missing)["reason"] == "person_treatment"
    assert module.person_alert_family(silver) == "silver_alert"
    assert module.classify_record(silver)["reason"] == "person_treatment"


def test_silver_in_a_fire_event_is_not_silver_alert():
    fire = _record(
        originalMessage=(
            "<alert><msgType>Alert</msgType><event>Pipeline-Silver Saddle-GO</event>"
            "<eventCode><valueName>SAME</valueName><value>006001</value></eventCode>"
            "<parameter><valueName>CMAMtext</valueName><value>Evacuation</value>"
            "</parameter></alert>"
        )
    )
    assert module.person_alert_family(fire) is None
    assert module.classify_record(fire)["reason"] == "keep"


def test_amber_weahandling_missing_child_is_not_a_non_amber_control():
    rec = _record(
        originalMessage=(
            "<alert><msgType>Alert</msgType><event>Administrative Message</event>"
            "<headline>Missing Child</headline>"
            "<eventCode><valueName>SAME</valueName><value>006001</value></eventCode>"
            "<parameter><valueName>WEAHandling</valueName><value>Amber</value></parameter>"
            "<parameter><valueName>CMAMtext</valueName><value>Missing child</value>"
            "</parameter></alert>"
        )
    )
    assert module.person_alert_family(rec) is None
    assert module.classify_record(rec)["reason"] == "amber"


def test_weather_classification_uses_cap_category_not_free_text():
    weather = _record(
        originalMessage=(
            "<alert><msgType>Alert</msgType><category>Met</category>"
            "<event>Flash Flood Warning</event>"
            "<eventCode><valueName>SAME</valueName><value>006001</value></eventCode>"
            "<parameter><valueName>CMAMtext</valueName><value>Flood warning</value>"
            "</parameter></alert>"
        )
    )
    non_weather = _record(
        originalMessage=(
            "<alert><msgType>Alert</msgType><category>Safety</category>"
            "<event>Law Enforcement Warning</event>"
            "<eventCode><valueName>SAME</valueName><value>006001</value></eventCode>"
            "<parameter><valueName>CMAMtext</valueName><value>Police activity</value>"
            "</parameter></alert>"
        )
    )
    assert module.is_weather_wea(weather)
    assert not module.is_weather_wea(non_weather)


def test_local_night_date_uses_county_timezone_and_evening_rollover():
    sent = pd.Timestamp("2024-01-02T04:00:00Z")
    assert module.effective_night_date(sent, "America/New_York") == pd.Timestamp("2024-01-02")
    assert module.effective_night_date(sent, "America/Chicago") == pd.Timestamp("2024-01-02")
    assert module.effective_night_date(sent, "America/Los_Angeles") is None
    sent = pd.Timestamp("2024-01-02T08:00:00Z")
    assert module.effective_night_date(sent, "America/New_York") == pd.Timestamp("2024-01-02")


def test_parse_record_keeps_standard_county_and_state_codes():
    rec = _record(
        originalMessage=(
            "<alert><msgType>Alert</msgType>"
            "<eventCode><valueName>SAME</valueName><value>006001</value></eventCode>"
            "<eventCode><valueName>SAME</valueName><value>006000</value></eventCode>"
            "</alert>"
        )
    )
    rows = module.parse_same_rows(rec)
    assert {row["fips"] for row in rows} == {"06001", "06000"}
