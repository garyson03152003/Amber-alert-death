import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "code" / "build_amber_missing_dataset.py"
SPEC = importlib.util.spec_from_file_location("build_amber_missing_dataset", MODULE_PATH)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


def _record(event: str, *, msg_type: str = "Alert", alert_id: str = "id-1") -> dict:
    return {
        "id": alert_id,
        "sent": "2024-01-02T04:00:00Z",
        "status": "Actual",
        "msgType": msg_type,
        "scope": "Public",
        "originalMessage": (
            f"<alert><msgType>{msg_type}</msgType><event>{event}</event>"
            "<eventCode><valueName>SAME</valueName><value>006001</value></eventCode>"
            "<parameter><valueName>CMAMtext</valueName><value>Urgent notice</value>"
            "</parameter></alert>"
        ),
    }


def test_person_rows_preserve_family_and_cancellations():
    rows = module.person_rows_from_records([
        _record("Missing Person", alert_id="missing"),
        _record("Silver Alert", msg_type="Cancel", alert_id="silver-cancel"),
        _record("Pipeline-Silver Saddle-GO", alert_id="fire"),
    ])

    assert set(rows["alert_family"]) == {"missing_person", "silver_alert"}
    assert set(rows["msg_type"]) == {"Alert", "Cancel"}
    assert "fire" not in set(rows["alert_id"])


def test_combined_dataset_deduplicates_alert_county_pairs():
    rows = module.person_rows_from_records([
        _record("Missing Person", alert_id="missing"),
        _record("Missing Person", alert_id="missing"),
    ])
    assert len(rows) == 1
    assert rows.loc[0, "fips"] == "06001"
