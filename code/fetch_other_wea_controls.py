"""Build county-day controls from non-AMBER public WEA-eligible CAP alerts.

The FEMA IPAWS archive contains every CAP message, not only messages that
reached a handset.  This module identifies phone-delivered WEA products from
the CAP delivery payload (``WEAHandling``, ``CMAMtext``, or
``CMAMlongtext``), then excludes records that explicitly carry
``BLOCKCHANNEL=CMAS``.  AMBER and the reviewed missing-person/Silver-Alert
families are kept out of this *control* because they are treatment; other
alerts, updates, and cancellations remain eligible when they carry the WEA
payload.  Tests and messages without a numeric SAME county geocode are
excluded.  The filter is recorded in an
audit table so the result is reproducible and its limitations are visible.

Only the overnight county-day exposure is retained.  A message sent at
22:00--23:59 local time is assigned to the following driving date; a message
sent at 00:00--05:59 is assigned to its calendar date.  Statewide SAME codes
are expanded with the same outcome-compatible county universe used by the
AMBER treatment.

Outputs
-------
``data/processed/other_wea_night_controls.parquet``
    One row per county/effective date, with a binary alert indicator and
    number of distinct source alert IDs.
``output/tables/other_wea_alert_filter_summary.csv``
    Record-level inclusion/exclusion counts and county-date totals.

The default source is the OpenFEMA API, queried one month at a time for each
WEA delivery parameter.  FEMA's indexed ArcGIS archive remains available as
an event-indexed fallback with ``--source arcgis``; the bulk JSONL archive is
available with ``--source bulk``.  Use ``--max-records`` for a quick parser
smoke test.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.util
import json
import os
import re
import sys
import time
import threading
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent))
import run_state_dot_analysis_fixed as amber_base
from config import DATA_PROC, DATA_RAW, OUTPUT_TABS, STUDY_YEARS
from utils import get_logger

log = get_logger("other_wea_controls")

API_URL = "https://www.fema.gov/api/open/v1/IpawsArchivedAlerts"
BULK_URL = API_URL + ".jsonl"
ARCGIS_QUERY_URL = (
    "https://gis.fema.gov/arcgis/rest/services/FEMA/IPAWS_Archive/"
    "FeatureServer/1/query"
)
API_TOP = 1000
API_SLEEP_SECONDS = 0.15
API_RETRY_DELAYS = (2, 5, 10)
CONTROL_PATH = DATA_PROC / "other_wea_night_controls.parquet"
SUMMARY_PATH = OUTPUT_TABS / "other_wea_alert_filter_summary.csv"
WEATHER_EXCLUDED_CONTROL_PATH = DATA_PROC / "other_wea_night_controls_no_weather.parquet"
WEATHER_EXCLUDED_SUMMARY_PATH = OUTPUT_TABS / "other_wea_alert_filter_summary_no_weather.csv"
MONTH_CACHE_DIR = DATA_RAW / "other_wea" / "api_months_v3"

_PARSER_SPEC = importlib.util.spec_from_file_location(
    "ipaws_amber_parser", Path(__file__).with_name("02c_fetch_openfema_ipaws.py")
)
if _PARSER_SPEC is None or _PARSER_SPEC.loader is None:
    raise ImportError("could not load the shared IPAWS SAME parser")
_PARSER_MODULE = importlib.util.module_from_spec(_PARSER_SPEC)
_PARSER_SPEC.loader.exec_module(_PARSER_MODULE)

EVENT_CODE_RE = re.compile(
    r"<eventCode>\s*<valueName>\s*([^<]+?)\s*</valueName>\s*"
    r"<value>\s*([^<]+?)\s*</value>\s*</eventCode>",
    re.IGNORECASE | re.DOTALL,
)
MSGTYPE_RE = re.compile(r"<msgType>\s*(Alert|Update|Cancel)\s*</msgType>", re.IGNORECASE)
SCOPE_RE = re.compile(r"<scope>\s*([^<]+?)\s*</scope>", re.IGNORECASE)
STATUS_RE = re.compile(r"<status>\s*([^<]+?)\s*</status>", re.IGNORECASE)
EVENT_RE = re.compile(r"<event>\s*([^<]+?)\s*</event>", re.IGNORECASE | re.DOTALL)
CATEGORY_RE = re.compile(r"<category>\s*([^<]+?)\s*</category>", re.IGNORECASE | re.DOTALL)
BLOCK_CMAS_RE = re.compile(
    r"<valueName>\s*BLOCKCHANNEL\s*</valueName>\s*"
    r"<value>\s*CMAS\s*</value>", re.IGNORECASE | re.DOTALL
)

# These are broadcast/test products, not an actual alert that should disturb
# sleep.  The status filter removes most of them; event-code filtering keeps
# the few that are marked Actual in the archive from entering the control.
TEST_EVENT_CODES = {"RWT", "RMT", "DMO", "PRA", "TST", "TST1", "TST2"}

# WEA products carry one or more of these CAP parameter names.  Querying each
# token separately keeps FEMA's OData planner from timing out on a large OR
# expression; records are deduplicated by identifier before local filtering.
WEA_SIGNAL_TOKENS = ("WEAHandling", "CMAMtext", "CMAMlongtext")
# The archive's older CAP records predate the WEAHandling parameter, but their
# handset payload still carries CMAMtext.  Avoid expensive unindexed queries
# for a parameter that cannot occur in those historical records.
WEA_HANDLING_START_YEAR = 2020
# Before WEAHandling was consistently populated, the archive's CMAMtext
# records can be found through the indexed CAP/SAME event codes.  This is a
# deliberately broad union of FEMA NWEM/WEA codes and NWS warning codes; the
# local WEA payload check below removes ordinary EAS/NWEM products that happen
# to share an event code.
WEA_QUERY_EVENT_CODES = (
    "AVW", "BLU", "BZW", "CAE", "CDW", "CEM", "CFW", "DSW", "EQW",
    "EVI", "EWW", "FFW", "FLW", "FRW", "HMW", "HWW", "ISW", "LAE",
    "LEW", "MEP", "NUW", "RHW", "RMT", "RWT", "SMW", "SPW", "SQW",
    "SSW", "SVR", "TOE", "TOR", "TRW", "TSW", "VOW", "WSW", "HUW",
)
WEA_PARAMETER_RE = re.compile(
    r"<parameter>\s*<valueName>\s*([^<]+?)\s*</valueName>\s*"
    r"<value>\s*(.*?)\s*</value>\s*</parameter>",
    re.IGNORECASE | re.DOTALL,
)
WEA_PARAMETER_NAMES = {"WEAHANDLING", "CMAMTEXT", "CMAMLONGTEXT"}
WEA_EVENT_NAMES = (
    "Civil Emergency Message", "Civil Danger Warning", "Evacuation Immediate",
    "Fire Warning", "Hazardous Materials Warning", "Local Area Emergency",
    "Law Enforcement Warning", "Shelter In Place Warning",
    "911 Telephone Outage Emergency", "Emergency Action Notification",
    "Emergency Action Termination", "Nuclear Power Plant Warning",
    "Radiological Hazard Warning",
)

# Historical IPAWS senders did not have a dedicated Silver/missing-person
# event code.  The small set below is therefore deliberately based on the
# CAP <event> label,
# rather than a broad free-text search of the headline/body.  In particular,
# matching ``SILVER`` alone would misclassify the Coconino County fire
# evacuation ``Pipeline-Silver Saddle-GO`` as a Silver Alert.
MISSING_PERSON_EVENT_RE = re.compile(
    r"(?:\bMISSING\s+(?:PERSONS?|CHILD(?:REN)?|ADULTS?)\b|"
    r"\bENDANGERED\s+MISSING\b)", re.IGNORECASE,
)
SILVER_ALERT_EVENT_RE = re.compile(r"\bSILVER\s+ALERT\b", re.IGNORECASE)


def _original(rec: dict) -> str:
    return str(rec.get("originalMessage") or "")


def _parse_sent(raw: object) -> pd.Timestamp:
    """Fast, scalar ISO-8601 parser for the archive's sent field."""
    if raw is None:
        return pd.NaT
    try:
        text = str(raw).strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return pd.Timestamp(parsed).tz_convert("UTC")
    except (TypeError, ValueError, OverflowError):
        return pd.NaT


def _field_or_xml(rec: dict, field: str, pattern: re.Pattern[str]) -> str:
    value = rec.get(field)
    if value is not None and str(value).strip():
        return str(value).strip()
    match = pattern.search(_original(rec))
    return match.group(1).strip() if match else ""


def event_codes(rec: dict) -> dict[str, set[str]]:
    """Return CAP event-code values grouped by upper-case valueName."""
    out: dict[str, set[str]] = defaultdict(set)
    for name, value in EVENT_CODE_RE.findall(_original(rec)):
        out[name.strip().upper()].add(value.strip().upper())
    events = EVENT_RE.findall(_original(rec))
    if events:
        out.setdefault("EVENT", set()).update(event.strip().upper() for event in events)
    # OpenFEMA also exposes the nested geocodes in the structured ``info``
    # field; event codes other than SAME are sometimes only present in XML,
    # while SAME is often present in both.
    for info in rec.get("info", []) or []:
        for area in info.get("areas", []) or []:
            for code in area.get("geocode", []) or []:
                name = str(code.get("valueName", "")).strip().upper()
                value = str(code.get("value", "")).strip().upper()
                if name and value:
                    out[name].add(value)
    return dict(out)


def cap_categories(rec: dict) -> set[str]:
    """Return CAP ``category`` values, normalized to upper case."""
    out = {value.strip().upper() for value in CATEGORY_RE.findall(_original(rec)) if value.strip()}
    for info in rec.get("info", []) or []:
        value = str(info.get("category") or "").strip()
        if value:
            out.add(value.upper())
    return out


def is_weather_wea(rec: dict) -> bool:
    """Whether the WEA is meteorological under the CAP category field.

    ``Met`` is the CAP category intended for weather products.  Restricting
    this sensitivity to the structured category avoids guessing from words
    such as ``storm`` in a headline (and leaves wildfire, earthquake, and
    evacuation messages in the non-weather public-safety control).
    """
    return "MET" in cap_categories(rec)


def wea_parameters(rec: dict) -> dict[str, set[str]]:
    """Return CAP WEA parameter values grouped by upper-case valueName."""
    out: dict[str, set[str]] = defaultdict(set)
    for name, value in WEA_PARAMETER_RE.findall(_original(rec)):
        out[name.strip().upper()].add(value.strip())
    # OpenFEMA exposes these as ``info[].parameter`` in its structured copy.
    for info in rec.get("info", []) or []:
        for parameter in info.get("parameter", []) or info.get("parameters", []) or []:
            name = str(parameter.get("name") or parameter.get("valueName") or "").strip().upper()
            value = str(parameter.get("value", "")).strip()
            if name and value:
                out[name].add(value)
    return dict(out)


def has_wea_payload(rec: dict) -> bool:
    """Whether the CAP contains a WEA phone-delivery payload.

    The synthetic fixtures used by the project also use a ``WEA`` event-code
    marker, so retain that as a compatibility signal.  Production IPAWS rows
    are identified by the standard WEAHandling/CMAM text parameters.
    """
    params = wea_parameters(rec)
    if WEA_PARAMETER_NAMES & set(params):
        return True
    return bool(event_codes(rec).get("WEA", set()))


def has_cmas_block(rec: dict) -> bool:
    """Whether CAP explicitly prevents dissemination on the CMAS channel."""
    if BLOCK_CMAS_RE.search(_original(rec)):
        return True
    params = wea_parameters(rec)
    return any(value.casefold() == "cmas" for value in params.get("BLOCKCHANNEL", set()))


def person_alert_family(rec: dict) -> str | None:
    """Return the high-confidence missing-person treatment family, if any.

    The archive has no stable pre-2025 Silver-Alert event code, so treatment
    classification is intentionally conservative and transparent:

    * ``silver_alert`` requires the complete phrase ``Silver Alert`` in the
      CAP ``<event>`` label (not merely the word ``silver``).
    * ``missing_person`` uses explicit missing-person/child/adult (including
      plural forms) or endangered-missing labels.

    Generic body text is not used here; those records remain available for a
    separate, manually reviewed text-screening dataset rather than silently
    entering either the treatment or the non-AMBER control.
    """
    original = _original(rec)
    event_match = EVENT_RE.search(original)
    event_text = event_match.group(1).strip() if event_match else ""
    if SILVER_ALERT_EVENT_RE.search(event_text):
        return "silver_alert"
    if MISSING_PERSON_EVENT_RE.search(event_text):
        return "missing_person"
    return None


def classify_record(rec: dict, *, start: pd.Timestamp | None = None,
                    end: pd.Timestamp | None = None) -> dict[str, object]:
    """Classify one archive record and return a reason plus parsed metadata."""
    sent = _parse_sent(rec.get("sent"))
    if pd.isna(sent):
        return {"reason": "invalid_sent", "sent": sent}
    if start is not None and sent < start:
        return {"reason": "outside_period", "sent": sent}
    if end is not None and sent >= end:
        return {"reason": "outside_period", "sent": sent}

    status = _field_or_xml(rec, "status", STATUS_RE).casefold()
    if status != "actual":
        return {"reason": "non_actual", "sent": sent, "status": status}

    scope = _field_or_xml(rec, "scope", SCOPE_RE).casefold()
    if scope != "public":
        return {"reason": "non_public", "sent": sent, "scope": scope}

    msg_type = _field_or_xml(rec, "msgType", MSGTYPE_RE).capitalize()
    if msg_type not in {"Alert", "Update", "Cancel"}:
        return {"reason": "unknown_msg_type", "sent": sent, "msg_type": msg_type}

    codes = event_codes(rec)
    events = codes.get("EVENT", set())
    if "CAE" in codes.get("SAME", set()) or "CAE" in codes.get("EVENTCODE", set()):
        return {"reason": "amber", "sent": sent, "msg_type": msg_type}
    # Some senders omit the CAE event code but retain the canonical event text.
    if any("CHILD ABDUCTION" in str(event).upper() for event in events):
        return {"reason": "amber", "sent": sent, "msg_type": msg_type}
    if codes.get("SAME", set()) & TEST_EVENT_CODES:
        return {"reason": "test_event", "sent": sent, "msg_type": msg_type}
    params = wea_parameters(rec)
    handling = {value.casefold() for value in params.get("WEAHANDLING", set())}
    if "wea test" in handling:
        return {"reason": "test_event", "sent": sent, "msg_type": msg_type}
    if has_cmas_block(rec):
        return {"reason": "cmas_blocked", "sent": sent, "msg_type": msg_type}
    if not has_wea_payload(rec):
        return {"reason": "no_wea_payload", "sent": sent, "msg_type": msg_type}

    family = person_alert_family(rec)
    if family is not None:
        return {
            "reason": "person_treatment", "sent": sent, "msg_type": msg_type,
            "alert_family": family, "event_codes": codes, "wea_parameters": params,
        }
    # A WEAHandling=Amber row without CAE is still an AMBER product.  It must
    # not be used as a non-AMBER control; a missing-child headline is handled
    # above as the explicit missing-person treatment family.
    if "amber" in handling:
        return {"reason": "amber", "sent": sent, "msg_type": msg_type,
                "event_codes": codes, "wea_parameters": params}

    return {
        "reason": "keep", "sent": sent, "msg_type": msg_type,
        "event_codes": codes, "wea_parameters": params,
    }


def parse_same_rows(rec: dict) -> list[dict[str, object]]:
    """Parse standard/non-standard SAME codes using the AMBER parser."""
    # ``02c_fetch_openfema_ipaws.py`` has the project's vetted SAME parser,
    # including areaDesc resolution for non-standard EAS-zone prefixes.
    return _PARSER_MODULE._parse_record(rec)


def effective_night_date(sent: pd.Timestamp, tz_name: str,
                         night_start: int = 22, night_end: int = 6) -> pd.Timestamp | None:
    """Return the outcome date for a local overnight send, or ``None``."""
    if not isinstance(sent, pd.Timestamp):
        sent = pd.Timestamp(sent)
    if sent.tzinfo is None:
        sent = sent.tz_localize("UTC")
    local = sent.tz_convert(tz_name)
    hour = int(local.hour)
    if not (hour >= night_start or hour < night_end):
        return None
    date = local.normalize().tz_localize(None)
    if hour >= night_start:
        date += pd.Timedelta(days=1)
    return date


def _county_targets(fips: str, state_map: dict[str, list[str]]) -> list[str]:
    fips = str(fips).zfill(5)
    if not re.fullmatch(r"\d{5}", fips):
        return []
    if fips[2:] == "000":
        return state_map.get(fips[:2], [])
    return [fips]


def process_records(records: Iterable[dict], timezone_map: dict[str, str],
                    state_map: dict[str, list[str]], *, start: pd.Timestamp,
                    end: pd.Timestamp, max_records: int | None = None,
                    exclude_weather: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter records and aggregate unique other-WEA overnight exposures."""
    county_date_ids: dict[tuple[str, pd.Timestamp], set[str]] = defaultdict(set)
    counts: Counter[str] = Counter()
    seen = 0
    for rec in records:
        if max_records is not None and seen >= max_records:
            break
        seen += 1
        classified = classify_record(rec, start=start, end=end)
        reason = str(classified["reason"])
        counts[reason] += 1
        if reason != "keep":
            if seen % 100_000 == 0:
                log.info("Processed %d archive records; kept %d county-night rows", seen,
                         counts["kept_night_county_rows"])
            continue
        if exclude_weather and is_weather_wea(rec):
            counts["weather_excluded"] += 1
            continue
        rows = parse_same_rows(rec)
        if not rows:
            counts["no_numeric_same"] += 1
            continue
        alert_id = str(rec.get("id") or rec.get("identifier") or "")
        if not alert_id:
            counts["missing_alert_id"] += 1
            continue
        row_targets = set()
        for row in rows:
            row_targets.update(_county_targets(str(row.get("fips", "")), state_map))
        if not row_targets:
            counts["no_outcome_county"] += 1
            continue
        for fips in row_targets:
            tz_name = timezone_map.get(fips)
            if not tz_name:
                counts["missing_timezone"] += 1
                continue
            eff_date = effective_night_date(classified["sent"], tz_name)  # type: ignore[arg-type]
            if eff_date is None:
                counts["not_night_local"] += 1
                continue
            county_date_ids[(fips, eff_date)].add(alert_id)
            counts["kept_night_county_rows"] += 1
        if seen % 100_000 == 0:
            log.info("Processed %d archive records; kept %d county-night rows", seen,
                     counts["kept_night_county_rows"])

    controls = pd.DataFrame(
        [
            {"fips": fips, "date": date, "other_wea_night_alert": 1,
             "other_wea_night_count": len(ids)}
            for (fips, date), ids in county_date_ids.items()
        ],
        columns=["fips", "date", "other_wea_night_alert", "other_wea_night_count"],
    )
    if controls.empty:
        controls = pd.DataFrame(columns=["fips", "date", "other_wea_night_alert", "other_wea_night_count"])
    else:
        controls = controls.sort_values(["fips", "date"]).reset_index(drop=True)
        controls["fips"] = controls["fips"].astype(str).str.zfill(5)
        controls["date"] = pd.to_datetime(controls["date"])
    counts["input_records_seen"] = seen
    counts["county_dates"] = len(controls)
    summary = pd.DataFrame([{"reason": k, "count": int(v)} for k, v in sorted(counts.items())])
    return controls, summary


def iter_bulk_records(session: requests.Session, url: str = BULK_URL) -> Iterator[dict]:
    """Stream JSONL records from FEMA's compressed bulk archive."""
    with session.get(url, stream=True, timeout=(30, 300)) as response:
        response.raise_for_status()
        for line in response.iter_lines(chunk_size=1024 * 1024, decode_unicode=True):
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                log.warning("Skipping malformed JSONL record (%d bytes)", len(line))


def _api_get(session: requests.Session, params: dict[str, object]) -> dict:
    """GET one month/page with retry for FEMA's intermittent 5xx/timeouts."""
    for attempt, delay in enumerate((0, *API_RETRY_DELAYS), start=1):
        if delay:
            time.sleep(delay)
        try:
            # The archive normally responds within a few seconds.  A bounded
            # read timeout prevents one stale historical query from blocking
            # the month-by-month run for several minutes; retries preserve
            # completeness when FEMA briefly throttles a request.
            response = session.get(API_URL, params=params, timeout=(30, 30))
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, list):
                return {"IpawsArchivedAlerts": payload}
            return payload
        except (requests.RequestException, ValueError) as exc:
            log.warning("IPAWS API attempt %d failed for skip=%s: %s", attempt,
                        params.get("$skip"), exc)
    raise RuntimeError(f"IPAWS API retries exhausted for params={params}")


def _arcgis_get(session: requests.Session, params: dict[str, object]) -> dict:
    """Fetch one ArcGIS archive page (fast index-backed alternative)."""
    for attempt, delay in enumerate((0, 2, 5, 10), start=1):
        if delay:
            time.sleep(delay)
        try:
            response = session.get(ARCGIS_QUERY_URL, params=params, timeout=(20, 45))
            response.raise_for_status()
            payload = response.json()
            if "error" in payload:
                raise RuntimeError(payload["error"])
            return payload
        except (requests.RequestException, ValueError, RuntimeError) as exc:
            log.warning("ArcGIS archive attempt %d failed for offset=%s: %s", attempt,
                        params.get("resultOffset"), exc)
    raise RuntimeError(f"ArcGIS archive retries exhausted for params={params}")


def _fetch_api_month(year: int, month: int, *, top: int = API_TOP) -> tuple[int, int, list[dict]]:
    """Fetch one month of WEA payloads in an isolated session.

    Each payload token is queried separately because FEMA's OData backend can
    time out on the equivalent three-way OR expression.  The returned records
    are deduplicated locally, then ``classify_record`` applies the definitive
    delivery and treatment exclusions.
    """
    next_month = month + 1 if month < 12 else 1
    next_year = year if month < 12 else year + 1
    period_start = pd.Timestamp(datetime(year, month, 1, tzinfo=timezone.utc))
    period_end = pd.Timestamp(datetime(next_year, next_month, 1, tzinfo=timezone.utc))
    if year < WEA_HANDLING_START_YEAR:
        # Keep historical event-code result sets below FEMA's unreliable
        # $skip=1000 boundary.  Three-day shards are still consumed as one
        # month by the caller and make each retry independently recoverable.
        date_filters = []
        cursor = period_start
        while cursor < period_end:
            window_end = min(cursor + pd.Timedelta(days=3), period_end)
            date_filters.append(
                f"sent ge '{cursor.strftime('%Y-%m-%dT%H:%M:%SZ')}' and "
                f"sent lt '{window_end.strftime('%Y-%m-%dT%H:%M:%SZ')}'"
            )
            cursor = window_end
    else:
        date_filters = [
            f"sent ge '{period_start.strftime('%Y-%m-%dT%H:%M:%SZ')}' and "
            f"sent lt '{period_end.strftime('%Y-%m-%dT%H:%M:%SZ')}'"
        ]
    session = requests.Session()
    session.headers.update({
        "User-Agent": "amber-alert-research/1.0 (academic; IPAWS controls)",
        "Accept": "application/json",
    })
    by_id: dict[str, dict] = {}
    for date_filter in date_filters:
        base_filter = (
            f"({date_filter}) and status eq 'Actual' and scope eq 'Public' and "
            "(msgType eq 'Alert' or msgType eq 'Update' or msgType eq 'Cancel')"
        )
        if year >= WEA_HANDLING_START_YEAR:
            # The WEA payload fields are reliable in current records, including
            # custom missing-person and weather alerts not represented by a
            # small fixed event-code list.
            query_filters = [
                f"{base_filter} and contains(originalMessage, '{token}')"
                for token in WEA_SIGNAL_TOKENS
            ]
        else:
            # Older records often omit WEAHandling and the direct CMAMtext
            # search is unindexed (and can take minutes).  The event-code index
            # is fast; local parsing of the returned CAP still requires a WEA
            # payload.
            query_filters = []
            for start_idx in range(0, len(WEA_QUERY_EVENT_CODES), 8):
                code_group = WEA_QUERY_EVENT_CODES[start_idx:start_idx + 8]
                event_terms = " or ".join(
                    f"(contains(originalMessage, '>{code}<') and "
                    "contains(originalMessage, 'CMAMtext'))"
                    for code in code_group
                )
                query_filters.append(f"{base_filter} and ({event_terms})")

        for combined in query_filters:
            skip = 0
            while True:
                params = {
                    "$top": top,
                    "$skip": skip,
                    "$filter": combined,
                    "$select": "id,identifier,sent,status,msgType,scope,originalMessage,info",
                    "$metadata": "false",
                }
                payload = _api_get(session, params)
                records = payload.get("IpawsArchivedAlerts", []) or []
                if not records:
                    break
                for record in records:
                    key = str(record.get("id") or record.get("identifier") or "")
                    if key:
                        by_id[key] = record
                if len(records) < top:
                    break
                skip += top
                if skip > 20_000:
                    raise RuntimeError(
                        f"more than 20,000 WEA-family records in {year}-{month:02d}; "
                        "refusing an unbounded API loop"
                    )
                time.sleep(API_SLEEP_SECONDS)
    return year, month, list(by_id.values())


def _month_cache_path(year: int, month: int) -> Path:
    return MONTH_CACHE_DIR / f"{year:04d}-{month:02d}.jsonl"


def _load_month_cache(year: int, month: int) -> list[dict] | None:
    """Load a completed monthly raw batch, if present."""
    path = _month_cache_path(year, month)
    if not path.is_file():
        return None
    records: list[dict] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("Ignoring incomplete monthly cache %s: %s", path, exc)
        return None
    return records


def _fetch_api_month_cached(year: int, month: int, *, top: int = API_TOP) -> tuple[int, int, list[dict]]:
    """Fetch one month, persisting a completed raw batch for safe resumption."""
    cached = _load_month_cache(year, month)
    if cached is not None:
        return year, month, cached
    fetched_year, fetched_month, records = _fetch_api_month(year, month, top=top)
    MONTH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _month_cache_path(year, month)
    temp_path = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    with temp_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temp_path, path)
    return fetched_year, fetched_month, records


def _fetch_arcgis_month(year: int, month: int, *, top: int = 2000) -> tuple[int, int, list[dict]]:
    """Fetch public-safety WEA-family rows and hydrate their CAP XML in batches."""
    next_month = month + 1 if month < 12 else 1
    next_year = year if month < 12 else year + 1
    names = ",".join("'" + name.replace("'", "''") + "'" for name in WEA_EVENT_NAMES)
    custom_missing = (
        "info_event LIKE '%Missing%' OR info_event LIKE '%MISSING%' OR "
        "info_event LIKE '%Blue Alert%' OR info_event LIKE '%BLUE Alert%' OR "
        "info_event LIKE '%Silver Alert%' OR info_event LIKE '%SILVER Alert%' OR "
        "info_event LIKE '%Golden Alert%' OR info_event LIKE '%GOLDEN Alert%'"
    )
    event_scope = f"(info_category IN ('Safety','Security') OR {custom_missing} OR info_event IN ({names}))"
    where = (
        f"sent >= DATE '{year}-{month:02d}-01' AND "
        f"sent < DATE '{next_year}-{next_month:02d}-01' AND "
        "status = 'Actual' AND scope = 'Public' AND msgtype IN ('Alert','Update','Cancel') AND "
        + event_scope
    )
    session = requests.Session()
    session.headers.update({"User-Agent": "amber-alert-research/1.0 (academic; IPAWS controls)"})
    attrs: list[dict] = []
    offset = 0
    while True:
        payload = _arcgis_get(session, {
            "where": where,
            "outFields": "id,identifier,sent,status,msgtype,scope,info_event,originalmessage",
            "returnGeometry": "false",
            "resultRecordCount": top,
            "resultOffset": offset,
            "f": "json",
        })
        features = payload.get("features", []) or []
        attrs.extend(feature.get("attributes", {}) for feature in features)
        if len(features) < top or not payload.get("exceededTransferLimit"):
            break
        offset += top

    # ArcGIS stores a link to the raw CAP XML, so hydrate only the small set
    # of event-family candidates rather than downloading the whole archive.
    ids = [str(row.get("identifier") or "") for row in attrs if row.get("identifier")]
    by_id: dict[str, dict] = {}
    # OpenFEMA rejects long OR expressions for some vendor-generated
    # identifiers (notably OnSolve).  Batches of 20 stay below that parser
    # limit while still avoiding one HTTP request per alert.
    # FEMA's API intermittently returns 503 for larger identifier filters.  Keep
    # hydration batches small enough to be reliable while still avoiding one
    # request per candidate alert.
    for start_idx in range(0, len(ids), 5):
        batch = ids[start_idx:start_idx + 5]
        identifier_filter = " or ".join(
            "identifier eq '" + ident.replace("'", "''") + "'" for ident in batch
        )
        payload = _api_get(session, {
            "$top": 1000,
            "$skip": 0,
            "$filter": identifier_filter,
            "$select": "id,identifier,sent,status,msgType,scope,originalMessage,info",
            "$metadata": "false",
        })
        for record in payload.get("IpawsArchivedAlerts", []) or []:
            key = str(record.get("identifier") or record.get("id") or "")
            if key:
                by_id[key] = record
    missing = [ident for ident in ids if ident not in by_id]
    if missing:
        raise RuntimeError(f"ArcGIS candidates missing raw CAP records (first IDs: {missing[:5]})")
    return year, month, list(by_id.values())


def iter_api_records(session: requests.Session, start_year: int, end_year: int,
                     *, top: int = API_TOP, workers: int = 4,
                     max_months: int | None = None) -> Iterator[dict]:
    """Yield WEA-family records month by month from the OpenFEMA API."""
    del session  # each worker owns a session; retained in the signature for callers
    months = [(year, month) for year in range(start_year, end_year + 1)
              for month in range(1, 13)]
    if max_months is not None:
        months = months[:max_months]
    if workers <= 1:
        for year, month in months:
            fetched_year, fetched_month, records = _fetch_api_month_cached(year, month, top=top)
            log.info("%d-%02d: %d WEA-family archive records", fetched_year, fetched_month, len(records))
            yield from records
        return
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [executor.submit(_fetch_api_month_cached, year, month, top=top)
                   for year, month in months]
        for future in as_completed(futures):
            year, month, records = future.result()
            log.info("%d-%02d: %d WEA-family archive records", year, month, len(records))
            yield from records


def iter_arcgis_records(start_year: int, end_year: int, *, workers: int = 1,
                        max_months: int | None = None) -> Iterator[dict]:
    """Yield records from the indexed FEMA ArcGIS archive month by month."""
    months = [(year, month) for year in range(start_year, end_year + 1)
              for month in range(1, 13)]
    if max_months is not None:
        months = months[:max_months]
    if workers <= 1:
        for year, month in months:
            fetched_year, fetched_month, records = _fetch_arcgis_month(year, month)
            log.info("%d-%02d: %d ArcGIS WEA-family archive records", fetched_year, fetched_month, len(records))
            yield from records
        return
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [executor.submit(_fetch_arcgis_month, year, month) for year, month in months]
        for future in as_completed(futures):
            year, month, records = future.result()
            log.info("%d-%02d: %d ArcGIS WEA-family archive records", year, month, len(records))
            yield from records


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=min(STUDY_YEARS))
    parser.add_argument("--end-year", type=int, default=max(STUDY_YEARS))
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--url", default=BULK_URL)
    parser.add_argument("--bulk", action="store_true",
                        help="use the compressed bulk JSONL endpoint instead of month-batched API")
    parser.add_argument("--source", choices=("api", "arcgis", "bulk"), default="api",
                        help="archive source; API payload filtering is exhaustive (default)")
    parser.add_argument(
        "--exclude-weather", action="store_true",
        help="write a sensitivity control excluding CAP category=Met WEA records",
    )
    parser.add_argument("--workers", type=int, default=1,
                        help="concurrent month API requests (default: 1; use cautiously with FEMA)")
    parser.add_argument("--force", action="store_true", help="rebuild existing output")
    args = parser.parse_args(argv)
    if args.end_year < args.start_year:
        parser.error("--end-year must be at least --start-year")
    control_path = WEATHER_EXCLUDED_CONTROL_PATH if args.exclude_weather else CONTROL_PATH
    summary_path = WEATHER_EXCLUDED_SUMMARY_PATH if args.exclude_weather else SUMMARY_PATH
    if control_path.exists() and not args.force:
        log.info("Control file already exists at %s; use --force to rebuild", control_path)
        return

    start = pd.Timestamp(datetime(args.start_year, 1, 1, tzinfo=timezone.utc))
    end = pd.Timestamp(datetime(args.end_year + 1, 1, 1, tzinfo=timezone.utc))
    timezone_map = amber_base.county_timezone_map(DATA_PROC / "county_pop_centroids.parquet")
    state_map = amber_base._state_county_map()
    session = requests.Session()
    session.headers.update({
        "User-Agent": "amber-alert-research/1.0 (academic; IPAWS controls)",
        "Accept": "application/jsonl+json",
    })
    source = "bulk" if args.bulk else args.source
    log.info("Fetching FEMA IPAWS archive %s–%s (month-batched %s)", args.start_year,
             args.end_year, source)
    started = time.monotonic()
    if source == "bulk":
        records = iter_bulk_records(session, args.url)
    elif source == "api":
        records = iter_api_records(session, args.start_year, args.end_year, workers=args.workers,
                                   max_months=1 if args.max_records is not None else None)
    else:
        records = iter_arcgis_records(args.start_year, args.end_year, workers=args.workers,
                                      max_months=1 if args.max_records is not None else None)
    controls, summary = process_records(
        records, timezone_map, state_map, start=start, end=end, max_records=args.max_records,
        exclude_weather=args.exclude_weather,
    )
    DATA_PROC.mkdir(parents=True, exist_ok=True)
    OUTPUT_TABS.mkdir(parents=True, exist_ok=True)
    controls.to_parquet(control_path, index=False)
    summary.to_csv(summary_path, index=False)
    log.info("Saved %s (%d county-dates) and %s in %.1fs", control_path,
             len(controls), summary_path, time.monotonic() - started)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
