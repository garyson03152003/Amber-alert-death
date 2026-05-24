# HANDOFF: Amber Alert Traffic Fatality Analysis

**Paste this at the start of a new Claude Code session.**  
**GitHub repo**: `garyson03152003/amber-alert-death`  
**Working branch**: `claude/amber-alerts-traffic-fatalities-c0nxz`  
**PR #1**: open draft at https://github.com/garyson03152003/Amber-alert-death/pull/1

---

## Project Overview

An econometric study testing whether nighttime AMBER Alerts increase next-day traffic fatalities through driver distraction/sleep disruption. Uses FARS 2013–2024 × IPAWS alert data in a county-day panel.

**Key numbers**:
- Panel: 1,655 counties × 4,372 days = 7,234,005 county-days (filtered: ≥5 fatals/yr)
- Treated nights: 9,621 county-nights with a `night_alert` (8pm–6am)
- Outcome: `fatals_next_commute` — fatalities on day t+1 aligned to alert timing

---

## Repo Structure

```
code/
  config.py                    # DATA_RAW, DATA_PROC, OUTPUT_TABS paths
  utils.py                     # get_logger()
  analysis_lib.py              # prep_panel() — applies county filter, adds state_code, year_str
  05_analysis.py               # load_panel(), add_aligned_outcome(), run_commuting_spillover()
  06_figures.py                # figure generation

  # Data build scripts (run once; data already on disk)
  01_download_fars.py          # downloads FARS ZIPs → data/raw/fars/
  01b_extract_serious_injuries.py
  01c_fetch_weather.py / 01d_merge_weather.py
  01e_fetch_car_commuters.py   # ACS B08301 → county_car_commuters.parquet
  01f_fetch_cell_connectivity.py
  01g_build_coverage_weight.py
  04_build_panel.py            # builds panel_county_day.parquet

  # Analysis runners (main results)
  run_poisson_fe.py            # Poisson PPML (P1–P6) → reg_poisson.csv
  run_affected_commuters.py    # Commuter dosage (AC1–AC4) → reg_affected_commuters.csv
  run_commute_exposure_dosage.py  # alt commuter dosage (CD1–CD5) → reg_commute_dosage.csv
  run_commuting_spillover.py   # cross-county spillover → reg_commuting_spillover.csv
  run_time_window_analysis.py  # FARS hourly time windows (W0–W4) → reg_time_window.csv
  run_weather_robustness.py
  run_weekend_puzzle.py
  make_latex_tables.py         # → output/tables/tab1–tab6.tex

  # State DOT crash data (need internet to run — see data/README_state_dot_data.md)
  build_california_ccrs.py     # CA CCRS (data.ca.gov, no auth, 2016–2024)
  build_illinois_idot.py       # IL IDOT (ArcGIS, no auth, 2016–2024)
  build_commuting_weights.py   # ACS commuting flows → county_commuting_weights.parquet

data/
  raw/fars/                    # FARS2013–2024NationalCSV.zip (ALL on disk)
  processed/
    panel_county_day.parquet   # 13.75M rows, 3,146 counties (unfiltered)
    amber_alerts_clean.parquet # cols: alert_id, state_fips, county_fips, issued_utc,
                               #       issued_local, hour_local, is_night, night_band
    fars_county_day.parquet    # daily fatality counts by county
    fars_serious_injuries.parquet  # county-day: fips, date, serious_injuries (72k rows)
    fars_hourly.parquet        # 379,563 crash records with HOUR field (built by run_time_window)
    county_car_commuters.parquet   # cols: fips, car_total
    county_population.parquet
    county_centroids.parquet
    commuting/
      county_commuting_weights.parquet  # cols: fips_home, fips_work, workers, weight
                                        # 119,161 OD pairs, 3,144 counties
  README_state_dot_data.md     # download instructions for CA/TX/IL/FL state data

output/
  tables/                      # CSVs + LaTeX for all regressions
  figures/                     # fig1–fig10 as PDF + PNG
```

---

## The Panel Loading Pattern

Every runner does this:
```python
import sys, importlib.util
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_PROC, OUTPUT_TABS
from utils import get_logger
from analysis_lib import prep_panel

spec = importlib.util.spec_from_file_location("a05", Path(__file__).parent / "05_analysis.py")
a05  = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a05)

df = a05.load_panel()       # loads panel_county_day.parquet (13.75M rows)
df = prep_panel(df)         # applies ≥5 fatals/yr filter → 1,655 counties, 7.2M rows
df = a05.add_aligned_outcome(df)  # adds fatals_next_commute, serious_inj_next_commute
df["fips"]       = df["fips"].astype(str)
df["state_code"] = df["state_code"].astype(str)
df["year_str"]   = pd.to_datetime(df["date"]).dt.year.astype(str)
```

Key columns after prep:
- `fips` — 5-digit string FIPS
- `date` — datetime64
- `night_alert` — 0/1 (alert fired 8pm–6am)
- `alert_hour` — local hour of alert (0–23)
- `night_band` — "late_night" / "deep_night" / NaN
- `alert_breadth` — number of counties in same alert
- `fatals_t0/t1/tm1/t2/t3` — fatalities on day D, D+1, D-1, D+2, D+3
- `fatals_next_commute` — timing-aligned outcome (t1 for evening alerts, t0 for midnight alerts)
- `serious_inj_next_commute` — same timing for serious injuries
- `prcp_mm`, `tmax_c` — weather controls
- `is_holiday` — holiday indicator
- `population` — county population

Regression pattern:
```python
import pyfixest as pf
fit = pf.feols(
    "fatals_next_commute ~ night_alert + is_holiday + prcp_mm + tmax_c | fips + year_str",
    data=df, vcov={"CRV1": "state_code"}, lean=True
)
```

**Memory**: 7.2M rows is near OOM limit. Always use `lean=True`. Add `del fit, sub; gc.collect()` after each regression. Never use `fips:year_str` interaction FE (OOM). Use additive `fips + year_str`.

---

## All Regression Results

### Baseline (OLS, binary night_alert, county+year FE)
| Spec | β | SE | p |
|------|---|----|---|
| Pooled OLS (no FE) | +0.0308 | 0.0104 | 0.003 |
| County FE only | +0.0050 | 0.0047 | 0.289 |
| County + Year FE (main) | +0.0044 | 0.0037 | **0.244 n.s.** |
| + DoW×Year FE | +0.0056 | 0.0047 | 0.229 |

### Dosage: log_breadth (OLS, county+year FE)
| Spec | β | SE | p |
|------|---|----|---|
| Binary baseline | +0.00453 | 0.00377 | 0.236 |
| Log-breadth | +0.00147 | 0.00070 | **0.041** |
| Log-breadth + lags | +0.00147 | 0.00070 | **0.042** |
| Log-breadth + State×Year FE | +0.00185 | 0.00087 | **0.039** |
| Log-breadth (WLS binary baseline) | +0.0176 | 0.0093 | 0.065 |

### Commuter dosage: log_affected_commuters
`affected_{i,t} = car_total_i × alert_{i,t} + Σ_{j≠i: alerted} workers_{j→i}`
| Spec | β | SE | p | n |
|------|---|----|---|---|
| AC1 OLS count TWFE1 | +0.000742 | 0.000358 | **0.043** | 7,177,182 |
| AC2 WLS rate/100k | +0.000049 | 0.000049 | 0.329 n.s. | 5,980,164 |
| AC3 Poisson count | +0.003382 | 0.001821 | 0.063* | 7,177,182 |
| AC4 Poisson rate/100k pop-wt | +0.002529 | 0.001876 | 0.178 n.s. | 5,980,164 |
| BIN night_alert (benchmark) | +0.005653 | 0.004776 | 0.242 n.s. | 7,177,182 |

### Commuter dosage: log_commute_dosage (all commuters, including own-county)
| Spec | β | SE | p |
|------|---|----|---|
| CD1 OLS count TWFE1 | +0.000749 | 0.000364 | **0.045** |
| CD3 WLS rate/100k | +0.000047 | 0.000050 | 0.343 n.s. |
| CD4 Poisson count | +0.003355 | 0.001829 | 0.067* |
| CD5 Poisson rate/100k pop-wt | +0.002482 | 0.001884 | 0.188 n.s. |

### Poisson PPML (pyfixest.fepois, county+year FE)
| Spec | β | SE | p | IRR |
|------|---|----|---|-----|
| P1 Binary, count | +0.0628 | 0.0659 | 0.341 n.s. | 1.065 |
| P2 Log-breadth, count | +0.0281 | 0.0136 | **0.038** | 1.029 |
| P5 Binary, rate/100k pop-wt | +0.2068 | 0.0847 | **0.015** | 1.230 |
| P6 Log-breadth, rate/100k pop-wt | +0.0540 | 0.0143 | **<0.001** | 1.056 |
| NB Binary (100-cty subsample) | +0.0969 | 0.3724 | 0.795 n.s. | 1.102 |

### Time-window (binary night_alert, TWFE OLS, FARS hourly data)
Window = crash hours relative to alert night D
| Window | Fatal β | p | Serious β | p |
|--------|---------|---|-----------|---|
| W0 Same night (D 20:00–D+1 06:00) | +0.00271 | 0.256 | +0.00109 | 0.319 |
| W1 Morning commute (D+1 06:00–10:00) | -0.00057 | 0.698 | -0.00075 | 0.189 |
| W2 Midday (D+1 10:00–16:00) | -0.00157 | 0.574 | +0.00112 | 0.406 |
| W3 Evening (D+1 16:00–20:00) | -0.00185 | 0.147 | +0.00038 | 0.726 |
| W4 Placebo D+2 morning | -0.00010 | **0.924** | -0.00091 | 0.006* |
→ No time-window is significantly positive. W4 placebo is clean for fatals.
→ W4 serious injury p=0.006 is likely multiple-testing artifact (10 tests, negative direction).
→ Binary treatment consistently null in every window.

---

## Key Scientific Findings

1. **Binary alert has no significant effect** (p≈0.24 OLS, p=0.34 Poisson). Alert presence alone doesn't predict fatalities.

2. **Alert reach (log_breadth) is significant** at 5% in both OLS (p=0.041) and Poisson (p=0.038). Larger geographic footprint alerts → more fatalities. But `log_breadth` is an alert-level property — same for all 97 counties in same alert.

3. **Commuter dosage (log_affected_commuters) is significant** at 5% in OLS count (p=0.043). This varies at the county level within the same alert night (CV=0.141). More directly interpretable mechanism.

4. **No time-window evidence for mechanism**: Neither sleep disruption (W1 morning commute) nor immediate disruption (W0 same night) is significant with binary treatment. The effect may operate through a different channel, or only through large-reach alerts.

5. **Rate models inconsistent**: Effect significant in count OLS (p=0.043) but not in rate/100k WLS (p=0.329). Suggests effect is concentrated in high-count (high-population, high-traffic) counties — not a uniform rate increase.

6. **Multiple testing concern**: ~25 specifications run. Several at p≈0.04–0.06. A Bonferroni/Romano-Wolf correction would likely make most non-significant. The consistent direction across OLS/Poisson for log_breadth and commuter dosage is the main protection.

---

## Pending Tasks

### High priority
1. **Texas CRIS registration**: Texas has 24,737 alerts (by far the most). Register FREE at https://cris.dot.state.tx.us/ → self-register → request CSV extract 2013–2024. Takes ~24h. This would dramatically increase power.

2. **Write build_texas_cris.py**: Once CRIS data is on disk, build county-day panel. Key columns: `Crash_Date`, `County` (name), `Crash_Sev_ID` (1=Fatal, 2=Incapacitating). Map county names to FIPS, aggregate to county-day.

3. **Run state serious injuries regression**: Once CA CCRS or TX CRIS is available, run:
   ```python
   pf.feols("serious_injuries ~ night_alert + ... | fips + year_str", ...)
   ```
   This directly tests the causality hypothesis with a broader injury outcome.

### Medium priority
4. **Time-window with commuter dosage**: Re-run `run_time_window_analysis.py` using `log_affected_commuters` instead of binary `night_alert`. The dosage variable is significant in the daily spec (p=0.043) — does it show a morning-commute peak?

5. **Romano-Wolf multiple testing correction**: ~25 specs tested; need familywise error control for publication. The `rw` package in R or custom bootstrap in Python.

6. **California CCRS + Illinois IDOT downloads**: Scripts are ready. Need to run on a machine with internet access:
   ```bash
   python3 code/build_california_ccrs.py   # → data/processed/california_ccrs_county_day.parquet
   python3 code/build_illinois_idot.py     # → data/processed/illinois_idot_county_day.parquet
   ```

### Already done / not needed
- NHTSA CRSS: no county IDs in public files → cannot use for county-level analysis
- North Carolina, Georgia, Tennessee: dashboard/PDF only → no bulk download path
- FARS time-window (hourly, W0–W4): done → reg_time_window.csv

---

## Environment Notes

- **Python network access**: The remote execution environment has NO outbound internet access from Python (`urllib`, `requests` etc. all fail with DNS errors). WebFetch tool works for URL inspection. State DOT download scripts must be run locally.
- **Memory**: 7.2M row panel is near OOM. Use `lean=True`, `gc.collect()`, and avoid `fips:year_str` interaction FE. Full 13.75M panel OOM-kills even OLS.
- **Pyfixest version**: 0.50.1. No `offset=` parameter in `fepois()`. Encode population exposure in outcome: `fatals_rate_100k = fatals * 100000 / population`.
- **FARS ZIPs**: 2013–2024 on disk at `data/raw/fars/`. 2021–2022 have UTF-8 BOM → use `encoding='latin1'` and strip `ï»¿` from column names. 2013–2014 have nested directory structure inside ZIP.

---

## How to Continue

```bash
# Clone / pull the branch
git clone https://github.com/garyson03152003/Amber-alert-death
cd Amber-alert-death
git checkout claude/amber-alerts-traffic-fatalities-c0nxz

# Check what's already built
ls data/processed/
ls output/tables/

# Re-run a specific analysis
python3 code/run_affected_commuters.py       # takes ~12 min
python3 code/run_poisson_fe.py               # takes ~15 min
python3 code/run_time_window_analysis.py     # takes ~10 min (uses cached fars_hourly.parquet)

# After getting TX CRIS data:
# Place CSV in data/raw/texas_cris/ and write build_texas_cris.py

# Commit + push
git add -A && git commit -m "..." && git push -u origin claude/amber-alerts-traffic-fatalities-c0nxz
```
