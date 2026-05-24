# Nighttime AMBER Alerts and Next-Day Traffic Fatalities

Replication code for the paper:

> **Do Nighttime AMBER Alerts Increase Traffic Fatalities? Evidence from Wireless Emergency Alert Disruptions**

**Research question:** Do AMBER Alerts issued during nighttime hours (10 pm – 5 am) cause a measurable increase in traffic fatalities the following day, via population-level sleep disruption impairing next-day driving performance?

**Identification:** Quasi-random timing and county-level geographic variation in nighttime WEA alerts; county and day-of-week × month fixed effects; weather controls.

---

## Causal chain

```
Nighttime AMBER Alert (WEA — sounds even on silent phones)
        ↓
Population sleep disruption (~30–90 min lost)
        ↓
Next-morning impaired driving (reduced reaction time, attention lapses)
        ↓
Increased traffic fatalities (county, day t+1)
```

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Step 0: county population (Census PEP)
python code/00_download_population.py

# Step 1: NHTSA FARS fatality data (2013–2022)
python code/01_download_fars.py

# Step 2: AMBER Alert issuance records
#   — If you have FOIA / IPAWS data: place CSVs in data/raw/amber/foia/
#   — Otherwise the script falls back to GDELT news proxies
python code/02_collect_amber_alerts.py

# Step 3: NOAA GHCND county-day weather controls
python code/03_download_weather.py

# Step 4: Build the analysis panel
python code/04_build_panel.py

# Step 5: Regressions
python code/05_analysis.py

# Step 6: Figures
python code/06_figures.py
```

Results appear in `output/tables/` (CSV + LaTeX) and `output/figures/` (PDF + PNG).

---

## Data sources

| Dataset | Variable | Source | Notes |
|---------|----------|--------|-------|
| FARS | County-day traffic fatalities | NHTSA | Annual files, 2013–2022 |
| FEMA IPAWS | Alert timestamp, type, broadcast area | FEMA | Public REST feed; FOIA for full polygon logs |
| NCMEC / state DOJ | AMBER alert dates and times | Various | Place CSVs in `data/raw/amber/foia/` |
| NOAA GHCND | Precipitation, temperature | NOAA NCEI | Station-to-county via Census gazetteer |
| Census PEP | County population (2013–2022) | Census Bureau | For density heterogeneity |

### AMBER Alert data — obtaining better records

The cleanest source is **FEMA IPAWS WEA dispatch logs** (precise timestamps + CAP polygon broadcast areas). These can be obtained via FOIA request to FEMA. Address requests to:

> FEMA Office of the Chief Counsel, FOIA Division  
> 500 C Street SW, Washington, DC 20472  
> fema-foia@fema.dhs.gov

Alternatively, many states publish AMBER alert histories through their State Police / DOJ websites. Compile these and place them as CSVs in `data/raw/amber/foia/` — the loader in `02_collect_amber_alerts.py` will pick them up automatically.

---

## Estimation strategy

**Baseline specification:**

```
fatals_{c,t+1} = α + β·NightAlert_{c,t} + γ_c + δ_{dow×month} + X_{c,t}·θ + ε_{c,t}
```

- `NightAlert_{c,t}` = 1 if county *c* received a nighttime WEA AMBER alert (10 pm – 5 am, local time) on day *t*
- `γ_c` = county fixed effects
- `δ_{dow×month}` = day-of-week × month fixed effects
- `X_{c,t}` = precipitation, max temperature
- Standard errors clustered at the county level

**Heterogeneity:** Time-of-night band (early / deep / late night); next-day type (weekday vs. weekend); population density quartile.

**Placebo tests:** Outcome at *t*−1 and *t*+2; both should be null if identification is valid.

---

## Repository structure

```
code/
  00_download_population.py   Census county population estimates
  01_download_fars.py         NHTSA FARS county-day fatality panel
  02_collect_amber_alerts.py  AMBER alert records (IPAWS / FOIA / GDELT)
  03_download_weather.py      NOAA GHCND county-day weather
  04_build_panel.py           Merge all sources → analysis panel
  05_analysis.py              OLS regressions with county + dow×month FE
  06_figures.py               Event study, timing histogram, forest plot
  config.py                   Central parameters (years, night hours, paths)
  utils.py                    Shared utilities (logger, downloader, FIPS helper)

data/
  raw/                        Downloaded source files (gitignored)
    amber/foia/               Place FOIA / state-level alert CSVs here
  processed/                  Cleaned intermediate and final files

output/
  tables/                     Regression results (CSV + LaTeX)
  figures/                    Plots (PDF + PNG)
```

---

## Threats to identification and responses

| Threat | Response |
|--------|----------|
| Alerts cluster on weekends | Day-of-week × month FEs |
| Alerts cluster in high-crime counties | County FEs absorb time-invariant characteristics |
| Distracted driving from checking phones | Separate channel; test by crash hour-of-day |
| Alerts cancelled quickly (no sleep disruption) | Heterogeneity by alert duration; restrict to uncancelled alerts |
| Small county-day cell sizes | Poisson specification; aggregate to state-week as robustness |

---

## Citation

If you use this code, please cite: *[citation forthcoming]*
