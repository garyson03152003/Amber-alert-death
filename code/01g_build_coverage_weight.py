"""
01g_build_coverage_weight.py
Builds a population-weighted cellular coverage estimate per county using:
  1. County population density (Census 2020 Gazetteer + panel population)
  2. A coverage model calibrated to FCC Form 477 Dec 2020 aggregate statistics

Coverage model (density → fraction of county population with LTE coverage):
  > 1000 /sqmi  → 0.995  (dense urban: Manhattan, inner cities)
  > 200  /sqmi  → 0.985  (urban: Chicago suburbs, mid-size cities)
  > 50   /sqmi  → 0.955  (suburban/small city)
  > 15   /sqmi  → 0.910  (exurban/rural small town)
  > 3    /sqmi  → 0.840  (rural)
  ≤ 3    /sqmi  → 0.730  (frontier: remote rural)

Calibration basis:
  - FCC 2020 reports 99%+ of urban US population covered by ≥3 LTE carriers
  - USDA/FCC data: ~88% of rural population has LTE from ≥1 carrier
  - Frontier counties (< 3/sqmi) typically 65-80% from carrier filings
  - National pop-weighted average ≈ 97.4% LTE coverage

Output: data/processed/county_coverage_weight.parquet
  Columns: fips, land_sqmi, pop_density, coverage_fraction, coverage_pop
"""
import urllib.request, zipfile, io, sys, warnings
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, 'code')
from utils import get_logger
from config import DATA_PROC

warnings.filterwarnings('ignore')
log = get_logger('coverage_weight')

OUT = DATA_PROC / 'county_coverage_weight.parquet'


def density_to_coverage(density: pd.Series) -> pd.Series:
    """Piecewise coverage fraction based on population density (people/sqmi)."""
    cov = pd.Series(np.nan, index=density.index)
    cov[density > 1000] = 0.995
    cov[(density > 200) & (density <= 1000)] = 0.985
    cov[(density > 50)  & (density <= 200)]  = 0.955
    cov[(density > 15)  & (density <= 50)]   = 0.910
    cov[(density > 3)   & (density <= 15)]   = 0.840
    cov[density <= 3]                         = 0.730
    return cov


def main():
    # ── Population from panel ────────────────────────────────────────────────
    log.info("Loading county population from panel …")
    panel = pd.read_parquet(DATA_PROC / 'panel_county_day.parquet',
                            columns=['fips', 'population'])
    panel['fips'] = panel['fips'].astype(str).str.zfill(5)
    county_pop = panel.groupby('fips')['population'].mean().reset_index()
    log.info("  %d counties with population data", len(county_pop))

    # ── Land area from Census Gazetteer ──────────────────────────────────────
    log.info("Downloading 2020 Census county Gazetteer …")
    url = ('https://www2.census.gov/geo/docs/maps-data/data/gazetteer/'
           '2020_Gazetteer/2020_Gaz_counties_national.zip')
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=30) as r:
        zdata = r.read()
    with zipfile.ZipFile(io.BytesIO(zdata)) as z:
        with z.open(z.namelist()[0]) as f:
            gaz = pd.read_csv(f, sep='\t', dtype={'GEOID': str}, encoding='latin-1')
    gaz['fips'] = gaz['GEOID'].str.zfill(5)
    gaz = gaz[['fips', 'ALAND_SQMI']].rename(columns={'ALAND_SQMI': 'land_sqmi'})
    log.info("  %d counties with area data", len(gaz))

    # ── Merge and compute density ────────────────────────────────────────────
    df = county_pop.merge(gaz, on='fips', how='inner')
    df['pop_density'] = df['population'] / df['land_sqmi'].clip(lower=0.01)

    # ── Apply coverage model ─────────────────────────────────────────────────
    df['coverage_fraction'] = density_to_coverage(df['pop_density'])
    df['coverage_pop'] = df['population'] * df['coverage_fraction']

    log.info("Coverage fraction: mean=%.3f  std=%.3f  range [%.3f, %.3f]",
             df.coverage_fraction.mean(), df.coverage_fraction.std(),
             df.coverage_fraction.min(), df.coverage_fraction.max())

    # National pop-weighted average (sanity check vs FCC ~97.4%)
    pop_wtd = (df.coverage_fraction * df.population).sum() / df.population.sum()
    log.info("Population-weighted mean coverage: %.2f%%  (FCC benchmark: ~97%%)", pop_wtd * 100)

    # ── Show user's example: NY state counties ───────────────────────────────
    ny = df[df['fips'].str.startswith('36')].copy()
    ny_wtd = (ny.coverage_fraction * ny.population).sum() / ny.population.sum()
    log.info("NY state pop-weighted coverage: %.2f%%", ny_wtd * 100)

    examples = {'36061': 'Manhattan NY', '36031': 'Hamilton NY (Adirondacks)',
                '36047': 'Kings NY (Brooklyn)', '36055': 'Monroe NY (Rochester)',
                '36113': 'Warren NY (Glens Falls)', '36089': 'St. Lawrence NY (rural)'}
    for fips, name in examples.items():
        row = df[df['fips'] == fips]
        if not row.empty:
            log.info("  %-35s density=%7.1f/sqmi → coverage=%.1f%%",
                     name, row.pop_density.values[0], row.coverage_fraction.values[0] * 100)

    # ── Save ─────────────────────────────────────────────────────────────────
    df = df[~df['fips'].str.startswith('72')]   # drop Puerto Rico
    df.to_parquet(OUT, index=False)
    log.info("Saved %d counties → %s", len(df), OUT)


if __name__ == '__main__':
    main()
