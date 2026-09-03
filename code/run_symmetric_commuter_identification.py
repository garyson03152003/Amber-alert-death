"""Run the additional rich commuter identification and mechanism checks.

This runner deliberately writes new tables instead of replacing the existing
headline and six-family robustness outputs.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

import run_symmetric_commuter_fatigue as base
import run_symmetric_commuter_robustness as robustness
from config import OUTPUT_TABS
from utils import get_logger

log = get_logger("symmetric_commuter_identification")

IDENTIFICATION_FILENAME = "reg_symmetric_commuter_identification.csv"
NETWORK_PLACEBO_FILENAME = "symmetric_commuter_network_placebo.csv"
TIME_BLOCK_FILENAME = "reg_symmetric_commuter_time_blocks.csv"
NETWORK_PLACEBO_DRAWS = 199
NETWORK_PLACEBO_SEED = 20260901


def save_outputs(
    state_rows: list[dict],
    network_rows: list[dict],
    network_distribution: pd.DataFrame,
    time_rows: list[dict],
    *,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write the three robustness tables and return the two estimate tables."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    identification = pd.concat(
        [pd.DataFrame(state_rows), pd.DataFrame(network_rows)],
        ignore_index=True,
        sort=False,
    )
    time_blocks = pd.DataFrame(time_rows)
    identification.to_csv(output_dir / IDENTIFICATION_FILENAME, index=False)
    network_distribution.to_csv(output_dir / NETWORK_PLACEBO_FILENAME, index=False)
    time_blocks.to_csv(output_dir / TIME_BLOCK_FILENAME, index=False)
    return identification, time_blocks


def main(
    *,
    bootstrap_reps: int = base.BOOTSTRAP_REPS,
    placebo_draws: int = NETWORK_PLACEBO_DRAWS,
    placebo_seed: int = NETWORK_PLACEBO_SEED,
    swaps_per_edge: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the common panel once and run all three new robustness families."""
    panel, alerts, metadata = robustness._load_common_panel()
    state_rows = robustness.run_state_date_models(
        panel,
        bootstrap_reps=bootstrap_reps,
        seed=base.BOOTSTRAP_SEED + 700,
    )
    network_distribution, network_rows = robustness.run_network_placebos(
        panel,
        alerts,
        metadata,
        draws=placebo_draws,
        seed=placebo_seed,
        swaps_per_edge=swaps_per_edge,
    )
    time_rows = robustness.run_time_block_models(
        panel,
        bootstrap_reps=bootstrap_reps,
        seed=base.BOOTSTRAP_SEED + 900,
    )

    common = {
        "analysis_counties": len(metadata["active"]),
        "excluded_no_self_loop_counties": len(metadata["excluded"]),
        "exposure_spec": "year_matched_acs_and_lodes",
    }
    for rows in (state_rows, network_rows, time_rows):
        for row in rows:
            row.update(common)
    for column, value in common.items():
        network_distribution[column] = value

    identification, time_blocks = save_outputs(
        state_rows,
        network_rows,
        network_distribution,
        time_rows,
        output_dir=OUTPUT_TABS,
    )
    log.info(
        "Saved %d identification rows, %d network draws, and %d time-block rows",
        len(identification),
        len(network_distribution),
        len(time_blocks),
    )
    return identification, time_blocks


if __name__ == "__main__":
    main()

