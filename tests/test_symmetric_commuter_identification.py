import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))


def test_identification_output_writer_keeps_three_tables_separate(tmp_path):
    from run_symmetric_commuter_identification import save_outputs

    state_rows = [{"spec": "state_date_fixed_effects", "outcome": "fatal_crashes"}]
    network_rows = [
        {"spec": "observed_vs_placebo_network", "outcome": "fatal_crashes"}
    ]
    distribution = pd.DataFrame(
        [{"draw": 1, "outcome": "fatal_crashes", "cross_coef": 0.2}]
    )
    time_rows = [
        {"spec": "state_date_time_blocks", "outcome": "fatals_avg_0609"}
    ]

    identification, time_blocks = save_outputs(
        state_rows,
        network_rows,
        distribution,
        time_rows,
        output_dir=tmp_path,
    )

    assert identification["spec"].tolist() == [
        "state_date_fixed_effects", "observed_vs_placebo_network"
    ]
    assert time_blocks["outcome"].tolist() == ["fatals_avg_0609"]
    assert pd.read_csv(tmp_path / "symmetric_commuter_network_placebo.csv").shape[0] == 1
    assert (tmp_path / "reg_symmetric_commuter_identification.csv").exists()
    assert (tmp_path / "reg_symmetric_commuter_time_blocks.csv").exists()

