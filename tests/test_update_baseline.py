"""
Tests of update_baseline.py module
"""

import json
from pathlib import Path

from ogcore.parameters import Specifications

from ogzaf import update_baseline


class MockCalibration:
    """
    Minimal calibration stub for update_baseline.main().
    """

    def __init__(self, p, update_from_api):
        self.p = p
        self.update_from_api = update_from_api

    def get_dict(self):
        return {"frisch": 0.5, "g_y_annual": 0.03}


def test_main_json_updates_specifications(monkeypatch, tmp_path):
    """
    JSON written by main() can be loaded into Specifications without error.
    """
    output_dir = tmp_path / "baseline_output"
    output_dir.mkdir()

    monkeypatch.setattr(update_baseline, "Calibration", MockCalibration)
    monkeypatch.setattr(
        update_baseline.os.path,
        "realpath",
        lambda _: str(output_dir / "update_baseline.py"),
    )

    update_baseline.main()

    json_path = Path(output_dir) / "ogzaf_default_parameters.json"
    saved_params = json.loads(json_path.read_text(encoding="utf-8"))

    p = Specifications(baseline=True)
    p.update_specifications(saved_params)

    assert not p.errors
    assert p.frisch == saved_params["frisch"]
    assert p.g_y_annual == saved_params["g_y_annual"]
