"""Fast value-pinning tests for the packaged default parameters.

Load the shipped ogzaf_default_parameters.json and assert specific curated
calibration values with their sources. No model solve — these tests exist so
an accidental regeneration of the JSON (e.g. running update_baseline.py) or
a live-API refresh cannot silently change hand-curated values.
"""

import json
from importlib.resources import files


def load_defaults():
    content = (
        files("ogzaf")
        .joinpath("ogzaf_default_parameters.json")
        .read_text(encoding="utf-8")
    )
    return json.loads(content)


def test_pit_gs_params():
    # SARS 2025/26 statutory schedule: phi0 anchored to the 45% top
    # marginal rate; phi2 tuned so PIT collects 10.1% of GDP from the
    # filing half of earners (National Treasury Budget Review 2026, Ch. 4)
    defaults = load_defaults()
    assert defaults["tax_func_type"] == "GS"
    expected = [0.464, 1.39288, 1.5314e-08]
    for key in ["etr_params", "mtrx_params", "mtry_params"]:
        assert defaults[key] == [[expected]]


def test_income_tax_filer():
    # Bottom two lifetime-income groups (bottom 50% of earners) are outside
    # the PIT net: ~8.3m individuals above the tax threshold (Budget Review
    # 2026, Table 4.5) vs 16.8m employed (Stats SA QLFS Q1 2026)
    defaults = load_defaults()
    assert defaults["income_tax_filer"] == [
        [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    ]


def test_no_bequest_tax():
    # South Africa's estate duty is negligible (~0.1% of GDP); the docs
    # state the model's bequest tax is zero
    defaults = load_defaults()
    assert defaults["tau_bq"] == [0.0]
