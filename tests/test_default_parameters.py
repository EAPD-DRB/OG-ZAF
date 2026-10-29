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
    # marginal rate; phi2 tuned so PIT collects 10.15% of GDP from the
    # filing half of earners (MTBPS 2026, Table 3.1)
    defaults = load_defaults()
    assert defaults["tax_func_type"] == "GS"
    expected = [0.464, 1.39288, 1.65594e-08]
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


def test_bequest_tax_is_effective_rate():
    # Estate duty + donations tax collect 0.08% of GDP (Budget Review
    # 2026 annexure). tau_bq is that collection over the model's bequests,
    # well under 1%, never the 20% statutory estate duty rate
    defaults = load_defaults()
    assert 0.0 < defaults["tau_bq"][0] < 0.01


def test_initial_debt_is_start_of_year_stock():
    # start_year 2025 starts from gross loan debt at the end of fiscal
    # 2024/25: 77.0% of GDP (MTBPS 2026, Table 3.8). 78.9% was the
    # projection for March 2026, the end of the first model year
    defaults = load_defaults()
    assert defaults["start_year"] == 2025
    assert defaults["initial_debt_ratio"] == 0.770


def test_public_investment_and_depreciation_sourced():
    # alpha_I: general government GFCF 2.42% of GDP (SARB QB S-127, 2025).
    # delta: private CFC 11.3% of GDP (SARB) over private K/Y 2.80
    # (PWT 11.0 less the government's 26.7% share)
    defaults = load_defaults()
    assert defaults["alpha_I"] == [0.0242]
    assert defaults["delta_annual"] == 0.0405


def test_payroll_tax_is_explicit():
    # SDL + UIF = 0.67% of GDP over the model's labour share 0.528
    defaults = load_defaults()
    assert defaults["tau_payroll"] == [[0.0127]]


def test_long_run_population_growth():
    # g_n_ss comes from UN WPP 2024 rates frozen at 2098, where South
    # Africa's population has levelled off (ogzaf.calibrate.
    # DEMOG_FINAL_DATA_YEAR). A value near +0.4%/yr would mean the
    # short two-year window has crept back in.
    defaults = load_defaults()
    assert abs(defaults["g_n_ss"]) < 0.001


def test_mortality_varies_by_income_group():
    # Income-differentiated demographics are switched on: at a working age
    # the poorest group dies at a higher rate than the richest
    import numpy as np

    rho = np.asarray(load_defaults()["rho"])
    assert rho[0, 20, 0] > 2 * rho[0, 20, -1]
