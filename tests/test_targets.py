"""Tests for the steady-state dashboard in ogzaf/targets.py."""

import numpy as np
import pytest

from ogzaf.targets import TARGETS, ss_moments


def fake_ss():
    """A steady state with round numbers; Y = 100."""
    return {
        "Y": np.array([100.0]),
        "C": np.array([60.0]),
        "I_total": np.array([15.0]),
        "I_g": np.array([2.0]),
        "G": np.array([19.0]),
        "iit_revenue": np.array([10.0]),
        "payroll_tax_revenue": np.array([0.5]),
        "business_tax_revenue": np.array([5.0]),
        "cons_tax_revenue": np.array([9.0]),
        "wealth_tax_revenue": np.array([1.5]),
        "bequest_tax_revenue": np.array([0.1]),
        "total_tax_revenue": np.array([26.1]),
        "r_gov": np.array([0.037]),
        "D": np.array([76.5]),
        "D_f": np.array([19.125]),
        "TR": np.array([3.6]),
        "agg_pension_outlays": np.array([0.0]),
        # K summed across industries, as in a multi-industry solve
        "K": np.array([200.0, 80.0]),
        "K_f": np.array([56.0]),
        "B": np.array([277.0]),
        "BQ": np.array([1.0, 2.0]),
        "r": np.array([0.11]),
        "factor": np.array([150000.0]),
    }


def test_every_target_has_a_model_moment():
    m = ss_moments(fake_ss())
    assert set(TARGETS) <= set(m)


def test_ratios_to_output():
    m = ss_moments(fake_ss())
    assert m["PIT/Y"] == pytest.approx(0.10)
    assert m["D/Y"] == pytest.approx(0.765)
    assert m["D_f/D"] == pytest.approx(0.25)
    assert m["K/Y"] == pytest.approx(2.80)
    assert m["K_f/K"] == pytest.approx(0.20)
    assert m["BQ/Y"] == pytest.approx(0.03)
    # net exports are output less every use: 100 - 60 - 15 - 2 - 19
    assert m["NX/Y"] == pytest.approx(0.04)
    # consumption at purchaser prices includes consumption taxes
    assert m["C/Y purch."] == pytest.approx(0.69)
    assert m["r_gov"] == pytest.approx(0.037)


def test_no_debt_gives_no_foreign_share():
    ss = fake_ss()
    ss["D"] = np.array([0.0])
    assert np.isnan(ss_moments(ss)["D_f/D"])
