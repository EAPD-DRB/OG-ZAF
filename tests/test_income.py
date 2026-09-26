"""
Tests of income.py module
"""

import pytest
import numpy as np
from ogzaf import income


def test_arctan_func():
    """
    Test of arctan_func()
    """
    expected_vals = np.array([0.14677821, 0.083305594, 0.057901228])
    xvals = np.array([1, 2, 3])
    a = 1.3
    b = 2.2
    c = 0.5
    test_vals = income.arctan_func(xvals, a, b, c)

    assert np.allclose(test_vals, expected_vals)


def test_arctan_deriv_func():
    """
    Test of arctan_deriv_func()
    """
    expected_vals = np.array([-0.109814991, -0.036400091, -0.017707961])
    xvals = np.array([1, 2, 3])
    a = 1.3
    b = 2.2
    c = 0.5
    test_vals = income.arctan_deriv_func(xvals, a, b, c)

    assert np.allclose(test_vals, expected_vals)


def test_arc_error():
    """
    Test of arc_error()
    """
    expected_vals = np.array([30.19765553, -1.40779391, 14.19212336])
    a = 1.3
    b = 2.2
    c = 0.5
    abc_vals = (a, b, c)
    first_point = 30.2
    coef1 = 0.05995294
    coef2 = -0.00004086
    coef3 = -0.00000521
    abil_deprec = 0.47
    params = (first_point, coef1, coef2, coef3, abil_deprec)
    test_vals = income.arc_error(abc_vals, params)

    assert np.allclose(test_vals, expected_vals)


@pytest.mark.parametrize(
    "S,lambdas",
    [
        (80, np.array([0.25, 0.25, 0.2, 0.1, 0.1, 0.09, 0.01])),
        (
            80,
            np.array(
                [0.25, 0.25, 0.2, 0.1, 0.1, 0.09, 0.005, 0.004, 0.0009, 0.0001]
            ),
        ),
        (40, np.array([0.25, 0.25, 0.2, 0.1, 0.1, 0.09, 0.01])),
    ],
)
def test_get_e_interp_uses_joint_population_weights(monkeypatch, S, lambdas):
    """Income profiles are normalized by joint age-income weights."""
    source_e = np.arange(1, 80 * 7 + 1, dtype=float).reshape(80, 7)
    monkeypatch.setattr(income, "get_e_orig", lambda *args: source_e)
    J = len(lambdas)
    age_wgts = np.arange(1, S * J + 1, dtype=float).reshape(S, J)
    age_wgts /= age_wgts.sum()
    age_wgts_80 = np.arange(1, 80 * J + 1, dtype=float).reshape(80, J)
    age_wgts_80 /= age_wgts_80.sum()

    e = income.get_e_interp(20, S, age_wgts, age_wgts_80, lambdas)

    assert e.shape == (S, J)
    assert np.isclose((e * age_wgts).sum(), 1.0)


@pytest.mark.local
def test_arctan_fit():
    """
    Test arctan_fit() function
    """
    expected_vals = np.array(
        [
            22.196999891242,
            22.196999897747,
            22.196999904181,
            22.196999910546,
            22.196999916843,
            22.196999923072,
            22.196999929234,
            22.196999935332,
            22.196999941365,
            22.196999947335,
            22.196999953243,
            22.196999959089,
            22.196999964876,
            22.196999970603,
            22.196999976271,
            22.196999981882,
            22.196999987436,
            22.196999992935,
            22.196999998378,
            22.197000003768,
        ]
    )
    a = 1.3
    b = 2.2
    c = 0.5
    init_guesses = (a, b, c)
    first_point = 30.2
    coef1 = 0.05995294
    coef2 = -0.00004086
    coef3 = -0.00000521
    abil_deprec = 0.47
    test_vals = income.arctan_fit(
        first_point, coef1, coef2, coef3, abil_deprec, init_guesses
    )
    assert np.allclose(test_vals, expected_vals)
