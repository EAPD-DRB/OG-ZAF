"""
Tests of macro_params.py module
"""
from ogzaf import macro_params


def test_get_macro_params():
    test_dict = macro_params.get_macro_params()

    assert isinstance(test_dict, dict)
    assert list(test_dict.keys()).sort() == [
        "r_gov_shift",
        "r_gov_scale",
        "alpha_T",
        "alpha_G",
        "initial_debt_ratio",
        "g_y_annual",
        "gamma",
        "zeta_D",
        "initial_foreign_debt_ratio"
    ].sort()
