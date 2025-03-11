"""
Tests of macro_params.py module
"""

import pytest
from ogidn import macro_params


@pytest.mark.parametrize(
    "update_from_api",
    [True, False],
    ids=["update_from_api=True", "update_from_api=False"],
)
def test_get_macro_params(update_from_api):
    test_dict = macro_params.get_macro_params(update_from_api=update_from_api)

    assert isinstance(test_dict, dict)
    if update_from_api:
        assert (
            list(test_dict.keys()).sort()
            == [
                "r_gov_shift",
                "r_gov_scale",
                "alpha_T",
                "alpha_G",
                "initial_debt_ratio",
                "g_y_annual",
                "gamma",
                "zeta_D",
                "initial_foreign_debt_ratio",
            ].sort()
        )
    else:
        assert (
            list(test_dict.keys()).sort()
            == ["r_gov_shift", "r_gov_scale"].sort()
        )
