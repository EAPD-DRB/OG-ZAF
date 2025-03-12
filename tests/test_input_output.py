"""
Tests of input_output.py module
"""

import pandas as pd
import pytest
from ogzaf import input_output as io


sam_dict = {
    # "index": ["Beer", "Chocolate", "Car", "House"],
    "Ag": [30, 160, 0, 5],
    "Mining": [10, 0, 100, 100],
    "Manufacturing": [60, 40, 200, 295],
    "total": [100, 200, 300, 400],
    "row": [10, 20, 30, 40],
}
sam_df = pd.DataFrame(sam_dict, index=["Beer", "Chocolate", "Car", "House"])

cons_dict = {"Food": ["Beer", "Chocolate"], "Non-food": ["Car", "House"]}

prod_dict = {
    "Primary": ["Ag", "Mining"],
    "Secondary": ["Manufacturing"],
}


def test_read_sam():
    """
    Test of read_SAM() function
    """
    test_df = io.read_SAM()

    assert isinstance(test_df, pd.DataFrame)


@pytest.mark.parametrize(
    "sam_df, cons_dict",
    [
        (sam_df, cons_dict),
    ],
    ids=["Test 1"],
)
def test_get_alpha_c(sam_df, cons_dict):
    """
    Test of get_alpha_c() function
    """
    test_dict = io.get_alpha_c(sam=sam_df, cons_dict=cons_dict)

    assert isinstance(test_dict, dict)
    assert list(test_dict.keys()).sort() == ["Food", "Non-food"].sort()
    assert test_dict["Food"] == 270 / 900
    assert test_dict["Non-food"] == 630 / 900


@pytest.mark.parametrize(
    "sam_df, cons_dict, prod_dict",
    [
        (sam_df, cons_dict, prod_dict),
    ],
    ids=["Test 1"],
)
def test_get_io_matrix(sam_df, cons_dict, prod_dict):
    """
    Test of get_io_matrix()
    """
    test_df = io.get_io_matrix(
        sam=sam_df, cons_dict=cons_dict, prod_dict=prod_dict
    )

    assert isinstance(test_df, pd.DataFrame)
    assert list(test_df.columns).sort() == ["Primary", "Secondary"].sort()
    assert list(test_df.index).sort() == ["Food", "Non-food"].sort()
    assert test_df.loc["Food", "Primary"] == 2 / 3
    assert test_df.loc["Food", "Secondary"] == 1 / 3
