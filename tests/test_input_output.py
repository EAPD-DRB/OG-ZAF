"""
Tests of input_output.py module
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch
from ogzaf import input_output as io

# Toy SAM for the naive get_io_matrix and the household-based get_alpha_c.
# Household columns (hhd*) drive alpha_c; production columns drive io_matrix.
sam_dict = {
    # "index": ["Beer", "Chocolate", "Car", "House"],
    "Ag": [30, 160, 0, 5],
    "Mining": [10, 0, 100, 100],
    "Manufacturing": [60, 40, 200, 295],
    "total": [100, 200, 300, 400],
    "row": [10, 20, 30, 40],
    "hhd1": [15, 20, 30, 40],
    "hhd2": [15, 20, 30, 40],
}
sam_df = pd.DataFrame(sam_dict, index=["Beer", "Chocolate", "Car", "House"])

cons_dict = {"Food": ["Beer", "Chocolate"], "Non-food": ["Car", "House"]}

prod_dict = {
    "Primary": ["Ag", "Mining"],
    "Secondary": ["Manufacturing"],
}

# Small but structurally complete synthetic SAM (activities, commodities, two
# factor rows, a household column, an imports row, and make/use blocks) for the
# value-added construction functions. Convention: cell [row, col] is a payment
# from the column account to the row account.
_accounts = ["a1", "a2", "c1", "c2", "flab-p", "gos", "hhd1", "row"]
_syn = pd.DataFrame(0.0, index=_accounts, columns=_accounts)
_syn.loc[["a1", "a2"], ["c1", "c2"]] = [
    [80, 20],
    [10, 90],
]  # make (act x comm)
_syn.loc[["c1", "c2"], ["a1", "a2"]] = [
    [20, 15],
    [10, 25],
]  # use  (comm x act)
_syn.loc["flab-p", ["a1", "a2"]] = [40, 35]  # labour
_syn.loc["gos", ["a1", "a2"]] = [30, 25]  # capital
_syn.loc[["c1", "c2"], "hhd1"] = [55, 65]  # household consumption
_syn.loc["row", ["c1", "c2"]] = [5, 10]  # imports
syn_cons = {"G1": ["c1"], "G2": ["c2"]}
syn_prod = {"Ind1": ["a1"], "Ind2": ["a2"]}  # Ind2 is the numeraire (last)


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
    Test of get_alpha_c() function: shares of household expenditure (the
    hhd* columns) by consumption category.
    """
    test_dict = io.get_alpha_c(sam=sam_df, cons_dict=cons_dict)

    assert isinstance(test_dict, dict)
    assert sorted(test_dict.keys()) == sorted(["Food", "Non-food"])
    # Food hh = (15+15)+(20+20)=70; Non-food = (30+30)+(40+40)=140; total 210
    assert test_dict["Food"] == pytest.approx(70 / 210)
    assert test_dict["Non-food"] == pytest.approx(140 / 210)


@pytest.mark.parametrize(
    "sam_df, cons_dict, prod_dict",
    [
        (sam_df, cons_dict, prod_dict),
    ],
    ids=["Test 1"],
)
def test_get_io_matrix(sam_df, cons_dict, prod_dict):
    """
    Test of get_io_matrix() (the naive direct-mapping version)
    """
    test_df = io.get_io_matrix(
        sam=sam_df, cons_dict=cons_dict, prod_dict=prod_dict
    )

    assert isinstance(test_df, pd.DataFrame)
    assert sorted(test_df.columns) == sorted(["Primary", "Secondary"])
    assert sorted(test_df.index) == sorted(["Food", "Non-food"])
    assert test_df.loc["Food", "Primary"] == 2 / 3
    assert test_df.loc["Food", "Secondary"] == 1 / 3


def test_get_gamma():
    """get_gamma(): capital share of factor income by industry, with the
    optional rescale to a target value-added-weighted mean."""
    gamma = io.get_gamma(sam=_syn, prod_dict=syn_prod)
    # Ind1: gos 30 / (flab 40 + gos 30) = 0.42857; Ind2: 25 / 60 = 0.41667
    assert gamma["Ind1"] == pytest.approx(30 / 70)
    assert gamma["Ind2"] == pytest.approx(25 / 60)
    # rescale so the VA-weighted mean equals the target
    target = 0.5
    gamma_rs = io.get_gamma(sam=_syn, prod_dict=syn_prod, target_avg=target)
    va = np.array([70.0, 60.0])
    mean = np.average([gamma_rs["Ind1"], gamma_rs["Ind2"]], weights=va)
    assert mean == pytest.approx(target)


def test_get_io_matrix_value_added():
    """get_io_matrix_value_added(): I x M matrix whose rows sum to one."""
    io_df = io.get_io_matrix_value_added(
        sam=_syn, cons_dict=syn_cons, prod_dict=syn_prod
    )
    assert isinstance(io_df, pd.DataFrame)
    assert list(io_df.index) == ["G1", "G2"]
    assert list(io_df.columns) == ["Ind1", "Ind2"]
    np.testing.assert_allclose(io_df.values.sum(axis=1), [1.0, 1.0])
    assert (io_df.values >= 0).all()


def test_get_employment():
    """get_employment(): positive employment for every industry."""
    emp = io.get_employment()
    from ogzaf.constants import PROD_DICT

    assert sorted(emp.keys()) == sorted(PROD_DICT.keys())
    assert all(v > 0 for v in emp.values())


def test_get_Z():
    """get_Z(): numeraire (last) industry normalized to 1, others positive."""
    gamma = io.get_gamma(sam=_syn, prod_dict=syn_prod)
    Z = io.get_Z(
        sam=_syn,
        prod_dict=syn_prod,
        gamma=gamma,
        employment={"Ind1": 10.0, "Ind2": 20.0},
    )
    assert Z["Ind2"] == pytest.approx(1.0)  # last industry is the numeraire
    assert Z["Ind1"] > 0


@patch("ogzaf.input_output.read_SAM", return_value=None)
def test_get_alpha_c_raises_on_none_sam(mock_read_sam):
    """get_alpha_c() raises RuntimeError when SAM data is unavailable."""
    with pytest.raises(RuntimeError, match="Cannot compute alpha_c"):
        io.get_alpha_c()


@patch("ogzaf.input_output.read_SAM", return_value=None)
def test_get_io_matrix_raises_on_none_sam(mock_read_sam):
    """get_io_matrix() raises RuntimeError when SAM data is unavailable."""
    with pytest.raises(RuntimeError, match="Cannot compute io_matrix"):
        io.get_io_matrix()
