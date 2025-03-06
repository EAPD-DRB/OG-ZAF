import pandas as pd
import numpy as np
from ogzaf.utils import is_connected
from ogzaf.constants import CONS_DICT, PROD_DICT

"""
Read in Social Accounting Matrix (SAM) file
This is the most recent SAM for 2019 available from the following page as a downloadable zip folder from UNU WIDER:
https://www.wider.unu.edu/sites/default/files/Publications/Technical-note/tn2023-1-2019-SASAM-for-distribution.zip
"""
# Read in SAM file
storage_options = {"User-Agent": "Mozilla/5.0"}
SAM_path = "https://raw.githubusercontent.com/EAPD-DRB/SAM-files/main/Data/ZAF/tn2023-1-2019-SASAM-for-distribution.xlsx"

if is_connected():
    try:
        SAM = pd.read_excel(
            SAM_path,
            sheet_name="SASAM 2019 61Ind 4Educ", # Can alternatively use sheet_name="SASM 2019 61Ind4Occ"
            skiprows=6,
            index_col=0,
            storage_options=storage_options,
        )
        print("Successfully read SAM from Github repository.")
    except Exception as e:
        print(f"Failed to read from the GitHub repository: {e}")
        SAM = None
    # If both attempts fail, SAM will be None
    if SAM is None:
        print("Failed to read SAM from both sources.")
else:
    SAM = None
    print("No internet connection. SAM cannot be read.")


def get_alpha_c(sam=SAM, cons_dict=CONS_DICT):
    """
    Calibrate the alpha_c vector, showing the shares of household
    expenditures for each consumption category

    Args:
        sam (pd.DataFrame): SAM file
        cons_dict (dict): Dictionary of consumption categories

    Returns:
        alpha_c (dict): Dictionary of shares of household expenditures
    """
    alpha_c = {}
    overall_sum = 0
    for key, value in cons_dict.items():
        # note the subtraction of the row to focus on domestic consumption
        category_total = (
            sam.loc[sam.index.isin(value), "total"].sum()
            - sam.loc[sam.index.isin(value), "row"].sum()
        )
        alpha_c[key] = category_total
        overall_sum += category_total
    for key, value in cons_dict.items():
        alpha_c[key] = alpha_c[key] / overall_sum

    return alpha_c


def get_io_matrix(sam=SAM, cons_dict=CONS_DICT, prod_dict=PROD_DICT):
    """
    Calibrate the io_matrix array.  This array relates the share of each
    production category in each consumption category

    Args:
        sam (pd.DataFrame): SAM file
        cons_dict (dict): Dictionary of consumption categories
        prod_dict (dict): Dictionary of production categories

    Returns:
        io_df (pd.DataFrame): Dataframe of io_matrix
    """
    # Create initial matrix as dataframe of 0's to fill in
    io_dict = {}
    for key in prod_dict.keys():
        io_dict[key] = np.zeros(len(cons_dict.keys()))
    io_df = pd.DataFrame(io_dict, index=cons_dict.keys())
    # Fill in the matrix
    # Note, each cell in the SAM represents a payment from the columns
    # account to the row account
    # (see https://www.un.org/en/development/desa/policy/capacity/presentations/manila/6_sam_mams_philippines.pdf)
    # We are thus going to take the consumption categories from rows and
    # the production categories from columns
    for ck, cv in cons_dict.items():
        for pk, pv in prod_dict.items():
            io_df.loc[io_df.index == ck, pk] = sam.loc[
                sam.index.isin(cv), pv
            ].values.sum()
    # change from levels to share (where each row sums to one)
    io_df = io_df.div(io_df.sum(axis=1), axis=0)

    return io_df
