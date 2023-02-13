import pandas as pd
from ogzaf.constants import CON_DICT, PROD_DICT

'''
Read in SAM file
'''
# Read in SAM file
storage_options = {'User-Agent': 'Mozilla/5.0'}
SAM_path = "https://www.wider.unu.edu/sites/default/files/Data/SASAM-2015-Data-Resource.xlsx"
SAM = pd.read_excel(SAM_path, sheet_name="Micro SAM 2015", skiprows=6, index_col=0, storage_options=storage_options)


def get_alpha_c(SAM, cons_dict=CONS_DICT):
    '''
    Calibrate the alpha_c vector, showing the shares of household
    expenditures for each consumption category

    Args:
        SAM (pd.DataFrame): SAM file
        cons_dict (dict): Dictionary of consumption categories

    Returns:
        alpha_c (dict): Dictionary of shares of household expenditures
    '''
    overall_sum = 0
    for key, value in cons_dict.items():
        # note the subtraction of the row to focus on domestic consumption
        categroy_total = SAM.loc[SAM.index.isin(value), "total"].sum() - SAM.loc[SAM.index.isin(value), "row"].sum()
        alpha_c[key] = categroy_total
        overall_sum += categroy_total
    for key, value in cons_dict.items():
        alpha_c[key] = alpha_c[key] / overall_sum

    return alpha_c


def get_io_matrix(SAM, cons_dict=CONS_DICT, prod_dict=PROD_DICT):
    '''
    Calibrate the io_matrix array.  This array relates the share of each
    production category in each consumption category

    Args:
        SAM (pd.DataFrame): SAM file
        cons_dict (dict): Dictionary of consumption categories
        prod_dict (dict): Dictionary of production categories

    Returns:
        io_df (pd.DataFrame): Dataframe of io_matrix
    '''
    # Create initial matrix as dataframe of 0's to fill in
    io_dict = {}
    for key in prod_dict.keys():
        io_dict[key] = np.zeros(len(cons_dict.keys()))
    io_df = pd.DataFrame(
        io_dict, index=cons_dict.keys()
    )
    # Fill in the matrix
    # Note, each cell in the SAM represents a payment from the columns
    # account to the row account
    # (see https://www.un.org/en/development/desa/policy/capacity/presentations/manila/6_sam_mams_philippines.pdf)
    # We are thus going to take the consumption categories from rows and
    # the production categories from columns
    for ck, cv in cons_dict.items():
        for pk, pv in prod_dict.items():
            io_df.loc[io_df.index == ck, pk] = SAM.loc[SAM.index.isin(cv), pv].values.sum()
    # change from levesl to share (where each row sums to one)
    io_df = io_df.div(io_df.sum(axis=1), axis=0)

    return io_df
