"""
This module uses data from World Bank WDI, World Bank Quarterly Public
Sector Debt (QPSD) database, UN Data Portal, and FRED to find values for
parameters for the OG-ZAF model that rely on macro data for calibration.
"""

# imports
from pandas_datareader import wb
import pandas as pd
import numpy as np
import requests
import datetime
import statsmodels.api as sm
from io import StringIO


def get_macro_params(
        data_start_date=datetime.datetime(1947, 1, 1),
        data_end_date=datetime.date.today()):
    """
    Compute values of parameters that are derived from macro data

    Args:
        data_start_date (datetime): start date for data
        data_end_date (datetime): end date for data

    Returns:
        macro_parameters (dict): dictionary of parameter values
    """

    # set beginning and end dates for data
    # format is year (1940),month (1),day (1)
    country_iso = "ZAF"
    baseline_YYYYQ = data_end_date.strftime("%YQ%q")

    """
    This retrieves annual data from the World Bank World Development Indicators.
    """

    wb_a_variable_dict = {
        "GDP per capita (constant 2015 US$)": "NY.GDP.PCAP.KD",
        "Real GDP (constant 2015 US$)": "NY.GDP.MKTP.KD",
        "Nominal GDP (current US$)": "NY.GDP.MKTP.CD",
        "General government final consumption expenditure (current US$)": "NE.CON.GOVT.CD",
        # "General government debt (percentage of GDP)": "GC.DOD.TOTL.GD.ZS",
        # "GDP per capita (current US$)": "NY.GDP.PCAP.CD",
        # "GDP per person employed (constant 2017 PPP$)": "SL.GDP.PCAP.EM.KD",
    }

    # pull series of interest using pandas_datareader
    # wb_data_a = wb.download(
    #     indicator=wb_a_variable_dict.values(),
    #     country=country_iso,
    #     start=data_start_date,
    #     end=end,
    # )
    # wb_data_a.rename(
    #     columns=dict((y, x) for x, y in wb_a_variable_dict.items()),
    #     inplace=True,
    # )

    """
    This retrieves quarterly data from the World Bank Quarterly Public Sector Debt database.
    The command extracts all available data even when start and end dates are specified.
    """

    # wb_q_variable_dict = {
    #     "Gross PSD USD - domestic creditors": "DP.DOD.DECD.CR.PS.CD",
    #     "Gross PSD USD - external creditors": "DP.DOD.DECX.CR.PS.CD",
    #     "Gross PSD Gen Gov - percentage of GDP": "DP.DOD.DECT.CR.GG.Z1",
    # }

    # pull series of interest using pandas_datareader
    # wb_data_q = wb.download(
    #     indicator=wb_q_variable_dict.values(),
    #     country=country_iso,
    #     start=data_start_date,
    #     end=end,
    # )
    # wb_data_q.rename(
    #     columns=dict((y, x) for x, y in wb_q_variable_dict.items()),
    #     inplace=True,
    # )

    # # Remove the hierarchical index (country and year) of wb_data_q and create a single row index using year
    # wb_data_q = wb_data_q.reset_index()
    # wb_data_q = wb_data_q.set_index("year")

    """
    This retrieves labour share data from the ILOSTAT Data API
    (see https://rshiny.ilo.org/dataexplorer9/?lang=en)
    """

    target = (
        "https://rplumber.ilo.org/data/indicator/"
        + "?id=LAP_2GDP_NOC_RT_A"
        + "&ref_area="
        + str(country_iso)
        + "&timefrom="
        + str(data_start_date.year)
        + "&timeto="
        + str(data_end_date.year)
        + "&type=both&format=.csv"
    )

    response = requests.get(target)
    if response.status_code == 200:
        csv_content = StringIO(response.text)
        df_temp = pd.read_csv(csv_content)
    else:
        print(
            f"Failed to retrieve data. HTTP status code: {response.status_code}"
        )

    ilo_data = df_temp[["time", "obs_value"]]

    # initialize a dictionary of parameters
    macro_parameters = {}

    # print(fred_data.loc(str(baseline_date)))
    # find initial_debt_ratio
    # macro_parameters["initial_debt_ratio"] = (
    #     pd.Series(wb_data_q["Gross PSD Gen Gov - percentage of GDP"]).loc[
    #         baseline_YYYYQ
    #     ]
    # ) / 100

    # find initial_foreign_debt_ratio
    # macro_parameters["initial_foreign_debt_ratio"] = pd.Series(
    #     wb_data_q["Gross PSD USD - external creditors"]
    #     / (
    #         wb_data_q["Gross PSD USD - domestic creditors"]
    #         + wb_data_q["Gross PSD USD - external creditors"]
    #     )
    # ).loc[baseline_YYYYQ]

    # find zeta_D (Share of new debt issues from government that are purchased by foreigners)
    # macro_parameters["zeta_D"] = [
    #     pd.Series(
    #         wb_data_q["Gross PSD USD - external creditors"]
    #         / (
    #             wb_data_q["Gross PSD USD - domestic creditors"]
    #             + wb_data_q["Gross PSD USD - external creditors"]
    #         )
    #     ).loc[baseline_YYYYQ]
    # ]

    # alpha_T, non-social security benefits as a fraction of GDP
    # source: https://data.imf.org/?sk=b052f0f0-c166-43b6-84fa-47cccae3e219&hide_uv=1
    macro_parameters["alpha_T"] = [0.041 - 0.0]
    # alpha_G, gov't consumption expenditures as a fraction of GDP
    # source: https://data.imf.org/?sk=edb0cd70-0af3-40e1-a9c3-bdef83ee4d1e&hide_uv=1
    macro_parameters["alpha_G"] = [0.351 - 0.043 - 0.041]

    # find gamma, capital's share of income
    macro_parameters["gamma"] = [
        1
        - (
            (
                ilo_data.loc[
                    ilo_data["time"] == baseline_date.year, "obs_value"
                ].squeeze()
            )
            / 100
        )
    ]

    # find g_y
    # macro_parameters["g_y_annual"] = (
    #     wb_data_a["GDP per capita (constant 2015 US$)"].pct_change(-1).mean()
    # )

    """"
    We want to use the non linear relationship estimated by
    Li, Magud, Werner, Witte (2021), available here:
    https://www.imf.org/en/Publications/WP/Issues/2021/06/04/The-Long-Run-Impact-of-Sovereign-Yields-on-Corporate-Yields-in-Emerging-Markets-50224

    Steps:
    1) Generate modelled corporate yields (corp_yhat) for a range of
    sovereign yields (sov_y)  using the estimated equation in col 2 of
    table 8 (and figure 3). 2) Estimate the OLS using sovereign yields
    as the dependent variable
    """

    # # estimate r_gov_shift and r_gov_scale
    sov_y = np.arange(20, 120) / 10
    corp_yhat = 8.199 - (2.975 * sov_y) + (0.478 * sov_y**2)
    corp_yhat = sm.add_constant(corp_yhat)
    mod = sm.OLS(
        sov_y,
        corp_yhat,
    )
    res = mod.fit()
    # first term is the constant and needs to be divided by 100 to have
    # the correct unit. Second term is the coefficient
    macro_parameters["r_gov_shift"] = [
        (-res.params[0] / 100)
    ]  # constant = -0.0337662504
    macro_parameters["r_gov_scale"] = [
        res.params[1]
    ]  # coefficient = 0.24484764

    return macro_parameters
