"""
This module uses data from World Bank WDI, World Bank Quarterly Public
Sector Debt (QPSD) database, the IMF, and UN ILO to find values for
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
    data_end_date=datetime.datetime(2024, 12, 31),
    country_iso="ZAF",
    update_from_api=False,
):
    """
    Compute values of parameters that are derived from macro data

    Args:
        data_start_date (datetime): start date for data
        data_end_date (datetime): end date for data
        country_iso (str): ISO code for country

    Returns:
        macro_parameters (dict): dictionary of parameter values
    """
    # initialize a dictionary of parameters
    macro_parameters = {}
    # baseline date formatted for World Bank data
    baseline_YYYYQ = (
        str(data_end_date.year)
        + "Q"
        + str(pd.Timestamp(data_end_date).quarter)
    )

    """
    Retrieve data from the World Bank World Development Indicators.
    """
    # Dictionaries of variables and their corresponding World Bank codes
    # Annual data
    wb_a_variable_dict = {
        "GDP per capita (constant 2015 US$)": "NY.GDP.PCAP.KD",
        "Real GDP (constant 2015 US$)": "NY.GDP.MKTP.KD",
        "Nominal GDP (current US$)": "NY.GDP.MKTP.CD",
        "General government final consumption expenditure (current US$)": "NE.CON.GOVT.CD",
    }
    # Quarterly data
    wb_q_variable_dict = {
        "Gross PSD USD - domestic creditors": "DP.DOD.DECD.CR.PS.CD",
        "Gross PSD USD - external creditors": "DP.DOD.DECX.CR.PS.CD",
        "Gross PSD Gen Gov - percentage of GDP": "DP.DOD.DECT.CR.GG.Z1",
    }
    if update_from_api:
        try:
            # pull series of interest from the WB using pandas_datareader
            # Annual data
            wb_data_a = wb.download(
                indicator=wb_a_variable_dict.values(),
                country=country_iso,
                start=data_start_date,
                end=data_end_date,
            )
            wb_data_a.rename(
                columns=dict((y, x) for x, y in wb_a_variable_dict.items()),
                inplace=True,
            )
            # Quarterly data
            wb_data_q = wb.download(
                indicator=wb_q_variable_dict.values(),
                country=country_iso,
                start=data_start_date,
                end=data_end_date,
            )
            wb_data_q.rename(
                columns=dict((y, x) for x, y in wb_q_variable_dict.items()),
                inplace=True,
            )
            # Remove the hierarchical index (country and year) of
            # wb_data_q and create a single row index using year
            wb_data_q = wb_data_q.reset_index()
            wb_data_q = wb_data_q.set_index("year")

            # Function to get the latest valid data if baseline_YYYYQ is missing or NaN
            def get_valid_data(series, baseline_YYYYQ):
                value = series.get(baseline_YYYYQ, None)

                if pd.isna(value):
                    latest_non_nan = series.dropna().last_valid_index()

                    if latest_non_nan is not None:
                        print(
                            f"Warning: No data for {baseline_YYYYQ}. Using last available quarter: {latest_non_nan}"
                        )
                        value = series.get(latest_non_nan, None)
                    else:
                        print(
                            "Warning: No historical data available. Skipping update."
                        )
                        value = None

                return value

            # Compute macro parameters from WB data
            macro_parameters["initial_debt_ratio"] = get_valid_data(
                pd.Series(wb_data_q["Gross PSD Gen Gov - percentage of GDP"])
                / 100,
                baseline_YYYYQ,
            )
            print(
                f"initial_debt_ratio updated from World Bank API: {macro_parameters['initial_debt_ratio']}"
            )

            # Compute initial_foreign_debt_ratio safely
            if (
                "Gross PSD USD - external creditors" in wb_data_q.columns
                and "Gross PSD USD - domestic creditors" in wb_data_q.columns
            ):

                total_debt = (
                    wb_data_q["Gross PSD USD - domestic creditors"]
                    + wb_data_q["Gross PSD USD - external creditors"]
                )

                # Avoid division by zero
                wb_data_q["foreign_debt_ratio"] = wb_data_q[
                    "Gross PSD USD - external creditors"
                ] / total_debt.replace(0, np.nan)

                macro_parameters["initial_foreign_debt_ratio"] = (
                    get_valid_data(
                        wb_data_q["foreign_debt_ratio"], baseline_YYYYQ
                    )
                )
            else:
                print(
                    "Warning: Missing debt variables in World Bank data. Skipping update for initial_foreign_debt_ratio."
                )

            print(
                f"initial_foreign_debt_ratio updated from World Bank API: {macro_parameters['initial_foreign_debt_ratio']}"
            )

            # Compute zeta_D safely
            macro_parameters["zeta_D"] = [
                macro_parameters["initial_foreign_debt_ratio"]
            ]  # Since it's the same formula, we use the same calculated value

            print(
                f"zeta_D updated from World Bank API: {macro_parameters['zeta_D']}"
            )

            # Compute annual GDP growth safely
            if "GDP per capita (constant 2015 US$)" in wb_data_a.columns:
                g_y_series = wb_data_a[
                    "GDP per capita (constant 2015 US$)"
                ].pct_change(-1)

                # If all values are NaN, return None
                macro_parameters["g_y_annual"] = (
                    g_y_series.mean() if not g_y_series.isna().all() else None
                )
            else:
                print(
                    "Warning: Missing GDP per capita data in World Bank data. Skipping update for g_y_annual."
                )

            print(
                f"g_y_annual updated from World Bank API: {macro_parameters['g_y_annual']}"
            )
        except:
            print("Failed to retrieve data from World Bank")
            print("Will not update the following parameters:")
            print(
                "[initial_debt_ratio, initial_foreign_debt_ratio, zeta_D, g_y]"
            )
    else:
        print("Not updating from World Bank API")

    """
    Retrieve labour share data from the United Nations ILOSTAT Data API
    (see https://rshiny.ilo.org/dataexplorer9/?lang=en)
    The series code is SDG_1041_NOC_RT_A (capital share)
    Labor share (gamma) = 1 - capital share
    If this fails we will not update gamma in 'default_parameters.json'
    """
    if update_from_api:
        try:
            target = (
                "https://rplumber.ilo.org/data/indicator/"
                + "?id=SDG_1041_NOC_RT_A"
                + "&ref_area="
                + str(country_iso)
                + "&timefrom="
                + str(data_start_date.year)
                + "&timeto="
                + str(data_end_date.year)
                + "&type=both&format=.csv"
            )
            # Add headers
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }

            print("Attempting to update gamma from ILOSTAT")
            response = requests.get(target, headers=headers)
            if response.status_code != 200:
                print(f"Error: Received status code {response.status_code}")
            else:
                print("Request successful.")
            csv_content = StringIO(response.text)
            df_temp = pd.read_csv(csv_content)
            ilo_data = df_temp[["time", "obs_value"]]
            # find gamma, capital's share of income
            macro_parameters["gamma"] = [
                1
                - (
                    (
                        ilo_data.loc[
                            ilo_data["time"] == data_end_date.year, "obs_value"
                        ].squeeze()
                    )
                    / 100
                )
            ]
            print(
                f"gamma updated from ILOSTAT API: {macro_parameters['gamma']}"
            )
        except:
            print("Failed to retrieve data from ILOSTAT")
            print("Will not update gamma")
    else:
        print("Not updating from ILOSTAT API")

    """
    Calibrate parameters from IMF data
    """

    if update_from_api:
        # alpha_T, non-social security benefits as a fraction of GDP
        # source: https://data.imf.org/?sk=78d0bcc1-7a8f-44eb-8a2c-d4e472b8e64b&hide_uv=1
        # alpha_T = Employment-related social benefits expense - Social security benefits expense
        macro_parameters["alpha_T"] = [0.36 - 0.0]  # 2022 = 0.36

        # alpha_G, gov't consumption expenditures as a fraction of GDP
        # source: https://data.imf.org/?sk=23ca1c1d-e6a5-4f18-bc2e-7e215837f971&hide_uv=1
        # alpha_G = Expense	- Interest expense - Social benefits expense
        macro_parameters["alpha_G"] = [0.324 - 0.047 - 0.036]  # 2022 = 0.241

        """"
        Esimate the discount on sovereign yields relative to private debt
        Follow the methodology in Li, Magud, Werner, Witte (2021)
        available at:
        https://www.imf.org/en/Publications/WP/Issues/2021/06/04/The-Long-Run-Impact-of-Sovereign-Yields-on-Corporate-Yields-in-Emerging-Markets-50224
        discussion is here: https://github.com/EAPD-DRB/OG-ZAF/issues/22
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
        # First term is the constant and needs to be divided by 100 to have
        # the correct unit. Second term is the coefficient
        macro_parameters["r_gov_shift"] = [-res.params[0] / 100]
        macro_parameters["r_gov_scale"] = [res.params[1]]
        # Report new values
        print(f"alpha_T updated from IMF data: {macro_parameters['alpha_T']}")
        print(f"alpha_G updated from IMF data: {macro_parameters['alpha_G']}")
        print(
            f"r_gov_shift updated from IMF data: {macro_parameters['r_gov_shift']}"
        )
        print(
            f"r_gov_scale updated from IMF data: {macro_parameters['r_gov_scale']}"
        )
    else:
        print("Not updating alpha_T, alpha_G, r_gov_shift, r_gov_scale")

    return macro_parameters
