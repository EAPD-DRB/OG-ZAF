"""
-------------------------------------------------------------------------------
Functions for generating demographic objects necessary for the OG-ZAF model

This module contains the following functions:
    get_un_fert_data()
    get_un_mort_data()
    get_wb_infmort_rate()
    get_un_pop_data()
    get_fert()
    get_mort()
    pop_rebin()
    get_imm_resid()
    immsolve()
    get_pop_objs()
    extrap_exp_3()
    extrap_arctan_3()
    ab_zero_eqs_exp_func()
    b_zero_eq_artctan_func()
-------------------------------------------------------------------------------
"""
# Import packages
import os
import numpy as np
import pandas as pd
import scipy.optimize as opt
from ogcore import parameter_plots as pp
from pandas_datareader import wb
import matplotlib.pyplot as plt
from ogzaf.utils import get_legacy_session
from io import StringIO

# create output director for figures
CUR_PATH = os.path.split(os.path.abspath(__file__))[0]
DATA_DIR = os.path.join(CUR_PATH, "data", "demographic")
OUTPUT_DIR = os.path.join(CUR_PATH, "OUTPUT", "Demographics")
if os.access(OUTPUT_DIR, os.F_OK) is False:
    os.makedirs(OUTPUT_DIR)


def get_un_fert_data(
    country_id: str = "710",
    start_year: int = 2021,
    end_year: int = None,
    download: bool = True,
) -> pd.DataFrame:
    """
    Get UN fertility rate data for a country for some range of years (at least
    one year) and by age. The country_id=710 is for South Africa. These data
    come from the United Nations Data Portal API for UN population data (see
    https://population.un.org/dataportal/about/dataapi)

    Args:
        country_id (str): 3-digit country id (numerical)
        start_year (int): beginning year of the data
        end_year (int or None): end year of the data
        download (bool): whether to download the data from the UN Data Portal.
            If False, a path must be specified in the path_folder argument.
        path_folder (None or str): string path to folder where data are stored

    Returns:
        fert_rates_df (DataFrame): dataset with fertility rates by age
    """
    if end_year is None:
        end_year = start_year
    # UN variable code for Population by 1-year age groups and sex
    pop_code = "47"
    # UN variable code for Fertility rates by age of mother (1-year)
    fert_code = "68"

    if download:
        pop_url = (
            "https://population.un.org/dataportalapi/api/v1/data/indicators/"
            + pop_code
            + "/locations/"
            + country_id
            + "/start/"
            + str(start_year)
            + "/end/"
            + str(end_year)
            + "?format=csv"
        )
        response = get_legacy_session().get(pop_url)
        # Check if the request was successful before processing
        if response.status_code == 200:
            pop_target = StringIO(response.text)
        else:
            print(
                f"Failed to retrieve population data. HTTP status code: {response.status_code}"
            )

        fert_url = (
            "https://population.un.org/dataportalapi/api/v1/data/indicators/"
            + fert_code
            + "/locations/"
            + country_id
            + "/start/"
            + str(start_year)
            + "/end/"
            + str(end_year)
            + "?format=csv"
        )
        response = get_legacy_session().get(fert_url)
        # Check if the request was successful before processing
        if response.status_code == 200:
            fert_target = StringIO(response.text)
        else:
            print(
                f"Failed to retrieve fertility data. HTTP status code: {response.status_code}"
            )
    else:
        pop_target = os.path.join(DATA_DIR, "un_zaf_pop.csv")
        fert_target = os.path.join(DATA_DIR, "un_zaf_fert.csv")

    # Convert .csv file to Pandas DataFrame
    pop_df = pd.read_csv(
        pop_target,
        sep="|",
        header=1,
        usecols=["TimeLabel", "SexId", "Sex", "AgeStart", "Value"],
        float_precision="round_trip",
    )
    fert_rates_df = pd.read_csv(
        fert_target,
        sep="|",
        header=1,
        usecols=["TimeLabel", "AgeStart", "Value"],
        float_precision="round_trip",
    )

    # Rename variables in the population and fertility rates data
    pop_df.rename(
        columns={
            "TimeLabel": "year",
            "SexId": "sex_num",
            "Sex": "sex_str",
            "AgeStart": "age",
            "Value": "pop",
        },
        inplace=True,
    )
    fert_rates_df.rename(
        columns={
            "TimeLabel": "year",
            "AgeStart": "age",
            "Value": "births_p_1000f",
        },
        inplace=True,
    )

    # Merge in the male and female population by age data
    fert_rates_df = fert_rates_df.merge(
        pop_df[["year", "age", "pop"]][pop_df["sex_num"] == 1],
        how="left",
        on=["year", "age"],
    )
    fert_rates_df.rename(columns={"pop": "pop_male"}, inplace=True)
    fert_rates_df = fert_rates_df.merge(
        pop_df[["year", "age", "pop"]][pop_df["sex_num"] == 2],
        how="left",
        on=["year", "age"],
    )
    fert_rates_df.rename(columns={"pop": "pop_female"}, inplace=True)
    fert_rates_df["fert_rate"] = fert_rates_df["births_p_1000f"] / (
        1000 * (1 + (fert_rates_df["pop_male"] / fert_rates_df["pop_female"]))
    )
    fert_rates_df = fert_rates_df[
        (
            (fert_rates_df["year"] >= start_year)
            & (fert_rates_df["year"] <= end_year)
        )
    ]

    return fert_rates_df


def get_un_mort_data(
    country_id: str = "710",
    start_year: int = 2021,
    end_year: int = None,
    download: bool = True,
) -> pd.DataFrame:
    """
    Get UN mortality rate data for a country for some range of years (at least
    one year) and by age, and get infant mortality rate data. The
    country_id=710 is for South Africa. These data come from the United Nations
    Population Data Portal API for UN population data (see
    https://population.un.org/dataportal/about/dataapi)

    Args:
        country_id (str): 3-digit country id (numerical)
        start_year (int): beginning year of the data
        end_year (int or None): end year of the data
        download (bool): whether to download the data from the UN Data Portal.
            If False, a path must be specified in the path_folder argument.
        path_folder (None or str): string path to folder where data are stored

    Returns:
        infmort_rate_df (DataFrame): dataset with infant mortality rates by yr
        mort_rates_df(DataFrame): dataset with mortality rates by age
    """
    if end_year is None:
        end_year = start_year
    # UN variable code for Population by 1-year age groups and sex
    pop_code = "47"
    # # UN variable code for Age specific mortality rate
    # mort_code = "80"
    # We use deaths and population to compute mortality rates rather than the
    # mortality rates data so that we have the option to have totpers<100.
    # UN variable code for Deaths by 1-year age groups
    deaths_code = "69"
    # UN variable code for Age specific mortality rate
    infmort_code = "22"

    if download:
        pop_url = (
            "https://population.un.org/dataportalapi/api/v1/data/indicators/"
            + pop_code
            + "/locations/"
            + country_id
            + "/start/"
            + str(start_year)
            + "/end/"
            + str(end_year)
            + "?format=csv"
        )
        response = get_legacy_session().get(pop_url)
        # Check if the request was successful before processing
        if response.status_code == 200:
            pop_target = StringIO(response.text)
        else:
            print(
                f"Failed to retrieve population data. HTTP status code: {response.status_code}"
            )

        infmort_url = (
            "https://population.un.org/dataportalapi/api/v1/data/indicators/"
            + infmort_code
            + "/locations/"
            + country_id
            + "/start/"
            + str(start_year)
            + "/end/"
            + str(end_year)
            + "?format=csv"
        )
        response = get_legacy_session().get(infmort_url)
        # Check if the request was successful before processing
        if response.status_code == 200:
            infmort_target = StringIO(response.text)
        else:
            print(
                f"Failed to retrieve infant mortality data. HTTP status code: {response.status_code}"
            )

        deaths_url = (
            "https://population.un.org/dataportalapi/api/v1/data/indicators/"
            + deaths_code
            + "/locations/"
            + country_id
            + "/start/"
            + str(start_year)
            + "/end/"
            + str(end_year)
            + "?format=csv"
        )
        response = get_legacy_session().get(deaths_url)
        # Check if the request was successful before processing
        if response.status_code == 200:
            deaths_target = StringIO(response.text)
        else:
            print(
                f"Failed to retrieve death data. HTTP status code: {response.status_code}"
            )
    else:
        pop_target = os.path.join(DATA_DIR, "un_zaf_pop.csv")
        infmort_target = os.path.join(DATA_DIR, "un_zaf_infmort.csv")
        deaths_target = os.path.join(DATA_DIR, "un_zaf_deaths.csv")

    # Convert .csv file to Pandas DataFrame
    pop_df = pd.read_csv(
        pop_target,
        sep="|",
        header=1,
        usecols=["TimeLabel", "SexId", "Sex", "AgeStart", "Value"],
        float_precision="round_trip",
    )
    infmort_rate_df = pd.read_csv(
        infmort_target,
        sep="|",
        header=1,
        usecols=["TimeLabel", "SexId", "Sex", "Value"],
        float_precision="round_trip",
    )
    deaths_df = pd.read_csv(
        deaths_target,
        sep="|",
        header=1,
        usecols=["TimeLabel", "SexId", "Sex", "AgeStart", "Value"],
        float_precision="round_trip",
    )

    # Rename variables in the population and mortality rates data
    pop_df.rename(
        columns={
            "TimeLabel": "year",
            "SexId": "sex_num",
            "Sex": "sex_str",
            "AgeStart": "age",
            "Value": "pop",
        },
        inplace=True,
    )
    infmort_rate_df.rename(
        columns={
            "TimeLabel": "year",
            "SexId": "sex_num",
            "Sex": "sex_str",
            "Value": "inf_deaths_p_1000",
        },
        inplace=True,
    )
    deaths_df.rename(
        columns={
            "TimeLabel": "year",
            "SexId": "sex_num",
            "Sex": "sex_str",
            "AgeStart": "age",
            "Value": "deaths",
        },
        inplace=True,
    )

    # Merge in the male and female population by age data to the deaths_df
    deaths_df = deaths_df.merge(
        pop_df,
        how="left",
        on=["year", "sex_num", "sex_str", "age"],
    )
    deaths_df["mort_rate"] = deaths_df["deaths"] / deaths_df["pop"]
    deaths_df = deaths_df[
        ((deaths_df["year"] >= start_year) & (deaths_df["year"] <= end_year))
    ]
    mort_rates_df = deaths_df.copy()

    # Clean the data
    infmort_rate_df["infmort_rate"] = (
        infmort_rate_df["inf_deaths_p_1000"] / 1000
    )

    infmort_rate_df = infmort_rate_df[
        (
            (infmort_rate_df["year"] >= start_year)
            & (infmort_rate_df["year"] <= end_year)
        )
    ]

    return infmort_rate_df, mort_rates_df


def get_wb_infmort_rate(
    country: str = "ZAF",
    start_year: int = 2020,
    end_year: int = None,
    download: bool = True,
) -> np.float64:
    """
    Get World Bank infant mortality rate measure from neonatal mortality rate
    (deaths per 1,000 live births, divided by 1,0000)
    https://data.worldbank.org/indicator/SH.DYN.NMRT

    Args:
        country (str): 3-digit country id (alphabetic)
        start_year (int): beginning year of the data
        end_year (int or None): end year of the data
        download (bool): whether to download the data from the UN Data Portal.
            If False, a path must be specified in the path_folder argument.

    Returns:
        wb_infmort_rate (float): neonatal infant mortality rate
    """
    if end_year is None:
        end_year = start_year
    if download:
        wb_infmort_rate = (
            wb.download(
                indicator="SH.DYN.NMRT",
                country=country,
                start=start_year,
                end=end_year,
            ).squeeze()
            / 1000
        )
    else:
        # Hard code the infant mortality rate for South Africa from the most
        # recent year (2020)
        wb_infmort_rate = 0.0106

    return wb_infmort_rate


def get_un_pop_data(
    country_id: str = "710",
    start_year: int = 2021,
    end_year: int = None,
    download: bool = True,
) -> pd.DataFrame:
    """
    Get UN population data for a country for some range of years (at least
    one year) and by age. The country_id=710 is for South Africa. These data
    come from the United Nations Data Portal API for UN population data (see
    https://population.un.org/dataportal/about/dataapi)

    Args:
        country_id (str): 3-digit country id (numerical)
        start_year (int): beginning year of the data
        end_year (int): end year of the data
        download (bool): whether to download the data from the UN Data Portal.
            If False, a path must be specified in the path_folder argument.
        path_folder (None or str): string path to folder where data are stored

    Returns:
        pop_df (DataFrame): dataset with total population by age
    """
    if end_year is None:
        end_year = start_year
    # UN variable code for Population by 1-year age groups and sex
    pop_code = "47"

    if download:
        pop_url = (
            "https://population.un.org/dataportalapi/api/v1/data/indicators/"
            + pop_code
            + "/locations/"
            + country_id
            + "/start/"
            + str(start_year)
            + "/end/"
            + str(end_year)
            + "?format=csv"
        )
        response = get_legacy_session().get(pop_url)
        # Check if the request was successful before processing
        if response.status_code == 200:
            pop_target = StringIO(response.text)
        else:
            print(
                f"Failed to retrieve population data. HTTP status code: {response.status_code}"
            )
    else:
        pop_target = os.path.join(DATA_DIR, "un_zaf_pop.csv")

    # Convert .csv file to Pandas DataFrame
    pop_df = pd.read_csv(
        pop_target,
        sep="|",
        header=1,
        usecols=["TimeLabel", "SexId", "Sex", "AgeStart", "Value"],
        float_precision="round_trip",
    )

    # Rename variables in the population data
    pop_df.rename(
        columns={
            "TimeLabel": "year",
            "SexId": "sex_num",
            "Sex": "sex_str",
            "AgeStart": "age",
            "Value": "pop",
        },
        inplace=True,
    )

    # Clean the data
    pop_df = pop_df[
        ((pop_df["year"] >= start_year) & (pop_df["year"] <= end_year))
    ]

    return pop_df


def get_fert(
    totpers, start_year=2021, end_year=None, download=False, graph=False
):
    """
    This function generates a vector of fertility rates by model period
    age that corresponds to the fertility rate data by age in years.

    Args:
        totpers (int): total number of agent life periods (E+S), >= 3
        start_year (int): first year data to download
        end_year (int or None): end year data to download
        graph (bool): =True if want graphical output

    Returns:
        fert_rates (Numpy array): fertility rates for each model period
            of life

    """
    if totpers > 100:
        err_msg = "ERROR get_fert(): totpers must be <= 100."
        raise ValueError(err_msg)

    # Get UN fertility rates for South Africa for ages 15-49
    ages_15_49 = np.arange(15, 50)
    fert_rates_15_49 = (
        get_un_fert_data(
            start_year=start_year, end_year=end_year, download=download
        )["fert_rate"]
        .to_numpy()
        .flatten()
    )

    # Extrapolate fertility rates for ages 1-14 and 50-100 using exponential
    # function
    ages_1_14 = np.arange(1, 15)
    slope_15 = (fert_rates_15_49[1] - fert_rates_15_49[0]) / (
        ages_15_49[1] - ages_15_49[0]
    )
    fert_rates_1_14 = extrap_exp_3(
        ages_1_14, (15, fert_rates_15_49[0]), slope_15, (9, 0.0001), low=True
    )
    ages_50_100 = np.arange(50, 101)
    slope_49 = (fert_rates_15_49[-1] - fert_rates_15_49[-2]) / (
        ages_15_49[-1] - ages_15_49[-2]
    )
    fert_rates_50_100 = extrap_exp_3(
        ages_50_100,
        (49, fert_rates_15_49[-1]),
        slope_49,
        (57, 0.0001),
        low=False,
    )
    fert_rates_1_100 = np.hstack(
        (fert_rates_1_14, fert_rates_15_49, fert_rates_50_100)
    )
    if totpers == 100:
        fert_rates = fert_rates_1_100.copy()
        ages = np.arange(1, 101)
    elif totpers < 100:
        # Create population weighted average fertility rates across bins
        # Get population data for ages 1-100
        pop_df = get_un_pop_data(
            start_year=start_year, end_year=end_year, download=download
        )
        pop_1_100 = (
            pop_df[((pop_df["age"] < 100) & (pop_df["sex_num"] == 3))]["pop"]
            .to_numpy()
            .flatten()
        )
        fert_rates = np.zeros(totpers)
        len_subbins = len_subbins = np.float64(100 / totpers)
        end_sub_bin = int(0)
        end_pct = 0.0
        for i in range(totpers):
            if end_pct < 1.0:
                beg_sub_bin = int(end_sub_bin)
                beg_pct = 1 - end_pct
            elif end_pct == 1.0:
                beg_sub_bin = 1 + int(end_sub_bin)
                beg_pct = 1.0
            end_sub_bin = int((i + 1) * len_subbins)
            if (i + 1) * len_subbins - end_sub_bin == 0.0:
                end_sub_bin = end_sub_bin - 1
                end_pct = 1
            elif (i + 1) * len_subbins - end_sub_bin > 0.0:
                end_pct = (i + 1) * len_subbins - end_sub_bin
            fert_rates_sub = fert_rates_1_100[beg_sub_bin : end_sub_bin + 1]
            pop_sub = pop_1_100[beg_sub_bin : end_sub_bin + 1]
            pop_sub[0] = beg_pct * pop_sub[0]
            pop_sub[-1] = end_pct * pop_sub[-1]
            fert_rates[i] = ((fert_rates_sub * pop_sub) / pop_sub.sum()).sum()
        ages = np.arange(1, totpers + 1)

    if graph:  # Plot fertility rates
        plt.plot(ages, fert_rates)
        plt.scatter(ages, fert_rates, marker="d")
        plt.xlabel(r"Age $s$")
        plt.ylabel(r"Fertility rate $f_{s}$")
        plt.text(
            -0,
            -0.023,
            "Source: UN Population Data",
            fontsize=9,
        )
        plt.tight_layout(rect=(0, 0.035, 1, 1))
        output_path = os.path.join(OUTPUT_DIR, "fert_rates")
        plt.savefig(output_path)
        plt.close()

    return fert_rates


def get_mort(
    totpers, start_year=2021, end_year=None, download=True, graph=False
):
    """
    This function generates a vector of mortality rates by model period
    age. Source: UN Population Data portal.

    Args:
        totpers (int): total number of agent life periods (E+S), >= 3
        start_year (int): first year data to download
        end_year (int or None): end year data to download
        graph (bool): =True if want graphical output

    Returns:
        mort_rates (Numpy array) mortality rates that correspond to each
            period of life
        infmort_rate (scalar): infant mortality rate

    """
    if totpers > 100:
        err_msg = "ERROR get_mort(): totpers must be <= 100."
        raise ValueError(err_msg)

    # Get UN infant mortality and mortality rate data by age
    un_infmort_rate_df, mort_rates_df = get_un_mort_data(
        start_year=start_year, end_year=end_year, download=download
    )
    un_infmort_rate = un_infmort_rate_df["infmort_rate"][
        un_infmort_rate_df["sex_num"] == 3
    ].to_numpy()[0]

    # Use World Bank infant mortality rate data (neonatal mortality rate) from
    # World Bank World Development Indicators database
    most_recent_wb_infmort_datayear = 2020
    if start_year > most_recent_wb_infmort_datayear:
        wb_infmort_rate = get_wb_infmort_rate(
            start_year=most_recent_wb_infmort_datayear, download=download
        )
    else:
        wb_infmort_rate = get_wb_infmort_rate(
            start_year=start_year, download=download
        )
    infmort_rate = wb_infmort_rate
    if totpers == 100:
        mort_rates = (
            mort_rates_df["mort_rate"][
                (
                    (mort_rates_df["sex_num"] == 3)
                    & (mort_rates_df["age"] < 100)
                )
            ]
            .to_numpy()
            .flatten()
        )

    elif totpers < 100:
        # Create population weighted average mortality rates across bins
        mort_rates = np.zeros(totpers)
        len_subbins = np.float64(100 / totpers)
        end_sub_bin = int(0)
        end_pct = 0.0
        deaths_0_99 = (
            mort_rates_df[
                (
                    (mort_rates_df["sex_num"] == 3)
                    & (mort_rates_df["age"] < 100)
                )
            ]["deaths"]
            .to_numpy()
            .flatten()
        )
        pop_0_99 = (
            mort_rates_df[
                (
                    (mort_rates_df["sex_num"] == 3)
                    & (mort_rates_df["age"] < 100)
                )
            ]["pop"]
            .to_numpy()
            .flatten()
        )
        deaths_pop_0_99 = mort_rates_df[
            ((mort_rates_df["sex_num"] == 3) & (mort_rates_df["age"] < 100))
        ][["age", "deaths", "pop"]]
        for i in range(totpers):
            if end_pct < 1.0:
                beg_sub_bin = int(end_sub_bin)
                beg_pct = 1 - end_pct
            elif end_pct == 1.0:
                beg_sub_bin = 1 + int(end_sub_bin)
                beg_pct = 1.0
            end_sub_bin = int((i + 1) * len_subbins)
            if (i + 1) * len_subbins - end_sub_bin == 0.0:
                end_sub_bin = end_sub_bin - 1
                end_pct = 1
            elif (i + 1) * len_subbins - end_sub_bin > 0.0:
                end_pct = (i + 1) * len_subbins - end_sub_bin
            deaths_sub = deaths_0_99[beg_sub_bin : end_sub_bin + 1]
            pop_sub = pop_0_99[beg_sub_bin : end_sub_bin + 1]
            deaths_sub[0] = beg_pct * deaths_sub[0]
            pop_sub[0] = beg_pct * pop_sub[0]
            deaths_sub[-1] = end_pct * deaths_sub[-1]
            pop_sub[-1] = end_pct * pop_sub[-1]
            mort_rates[i] = deaths_sub.sum() / pop_sub.sum()
    # Artificially set the mortality rate of the oldest age to 1.
    orig_end_mort_rate = mort_rates[-1]
    mort_rates[-1] = 1.0
    ages = np.arange(1, totpers + 1)

    if graph:
        mort_rates_all = np.hstack((infmort_rate, mort_rates))
        mort_rates_all[-1] = orig_end_mort_rate
        plt.plot(np.arange(0, totpers + 1), mort_rates_all)
        plt.scatter(
            0,
            infmort_rate,
            c="green",
            marker="d",
            label="Infant mortality rate",
        )
        plt.scatter(
            ages,
            np.hstack((mort_rates[:-1], orig_end_mort_rate)),
            c="blue",
            marker="d",
            label="Mortality rates, model ages 1 to " + str(totpers),
        )
        plt.scatter(
            totpers,
            1.0,
            c="red",
            marker="d",
            label="Artificial mortality limit, model age " + str(totpers),
        )
        plt.xlabel(r"Age $s$")
        plt.ylabel(r"Mortality rate $\rho_{s}$")
        plt.legend(loc="upper left")
        plt.text(
            0,
            -0.23,
            "Source: UN Population Data",
            fontsize=9,
        )
        plt.tight_layout(rect=(0, 0.035, 1, 1))
        output_path = os.path.join(OUTPUT_DIR, "mort_rates")
        plt.savefig(output_path)
        plt.close()

    return mort_rates, infmort_rate


def pop_rebin(curr_pop_dist, totpers_new):
    """
    For cases in which totpers (E+S) is less than the number of periods
    in the population distribution data, this function calculates a new
    population distribution vector with totpers (E+S) elements.

    Args:
        curr_pop_dist (Numpy array): population distribution over N
            periods
        totpers_new (int): number of periods to which we are
            transforming the population distribution, >= 3

    Returns:
        curr_pop_new (Numpy array): new population distribution over
            totpers (E+S) periods that approximates curr_pop_dist

    """
    assert totpers_new >= 3
    # Number of periods in original data
    totpers_orig = len(curr_pop_dist)
    if int(totpers_new) == totpers_orig:
        curr_pop_new = curr_pop_dist
    elif int(totpers_new) < totpers_orig:
        curr_pop_new = np.zeros(totpers_new)
        len_subbins = np.float64(totpers_orig / totpers_new)
        end_sub_bin = int(0)
        end_pct = 0.0
        for i in range(totpers_new):
            if end_pct < 1.0:
                beg_sub_bin = int(end_sub_bin)
                beg_pct = 1 - end_pct
            elif end_pct == 1.0:
                beg_sub_bin = 1 + int(end_sub_bin)
                beg_pct = 1.0
            end_sub_bin = int((i + 1) * len_subbins)
            if (i + 1) * len_subbins - end_sub_bin == 0.0:
                end_sub_bin = end_sub_bin - 1
                end_pct = 1
            elif (i + 1) * len_subbins - end_sub_bin > 0.0:
                end_pct = (i + 1) * len_subbins - end_sub_bin
            curr_pop_sub = curr_pop_dist[beg_sub_bin : end_sub_bin + 1]
            curr_pop_sub[0] = beg_pct * curr_pop_sub[0]
            curr_pop_sub[-1] = end_pct * curr_pop_sub[-1]
            curr_pop_new[i] = curr_pop_sub.sum()

    return curr_pop_new


def get_imm_resid(totpers, start_year=2021, end_year=None, graph=False):
    """
    Calculate immigration rates by age as a residual given population levels in
    different periods, then output average calculated immigration rate. We have
    to replace the first mortality rate in this function in order to adjust the
    first implied immigration rate. Source: UN Population Data portal.

    Args:
        totpers (int): total number of agent life periods (E+S), >= 3
        start_year (int): first year data to download
        end_year (int or None): end year data to download
        graph (bool): =True if want graphical output

    Returns:
        imm_rates (Numpy array):immigration rates that correspond to
            each period of life, length E+S

    """
    pop_df = get_un_pop_data(start_year=2019, end_year=2021, download=False)
    pop_2019 = (
        pop_df["pop"][
            (
                (pop_df["sex_num"] == 3)
                & (pop_df["year"] == 2019)
                & (pop_df["age"] < 100)
            )
        ]
        .to_numpy()
        .flatten()
    )
    pop_2020 = (
        pop_df["pop"][
            (
                (pop_df["sex_num"] == 3)
                & (pop_df["year"] == 2020)
                & (pop_df["age"] < 100)
            )
        ]
        .to_numpy()
        .flatten()
    )
    pop_2021 = (
        pop_df["pop"][
            (
                (pop_df["sex_num"] == 3)
                & (pop_df["year"] == 2021)
                & (pop_df["age"] < 100)
            )
        ]
        .to_numpy()
        .flatten()
    )
    pop_2019_EpS = pop_rebin(pop_2019, totpers)
    pop_2020_EpS = pop_rebin(pop_2020, totpers)
    pop_2021_EpS = pop_rebin(pop_2021, totpers)

    fert_rates = get_fert(totpers, start_year=start_year, end_year=end_year)
    mort_rates, infmort_rate = get_mort(
        totpers, start_year=start_year, end_year=end_year
    )

    imm_rate_1_2020 = (
        pop_2021_EpS[0]
        - (1 - infmort_rate) * (fert_rates * pop_2020_EpS).sum()
    ) / pop_2020_EpS[0]
    imm_rate_1_2019 = (
        pop_2020_EpS[0]
        - (1 - infmort_rate) * (fert_rates * pop_2019_EpS).sum()
    ) / pop_2019_EpS[0]
    imm_rate_1 = (imm_rate_1_2020 + imm_rate_1_2019) / 2

    imm_rates_sp1_2020 = (
        pop_2021_EpS[1:] - (1 - mort_rates[:-1]) * pop_2020_EpS[:-1]
    ) / pop_2020_EpS[:-1]
    imm_rates_sp1_2019 = (
        pop_2020_EpS[1:] - (1 - mort_rates[:-1]) * pop_2019_EpS[:-1]
    ) / pop_2019_EpS[:-1]
    imm_rates_sp1 = (imm_rates_sp1_2020 + imm_rates_sp1_2019) / 2
    imm_rates = np.hstack((imm_rate_1, imm_rates_sp1))
    if graph:
        ages = np.arange(1, totpers + 1)
        plt.plot(ages, imm_rates, label="Residual data")
        plt.xlabel(r"Age $s$")
        plt.ylabel(r"immigration rates $\i_s$")
        output_path = os.path.join(OUTPUT_DIR, "imm_rates")
        plt.savefig(output_path)
        plt.close()

    return imm_rates


def immsolve(imm_rates, *args):
    """
    This function generates a vector of errors representing the
    difference in two consecutive periods stationary population
    distributions. This vector of differences is the zero-function
    objective used to solve for the immigration rates vector, similar to
    the original immigration rates vector from get_imm_resid(), that
    sets the steady-state population distribution by age equal to the
    population distribution in period int(1.5*S)

    Args:
        imm_rates (Numpy array):immigration rates that correspond to
            each period of life, length E+S
        args (tuple): (fert_rates, mort_rates, infmort_rate, omega_cur,
            g_n_ss)

    Returns:
        omega_errs (Numpy array): difference between omega_new and
            omega_cur_pct, length E+S

    """
    fert_rates, mort_rates, infmort_rate, omega_cur_lev, g_n_ss = args
    omega_cur_pct = omega_cur_lev / omega_cur_lev.sum()
    totpers = len(fert_rates)
    OMEGA = np.zeros((totpers, totpers))
    OMEGA[0, :] = (1 - infmort_rate) * fert_rates + np.hstack(
        (imm_rates[0], np.zeros(totpers - 1))
    )
    OMEGA[1:, :-1] += np.diag(1 - mort_rates[:-1])
    OMEGA[1:, 1:] += np.diag(imm_rates[1:])
    omega_new = np.dot(OMEGA, omega_cur_pct) / (1 + g_n_ss)
    omega_errs = omega_new - omega_cur_pct

    return omega_errs


def get_pop_objs(E, S, T, curr_year, download=False, GraphDiag=False):
    """
    This function produces the demographics objects to be used in the OG-ZAF
    model package.

    Args:
        E (int): number of model periods in which agent is not
            economically active, >= 1
        S (int): number of model periods in which agent is economically
            active, >= 3
        T (int): number of periods to be simulated in TPI, > 2*S
        curr_year (int): current year for which analysis will begin,
            >= 2016
        GraphDiag (bool): =True if want graphical output and printed
                diagnostics

    Returns:
        pop_dict (dict): includes:
            omega_path_S (Numpy array), time path of the population
                distribution from the current state to the steady-state,
                size T+S x S
            g_n_ss (scalar): steady-state population growth rate
            omega_SS (Numpy array): normalized steady-state population
                distribution, length S
            mort_rates (Numpy array): mortality rates that correspond to
                each model period of life, length S
            g_n_path (Numpy array): population growth rates over the time
                path, length T + S

    """
    assert curr_year >= 2021
    most_recent_data_year = 2021
    hardcode_start_year = min(curr_year, most_recent_data_year)
    fert_rates = get_fert(
        E + S, start_year=hardcode_start_year, download=download
    )
    mort_rates, infmort_rate = get_mort(
        E + S, start_year=hardcode_start_year, download=download
    )
    mort_rates_S = mort_rates[-S:]
    imm_rates_orig = get_imm_resid(E + S, start_year=hardcode_start_year)
    OMEGA_orig = np.zeros((E + S, E + S))
    OMEGA_orig[0, :] = (1 - infmort_rate) * fert_rates
    OMEGA_orig[1:, :-1] += np.diag(1 - mort_rates[:-1])
    OMEGA_orig += np.diag(imm_rates_orig)

    # Solve for steady-state population growth rate and steady-state
    # population distribution by age using eigenvalue and eigenvector
    # decomposition
    eigvalues_orig, eigvectors_orig = np.linalg.eig(OMEGA_orig)
    g_n_ss_orig = (eigvalues_orig[np.isreal(eigvalues_orig)].real).max() - 1
    eigvec_raw_orig = eigvectors_orig[
        :, (eigvalues_orig[np.isreal(eigvalues_orig)].real).argmax()
    ].real
    omega_SS_orig = eigvec_raw_orig / eigvec_raw_orig.sum()

    # Generate time path of the nonstationary population distribution
    omega_path_lev = np.zeros((E + S, T + S))
    pop_df = get_un_pop_data(start_year=2019, end_year=2021, download=download)
    pop_2020 = (
        pop_df["pop"][
            (
                (pop_df["sex_num"] == 3)
                & (pop_df["year"] == 2020)
                & (pop_df["age"] < 100)
            )
        ]
        .to_numpy()
        .flatten()
    )
    pop_2021 = (
        pop_df["pop"][
            (
                (pop_df["sex_num"] == 3)
                & (pop_df["year"] == 2021)
                & (pop_df["age"] < 100)
            )
        ]
        .to_numpy()
        .flatten()
    )
    age_per_EpS = np.arange(1, E + S + 1)
    pop_2020_EpS = pop_rebin(pop_2020, E + S)
    pop_2021_EpS = pop_rebin(pop_2021, E + S)
    pop_2021_pct = pop_2021_EpS / pop_2021_EpS.sum()
    # Age most recent population data to the current year of analysis
    pop_curr = pop_2020_EpS.copy()
    # pop_past = pop_2020_EpS.copy()
    if curr_year == most_recent_data_year:
        pop_past = pop_curr.copy()
        pop_curr = np.dot(OMEGA_orig, pop_past)
        g_n_curr = (pop_curr[-S:].sum() - pop_past[-S:].sum()) / pop_past[
            -S:
        ].sum()
        omega_path_lev[:, 0] = pop_curr
    elif curr_year > most_recent_data_year:
        for per in range(curr_year - most_recent_data_year):
            pop_past = pop_curr.copy()
            pop_curr = np.dot(OMEGA_orig, pop_past)
            g_n_curr = (pop_curr[-S:].sum() - pop_past[-S:].sum()) / pop_past[
                -S:
            ].sum()
        omega_path_lev[:, 0] = pop_curr
    for per in range(1, T + S):
        pop_next = np.dot(OMEGA_orig, pop_curr)
        omega_path_lev[:, per] = pop_next.copy()
        pop_curr = pop_next.copy()

    # Force the population distribution after 1.5*S periods to be the
    # steady-state distribution by adjusting immigration rates, holding
    # constant mortality, fertility, and SS growth rates
    imm_tol = 1e-14
    fixper = int(1.5 * S)
    omega_SSfx = omega_path_lev[:, fixper] / omega_path_lev[:, fixper].sum()
    imm_objs = (
        fert_rates,
        mort_rates,
        infmort_rate,
        omega_path_lev[:, fixper],
        g_n_ss_orig,
    )
    imm_fulloutput = opt.fsolve(
        immsolve,
        imm_rates_orig,
        args=(imm_objs),
        full_output=True,
        xtol=imm_tol,
    )
    imm_rates_adj = imm_fulloutput[0]
    imm_diagdict = imm_fulloutput[1]
    omega_path_S = omega_path_lev[-S:, :] / np.tile(
        omega_path_lev[-S:, :].sum(axis=0), (S, 1)
    )
    omega_path_S[:, fixper:] = np.tile(
        omega_path_S[:, fixper].reshape((S, 1)), (1, T + S - fixper)
    )
    g_n_path = np.zeros(T + S)
    g_n_path[0] = g_n_curr.copy()
    g_n_path[1:] = (
        omega_path_lev[-S:, 1:].sum(axis=0)
        - omega_path_lev[-S:, :-1].sum(axis=0)
    ) / omega_path_lev[-S:, :-1].sum(axis=0)
    # Compute adjusted population growth rate
    OMEGA2 = np.zeros((E + S, E + S))
    OMEGA2[0, :] = (1 - infmort_rate) * fert_rates
    OMEGA2[1:, :-1] += np.diag(1 - mort_rates[:-1])
    OMEGA2 += np.diag(imm_rates_adj)
    eigvalues2, eigvectors2 = np.linalg.eig(OMEGA2)
    g_n_ss_adj = (eigvalues2[np.isreal(eigvalues2)].real).max() - 1
    g_n_ss = g_n_ss_adj.copy()
    # g_n_path[fixper + 1 :] = g_n_ss
    omega_S_preTP = pop_past[-S:] / pop_past[-S:].sum()
    imm_rates_mat = np.hstack(
        (
            np.tile(np.reshape(imm_rates_orig[E:], (S, 1)), (1, fixper)),
            np.tile(
                np.reshape(imm_rates_adj[E:], (S, 1)), (1, T + S - fixper)
            ),
        )
    )

    if GraphDiag:
        # Check whether original SS population distribution is close to
        # the period-T population distribution
        omegaSSmaxdif = np.absolute(
            omega_SS_orig - (omega_path_lev[:, T] / omega_path_lev[:, T].sum())
        ).max()
        if omegaSSmaxdif > 0.0003:
            print(
                "POP. WARNING: Max. abs. dist. between original SS "
                + "pop. dist'n and period-T pop. dist'n is greater than"
                + " 0.0003. It is "
                + str(omegaSSmaxdif)
                + "."
            )
        else:
            print(
                "POP. SUCCESS: orig. SS pop. dist is very close to "
                + "period-T pop. dist'n. The maximum absolute "
                + "difference is "
                + str(omegaSSmaxdif)
                + "."
            )

        # Plot the adjusted steady-state population distribution versus
        # the original population distribution. The difference should be
        # small
        omegaSSvTmaxdiff = np.absolute(omega_SS_orig - omega_SSfx).max()
        if omegaSSvTmaxdiff > 0.0003:
            print(
                "POP. WARNING: The maximimum absolute difference "
                + "between any two corresponding points in the original"
                + " and adjusted steady-state population "
                + "distributions is"
                + str(omegaSSvTmaxdiff)
                + ", "
                + "which is greater than 0.0003."
            )
        else:
            print(
                "POP. SUCCESS: The maximum absolute difference "
                + "between any two corresponding points in the original"
                + " and adjusted steady-state population "
                + "distributions is "
                + str(omegaSSvTmaxdiff)
            )

        # Print whether or not the adjusted immigration rates solved the
        # zero condition
        immtol_solved = np.absolute(imm_diagdict["fvec"].max()) < imm_tol
        if immtol_solved:
            print(
                "POP. SUCCESS: Adjusted immigration rates solved "
                + "with maximum absolute error of "
                + str(np.absolute(imm_diagdict["fvec"].max()))
                + ", which is less than the tolerance of "
                + str(imm_tol)
            )
        else:
            print(
                "POP. WARNING: Adjusted immigration rates did not "
                + "solve. Maximum absolute error of "
                + str(np.absolute(imm_diagdict["fvec"].max()))
                + " is greater than the tolerance of "
                + str(imm_tol)
            )

        # Test whether the steady-state growth rates implied by the
        # adjusted OMEGA matrix equals the steady-state growth rate of
        # the original OMEGA matrix
        if np.max(np.absolute(g_n_ss_adj - g_n_ss_orig)) > 10 ** (-8):
            print(
                "FAILURE: The steady-state population growth rate"
                + " from adjusted OMEGA is different (diff is "
                + str(g_n_ss_adj - g_n_ss_orig)
                + ") than the steady-"
                + "state population growth rate from the original"
                + " OMEGA."
            )
        elif np.max(np.absolute(g_n_ss_adj - g_n_ss_orig)) <= 10 ** (-8):
            print(
                "SUCCESS: The steady-state population growth rate"
                + " from adjusted OMEGA is close to (diff is "
                + str(g_n_ss_adj - g_n_ss_orig)
                + ") the steady-"
                + "state population growth rate from the original"
                + " OMEGA."
            )

        # Do another test of the adjusted immigration rates. Create the
        # new OMEGA matrix implied by the new immigration rates. Plug in
        # the adjusted steady-state population distribution. Hit is with
        # the new OMEGA transition matrix and it should return the new
        # steady-state population distribution
        omega_new = np.dot(OMEGA2, omega_SSfx)
        omega_errs = np.absolute(omega_new - omega_SSfx)
        print(
            "The maximum absolute difference between the adjusted "
            + "steady-state population distribution and the "
            + "distribution generated by hitting the adjusted OMEGA "
            + "transition matrix is "
            + str(omega_errs.max())
        )

        # Plot the original immigration rates versus the adjusted
        # immigration rates
        immratesmaxdiff = np.absolute(imm_rates_orig - imm_rates_adj).max()
        print(
            "The maximum absolute distance between any two points "
            + "of the original immigration rates and adjusted "
            + "immigration rates is "
            + str(immratesmaxdiff)
        )

        # plots
        pp.plot_omega_fixed(
            age_per_EpS, omega_SS_orig, omega_SSfx, E, S, output_dir=OUTPUT_DIR
        )
        pp.plot_imm_fixed(
            age_per_EpS,
            imm_rates_orig,
            imm_rates_adj,
            E,
            S,
            output_dir=OUTPUT_DIR,
        )
        pp.plot_population_path(
            age_per_EpS,
            pop_2021_pct,
            omega_path_lev,
            omega_SSfx,
            curr_year,
            E,
            S,
            output_dir=OUTPUT_DIR,
        )

    # return omega_path_S, g_n_ss, omega_SSfx,
    # mort_rates_S, and g_n_path
    pop_dict = {
        "omega": omega_path_S.T,
        "g_n_ss": g_n_ss,
        "omega_SS": omega_SSfx[-S:] / omega_SSfx[-S:].sum(),
        "rho": [mort_rates_S],
        "g_n": g_n_path,
        "imm_rates": imm_rates_mat.T,
        "omega_S_preTP": omega_S_preTP,
    }

    return pop_dict


def extrap_exp_3(
    x_vals, con_val: tuple, con_slope: float, eps_val: tuple, low: bool = True
):
    """
    This function fits a smooth exponential extrapolation to either the low end
    of data or the high end of data, both of which are monotonically
    asymptoting to zero. For the exponential function extrapolation on both
    ends of the distribution, we use the function:

    f(x) = e ** (a * (x ** 2) + b * x + c)
    s.t.    (i) f(x_con) = y_con
           (ii) f'(x_con) = con_slope
          (iii) f'(x_eps) = eps_low (>0) or eps_high (<0)

    Args:
        x_vals (array_like): array of x values to be extrapolated
        con_val (tuple): (x, y) coordinate at which the function must connect
            to the data
        con_slope (float): slope of the data at the connecting value
        eps_val (tuple): (x, y) coordinate at which the function must be close
            to zero
        low (bool): If True, the function is fit to the low end of the data.
            If False, the function is fit to the high end of the data.

    Returns:
        y_vals (array_like): extrapolated y values corresponding to x values
    """
    if low:
        if con_slope <= 0:
            err_msg = (
                "ERROR extrap_exp_3: con_slope must be positive if "
                + "extrapolating to the low end of the data."
            )
            raise ValueError(err_msg)
    else:
        if con_slope >= 0:
            err_msg = (
                "ERROR extrap_exp_3: con_slope must be negative if "
                + "extrapolating to the high end of the data."
            )
            raise ValueError(err_msg)

    eps_slope_low = 0.0001
    eps_slope_high = -eps_slope_low

    # Unpack the coordinates
    x_con, y_con = con_val
    x_eps, y_eps = eps_val

    # check if linear extrapolation intersects zero beyond x_eps
    lin_y_intercept = y_con - con_slope * x_con
    x_intercept = -lin_y_intercept / con_slope
    if low:
        lin_extrap_overshoot = x_intercept < x_eps
    else:
        lin_extrap_overshoot = x_intercept > x_eps
    if lin_extrap_overshoot:
        # Estimate an arctangent function to fit the data
        print(
            "WARNING: extrap_exp_3: Linear extrapolation overshoots "
            + "furthest value. Using arctangent function instead."
        )
        y_vals = extrap_arctan_3(x_vals, con_slope, x_con, y_con, x_eps, low)
    else:
        # Estimate an exponential function to fit the data
        if low:
            params = [con_slope, x_con, y_con, x_eps, eps_slope_low]
        else:
            params = [con_slope, x_con, y_con, x_eps, eps_slope_high]
        a_guess = 0.1
        b_guess = 0.1
        ab_guess = np.array([a_guess, b_guess])
        solution = opt.root(
            ab_zero_eqs_exp_func, ab_guess, args=params, method="lm"
        )
        if not solution.success:
            err_msg = (
                "ERROR extrap_exp_3: Root finder failed in "
                + "ab_zero_eqs_exp_func."
            )
            raise ValueError(err_msg)
        a, b = solution.x
        if low:
            # a = np.log(con_slope / eps_low) / (x_con - x_eps)
            y_pos_ind = x_vals >= x_eps
        else:
            # a = np.log(con_slope / eps_high) / (x_con - x_eps)
            y_pos_ind = x_vals <= x_eps
        # b = np.log(con_slope / (a * np.exp(a * x_con)))
        # c = y_con - np.exp(a * x_con + b)
        c = np.log(y_con) - a * (x_con**2) - b * x_con

        len_x_vals = len(x_vals)
        len_y_pos_ind = y_pos_ind.sum()
        if low:
            y_vals = np.hstack(
                (
                    np.zeros(len_x_vals - len_y_pos_ind),
                    np.exp(
                        a * (x_vals[y_pos_ind] ** 2)
                        + b * x_vals[y_pos_ind]
                        + c
                    ),
                )
            )
        else:
            y_vals = np.hstack(
                (
                    np.exp(
                        a * (x_vals[y_pos_ind] ** 2)
                        + b * x_vals[y_pos_ind]
                        + c
                    ),
                    np.zeros(len_x_vals - len_y_pos_ind),
                )
            )

    return y_vals


def extrap_arctan_3(
    x_vals, con_slope: float, x_con, y_con, x_eps, low: bool = True
):
    """
    This function fits an arctangent function to extrapolate data that
    monotonically decrease to zero and start with small absolute slope, then
    absolute slope increases, then absolute slope decreases to zero. The
    arctangent function is the following with the three following identifying
    conditions:

    if extrapolating to the low end of the data:
    f(x) = (a / pi) * arctan(b * x + c) + (a / 2) s.t. a, b > 0
    where f'(x) =  (a * b) / (pi * (1 + (b * x + c)^2))

    if extrapolating to the high end of the data:
    f(x) = (-a / pi) * arctan(b * x + c) + (a / 2) s.t. a, b > 0
    where f'(x) =  (-a * b) / (pi * (1 + (b * x + c)^2))

    s.t.   (i) f(x_con) = y_con
    and   (ii) f'(x_con) = con_slope
    and  (iii) b * (2/3 * x_con + 1/3 * x_eps) + c = 0

    The solution to this problem can be reduced to finding the root of a
    univariate equation in the parameter b.

    Args:
        x_vals (array_like): array of x values to be extrapolated
        con_slope (float): slope of the data at the connecting value
        x_con (float): x value at which the function must connect to the data
        y_con (float): y value at which the function must connect to the data
        x_eps (float): x value at which the function must be close to zero
        low (bool): If True, the function is fit to the low end of the data.
            If False, the function is fit to the high end of the data.

    Returns:
        y_vals (array_like): extrapolated y values corresponding to x values
    """
    y_vals = np.zeros_like(x_vals)

    # Solve for the parameter b
    params = [con_slope, x_con, y_con, x_eps, low]
    b_guess = 20.0
    solution = opt.root(b_zero_eq_arctan_func, b_guess, args=params)
    if not solution.success:
        err_msg = (
            "ERROR extrap_arctan_3: Root finder failed in "
            + "b_zero_eq_arctan_func."
        )
        raise ValueError(err_msg)
    b = solution.x

    len_x_vals = len(x_vals)

    if low:
        a = y_con / (
            (1 / np.pi) * np.arctan((b / 3) * (x_con - x_eps)) + (1 / 2)
        )
        c = -b * ((2 / 3) * x_con + (1 / 3) * x_eps)
        y_pos_ind = x_vals >= x_eps
        len_y_pos_ind = y_pos_ind.sum()
        y_vals = np.hstack(
            (
                np.zeros(len_x_vals - len_y_pos_ind),
                (a / np.pi) * np.arctan(b * x_vals[y_pos_ind] + c) + (a / 2),
            )
        )
    else:
        a = y_con / (
            (-1 / np.pi) * np.arctan((b / 3) * (x_con - x_eps)) + (1 / 2)
        )
        c = -b * ((2 / 3) * x_con + (1 / 3) * x_eps)
        y_pos_ind = x_vals <= x_eps
        len_y_pos_ind = y_pos_ind.sum()
        y_vals = np.hstack(
            (
                (-a / np.pi) * np.arctan(b * x_vals[y_pos_ind] + c) + (a / 2),
                np.zeros(len_x_vals - len_y_pos_ind),
            )
        )

    return y_vals


def ab_zero_eqs_exp_func(ab_vals, params):
    """ "
    This function returns a vector of error values for the two zero equations
    in terms of parameters a and b for given values of a and b.
    """
    con_slope, x_con, y_con, x_eps, eps = params
    a, b = ab_vals

    c = np.log(y_con) - a * (x_con**2) - b * x_con
    error_1 = (2 * a * x_con + b) * np.exp(
        a * (x_con**2) + b * x_con + c
    ) - con_slope
    error_2 = (2 * a * x_eps + b) * np.exp(
        a * (x_eps**2) + b * x_eps + c
    ) - eps

    error_vec = np.array([error_1, error_2])

    return error_vec


def b_zero_eq_arctan_func(b, params):
    """ "
    This function returns a scalar error value of the univariate error function
    in parameter b for given values of b.
    """
    con_slope, x_con, y_con, x_eps, low = params

    if low:
        a = y_con / (
            (1 / np.pi) * np.arctan((b / 3) * (x_con - x_eps)) + (1 / 2)
        )
        a_other = (
            con_slope * np.pi * (1 + ((b / 3) ** 2) * ((x_con - x_eps) ** 2))
        ) / b
        error_val = a_other - a
    else:
        a = y_con / (
            (-1 / np.pi) * np.arctan((b / 3) * (x_con - x_eps)) + (1 / 2)
        )
        a_other = (
            -con_slope * np.pi * (1 + ((b / 3) ** 2) * ((x_con - x_eps) ** 2))
        ) / b
        error_val = a_other - a

    return error_val
