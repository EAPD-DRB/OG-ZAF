"""
-------------------------------------------------------------------------------
Functions for generating demographic objects necessary for the OG-ZAF model
-------------------------------------------------------------------------------
"""
# Import packages
import os
import numpy as np
import pandas as pd
import scipy.optimize as opt
import scipy.interpolate as si
from ogcore import parameter_plots as pp
import matplotlib.pyplot as plt

# create output director for figures
CUR_PATH = os.path.split(os.path.abspath(__file__))[0]
DATA_DIR = os.path.join(CUR_PATH, "data", "demographic")
OUTPUT_DIR = os.path.join(CUR_PATH, "OUTPUT", "Demographics")
if os.access(OUTPUT_DIR, os.F_OK) is False:
    os.makedirs(OUTPUT_DIR)

"""
------------------------------------------------------------------------
Define functions
------------------------------------------------------------------------
"""


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
        end_year (int): end year of the data
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
        pop_target = (
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
        fert_target = (
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
    else:
        pop_target = os.path.join(DATA_DIR, "un_zaf_pop.csv")
        fert_target = os.path.join(DATA_DIR, "un_zaf_fert.csv")

    # Convert .csv file to Pandas DataFrame
    pop_df = pd.read_csv(
        pop_target,
        sep="|",
        header=1,
        usecols=["TimeLabel", "SexId", "Sex", "AgeMid", "Value"],
        float_precision="round_trip",
    )
    fert_rates_df = pd.read_csv(
        fert_target,
        sep="|",
        header=1,
        usecols=["TimeLabel", "AgeMid", "Value"],
        float_precision="round_trip",
    )

    # Rename variables in the population and fertility rates data
    pop_df.rename(
        columns={
            "TimeLabel": "year",
            "SexId": "sex_num",
            "Sex": "sex_str",
            "AgeMid": "age",
            "Value": "pop",
        },
        inplace=True,
    )
    fert_rates_df.rename(
        columns={
            "TimeLabel": "year",
            "AgeMid": "age",
            "Value": "births_p_1000f",
        },
        inplace=True,
    )

    # Clean the data
    # I don't know why in the pop_df population data by age and sex and year
    # there are 10 different population numbers for each sex and age and year
    # and all the other variables are equal. I just average them here.
    pop_df = (
        pop_df.groupby(["year", "sex_num", "sex_str", "age"])
        .mean()
        .reset_index()
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
        end_year (int): end year of the data
        download (bool): whether to download the data from the UN Data Portal.
            If False, a path must be specified in the path_folder argument.
        path_folder (None or str): string path to folder where data are stored

    Returns:
        fert_rates_df (DataFrame): dataset with fertility rates by age
    """
    if end_year is None:
        end_year = start_year
    # UN variable code for Age specific mortality rate
    mort_code = "80"
    # UN variable code for Age specific mortality rate
    infmort_code = "22"

    if download:
        infmort_target = (
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
        mort_target = (
            "https://population.un.org/dataportalapi/api/v1/data/indicators/"
            + mort_code
            + "/locations/"
            + country_id
            + "/start/"
            + str(start_year)
            + "/end/"
            + str(end_year)
            + "?format=csv"
        )
    else:
        infmort_target = os.path.join(DATA_DIR, "un_zaf_infmort.csv")
        mort_target = os.path.join(DATA_DIR, "un_zaf_mort.csv")

    # Convert .csv file to Pandas DataFrame
    infmort_rate_df = pd.read_csv(
        infmort_target,
        sep="|",
        header=1,
        usecols=["TimeLabel", "SexId", "Sex", "Value"],
        float_precision="round_trip",
    )
    mort_rates_df = pd.read_csv(
        mort_target,
        sep="|",
        header=1,
        usecols=["TimeLabel", "SexId", "Sex", "AgeStart", "Value"],
        float_precision="round_trip",
    )

    # Rename variables in the population and fertility rates data
    infmort_rate_df.rename(
        columns={
            "TimeLabel": "year",
            "SexId": "sex_num",
            "Sex": "sex_str",
            "Value": "inf_deaths_p_1000",
        },
        inplace=True,
    )
    mort_rates_df.rename(
        columns={
            "TimeLabel": "year",
            "SexId": "sex_num",
            "Sex": "sex_str",
            "AgeStart": "age",
            "Value": "mort_rate",
        },
        inplace=True,
    )

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
    mort_rates_df = mort_rates_df[
        (
            (mort_rates_df["year"] >= start_year)
            & (mort_rates_df["year"] <= end_year)
        )
    ]

    return infmort_rate_df, mort_rates_df


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
        fert_rates_df (DataFrame): dataset with fertility rates by age
    """
    if end_year is None:
        end_year = start_year
    # UN variable code for Population by 1-year age groups and sex
    pop_code = "47"

    if download:
        pop_target = (
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

    # Clean the data
    pop_df = pop_df[
        ((pop_df["year"] >= start_year) & (pop_df["year"] <= end_year))
    ]

    return pop_df


def get_fert(totpers, min_yr, max_yr, graph=False):
    """
    This function generates a vector of fertility rates by model period
    age that corresponds to the fertility rate data by age in years.

    Args:
        totpers (int): total number of agent life periods (E+S), >= 3
        min_yr (int): age in years at which agents are born, >= 0
        max_yr (int): age in years at which agents die with certainty,
            >= 4
        graph (bool): =True if want graphical output

    Returns:
        fert_rates (Numpy array): fertility rates for each model period
            of life

    """
    # Get UN fertility rates for India for ages 15-49
    ages_15_49 = np.arange(15, 50)
    fert_rates_15_49 = (
        get_un_fert_data(download=False)["fert_rate"].to_numpy().flatten()
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
    fert_rates = np.hstack(
        (fert_rates_1_14, fert_rates_15_49, fert_rates_50_100)
    )
    ages = np.arange(1, 101)

    if graph:  # Plot fertility rates
        plt.plot(ages, fert_rates)
        plt.xlabel(r"Age $s$")
        plt.ylabel(r"Fertility rate $f_{s}$")
        plt.text(
            -5,
            -0.023,
            "Source: UN Population Data",
            fontsize=9,
        )
        plt.tight_layout(rect=(0, 0.035, 1, 1))
        output_path = os.path.join(OUTPUT_DIR, "fert_rates")
        plt.savefig(output_path)
        plt.close()

    return fert_rates


def get_mort(totpers, min_yr, max_yr, graph=False):
    """
    This function generates a vector of mortality rates by model period
    age. Source: UN Population Data portal.

    Args:
        totpers (int): total number of agent life periods (E+S), >= 3
        min_yr (int): age in years at which agents are born, >= 0
        max_yr (int): age in years at which agents die with certainty,
            >= 4
        graph (bool): =True if want graphical output

    Returns:
        mort_rates (Numpy array) mortality rates that correspond to each
            period of life
        infmort_rate (scalar): infant mortality rate

    """
    # Get UN infant mortality and mortality rate data by age
    infmort_rate_df, mort_rates_df = get_un_mort_data(
        start_year=2021, download=False
    )
    infmort_rate = infmort_rate_df["infmort_rate"][
        infmort_rate_df["sex_num"] == 3
    ].to_numpy()[0]
    mort_rates = (
        mort_rates_df["mort_rate"][
            ((mort_rates_df["sex_num"] == 3) & (mort_rates_df["age"] < 100))
        ]
        .to_numpy()
        .flatten()
    )

    if graph:
        ages_all = np.arange(0, 101)
        mort_rates_all = np.hstack((infmort_rate, mort_rates))
        plt.plot(ages_all, mort_rates_all, label="Data")
        plt.scatter(
            0,
            infmort_rate,
            c="blue",
            marker="d",
            label="Infant mortality rate",
        )
        plt.scatter(
            100, 1.0, c="red", marker="d", label="Artificial mortality limit"
        )
        plt.xlabel(r"Age $s$")
        plt.ylabel(r"Mortality rate $\rho_{s}$")
        plt.legend(loc="upper left")
        plt.text(
            -5,
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
    # Number of periods in original data
    assert totpers_new >= 3
    # Number of periods in original data
    totpers_orig = len(curr_pop_dist)
    if int(totpers_new) == totpers_orig:
        curr_pop_new = curr_pop_dist
    elif int(totpers_new) < totpers_orig:
        num_sub_bins = float(10000)
        curr_pop_sub = np.repeat(
            np.float64(curr_pop_dist) / num_sub_bins, num_sub_bins
        )
        len_subbins = (np.float64(totpers_orig * num_sub_bins)) / totpers_new
        curr_pop_new = np.zeros(totpers_new, dtype=np.float64)
        end_sub_bin = 0
        for i in range(totpers_new):
            beg_sub_bin = int(end_sub_bin)
            end_sub_bin = int(np.rint((i + 1) * len_subbins))
            curr_pop_new[i] = curr_pop_sub[beg_sub_bin:end_sub_bin].sum()
        # Return curr_pop_new to single precision float (float32)
        # datatype
        curr_pop_new = np.float32(curr_pop_new)

    return curr_pop_new


def get_imm_resid(totpers, min_yr, max_yr):
    """
    Calculate immigration rates by age as a residual given population
    levels in different periods, then output average calculated
    immigration rate. We have to replace the first mortality rate in
    this function in order to adjust the first implied immigration rate
    Source: India Census, 2001 and 2011

    Args:
        totpers (int): total number of agent life periods (E+S), >= 3
        min_yr (int): age in years at which agents are born, >= 0
        max_yr (int): age in years at which agents die with certainty,
            >= 4
        graph (bool): =True if want graphical output

    Returns:
        imm_rates (Numpy array):immigration rates that correspond to
            each period of life, length E+S

    """
    pop_file = utils.read_file(
        CUR_PATH, os.path.join("data", "demographic", "india_pop_data.csv")
    )
    pop_data = pd.read_csv(pop_file, encoding="utf-8")
    pop_data_samp = pop_data[
        (pop_data["Age"] >= min_yr - 1) & (pop_data["Age"] <= max_yr - 1)
    ]
    age_year_all = pop_data_samp["Age"] + 1
    pop_2001, pop_2011 = (
        np.array(pop_data_samp["2001"], dtype="f"),
        np.array(pop_data_samp["2011"], dtype="f"),
    )
    pop_2001_EpS = pop_rebin(pop_2001, totpers)
    pop_2011_EpS = pop_rebin(pop_2011, totpers)
    # Create three years of estimated immigration rates for youngest age
    # individuals
    imm_mat = np.zeros((2, totpers))
    fert_rates = get_fert(totpers, min_yr, max_yr, False)
    mort_rates, infmort_rate = get_mort(totpers, min_yr, max_yr, False)
    newbornvec = np.dot(fert_rates, pop_2001_EpS).T
    # imm_mat[:, 0] = ((pop_2011_EpS[0] - (1 - infmort_rate) * newbornvec)
    #                  / pop_2001_EpS[0])
    imm_mat[:, 0] = 0
    # Estimate immigration rates for all other-aged
    # individuals
    mort_rate10 = np.zeros_like(mort_rates[:-10])  # 10-year mort rate
    for i in range(10):
        mort_rate10 = mort_rates[i : -10 + i] + mort_rate10
    mort_rate10[mort_rate10 > 1.0] = 1.0
    imm_mat[:, 10:] = (
        pop_2011_EpS[10:] - (1 - mort_rate10) * pop_2001_EpS[:-10]
    ) / pop_2001_EpS[10:]
    # Final estimated immigration rates are the averages over years
    imm_rates = imm_mat.mean(axis=0)
    neg_rates = imm_rates < 0
    # For India, data were 10 years apart, so make annual rate
    imm_rates = ((1 + np.absolute(imm_rates)) ** (1 / 10)) - 1
    imm_rates[neg_rates] = -1 * imm_rates[neg_rates]

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
            g_n_SS)

    Returns:
        omega_errs (Numpy array): difference between omega_new and
            omega_cur_pct, length E+S

    """
    fert_rates, mort_rates, infmort_rate, omega_cur_lev, g_n_SS = args
    omega_cur_pct = omega_cur_lev / omega_cur_lev.sum()
    totpers = len(fert_rates)
    OMEGA = np.zeros((totpers, totpers))
    OMEGA[0, :] = (1 - infmort_rate) * fert_rates + np.hstack(
        (imm_rates[0], np.zeros(totpers - 1))
    )
    OMEGA[1:, :-1] += np.diag(1 - mort_rates[:-1])
    OMEGA[1:, 1:] += np.diag(imm_rates[1:])
    omega_new = np.dot(OMEGA, omega_cur_pct) / (1 + g_n_SS)
    omega_errs = omega_new - omega_cur_pct

    return omega_errs


def get_pop_objs(E, S, T, min_yr, max_yr, curr_year, GraphDiag=False):
    """
    This function produces the demographics objects to be used in the
    OG-ZAF model package.

    Args:
        E (int): number of model periods in which agent is not
            economically active, >= 1
        S (int): number of model periods in which agent is economically
            active, >= 3
        T (int): number of periods to be simulated in TPI, > 2*S
        min_yr (int): age in years at which agents are born, >= 0
        max_yr (int): age in years at which agents die with certainty,
            >= 4
        curr_year (int): current year for which analysis will begin,
            >= 2016
        GraphDiag (bool): =True if want graphical output and printed
                diagnostics

    Returns:
        pop_dict (dict): includes:
            omega_path_S (Numpy array), time path of the population
                distribution from the current state to the steady-state,
                size T+S x S
            g_n_SS (scalar): steady-state population growth rate
            omega_SS (Numpy array): normalized steady-state population
                distribution, length S
            surv_rates (Numpy array): survival rates that correspond to
                each model period of life, length S
            mort_rates (Numpy array): mortality rates that correspond to
                each model period of life, length S
            g_n_path (Numpy array): population growth rates over the time
                path, length T + S

    """
    assert curr_year >= 2011
    # age_per = np.linspace(min_yr, max_yr, E+S)
    fert_rates = get_fert(E + S, min_yr, max_yr, graph=False)
    mort_rates, infmort_rate = get_mort(E + S, min_yr, max_yr, graph=False)
    mort_rates_S = mort_rates[-S:]
    imm_rates_orig = get_imm_resid(E + S, min_yr, max_yr)
    OMEGA_orig = np.zeros((E + S, E + S))
    OMEGA_orig[0, :] = (1 - infmort_rate) * fert_rates + np.hstack(
        (imm_rates_orig[0], np.zeros(E + S - 1))
    )
    OMEGA_orig[1:, :-1] += np.diag(1 - mort_rates[:-1])
    OMEGA_orig[1:, 1:] += np.diag(imm_rates_orig[1:])

    # Solve for steady-state population growth rate and steady-state
    # population distribution by age using eigenvalue and eigenvector
    # decomposition
    eigvalues, eigvectors = np.linalg.eig(OMEGA_orig)
    g_n_SS = (eigvalues[np.isreal(eigvalues)].real).max() - 1
    eigvec_raw = eigvectors[
        :, (eigvalues[np.isreal(eigvalues)].real).argmax()
    ].real
    omega_SS_orig = eigvec_raw / eigvec_raw.sum()

    # Generate time path of the nonstationary population distribution
    omega_path_lev = np.zeros((E + S, T + S))
    pop_file = utils.read_file(
        CUR_PATH, os.path.join("data", "demographic", "india_pop_data.csv")
    )
    pop_data = pd.read_csv(pop_file, encoding="utf-8")
    pop_data_samp = pop_data[
        (pop_data["Age"] >= min_yr - 1) & (pop_data["Age"] <= max_yr - 1)
    ]
    pop_2011 = np.array(pop_data_samp["2011"], dtype="f")
    # Generate the current population distribution given that E+S might
    # be less than max_yr-min_yr+1
    age_per_EpS = np.arange(1, E + S + 1)
    pop_2011_EpS = pop_rebin(pop_2011, E + S)
    pop_2011_pct = pop_2011_EpS / pop_2011_EpS.sum()
    # Age most recent population data to the current year of analysis
    pop_curr = pop_2011_EpS.copy()
    data_year = 2019
    pop_next = np.dot(OMEGA_orig, pop_curr)
    g_n_curr = (pop_next[-S:].sum() - pop_curr[-S:].sum()) / pop_curr[
        -S:
    ].sum()  # g_n in 2019
    pop_past = pop_curr  # assume 2018-2019 pop
    # Age the data to the current year
    for per in range(curr_year - data_year):
        pop_next = np.dot(OMEGA_orig, pop_curr)
        g_n_curr = (pop_next[-S:].sum() - pop_curr[-S:].sum()) / pop_curr[
            -S:
        ].sum()
        pop_past = pop_curr
        pop_curr = pop_next

    # Generate time path of the population distribution
    omega_path_lev[:, 0] = pop_curr.copy()
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
        g_n_SS,
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
    g_n_path[fixper + 1 :] = g_n_SS
    omega_S_preTP = (pop_past.copy()[-S:]) / (pop_past.copy()[-S:].sum())
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
        OMEGA2 = np.zeros((E + S, E + S))
        OMEGA2[0, :] = (1 - infmort_rate) * fert_rates + np.hstack(
            (imm_rates_adj[0], np.zeros(E + S - 1))
        )
        OMEGA2[1:, :-1] += np.diag(1 - mort_rates[:-1])
        OMEGA2[1:, 1:] += np.diag(imm_rates_adj[1:])
        eigvalues2, eigvectors2 = np.linalg.eig(OMEGA2)
        g_n_SS_adj = (eigvalues[np.isreal(eigvalues2)].real).max() - 1
        if np.max(np.absolute(g_n_SS_adj - g_n_SS)) > 10 ** (-8):
            print(
                "FAILURE: The steady-state population growth rate"
                + " from adjusted OMEGA is different (diff is "
                + str(g_n_SS_adj - g_n_SS)
                + ") than the steady-"
                + "state population growth rate from the original"
                + " OMEGA."
            )
        elif np.max(np.absolute(g_n_SS_adj - g_n_SS)) <= 10 ** (-8):
            print(
                "SUCCESS: The steady-state population growth rate"
                + " from adjusted OMEGA is close to (diff is "
                + str(g_n_SS_adj - g_n_SS)
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
            pop_2011_pct,
            omega_path_lev,
            omega_SSfx,
            curr_year,
            E,
            S,
            output_dir=OUTPUT_DIR,
        )

    # return omega_path_S, g_n_SS, omega_SSfx, survival rates,
    # mort_rates_S, and g_n_path
    pop_dict = {
        "omega": omega_path_S.T,
        "g_n_SS": g_n_SS,
        "omega_SS": omega_SSfx[-S:] / omega_SSfx[-S:].sum(),
        "surv_rate": 1 - mort_rates_S,
        "rho": mort_rates_S,
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
