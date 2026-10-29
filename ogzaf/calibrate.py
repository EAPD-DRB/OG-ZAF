from ogzaf import macro_params, income
from ogzaf import input_output as io
import os
import warnings
import numpy as np
import datetime

# UN M49 code for South Africa, used for every demographic download.
UN_COUNTRY_CODE = "710"
# Last year of UN data read by the demographics. ogcore freezes fertility,
# mortality and immigration at this year forever, so it sets the long-run
# population growth rate: a two-year window froze 2026 rates (g_n_ss about
# +0.42%/yr), while the UN WPP 2024 projection levels South Africa's population
# off this century (g_n_ss about -0.03%/yr with the full window). 2098 is one
# year before the last year in the UN data (ogcore reads final_data_year + 1)
# and within ogcore's start_year + 74 limit.
DEMOG_FINAL_DATA_YEAR = 2098

# Income gradients in demographic rates, from EAPD-DRB/Demographic-Gradients
# (docs/countries/south-africa.md). Each tilt is the slope of the log rate on
# wealth rank (0 = poorest, 1 = richest); ogcore takes the slope per centered
# percentile point, so every tilt is divided by 100. Negative = poorer groups
# have higher rates. ogcore keeps the national rate at each age and only
# tilts it across lifetime-income groups.
# Fertility: total fertility rate by wealth quintile, DHS 2016.
FERT_TILT = -0.5054
# Infant mortality: DHS 2016 (IMR, not U5MR).
INFMORT_TILT = -0.5443
# Adult mortality by age band: mean of the 2001 and 2007 census household-
# deaths tilts (asset-index ranking, all deaths linked; IPUMS International /
# Statistics South Africa). The 2011 census agrees on the 15-59 summary
# (45q15 tilt -0.80 in all three censuses) but links only 9% of deaths and
# has no band-level rows outside 30-44.
ADULT_MORT_TILTS = {
    (15, 29): (-0.9155 - 1.2085) / 2,
    (30, 44): (-1.1703 - 1.2545) / 2,
    (45, 59): (-0.8629 - 1.2122) / 2,
    (60, 74): (-0.3201 - 0.5650) / 2,
}
# Ages 1-14 have no adult-style measure; the DHS 2016 under-five tilt is the
# closest measured child gradient.
CHILD_MORT_TILT = -0.7036


def mortality_gradient(max_age=99):
    """
    Age profile of the mortality tilt for ages 0 to ``max_age``, in ogcore
    units (per centered percentile point).

    Age 0 takes the infant tilt, ages 1-14 the under-five tilt, ages 15-74
    the census age-band tilts. Above 74 there is no measurement; the gradient
    fades with age in every census measured, so it declines linearly from
    the 60-74 value to zero at ``max_age``.
    """
    ages = np.arange(max_age + 1)
    tilt = np.zeros(max_age + 1)
    tilt[0] = INFMORT_TILT
    tilt[1:15] = CHILD_MORT_TILT
    for (lo, hi), value in ADULT_MORT_TILTS.items():
        tilt[lo : hi + 1] = value
    old = ages > 74
    tilt[old] = (
        ADULT_MORT_TILTS[(60, 74)] * (max_age - ages[old]) / (max_age - 74)
    )
    return tilt / 100


def get_demographics_and_e(p, demographic_data_path=None, output_path=None):
    """
    Download the demographic objects and build the earnings matrix from
    them, so the two always come from the same population weights.

    Args:
        p (OG-Core Specifications object): model parameters
        demographic_data_path (str): path to save demographic data
        output_path (str): path to save the earnings-profile plots to

    Returns:
        tuple: (dict of demographic parameters, ``e`` array)
    """
    from ogcore import demographics

    window = dict(
        country_id=UN_COUNTRY_CODE,
        initial_data_year=p.start_year - 1,
        final_data_year=DEMOG_FINAL_DATA_YEAR,
        income_percentiles=p.lambdas.flatten(),
        fert_gradient=FERT_TILT / 100,
        mort_gradient=mortality_gradient(),
        infmort_gradient=INFMORT_TILT / 100,
        GraphDiag=False,
    )
    demog = demographics.get_pop_objs(
        p.E,
        p.S,
        p.T,
        0,
        99,
        download_path=demographic_data_path,
        **window,
    )
    # demographics for 80 period lives (needed for getting e below)
    demog80 = demographics.get_pop_objs(20, 80, p.T, 0, 99, **window)
    e = income.get_e_interp(
        p.E,
        p.S,
        demog["omega_SS"],
        demog80["omega_SS"],
        p.lambdas,
        plot_path=output_path,
    )
    return demog, e


class Calibration:
    """OG-ZAF calibration class"""

    def __init__(
        self,
        p,
        macro_data_start_year=datetime.datetime(1947, 1, 1),
        macro_data_end_year=datetime.datetime(2024, 12, 31),
        demographic_data_path=None,
        output_path=None,
        # Set True to update from World Bank and UN APIs
        update_from_api=False,
    ):
        """
        Constructor for the Calibration class.

        Args:
            p (OG-Core Specifications object): model parameters
            demographic_data_path (str): path to save demographic data
            output_path (str): path to save output to
            update_from_api (bool): Set True if you want to pull
                updated macro data from World Bank and UN APIs

        Returns:
            None

        """
        # Create output_path if it doesn't exist
        if output_path is not None:
            if not os.path.exists(output_path):
                os.makedirs(output_path)

        # Initialize attributes — populated only when update succeeds
        self.macro_params = {}
        self.demographic_params = {}
        self.e = None
        self.alpha_c = np.array([1.0]) if p.I == 1 else None
        self.io_matrix = np.array([[1.0]]) if p.M == 1 else None

        if not update_from_api:
            return

        # --- Online path: try each source independently ---

        # Macro estimation
        try:
            self.macro_params = macro_params.get_macro_params(
                macro_data_start_year,
                macro_data_end_year,
                update_from_api=update_from_api,
            )
        except Exception as exc:
            warnings.warn(f"Macro params update failed: {exc}", stacklevel=2)

        # io matrix and alpha_c (multi-sector only)
        if p.I > 1:
            try:
                alpha_c_dict = io.get_alpha_c()
                assert p.I == len(list(alpha_c_dict.keys()))
                self.alpha_c = np.array(list(alpha_c_dict.values()))
            except Exception as exc:
                warnings.warn(f"alpha_c update failed: {exc}", stacklevel=2)
        if p.M > 1:
            try:
                io_df = io.get_io_matrix()
                assert p.M == len(list(io_df.keys()))
                self.io_matrix = io_df.values
            except Exception as exc:
                warnings.warn(f"io_matrix update failed: {exc}", stacklevel=2)

        # Demographics + income (atomic — e depends on demographic output)
        try:
            self.demographic_params, self.e = get_demographics_and_e(
                p, demographic_data_path, output_path
            )
        except Exception as exc:
            warnings.warn(
                f"Demographics/income update failed: {exc}", stacklevel=2
            )
            self.demographic_params = {}
            self.e = None

    # method to return all newly calibrated parameters in a dictionary
    def get_dict(self):
        d = {}
        d.update(self.macro_params)
        d.update(self.demographic_params)
        if self.e is not None:
            d["e"] = self.e
        if self.alpha_c is not None:
            d["alpha_c"] = self.alpha_c
        if self.io_matrix is not None:
            d["io_matrix"] = self.io_matrix
        return d
