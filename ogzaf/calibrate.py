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
