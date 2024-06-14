from ogzaf import macro_params, income
from ogzaf import input_output as io
import os
import numpy as np
from ogcore import txfunc, demographics
from ogcore.utils import safe_read_pickle, mkdirs
import pkg_resources


class Calibration:
    """OG-ZAF calibration class"""

    def __init__(
        self,
        p,
        estimate_tax_functions=False,
        estimate_chi_n=False,
        demographic_data_path=None,
        output_path=None,
    ):
        """
        Constructor for the Calibration class.

        Args:
            p (OG-Core Specifications object): model parameters
            estimate_tax_functions (bool): whether to estimate tax
                function parameters
            estimate_chi_n (bool): whether to estimate chi_n
            demographic_data_path (str): path to save demographic data
            output_path (str): path to save output to

        Returns:
            None

        """
        # Create output_path if it doesn't exist
        if output_path is not None:
            if not os.path.exists(output_path):
                os.makedirs(output_path)
        self.estimate_tax_functions = estimate_tax_functions
        self.estimate_chi_n = estimate_chi_n

        # Macro estimation
        self.macro_params = macro_params.get_macro_params()

        # io matrix and alpha_c
        if p.I > 1:  # no need if just one consumption good
            alpha_c_dict = io.get_alpha_c()
            # check that model dimensions are consistent with alpha_c
            assert p.I == len(list(alpha_c_dict.keys()))
            self.alpha_c = np.array(list(alpha_c_dict.values()))
        else:
            self.alpha_c = np.array([1.0])
        if p.M > 1:  # no need if just one production good
            io_df = io.get_io_matrix()
            # check that model dimensions are consistent with io_matrix
            assert p.M == len(list(io_df.keys()))
            self.io_matrix = io_df.values
        else:
            self.io_matrix = np.array([[1.0]])

        # demographics
        self.demographic_params = demographics.get_pop_objs(
            p.E,
            p.S,
            p.T,
            0,
            99,
            initial_data_year=p.start_year - 1,
            final_data_year=p.start_year + 1,
            GraphDiag=False,
            download_path=demographic_data_path,
        )

        # demographics for 80 period lives (needed for getting e below)
        demog80 = demographics.get_pop_objs(
            20,
            80,
            p.T,
            0,
            99,
            initial_data_year=p.start_year - 1,
            final_data_year=p.start_year + 1,
            GraphDiag=False,
        )

        # earnings profiles
        self.e = income.get_e_interp(
            p.S,
            self.demographic_params["omega_SS"],
            demog80["omega_SS"],
            p.lambdas,
            plot_path=output_path,
        )

    # method to return all newly calibrated parameters in a dictionary
    def get_dict(self):
        dict = {}
        # if self.estimate_chi_n:
        #     dict["chi_n"] = self.chi_n
        dict.update(self.macro_params)
        dict["e"] = self.e
        dict["alpha_c"] = self.alpha_c
        dict["io_matrix"] = self.io_matrix
        dict.update(self.demographic_params)

        return dict
