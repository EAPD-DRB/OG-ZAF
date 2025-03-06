# Need to fix references to Calculator, reform json, and substitute new tax
# function call
import multiprocessing
from distributed import Client
import importlib.resources
import os
import json
import time
import copy
import numpy as np

# from taxcalc import Calculator
from ogzaf.calibrate import Calibration
from ogcore.parameters import Specifications
from ogcore import output_tables as ot
from ogcore import output_plots as op
from ogcore.execute import runner
from ogcore.utils import safe_read_pickle


def main():
    # Define parameters to use for multiprocessing
    client = Client()
    num_workers = min(multiprocessing.cpu_count(), 7)
    print("Number of workers = ", num_workers)

    # Directories to save data
    CUR_DIR = os.path.dirname(os.path.realpath(__file__))
    base_dir = os.path.join(
        CUR_DIR, "OG-ZAF-energy_tax_Example", "OUTPUT_BASELINE"
    )
    reform_dir = os.path.join(
        CUR_DIR, "OG-ZAF-energy_tax_Example", "OUTPUT_REFORM"
    )

    """
    ---------------------------------------------------------------------------
    Run baseline policy
    ---------------------------------------------------------------------------
    """
    # Set up baseline parameterization
    p = Specifications(
        baseline=True,
        num_workers=num_workers,
        baseline_dir=base_dir,
        output_base=base_dir,
    )
    # Update parameters for baseline from default json file
    with importlib.resources.open_text(
        "ogzaf", "ogzaf_default_parameters.json"
    ) as file:
        defaults = json.load(file)
    p.update_specifications(defaults)
    # Update parameters from calibrate.py Calibration class for multiple industries
    p.M = 4
    p.I = 5
    c = Calibration(p)
    updated_params = c.get_dict()
    p.update_specifications(updated_params)
    updated_params_tax = {
        "cit_rate": [[0.28, 0.28, 0.28, 0.28]],
        # order of industries is primary, energy, tertiary, secondary ex energy
        "Z": [[0.5, 0.4, 1.7, 1.0]],
        "epsilon": [1.0, 1.0, 1.0, 1.0],
        "gamma": [0.67, 0.50, 0.45, 0.53],
        "gamma_g": [0.0, 0.0, 0.0, 0.0],
        "alpha_c": c.alpha_c,
        "io_matrix": c.io_matrix,
        "tG1": 23,
        "debt_ratio_ss": 1.8,
    }
    p.update_specifications(updated_params_tax)

    # Run model
    start_time = time.time()
    runner(p, time_path=True, client=client)
    print("run time = ", time.time() - start_time)

    """
    ---------------------------------------------------------------------------
    Run reform policy
    ---------------------------------------------------------------------------
    """

    # create new Specifications object for reform simulation
    p2 = copy.deepcopy(p)
    p2.baseline = False
    p2.output_base = reform_dir

    # additional parameters to change
    updated_params_ref = {
        "tau_c": [
            [0.15, 0.15, 0.15, 0.15, 0.15],
            [0.15, 0.17, 0.15, 0.15, 0.15],
            [0.15, 0.20, 0.15, 0.15, 0.15],
        ],
        "baseline_spending": True,
    }
    p2.update_specifications(updated_params_ref)

    # Run model
    start_time = time.time()
    runner(p2, time_path=True, client=client)
    print("run time = ", time.time() - start_time)
    client.close()

    """
    ---------------------------------------------------------------------------
    Save some results of simulations
    ---------------------------------------------------------------------------
    """
    base_tpi = safe_read_pickle(os.path.join(base_dir, "TPI", "TPI_vars.pkl"))
    base_params = safe_read_pickle(os.path.join(base_dir, "model_params.pkl"))
    reform_tpi = safe_read_pickle(
        os.path.join(reform_dir, "TPI", "TPI_vars.pkl")
    )
    reform_params = safe_read_pickle(
        os.path.join(reform_dir, "model_params.pkl")
    )
    ans = ot.macro_table(
        base_tpi,
        base_params,
        reform_tpi=reform_tpi,
        reform_params=reform_params,
        var_list=["Y", "C", "K", "L", "r", "w"],
        output_type="pct_diff",
        num_years=10,
        start_year=base_params.start_year,
    )

    # create plots of output
    op.plot_all(
        base_dir,
        reform_dir,
        os.path.join(CUR_DIR, "OG-ZAF_energy_tax_multi_industry_plots"),
    )

    print("Percentage changes in aggregates:", ans)
    # save percentage change output to csv file
    ans.to_csv("ogzaf_energy_tax_multi_industry_output.csv")


if __name__ == "__main__":
    # execute only if run as a script
    main()
