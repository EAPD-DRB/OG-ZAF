# Need to fix references to Calculator, reform json, and substitute new tax
# function call
import multiprocessing
from distributed import Client
import os
import json
import time
import copy
import numpy as np
from importlib.resources import files
import matplotlib.pyplot as plt
from ogzaf.calibrate import Calibration
from ogcore.parameters import Specifications
from ogcore import output_tables as ot
from ogcore import output_plots as op
import ogcore
from ogcore.execute import runner
from ogcore.utils import safe_read_pickle


# Use a custom matplotlib style file for plots
plt.style.use("ogcore.OGcorePlots")


def main():
    # Define parameters to use for multiprocessing
    num_workers = min(multiprocessing.cpu_count(), 7)
    client = Client(n_workers=num_workers, threads_per_worker=1)
    print("Number of workers = ", num_workers)

    # Directories to save data
    CUR_DIR = os.path.dirname(os.path.realpath(__file__))
    save_dir = os.path.join(CUR_DIR, "OG-ZAF-Education")
    base_dir = os.path.join(save_dir, "OUTPUT_BASELINE")
    reform_dir = os.path.join(save_dir, "OUTPUT_EDUC")
    reform_dir2 = os.path.join(save_dir, "OUTPUT_EDUC_SPEND")

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
    with (
        files("ogzaf")
        .joinpath("ogzaf_default_parameters.json")
        .open("r") as file
    ):
        defaults = json.load(file)
    p.update_specifications(defaults)
    # move closure rule out to 50 years since educaation phases in over 20 years
    p.tG1 = 50

    # Run model
    start_time = time.time()
    runner(p, time_path=True, client=client)
    print("run time = ", time.time() - start_time)

    """
    ---------------------------------------------------------------------------
    Run counterfactual simulation
    ---------------------------------------------------------------------------
    """

    # create new Specifications object for reform simulation
    p2 = copy.deepcopy(p)
    p2.baseline = False
    p2.output_base = reform_dir

    # adjust labor productivity to account for education investment
    # Public spending may be able to obtain universal enrollment in Africa: https://www.inderscienceonline.com/doi/abs/10.1504/IJEED.2015.075794
    # no time for paper, but let's assume this increases productivity of those bottom 70% by 20% for all ages 20+
    # Let's assume this phases in linearly over 20 years
    num_years = 20  # 20 years to phase in
    total_benefit = 0.2  # total effect on productivity when fully phased in
    benefits = np.linspace(0, total_benefit, num_years)
    for t, benefit in enumerate(benefits):
        p2.e[t, :, :3] = p.e[t, :, :3] * (
            1 + benefit
        )  # just apply to bottom 70%
    p2.e[num_years:, :, :3] = p.e[num_years:, :, :3] * (1 + total_benefit)

    # Run sim with just the benefits of education
    start_time = time.time()
    runner(p2, time_path=True, client=client)
    print("run time = ", time.time() - start_time)

    # Now increase spending and solve model again
    p3 = copy.deepcopy(p2)
    p3.baseline = False
    p3.output_base = reform_dir2
    # increase gov't spending to account for costs of education
    # Spending currently about 6.6% of GDP (source: https://data.worldbank.org/indicator/SE.XPD.TOTL.GD.ZS?locations=ZA)
    # Let's assume this increases to 10% of GDP
    p3.alpha_G = (
        p3.alpha_G + 0.034
    )  # counterfactual 10% of GDP - current 6.6% of GDP
    start_time = time.time()
    runner(p3, time_path=True, client=client)
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
        base_dir, reform_dir, os.path.join(save_dir, "OG-ZAF_example_plots")
    )

    print("Percentage changes in aggregates:", ans)
    # save percentage change output to csv file
    ans.to_csv(os.path.join(save_dir, "ogzaf_example_output.csv"))


if __name__ == "__main__":
    # execute only if run as a script
    main()
