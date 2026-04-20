"""
This script creates tables and figures from the OG-ZAF documentation.
"""

# import
import os
import numpy as np
import matplotlib.pyplot as plt
from importlib.resources import files
import json
from ogcore.parameters import Specifications
from ogcore import parameter_plots as pp
from ogcore import parameter_tables as pt
from ogcore import demographics as demog

CUR_DIR = os.path.dirname(os.path.realpath(__file__))
UN_COUNTRY_CODE = "710"
plot_path = os.path.join(CUR_DIR, "book", "content", "calibration", "images")


def main():
    # update path for demographics graphdiag plots
    demog.OUTPUT_DIR = plot_path

    # Use a custom matplotlib style file for plots
    plt.style.use("ogcore.OGcorePlots")

    """
    Load specifications object with default parameters
    """
    p = Specifications()
    # Update parameters for baseline from default json file
    content = (
        files("ogzaf")
        .joinpath("ogzaf_default_parameters.json")
        .read_text(encoding="utf-8")
    )
    defaults = json.loads(content)
    p.update_specifications(defaults)
    YEAR_TO_PLOT = int(p.start_year)
    """
    Demographics chapter
    """
    # Fertility rates
    _, _ = demog.get_fert(
        totpers=100,
        min_age=0,
        max_age=99,
        country_id=UN_COUNTRY_CODE,
        start_year=YEAR_TO_PLOT,
        end_year=YEAR_TO_PLOT,
        graph=True,
        plot_path=None,
        download_path=None,
    )
    plt.savefig(os.path.join(plot_path, "fert_rates.png"), dpi=300)
    # Mortality rates
    _, _, _ = demog.get_mort(
        totpers=100,
        min_age=0,
        max_age=99,
        country_id=UN_COUNTRY_CODE,
        start_year=YEAR_TO_PLOT,
        end_year=YEAR_TO_PLOT,
        graph=True,
        plot_path=None,
        download_path=None,
    )
    plt.xlabel(r"Age ($s$)")
    plt.ylabel(r"Mortality rate ($\rho_s$)")
    plt.savefig(os.path.join(plot_path, "mort_rates.png"), dpi=300)
    # Immigration rates
    _, _ = demog.get_imm_rates(
        totpers=100,
        min_age=0,
        max_age=99,
        fert_rates=None,
        mort_rates=None,
        infmort_rates=None,
        pop_dist=None,
        country_id=UN_COUNTRY_CODE,
        start_year=YEAR_TO_PLOT,
        end_year=YEAR_TO_PLOT + 50,
        graph=True,
        plot_path=None,
        download_path=None,
    )
    plt.xlabel(r"Age ($s$)")
    plt.ylabel(r"Immigration rate ($i_s$)")
    # give a little more before the plot source note

    plt.savefig(os.path.join(plot_path, "imm_rates.png"), dpi=300)
    # Fixed versus original population distribution
    demog.get_pop_objs(
        E=20,
        S=80,
        T=320,
        min_age=0,
        max_age=99,
        fert_rates=None,
        mort_rates=None,
        infmort_rates=None,
        imm_rates=None,
        infer_pop=False,
        pop_dist=None,
        country_id=UN_COUNTRY_CODE,
        initial_data_year=YEAR_TO_PLOT - 1,
        final_data_year=YEAR_TO_PLOT + 2,
        GraphDiag=True,
        download_path=None,
    )

    # Population growth
    pp.plot_pop_growth(
        p,
        start_year=YEAR_TO_PLOT,
        num_years_to_plot=150,
        include_title=False,
        path=None,
    )
    # Add average growth rate with this
    plt.plot(
        np.arange(YEAR_TO_PLOT, YEAR_TO_PLOT + 150),
        np.ones(150) * np.mean(p.g_n[:150]),
        linestyle="-",
        linewidth=1,
        color="red",
    )
    plt.xlabel(r"Model Period ($t$)")
    plt.ylabel(r"Population Growth Rate ($g_{n,t}$)")
    plt.savefig(
        os.path.join(plot_path, "population_growth_rates.png"), dpi=300
    )

    # Population distribution at different points in time
    pp.plot_population(
        p,
        years_to_plot=[
            YEAR_TO_PLOT,
            YEAR_TO_PLOT + 25,
            YEAR_TO_PLOT + 50,
            YEAR_TO_PLOT + 100,
        ],
        include_title=False,
        path=plot_path,
    )
    """
    Income chapter
    """
    # USA profiles
    pp.plot_ability_profiles(
        p, p2=None, t=None, log_scale=True, include_title=False, path=plot_path
    )

    """
    Create table for exogenous parameters
    """
    pt.param_table(
        p,
        table_format="md",
        path=os.path.join(plot_path, "exogenous_parameters_table.md"),
    )


if __name__ == "__main__":
    main()
