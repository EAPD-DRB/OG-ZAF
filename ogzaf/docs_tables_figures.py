"""
This script creates tables and figures from the OG-ZAF documentation.
"""

# import
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import ogzaf
from ogcore.parameters import Specifications
from ogcore import parameter_plots as pp
from ogcore import demographics

CUR_DIR = os.path.dirname(os.path.realpath(__file__))
UN_COUNTRY_CODE = "710"
plot_path = os.path.join(CUR_DIR, "..", "docs", "book", "content", "calibration", "images")
YEAR_TO_PLOT = 2025

# Use a custom matplotlib style file for plots
style_file_url = (
    "https://raw.githubusercontent.com/PSLmodels/OG-Core/"
    + "master/ogcore/OGcorePlots.mplstyle"
)
plt.style.use(style_file_url)

"""
Load specifications object with default parameters
"""
p = Specifications()
p.update_specifications(
        json.load(
            open(
                os.path.join(
                    CUR_DIR, "..", "ogzaf", "ogzaf_default_parameters.json"  # TODO: change to URL
                )
            )
        )
    )

# also load parameters from OG-USA for comparison
p2 = Specifications()
p2.update_specifications(
        json.load(
            open(
                os.path.join(
                    CUR_DIR, "..", "..", "OG-USA", "ogusa", "ogusa_default_parameters.json"  # TODO: change to URL
                )
            )
        )
    )

"""
Demographics chapter
"""
# Fertility rates
# pp.plot_fert_rates(
#         [fert_rates_2D],
#         start_year=start_year,
#         years_to_plot=[start_year, end_year],
#         path=plot_path,
#     )
# or
demographics.get_fert(
    totpers=100,
    min_age=0,
    max_age=99,
    country_id=UN_COUNTRY_CODE,
    start_year=YEAR_TO_PLOT,
    end_year=YEAR_TO_PLOT,
    graph=True,
    plot_path=plot_path,
    download_path=None,
)
# Mortality rates
# pp.plot_mort_rates_data(
#     mort_rates_2D,
#     start_year,
#     [start_year, end_year],
#     path=plot_path,
# )
demographics.get_mort(
    totpers=100,
    min_age=0,
    max_age=99,
    country_id=UN_COUNTRY_CODE,
    start_year=YEAR_TO_PLOT,
    end_year=YEAR_TO_PLOT,
    graph=True,
    plot_path=plot_path,
    download_path=None,
)
# Immigration rates
# pp.plot_imm_rates(
#     imm_rates_2D,
#     start_year,
#     [start_year, end_year],
#     path=plot_path,
# )
demographics.get_imm_rates(
    totpers=100,
    min_age=0,
    max_age=99,
    fert_rates=None,
    mort_rates=None,
    infmort_rates=None,
    pop_dist=None,
    country_id=UN_COUNTRY_CODE,
    start_year=YEAR_TO_PLOT,
    end_year=YEAR_TO_PLOT,
    graph=True,
    plot_path=plot_path,
    download_path=None,
)
# Fixed versus original population distribution
# pp.plot_omega_fixed(
#     age_per_EpS, omega_SS_orig, omega_SSfx, E, S, path=OUTPUT_DIR
# )
# Fixed versus adjusted immigration rates
# pp.plot_imm_fixed(
#     age_per_EpS,
#     imm_rates_orig[fixper - 1, :],
#     imm_rates_adj,
#     E,
#     S,
#     path=OUTPUT_DIR,
# )
demographics.get_pop_objs(
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
    pre_pop_dist=None,
    country_id=UN_COUNTRY_CODE,
    initial_data_year=YEAR_TO_PLOT - 1,
    final_data_year=YEAR_TO_PLOT + 2,
    GraphDiag=True,
    download_path=None,
)
# Population distribution at different points in time
# pp.plot_population_path(
#     age_per_EpS,
#     omega_path_lev,
#     omega_SSfx,
#     initial_data_year,
#     initial_data_year,
#     initial_data_year,
#     S,
#     path=OUTPUT_DIR,
# )
pp.plot_population(p, years_to_plot=[YEAR_TO_PLOT, YEAR_TO_PLOT + 25, YEAR_TO_PLOT + 50, YEAR_TO_PLOT + 100], include_title=False, path=plot_path)
# Population growth  # want average growth rate with this
pp.plot_pop_growth(
    p,
    start_year=YEAR_TO_PLOT,
    num_years_to_plot=150,
    include_title=False,
    path=plot_path,
)
"""
Income chapter
"""
pp.plot_ability_profiles(
    p, p2=None, t=None, log_scale=True, include_title=False, path=plot_path
)
# Plotting with USA also is too busy


#		* Fert rates
		# * Mort rates -- should we use survival function?
		# * Imm rates
		# * SS pop vs fixed  ** do we do this? **
		# * Adj vs not imm rates ** do we do this? **
		# * Pop dist at points in time
		# * Pop growth rate over time with avg rate
		# * Life time earning profiles
		# 	* Prob also want US for comparison
		# 	* Note gini in text
		# * Drop tables from income chapter