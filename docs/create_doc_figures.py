"""
This script creates tables and figures from the OG-ZAF documentation.
"""

# import
import os
import numpy as np
import matplotlib.pyplot as plt
import ogcore
from ogcore.parameters import Specifications
from ogcore import parameter_plots as pp
from ogcore import demographics as demog


CUR_DIR = os.path.dirname(os.path.realpath(__file__))
UN_COUNTRY_CODE = "710"
plot_path = os.path.join(CUR_DIR, "book", "content", "calibration", "images")
YEAR_TO_PLOT = 2023
# update path for demographics graphdiag plots
demog.OUTPUT_DIR = plot_path

# Use a custom matplotlib style file for plots
plt.style.use("ogcore.OGcorePlots")

"""
Load specifications object with default parameters
"""
p = Specifications()
p.update_specifications(
    "github://EAPD-DRB:OG-ZAF@main/ogzaf/ogzaf_default_parameters.json"
)
p.start_year = YEAR_TO_PLOT

# also load parameters from OG-USA for comparison
p2 = Specifications()
p2.update_specifications(
    "github://PSLmodels:OG-USA@master/ogusa/ogusa_default_parameters.json"
)

"""
Demographics chapter
"""
# Fertility rates
fert_rates, fig = demog.get_fert(
    totpers=100,
    min_age=0,
    max_age=99,
    country_id="710",
    start_year=YEAR_TO_PLOT,
    end_year=YEAR_TO_PLOT,
    graph=True,
    plot_path=None,
    download_path=None,
)
plt.savefig(os.path.join(plot_path, "fert_rates.png"), dpi=300)
# Mortality rates
mort_rates, _, fig = demog.get_mort(
    totpers=100,
    min_age=0,
    max_age=99,
    country_id="710",
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
imm_rates, fig = demog.get_imm_rates(
    totpers=100,
    min_age=0,
    max_age=99,
    fert_rates=None,
    mort_rates=None,
    infmort_rates=None,
    pop_dist=None,
    country_id="710",
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
    pre_pop_dist=None,
    country_id=UN_COUNTRY_CODE,
    initial_data_year=YEAR_TO_PLOT - 1,
    final_data_year=YEAR_TO_PLOT + 2,
    GraphDiag=True,
    download_path=None,
)
# Population growth
fig = pp.plot_pop_growth(
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
plt.savefig(os.path.join(plot_path, "population_growth_rates.png"), dpi=300)

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
# ZAF profiles
pp.plot_ability_profiles(
    p, p2=None, t=None, log_scale=True, include_title=False, path=plot_path
)
# Plotting with USA also is too busy, so do separately
pp.plot_ability_profiles(
    p2,
    p2=None,
    t=None,
    log_scale=True,
    include_title=False,
    path=os.path.join(plot_path, "USA_plots"),
)
