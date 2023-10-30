---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: '0.8'
    jupytext_version: '1.4.1'
kernelspec:
  display_name: Python 3
  language: python
  name: ogzaf-dev
---

(Chap_LfEarn)=
# Lifetime Earnings Profiles

Among households in `OG-ZAF`, we model both age heterogeneity and within-age ability heterogeneity. We use this ability or productivity heterogeneity to generate the income heterogeneity that we see in the data.

Differences among workers' productivity in terms of ability is one of the key dimensions of heterogeneity to model in a micro-founded macroeconomy. In this chapter, we characterize this heterogeneity as deterministic lifetime productivity paths to which new cohorts of agents in the model are randomly assigned. In `OG-ZAF`, households' labor income comes from the equilibrium wage and the agent's endogenous quantity of labor supply. In this section, we augment the labor income expression with an individual productivity $e_{j,s}$, where $j$ is the index of the ability type or path of the individual and $s$ is the age of the individual with that ability path.

```{math}
:label: EqLaborIncome
  \text{labor income:}\quad x_{j,s,t}\equiv w_t e_{j,s}n_{j,s,t} \quad\forall j,t \quad\text{and}\quad E+1\leq s\leq E+S
```

In this specification, $w_t$ is an equilibrium wage representing a portion of labor income that is common to all workers. Individual quantity of labor supply is $n_{j,s,t}$, and $e_{j,s}$ represents a labor productivity factor that augments or diminishes the productivity of a worker's labor supply relative to average productivity.

We calibrate deterministic ability paths such that each lifetime income group has a different life-cycle profile of earnings. The distribution of income and wealth are often focal components of macroeconomic models. These calibrations require the use of microeconomic data on household incomes, but this level of data is not readily available for South Africa from public sources or surveys. To overcome this, we start with the proposition that estimated productivity curves calibrated for the [OG-USA](https://pslmodels.github.io/OG-USA/content/calibration/earnings.html) model, generated from micro-level earnings data, represent a generalized relationship between age and lifetime income {cite}`DeBackerEtAl:2017`. As such, our objective is to generate the curves for the U.S. and then adjust their generalized shapes to produce those for South Africa. In other words, our strategic approach is to apply South Africa's national distribution of income by age to the estimated U.S. data on income by age, and then use these re-distributed data to re-estimate the earning profile curves and use them for South Africa. This is done in two ways (in this order):
  1. Adjustment by income ($J$): adjust the gaps between the U.S. curves to match South Africa's distribution between the $J$-income groups, using data from the World Inequality Database (WID);
  2. Adjustment by age ($S$): adjust the shape of all the U.S. curves to match South Africa's national distribution of income per capita for each age year, using data from the National Transfer Accounts database (NTA).

The data for the U.S. come from the U.S. Internal Revenue Services's (IRS) Statistics of Income program (SOI) Continuous Work History Sample (CWHS). {cite}`DeBackerEtAl:2017` match the SOI data with Social Security Administration (SSA) data on age and Current Population Survey (CPS) data on hours in order to generate a non-top-coded measure of hourly wage. See {cite}`DeBackerRamnath:2017` for a detailed description of the methodology.

```{figure} ./images/ability_log_2D_ZAF.png
---
height: 350px
name: FigLogAbil
---
Exogenous life cycle income ability paths $\log(e_{j,s})$ with $S=80$ and $J=7$
```

<!-- +++
```{code-cell} ogzaf-dev
:tags: [hide-cell]
from myst_nb import glue
import ogzaf.parameter_plots as pp
from ogzaf import Specifications
p = Specifications()
fig = pp.plot_ability_profiles(p)
glue("earnings_profiles", fig, display=False)
```

```{glue:figure} earnings_profiles
:figwidth: 750px
:name: "FigLogAbil"

Exogenous life cycle income ability paths $\log(e_{j,s})$ with $S=80$ and $J=7$
```
-->


{numref}`Figure %s <FigLogAbil>` shows a calibration for $J=7$ deterministic lifetime ability paths $e_{j,s}$ corresponding to labor income percentiles $\boldsymbol{\lambda}=[0.25, 0.25, 0.20, 0.10, 0.10, 0.09, 0.01]$. Because there are few individuals above age 80 in the data, {cite}`DeBackerEtAl:2017` extrapolate these estimates for model ages 80-100 using an arctan function.

We calibrate the model such that each lifetime income group has a different life-cycle profile of earnings. Since the distribution on income and wealth are key aspects of our model, we calibrate these processes so that we can represent earners in the top 1 percent of the distribution of lifetime income.


(SecLFearnLifInc)=
## Lifetime Income

  In our model, labor supply and savings, and thus lifetime income, are endogenous. We therefore define lifetime income as the present value of lifetime labor endowments and not the value of lifetime labor earnings. Note that our data are at the tax filing unit.  We take this unit to be equivalent to a household.  Because of differences in household structure (i.e., singles versus couples), our definition of lifetime labor income will be in per-adult terms. In particular, for filing units with a primary and secondary filer, our imputed wage represents the average hourly earnings between the two. When calculating lifetime income, we assign single- and couple-households the same labor endowment.  This has the effect of making our lifetime income metric a per-adult metric; there is therefore not an over-representation of couple-households in the higher lifetime income groups simply because their time endowment is higher than for singles. We use the following approach to measure the lifetime income.

  First, since our panel data do not allow us to observe the complete life cycle of earnings for each household (because of sample attrition, death or the finite sample period of the data), we use an imputation to estimate wages in the years of the household's economic life for which they do not appear in the CWHS. To do this, we estimate the following equation, separately by household type (where household types are single male, single female, couple with male head, or couple with female head):

  ```{math}
  :label: wage_step1
    ln(w_{i,t}) = \alpha_{i} + \beta_{1}age_{i,t} + \beta_{2}age_{i,t}^{2} + \beta_{3}*age_{i,t}^{3} + \varepsilon_{i,t}
  ```

  The parameter estimates, including the household fixed effects, from Equation {eq}`wage_step1` are shown in {numref}`TabWage_step1`. These estimates are then used to impute values for log wages in years of each households' economic life for which we do not have data.  This creates a balanced panel of log wages of households with heads aged 21 to 80. The actual and imputed wage values are then used to calculate the net present value of lifetime labor endowments per adult for each household. Specifically, we define lifetime income for household $i$ as:

  ```{math}
  :label: eqn:LI
    LI_{i} = \sum_{t=21}^{80}\left(\frac{1}{1+r}\right)^{t-21}(w_{i,t}*4000)
  ```


  ```{list-table} **Initial log wage regressions.** Source: CWHS data, 1991-2009. \*\* Significant at the 5-percent level ($p<0.05$). \*\*\* Significant at the 1-percent level ($p<0.01$).
  :header-rows: 1
  :name: TabWage_step1
  * - Dependent variable
    - Single male
    - Single female
    - Married male head
    - Married female head
  * - $Age$
    - 0.177\*\*\*
    - 0.143\*\*\*
    - 0.134\*\*\*
    - 0.065\*\*
  * -
    - (0.006)
    - (0.005)
    - (0.004)
    - (0.027)
  * - $Age^2$
    - -0.003\*\*\*
    - -0.002\*\*\*
    - -0.002\*\*\*
    - -0.000
  * -
    - (0.000)
    - (0.000)
    - (0.000)
    - (0.001)
  * - $Age^3$
    - 0.000\*\*\*
    - 0.000\*\*\*
    - 0.000\*\*\*
    - 0.000
  * -
    - (0.000)
    - (0.000)
    - (0.000)
    - (0.000)
  * - $Constant$
    - -0.839\*\*\*
    - -0.648\*\*\*
    - -0.042
    - 1.004\*\*\*
  * -
    - (0.072)
    - (0.070)
    - (0.058)
    - (0.376)
  * - Adjusted $R^2$
    - -0.007
    - 0.011
    - -0.032
    - -0.324
  * - Observations
    - 88,833
    - 96,670
    - 141,564
    - 6,314
  ```


  Note that households all have the same time endowment in each year (4000 hours). Thus, the amount of the time endowment scales the lifetime income up or down, but does not change the lifetime income of one household relative to another. This is not the case with the interest rate, $r$, which we fix at 4\%. Changes in the interest rate differentially impact the lifetime income calculation for different individuals because they may face different earnings profiles. For example, a higher interest rate would reduced the discounted present value of lifetime income for those individuals whose wage profiles peaked later in their economic life by a larger amount than it would reduce the discounted present value of lifetime income for individuals whose wage profiles peaked earlier.


## Profiles by Lifetime Income

  With observations of lifetime income for each household, we next sort households and find the percentile of the lifetime income distribution within which each household falls.  With these percentiles, we create our lifetime income groupings.
  ```{math}
  :label: EqLfEarnLambda_j
    \lambda_{j}=[0.25, 0.25, 0.2, 0.1, 0.1, 0.09, 0.01]
  ```

  That is, lifetime income group one includes those in below the 25th percentile, group two includes those from the 25th to the median, group three includes those from the median to the 70th percentile, group four includes those from the 70th to the 80th percentile, group 5 includes those from the 80th to 90th percentile, group 6 includes those from the 90th to 99th percentile, and group 7 consists of the top one percent in the lifetime income distribution.


  To get a life-cycle profile of effective labor units for each group, we estimate the wage profile for each lifetime income group. We do this by estimating the following regression model Equation {eq}`` separately for each lifetime income group using data on actual (not imputed) wages:

  ```{math}
  :label: EqWage_profile
    ln(w_{j,t}) = \alpha_{j} + \beta_{1}age_{j,t} + \beta_{2}age_{j,t}^{2} + \beta_{3}*age_{j,t}^{3} + \varepsilon_{j,t}
  ```

  Life-cycle earnings profiles are implied by the parameters estimated from equation {eq}`EqWage_profile`, and are used to plot {numref}`Figure %s <FigLogAbil>`. The arctan function used to extrapolate these estimates for model ages 80-100 takes the following form:
  ```{math}
  :label: EqLfEarnArctan
      y = \left(\frac{-a}{\pi}\right)*arctan(bx+c)+\frac{a}{2}
  ```
  where $x$ is age, and $a$, $b$, and $c$ are the parameters we search over for the best fit of the function to the following three criteria: 1) the value of the function should match the value of the data at age 80, 2) the slope of the arctan should match the slope of the data at age 80, and 3) the value of the function should match the value of the data at age 100 times a constant. This constant is 0.5 for all lifetime income groups, except for the second highest ability at 0.7 (otherwise, the second highest has a lower income than the third highest ability group in the last few years).


## Income at the very top

In addition to lifecycle profiles of the seven percentile groups above, the model provides calibrations of income at the very top. This includes breaking out percentiles as fine as the top 0.01% of earners. The two alternative $\lambda$ vectors are $\lambda_{j}=[0.25, 0.25, 0.2, 0.1, 0.1, 0.09, 0.005, 0.004, 0.001]$ and $\lambda_{j}=[0.25, 0.25, 0.2, 0.1, 0.1, 0.09, 0.005, 0.004, 0.0009, 0.0001]$.

Because we do not have U.S. panel data that allow us to observe such top percentile groups, we make the following assumptions in calibrating income at the very top. First, we assume the shape of the lifecycle profile of these top earners is the same as the top 1% overall. Second, we use 2018 estimates from the methodology of {cite}`PikettySaez:2003` to provide factors to scale the earnings process we estimate for groups inside the top 1%.[^PS_note]


```{list-table} **Incomes at the very top** Source: Piketty and Saez (2003) 2018 estimates.
:header-rows: 1
:name: tab:top_incomes

* - Income Percentile Range
  - Ratio to Top 1%
  - Mean Income
* - Top 1%
  - $1,143,687
  - 1.000
* - Top 1-0.5%
  - $524,677
  - 0.459
* - Top 0.5-0.1%
  - $968,991
  - 0.847
* - Top 0.1-0.01%
  - $3,103,621
  - 2.714
* - Top 0.1%
  - $4,937,516
  - 4.317
* - Top 0.01%
  - $21,442,570
  - 18.749
```


[^PS_note]: These data are available from the website of Emmanuel Saez: [https://eml.berkeley.edu/~saez/](https://eml.berkeley.edu/~saez/).  We use numbers from Table0, Panel B, "Income excluding realized capital gains."
