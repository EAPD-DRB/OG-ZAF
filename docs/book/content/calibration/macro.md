(Chap_MacroCalib)=
# Calibration of Macroeconomic Parameters

## Economic Assumptions

As the default rate of labor augmenting technological change, $g_y$, we use a value of 0%.  The average annual growth rate in GDP per capita in South Africa between 2007 and 2023 is 0.0% per year.

## Open Economy Parameters

### Foreign holding of government debt in the initial period

The path of foreign holding of domestic debt is endogenous, but the initial period stock of debt held by foreign investors is exogenous.  We set this parameter, `initial_foreign_debt_ratio` to 0.26, consistent with data from the World Bank WDI.

### Foreign purchases of newly issued debt

We set $\zeta_D = 0.23$.  This is the average share of foreign purchases of newly issued government debt found from the World Bank WDI.

### Foreign holdings of excess capital

We set $\zeta_K = 0.9$. Note, this parameter is harder to pin down from the data as foreign purchases on "excess" capital demand is not typically directly measured or reported.  A value of 0.9 implies a high degree of openness to international capital flows.

## Government Debt, Spending and Transfers

### Government Debt

The path of government debt is endogenous.  But the initial value is exogenous.  To avoid converting between model units and dollars, we calibrate the initial debt to GDP ratio, rather than the dollar value of the debt.  This is the model parameter $\alpha_D$.  We compute this from the ratio of publicly held debt outstanding to GDP.  Based on 2023 values, this gives us a ratio of 0.59.


#### Interest rates on government debt

We assume a wedge between the real rate of return on private capital and the real interest rate on government debt, modeled as a scale and level shift. The real interest rate on government debt, $r_{gov,t}$, relates to the real rate of return on private capital, $r_t$, by

```{math}
:label: eqn:r_gov
    r_{gov,t} = (1-\tau_{d,t}) r_t + \mu_d
```

where $(1-\tau_d)$ is the pass-through coefficient and $\mu_d$ is the level shift. For South Africa we use $1-\tau_d = 0.24485$ (so $\tau_d = 0.75515$) and $\mu_d = 0.03377$.

These values come from {cite}`LMW2023`, who estimate the long-run pass-through of sovereign yields to corporate yields across 46 emerging economies using corporate yields from IHS Markit and sovereign yields from Bloomberg, predominantly U.S.-dollar secondary-market yields. They are therefore a cross-country emerging-market relationship rather than South Africa-specific bond data. Their preferred specification (Table 8, Column 2) fits a quadratic of the corporate yield on the sovereign yield of the same country:

```{math}
:label: eqn:lmw_quadratic
    y_{corp} = 8.199 - 2.975\, y_{sov} + 0.478\, y_{sov}^2
```

with both yields in percentage points. The quadratic captures the empirical fact that pass-through rises with the level of sovereign risk, consistent with the credit-risk and liquidity-premium channels the paper identifies. The paper is available as an IMF Working Paper, [Li, Magud, Werner, and Witte (2021)](https://www.imf.org/en/Publications/WP/Issues/2021/06/04/The-Long-Run-Impact-of-Sovereign-Yields-on-Corporate-Yields-in-Emerging-Markets-50224), and was later published in the *Journal of International Money and Finance* {cite}`LMW2023`.

OG-Core models the wedge in the opposite direction: it takes $r_t$ as an input and produces $r_{gov,t}$. We therefore invert the LMW relationship. We evaluate their quadratic on a grid of sovereign yields from 2% to 12%, compute the implied corporate yields, and then regress sovereign yields linearly on those corporate yields. Calling the resulting slope $b$ and intercept $a$ (both in percentage points), we identify $1-\tau_d = b$ and $\mu_d = a/100$.

OG-Core's operational formula is $r_{gov,t} = \max\!\big(\texttt{r\_gov\_scale}\cdot r_t - \texttt{r\_gov\_shift},\; 0\big)$, so the JSON stores `r_gov_scale = 1-\tau_d = 0.24485` and `r_gov_shift = -\mu_d = -0.03377`. The negative sign on `r_gov_shift` reflects the subtraction in the OG-Core rule, not a negative level shift in the theoretical equation.

Because the inputs to this inversion are deterministic and contain no South Africa-specific bond data, the resulting values do not change across calibration runs. The packaged values in `ogzaf/ogzaf_default_parameters.json` and `ogzaf/ogzaf_default_parameters_multisector.json` are the authoritative source. The snippet below reproduces them for transparency:

```python
import numpy as np
import statsmodels.api as sm

# LMW (2023) Table 8, Column 2: corp = 8.199 - 2.975 sov + 0.478 sov^2  (pct pts)
sov_y = np.arange(20, 120) / 10
corp_yhat = 8.199 - (2.975 * sov_y) + (0.478 * sov_y**2)

# Invert: regress sov on corp -> linear pass-through
X = sm.add_constant(corp_yhat)
res = sm.OLS(sov_y, X).fit()

r_gov_shift = -res.params[0] / 100  # -0.03377  (= -\mu_d in the theoretical equation)
r_gov_scale = res.params[1]         #  0.24485  (= 1-\tau_d in the theoretical equation)
```

If the LMW estimates are superseded, re-run the inversion above with the new coefficients and update the JSON values.

### Aggregate transfers

Aggregate (non-Social Security) transfers to households are set as a share of GDP with the parameter $\alpha_T$. We exclude Social Security from transfers since it is modeled specifically. With this definition, the share of transfers to GDP in 2015 is 0.04 according to [IMF data](https://data.imf.org/?sk=b052f0f0-c166-43b6-84fa-47cccae3e219&hide_uv=1).

### Government expenditures

Government spending on goods and services are also set as a share of GDP with the parameter $\alpha_G$. We define government spending as:
    <center>Government Spending = Total Outlays - Transfers - Net Interest on Debt - Social Security</center>
With this definition, the share of government expenditure to GDP is 0.267 based on [data from the IMF](https://data.imf.org/?sk=b052f0f0-c166-43b6-84fa-47cccae3e219&hide_uv=1).
