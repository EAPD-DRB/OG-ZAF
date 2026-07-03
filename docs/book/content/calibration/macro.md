(Chap_MacroCalib)=
# Calibration of Macroeconomic Parameters

## Economic Assumptions

As the default rate of labor augmenting technological change, $g_y$, we use the average annual growth rate of real GDP per capita in South Africa (World Bank, constant 2015 US$), about 0.6% per year. This is the one macro parameter `ogzaf.macro_params.get_macro_params` still refreshes from a live API (`ogzaf.update_baseline` re-pulls it), because the World Bank series is its documented source.

## Open Economy Parameters

### World interest rate

We set the annual world interest rate, `world_int_rate_annual`, to 6.3%: a 4% global risk-free rate plus a South African sovereign risk premium of roughly 2.3 percentage points. The premium is anchored by the National Treasury's own estimate — "the sovereign risk premium – the extra yield investors demand to hold government debt over 'risk-free' alternatives – has narrowed from 2.93 percentage points a year ago to 2.26 percentage points" ([2026 Budget Review, Chapter 7](https://www.treasury.gov.za/documents/National%20Budget/2026/review/Chapter%207.pdf)). The chapter does not define the benchmark instrument, so we corroborate the level with hard-currency evidence, which lands in the same band: South Africa's US-dollar bond spreads traded around 2.2–2.6 percentage points over Treasuries in 2025/26 (multi-year lows, after the December 2025 US$3.5 billion oversubscribed Eurobond), and the Damodaran country-default-spread table puts a Ba2/BB sovereign at roughly 2.7–3.2 percentage points. This is the rate at which foreign capital is supplied to the domestic economy; the previous value of 4% omitted the country premium entirely. South Africa remains sub-investment-grade (BB/Ba2 at all three agencies, upgraded during 2025–26 with positive outlooks), so its premium is materially larger than that of investment-grade emerging markets.

### Foreign holding of government debt in the initial period

The path of foreign holding of domestic debt is endogenous, but the initial period stock of debt held by foreign investors is exogenous. We set this parameter, `initial_foreign_debt_ratio`, to 0.25: foreign investors held 25.0% of South Africa's domestic government bonds in 2025, up from 24.6% in 2024 ([2026 Budget Review, Chapter 7](https://www.treasury.gov.za/documents/National%20Budget/2026/review/Chapter%207.pdf), Figure 7.4). Note this is the domestic-market headline measure; a residency-comprehensive measure that also counts the (roughly 10% of gross debt) foreign-currency loans held abroad would be closer to 0.33.

### Foreign purchases of newly issued debt

We set $\zeta_D = 0.25$, matching the foreign share of the domestic government bond market above, on the assumption that foreign investors absorb new issuance in proportion to their current holdings.

### Foreign holdings of excess capital

We set $\zeta_K = 0.16$, the normalized Chinn-Ito capital-account openness index (`ka_open`) for South Africa, which is 0.1626 in the [2022 update](https://web.pdx.edu/~ito/Chinn-Ito_website.htm) (unchanged since 2008). This parameter is hard to pin down directly since foreign purchases of "excess" capital demand are not measured; the Chinn-Ito index anchors it to South Africa's de jure openness, which is low because exchange controls still apply to residents. De facto participation by non-residents in South African capital markets is considerably higher, so 0.16 is a conservative anchor. (The previous value of 0.9 implied near-total foreign absorption of excess capital demand, which has no empirical basis for South Africa.)

## Government Debt, Spending and Transfers

### Government Debt

The path of government debt is endogenous. But the initial value is exogenous. To avoid converting between model units and rand, we calibrate the initial debt-to-GDP ratio rather than the rand value of the debt. This is the model parameter $\alpha_D$ (`initial_debt_ratio`), which we set to 0.789: national government gross loan debt is 78.9% of GDP in 2025/26 ([2026 Budget Review, Chapter 7](https://www.treasury.gov.za/documents/National%20Budget/2026/review/Chapter%207.pdf), Table 7.7). We use the National Treasury's national-government gross loan debt rather than the World Bank QPSD general-government series, which measures a wider perimeter (for 2025 the QPSD figure is roughly 84% of GDP); the Treasury figure is the one the fiscal framework targets and reports against.

The long-run debt target `debt_ratio_ss` is set to 0.765. The 2026 Budget Review projects gross loan debt to stabilize at 78.9% of GDP in 2025/26 — the first stabilization in 17 years — and to decline to 76.5% by 2028/29; we take the end of the medium-term expenditure framework as the steady-state anchor.


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

OG-Core's operational formula is $r_{gov,t} = \max\!\big(\texttt{r\_gov\_scale}\cdot r_t - \texttt{r\_gov\_shift} + \texttt{r\_gov\_DY}\cdot\tfrac{D_t}{Y_t} + \texttt{r\_gov\_DY2}\cdot\big(\tfrac{D_t}{Y_t}\big)^2,\; 0\big)$, so the JSON stores `r_gov_scale` $= 1-\tau_d = 0.24485$. The LMW level shift is $\mu_d = 0.03377$; the stored `r_gov_shift` additionally absorbs the constant term of the centered debt-elastic premium described below, so `r_gov_shift = -0.05718` rather than $-\mu_d$ alone. The negative sign on `r_gov_shift` reflects the subtraction in the OG-Core rule, not a negative level shift in the theoretical equation.

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

If the LMW estimates are superseded, re-run the inversion above with the new coefficients and update the JSON values. Note that `ogzaf.macro_params.get_macro_params` no longer recomputes these values on a live update: the inversion is deterministic, and the packaged `r_gov_shift` includes the premium recentering below, which a recompute would silently undo.

#### Debt-elastic sovereign premium

We enable a debt-elastic sovereign premium — the crowding-out-via-risk channel that OG-Core's defaults leave off (`r_gov_DY = r_gov_DY2 = 0`). It takes the *centered* convex form used in OG-PHL:

```{math}
:label: eqn:centered_premium
    \text{premium}_t = r_{gov,DY2}\,\Big(\frac{D_t}{Y_t} - \bar{d}\Big)^2
```

with $\bar{d} = 0.765$ (the `debt_ratio_ss` target) and $r_{gov,DY2} = 0.04$. Expanding the square maps this into OG-Core's linear-plus-quadratic terms:

- `r_gov_DY2` $= 0.04$
- `r_gov_DY` $= -2 \times 0.04 \times 0.765 = -0.0612$
- `r_gov_shift` $= -0.03377 - 0.04 \times 0.765^2 = -0.05718$ (the recentering constant $r_{gov,DY2}\,\bar{d}^2$ is absorbed into the shift)

The premium is exactly zero at the debt target, so the steady state is unchanged; along the transition path, debt above (or below) the target raises the sovereign rate convexly — about 7 basis points at $D/Y = 0.9$ and 22 basis points at $D/Y = 1.0$. This is the [Schmitt-Grohé and Uribe (2003)](https://www.nber.org/system/files/working_papers/w9270/w9270.pdf) debt-elastic premium in convex form, following the fiscal-limits literature ([Bi 2012](https://www.sciencedirect.com/science/article/abs/pii/S0014292111001085); [Ghosh et al. 2013](https://www.nber.org/system/files/working_papers/w16782/w16782.pdf)) and emerging-market spread-to-debt empirics of roughly 4–8 basis points per percentage point of debt ([Jaramillo and Weber 2012](https://www.imf.org/external/pubs/ft/wp/2012/wp12198.pdf)). It also matches South Africa's own experience: spreads were broadly stable while gross debt climbed through the 60s percent-of-GDP range, then blew out in 2020 when debt jumped toward 70% amid the downgrade to sub-investment grade — and compressed again (2.93 to 2.26 percentage points during 2025) as consolidation took hold. The value of $r_{gov,DY2} = 0.04$ matches OG-PHL's calibration of the same channel.

### Aggregate transfers

Aggregate (non-Social Security) transfers to households are set as a share of GDP with the parameter $\alpha_T$. We exclude Social Security from transfers since it is modeled specifically. With this definition, the share of transfers to GDP is 0.037, computed from the IMF Government Finance Statistics (non-interest transfers, latest available year) by `ogzaf.macro_params.get_macro_params`, which refreshes this parameter from the [IMF GFS API](https://data.imf.org/) on a live update.

### Government expenditures

Government spending on goods and services is also set as a share of GDP with the parameter $\alpha_G$. We define government spending as:
    <center>Government Spending = Total Outlays - Transfers - Net Interest on Debt - Social Security</center>
With this definition, the share of government expenditure to GDP is 0.233, computed from the IMF Government Finance Statistics (latest available year) by `ogzaf.macro_params.get_macro_params`, which refreshes this parameter from the [IMF GFS API](https://data.imf.org/) on a live update.
