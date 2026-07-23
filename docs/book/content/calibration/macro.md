(Chap_MacroCalib)=
# Calibration of Macroeconomic Parameters

## Economic Assumptions

As the default rate of labour-augmenting technological change, $g_y$, we use 1.4% per year (`g_y_annual = 0.014`). On the model's balanced growth path aggregate GDP grows at approximately $g_y + g_n$; with the model's steady-state population growth $g_n \approx 0.42\%$ this reproduces the **~1.8% medium-term real GDP growth** that both the IMF (2025 Article IV, "reaching 1.8 percent in the medium term") and the National Treasury (2026 Budget Review, averaging 1.8% over 2026–2028, rising to 2.0% by 2028) assume — the *same* growth path that underpins the debt-stabilisation plan we anchor `debt_ratio_ss` to below.

We deliberately do **not** use South Africa's *realized* productivity growth, which has been dismal — near zero, even negative, over the 2010s state-capture/load-shedding decade, when potential growth collapsed from ~4% to near zero and TFP subtracted about 1.3 percentage points from growth. The steady state is a long-run object, and pairing that stagnation with a *stabilising* debt target would be internally inconsistent: it is precisely the IMF's more pessimistic scenario, in which debt does **not** stabilise but drifts toward 84% of GDP. The ~1.8% recovery is reform-contingent (Operation Vulindlela — energy and logistics), and we adopt it as the sustainable long-run assumption that is consistent with the debt anchor. Because $g_y$ is now a curated forward-looking assumption rather than a realized-data series, it is held in the packaged parameters and no longer refreshed from the live World Bank pull.

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

where $(1-\tau_d)$ is the pass-through coefficient and $\mu_d$ is the level shift. For South Africa we take the *slope* $1-\tau_d = 0.24485$ from the Li-Magud-Werner (LMW) emerging-market estimate below, but we **re-anchor the level $\mu_d$ to South African data** — the raw LMW cross-country intercept ($0.03377$) over-predicts South Africa's actual borrowing cost, as explained after the derivation.

These values come from {cite}`LMW2023`, who estimate the long-run pass-through of sovereign yields to corporate yields across 46 emerging economies using corporate yields from IHS Markit and sovereign yields from Bloomberg, predominantly U.S.-dollar secondary-market yields. They are therefore a cross-country emerging-market relationship rather than South Africa-specific bond data. Their preferred specification (Table 8, Column 2) fits a quadratic of the corporate yield on the sovereign yield of the same country:

```{math}
:label: eqn:lmw_quadratic
    y_{corp} = 8.199 - 2.975\, y_{sov} + 0.478\, y_{sov}^2
```

with both yields in percentage points. The quadratic captures the empirical fact that pass-through rises with the level of sovereign risk, consistent with the credit-risk and liquidity-premium channels the paper identifies. The paper is available as an IMF Working Paper, [Li, Magud, Werner, and Witte (2021)](https://www.imf.org/en/Publications/WP/Issues/2021/06/04/The-Long-Run-Impact-of-Sovereign-Yields-on-Corporate-Yields-in-Emerging-Markets-50224), and was later published in the *Journal of International Money and Finance* {cite}`LMW2023`.

OG-Core models the wedge in the opposite direction: it takes $r_t$ as an input and produces $r_{gov,t}$. We therefore invert the LMW relationship. We evaluate their quadratic on a grid of sovereign yields from 2% to 12%, compute the implied corporate yields, and then regress sovereign yields linearly on those corporate yields. Calling the resulting slope $b$ and intercept $a$ (both in percentage points), we identify $1-\tau_d = b$ and $\mu_d = a/100$.

**Re-anchoring the level to South Africa's data.** The LMW inversion delivers two numbers: a *slope* (the pass-through elasticity $1-\tau_d = 0.24485$) and an *intercept* ($\mu_d = 0.03377$). The slope is a genuine estimated co-movement, and we keep it. The intercept, however, is a cross-country average that maps a *nominal, US-dollar* sovereign-yield level onto the model's *real* return on capital, and it over-predicts South Africa's actual borrowing cost — it implies a steady-state $r_{gov}$ of about 4.3% real, whereas South Africa's effective real rate on the debt stock is about 3.7%. Since $r_{gov}$ multiplies the *entire* debt stock in the government budget ($\texttt{debt\_service} = r_{gov}\cdot D$), the right target is the effective/average rate: the National Treasury's effective interest rate on the debt portfolio, **7.1% nominal** (total debt-service cost ÷ gross loan debt, [2026 Budget Review, Chapter 7](https://www.treasury.gov.za/documents/National%20Budget/2026/review/Chapter%207.pdf)), deflated by the ~3.4% GDP deflator to **~3.7% real**. We therefore re-anchor $\mu_d$ so the steady-state $r_{gov}$ equals this effective rate, giving $\mu_d \approx 0.0254$. (This is *not* the ~5% real *marginal* new-issue yield on 10-year bonds, which is the cost of new borrowing, not the average cost of the stock that $r_{gov}$ multiplies.)

OG-Core's operational formula is $r_{gov,t} = \max\!\big(\texttt{r\_gov\_scale}\cdot r_t - \texttt{r\_gov\_shift} + \texttt{r\_gov\_DY}\cdot\tfrac{D_t}{Y_t} + \texttt{r\_gov\_DY2}\cdot\big(\tfrac{D_t}{Y_t}\big)^2,\; 0\big)$, so the JSON stores `r_gov_scale` $= 1-\tau_d = 0.24485$ and folds the re-anchored level $\mu_d = 0.0254$ together with the centred-premium constant below into `r_gov_shift` $= -0.0488$. The negative sign reflects the subtraction in the OG-Core rule, not a negative level shift in the theoretical equation.

The slope's inputs are deterministic and contain no South Africa-specific data; the *level* is re-anchored to South Africa's effective rate as above. The packaged values in the JSON are authoritative; `ogzaf.macro_params.get_macro_params` does not recompute them on a live update (a recompute would restore the raw LMW intercept and de-centre the premium). The snippet below reproduces the LMW *slope* derivation for transparency:

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
- `r_gov_shift` $= -\mu_d - r_{gov,DY2}\,\bar{d}^2 = -0.0254 - 0.04 \times 0.765^2 = -0.0488$ (the re-anchored level $\mu_d = 0.0254$ and the recentering constant $r_{gov,DY2}\,\bar{d}^2$, both absorbed into the shift)

The premium is exactly zero at the debt target, so the steady state is unchanged; along the transition path, debt above (or below) the target raises the sovereign rate convexly — about 7 basis points at $D/Y = 0.9$ and 22 basis points at $D/Y = 1.0$. This is the [Schmitt-Grohé and Uribe (2003)](https://www.nber.org/system/files/working_papers/w9270/w9270.pdf) debt-elastic premium in convex form, following the fiscal-limits literature ([Bi 2012](https://www.sciencedirect.com/science/article/abs/pii/S0014292111001085); [Ghosh et al. 2013](https://www.nber.org/system/files/working_papers/w16782/w16782.pdf)) and emerging-market spread-to-debt empirics of roughly 4–8 basis points per percentage point of debt ([Jaramillo and Weber 2012](https://www.imf.org/external/pubs/ft/wp/2012/wp12198.pdf)). It also matches South Africa's own experience: spreads were broadly stable while gross debt climbed through the 60s percent-of-GDP range, then blew out in 2020 when debt jumped toward 70% amid the downgrade to sub-investment grade — and compressed again (2.93 to 2.26 percentage points during 2025) as consolidation took hold. The value of $r_{gov,DY2} = 0.04$ matches OG-PHL's calibration of the same channel.

### Aggregate transfers

Aggregate (non-Social Security) transfers to households are set as a share of GDP with the parameter $\alpha_T$. We exclude Social Security from transfers since it is modeled specifically. With this definition, the share of transfers to GDP is 0.037, computed from the IMF Government Finance Statistics (non-interest transfers, latest available year) by `ogzaf.macro_params.get_macro_params`, which refreshes this parameter from the [IMF GFS API](https://data.imf.org/) on a live update.

### Government expenditures

Government spending on goods and services is also set as a share of GDP with the parameter $\alpha_G$. We define government spending as:
    <center>Government Spending = Total Outlays - Transfers - Net Interest on Debt - Social Security</center>
The IMF Government Finance Statistics put this share near 0.23–0.27. We instead set $\alpha_G = 0.19$ — the level at which the government budget is *consistent with the steady-state debt target* (see **Fiscal consistency** below). This sits just below South Africa's actual general-government consumption (~0.21 of GDP) because a steady state that holds debt at 0.765 must run the debt-stabilising primary surplus, which South Africa's current, deficit-running fiscal stance does not yet deliver: the model's steady state is the *sustainable, post-consolidation* state the Budget Review is steering toward, not today's stance. Because $\alpha_G$ is pinned to this consistency requirement rather than the raw GFS figure, it is held in the packaged parameters and not refreshed live.

## Fiscal consistency

The spending shares ($\alpha_G$, $\alpha_T$), the revenue the tax system raises, and the debt target `debt_ratio_ss` are not free of one another: for debt to hold at $\bar{d} = 0.765$ in the steady state, the government must run a primary surplus $pb^\* = \tfrac{r_{gov}-g}{1+g}\,\bar{d}$, so **primary spending must equal revenue minus $pb^\*$**. With the calibration above ($r_{gov} \approx 3.7\%$, $g \approx 1.8\%$, so $pb^\* \approx 1.4\%$ of GDP) and total tax revenue of ~24% of GDP, consistent primary spending is ~22.5% of GDP, i.e. $\alpha_G + \alpha_T \approx 0.227$ — hence $\alpha_G = 0.19$.

This identity is why the calibration is internally reconciled rather than a collection of independent point estimates. Two earlier settings had *masked* the fact that the block did not balance: a flat 22% personal income tax that over-collected (~16% of GDP versus South Africa's actual ~10%), and a stray 20% bequest tax (`tau_bq = 0.2`) that collected ~3.9% of GDP with no basis in South African law. Both inflated revenue enough to paper over a spending-above-revenue gap; with the correct progressive PIT and the bequest tax removed, the gap surfaced, and — because OG-Core's steady-state closure silently defers it to the transition while the debt-elastic premium prices the resulting overshoot — it made the baseline transition diverge until the spending side was reconciled to the identity. The debt-dynamics arithmetic is also the honest reason $g_y$ and $r_{gov}$ are calibrated as above: a model with a hotter interest-growth differential than South Africa's ($r_{gov}-g$) would demand a larger primary surplus and force $\alpha_G$ implausibly low.

South Africa's debt is not a solvency or default question for this purpose — it is ~90% rand-denominated, long-maturity and domestically financed, rated sub-investment-grade-but-*stabilising* (BB/Ba2, all three agencies on positive outlook in 2025–26), and the IMF assesses debt-distress risk as moderate with no restructuring risk. The stable-debt steady state is therefore a modelling device for the sustainable long run, consistent with the National Treasury's own stabilisation-at-~76.5% plan, not a claim that default is off the table or that current policy already stabilises debt (the IMF's more cautious baseline has debt still rising absent a sustained ~3% primary surplus).

### Steady-state validation

The calibrated steady state, solved offline from the packaged parameters, lands close to South African data instrument by instrument and moment by moment:

| Moment | Model SS | South Africa (data) | Source |
|---|---|---|---|
| Personal income tax / GDP | 10.1% | 10.1% | Budget Review 2026 / SARS |
| Corporate income tax / GDP | 4.5% | 4.5% | Budget Review 2026 |
| Consumption (indirect) tax / GDP | 9.5% | ~10% | SARS (VAT + fuel + excise + customs) |
| Government debt $D/Y$ | 0.765 | 0.765 (2028/29 target) | Budget Review 2026 |
| Foreign-held debt share $D_f/D$ | 0.25 | 0.25 | Budget Review 2026 |
| Sovereign real rate $r_{gov}$ | 3.7% | ~3.7% effective | Budget Review 2026 (deflated) |
| Real GDP growth $g$ | 1.8% | 1.8% (medium-term) | IMF / Budget Review 2026 |
| Primary surplus | 1.5% | 1.4% stabilising; 2.3% targeted by 2028/29 | Budget Review 2026 |
| `factor` (mean income) | R124,371 | R123,809 (data anchor) | model diagnostic |
| Capital-output $K/Y$ | 4.35 | — (OG-Core family ~4–5) | — |

The `factor` diagnostic — the model's solved mean income against the R123,809 data anchor — matches to within 0.5%, confirming the income level is well aligned.
