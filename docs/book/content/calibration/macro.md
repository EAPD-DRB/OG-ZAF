(Chap_MacroCalib)=
# Calibration of Macroeconomic Parameters

## Economic Assumptions

As the default rate of labour-augmenting technological change, $g_y$, we use 1.4% per year (`g_y_annual = 0.014`), South Africa's measured long-run productivity growth. In PWT 11.0, real GDP per person engaged (`rgdpna`/`emp`) grew 1.44% a year over 1994–2019 and over 2000–2019, both windows ending before the Covid shock. Hours per worker are available only from 2008; they fell about 0.3% a year over 2008–2019, so growth per hour, the measure the model needs, is if anything a little higher. The interest-growth differential is consistent with the Treasury's: with the steady-state real rate on debt of 3.7% and steady-state growth of $g_y + g_n \approx 1.4\%$ (long-run population growth is about zero, see the demographics chapter), $r_{gov} - g \approx 2.3$ percentage points, against about 2.1 points in the Treasury's medium-term numbers (a 7.1% nominal effective rate, a 3% inflation target and 2.0% real growth; [MTBPS 2026](https://www.treasury.gov.za/MediumTermBudget/Medium%20Term%20Budget%20Policy%20Statement%20Full%20document.pdf), Tables 1.1 and 3.5).

Near-term growth in the model is higher than the forecasts, for a reason $g_y$ cannot fix. The model's adult population grows about 1.4% a year in the first decade (UN WPP 2024) and every adult supplies labour, so effective labour grows about 2.2% a year and GDP about 3.3%. South Africa's employment grew 1.2% a year over 1994–2019 (PWT) and unemployment is a third of the labour force; the Treasury forecasts 1.2–2.0% growth for 2026–2029. OG-Core has no unemployment, so the model turns the youth bulge into output that South Africa does not produce. See the fiscal path below for what this does to debt.

We deliberately do **not** use South Africa's *realized* productivity growth, which has been dismal — near zero, even negative, over the 2010s state-capture/load-shedding decade, when potential growth collapsed from ~4% to near zero and TFP subtracted about 1.3 percentage points from growth. The steady state is a long-run object, and pairing that stagnation with a *stabilising* debt target would be internally inconsistent: it is precisely the IMF's more pessimistic scenario, in which debt does **not** stabilise but drifts toward 84% of GDP. The ~1.8% recovery is reform-contingent (Operation Vulindlela — energy and logistics), and we adopt it as the sustainable long-run assumption that is consistent with the debt anchor. Because $g_y$ is now a curated forward-looking assumption rather than a realized-data series, it is held in the packaged parameters and no longer refreshed from the live World Bank pull.

## Open Economy Parameters

### World interest rate

We set the annual world interest rate, `world_int_rate_annual`, to 6.3%: a 4% global risk-free rate plus a South African sovereign risk premium of roughly 2.3 percentage points. The premium is anchored by the National Treasury's own estimate — "the sovereign risk premium – the extra yield investors demand to hold government debt over 'risk-free' alternatives – has narrowed from 2.93 percentage points a year ago to 2.26 percentage points" ([2026 Budget Review, Chapter 7](https://www.treasury.gov.za/documents/National%20Budget/2026/review/Chapter%207.pdf)). The chapter does not define the benchmark instrument, so we corroborate the level with hard-currency evidence, which lands in the same band: South Africa's US-dollar bond spreads traded around 2.2–2.6 percentage points over Treasuries in 2025/26 (multi-year lows, after the December 2025 US$3.5 billion oversubscribed Eurobond), and the Damodaran country-default-spread table puts a Ba2/BB sovereign at roughly 2.7–3.2 percentage points. This is the rate at which foreign capital is supplied to the domestic economy. South Africa remains sub-investment-grade (BB/Ba2 at all three agencies, upgraded during 2025–26 with positive outlooks), so its premium is materially larger than that of investment-grade emerging markets.

### Foreign holding of government debt in the initial period

The path of foreign holding of domestic debt is endogenous, but the initial period stock of debt held by foreign investors is exogenous. We set this parameter, `initial_foreign_debt_ratio`, to 0.25: foreign investors held 25.0% of South Africa's domestic government bonds in 2025, up from 24.6% in 2024 ([2026 Budget Review, Chapter 7](https://www.treasury.gov.za/documents/National%20Budget/2026/review/Chapter%207.pdf), Figure 7.4). Note this is the domestic-market headline measure; a residency-comprehensive measure that also counts the (roughly 10% of gross debt) foreign-currency loans held abroad would be closer to 0.33.

### Foreign purchases of newly issued debt

We set $\zeta_D = 0.25$, matching the foreign share of the domestic government bond market above, on the assumption that foreign investors absorb new issuance in proportion to their current holdings.

### Foreign holdings of excess capital

We set $\zeta_K = 0.16$, the normalized Chinn-Ito capital-account openness index (`ka_open`) for South Africa, which is 0.1626 in the [2022 update](https://web.pdx.edu/~ito/Chinn-Ito_website.htm) (unchanged since 2008). This parameter is hard to pin down directly since foreign purchases of "excess" capital demand are not measured; the Chinn-Ito index anchors it to South Africa's de jure openness, which is low because exchange controls still apply to residents. De facto participation by non-residents in South African capital markets is considerably higher, so 0.16 is a conservative anchor. We check the level the steady state produces against the balance of payments. Read gross, as the model's capital identity requires, foreign-owned capital is inward direct-investment equity (about 22.7% of GDP) plus portfolio equity (about 34.8%) at the end of 2025 ([SARB IIP](https://www.resbank.co.za/), equity splits from QB S-98), divided by private capital of 2.80 times GDP: 20.5% of the private capital stock. This is an upper bound, because the portfolio-equity figure includes dual-listed companies whose capital sits abroad; direct-investment equity alone gives 8%. At $\zeta_K = 0.16$ the steady state lands at 21.1%, on the gross measure, so the prior needs no adjustment. The sign is right because the domestic return on capital (11.0%) sits above the 6.3% world rate, so foreign capital flows in.

## Government Debt, Spending and Transfers

### Government Debt

The path of government debt is endogenous. But the initial value is exogenous. To avoid converting between model units and rand, we calibrate the initial debt-to-GDP ratio rather than the rand value of the debt. This is the model parameter $\alpha_D$ (`initial_debt_ratio`), which we set to 0.770: national government gross loan debt at the end of fiscal year 2024/25, the stock the model's first year (2025) starts from ([MTBPS 2026](https://www.treasury.gov.za/MediumTermBudget/Medium%20Term%20Budget%20Policy%20Statement%20Full%20document.pdf), Table 3.8). We use the National Treasury's national-government gross loan debt rather than the World Bank QPSD general-government series, which measures a wider perimeter; the Treasury figure is the one the fiscal framework targets and reports against.

The long-run debt target `debt_ratio_ss` is set to 0.765. The MTBPS 2026 has gross loan debt peaking at 78.4% of GDP in 2025/26 and falling to 76.6% by 2029/30, the end of its projection; we take that end point as the steady-state anchor.

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

**Re-anchoring the level to South Africa's data.** The LMW inversion delivers two numbers: a *slope* (the pass-through elasticity $1-\tau_d = 0.24485$) and an *intercept* ($\mu_d = 0.03377$). The slope is a genuine estimated co-movement, and we keep it. The intercept, however, is a cross-country average that maps a *nominal, US-dollar* sovereign-yield level onto the model's *real* return on capital, and it over-predicts South Africa's actual borrowing cost — with the steady-state return on capital of 11.0% it implies $r_{gov}$ of about 6.1% real, whereas South Africa's effective real rate on the debt stock is about 3.7%. Since $r_{gov}$ multiplies the *entire* debt stock in the government budget ($\texttt{debt\_service} = r_{gov}\cdot D$), the right target is the effective/average rate: the National Treasury's effective interest rate on the debt portfolio, **7.1% nominal** (total debt-service cost ÷ gross loan debt, [2026 Budget Review, Chapter 7](https://www.treasury.gov.za/documents/National%20Budget/2026/review/Chapter%207.pdf)), deflated by the ~3.4% GDP deflator to **~3.7% real**. We therefore re-anchor $\mu_d$ so the steady-state $r_{gov}$ equals this effective rate, giving $\mu_d \approx 0.0100$ (tuned jointly with the other steady-state targets). (This is *not* the ~5% real *marginal* new-issue yield on 10-year bonds, which is the cost of new borrowing, not the average cost of the stock that $r_{gov}$ multiplies.)

OG-Core's operational formula is $r_{gov,t} = \max\!\big(\texttt{r\_gov\_scale}\cdot r_t - \texttt{r\_gov\_shift} + \texttt{r\_gov\_DY}\cdot\tfrac{D_t}{Y_t} + \texttt{r\_gov\_DY2}\cdot\big(\tfrac{D_t}{Y_t}\big)^2,\; 0\big)$, so the JSON stores `r_gov_scale` $= 1-\tau_d = 0.24485$ and folds the re-anchored level $\mu_d = 0.0100$ together with the centred-premium constant below into `r_gov_shift` $= -0.0334$. The negative sign reflects the subtraction in the OG-Core rule, not a negative level shift in the theoretical equation.

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

r_gov_shift = (
    -res.params[0] / 100
)  # -0.03377  (= -\mu_d in the theoretical equation)
r_gov_scale = res.params[
    1
]  #  0.24485  (= 1-\tau_d in the theoretical equation)
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
- `r_gov_shift` $= -\mu_d - r_{gov,DY2}\,\bar{d}^2 = -0.0100 - 0.04 \times 0.765^2 = -0.0334$ (the re-anchored level $\mu_d = 0.0100$ and the recentering constant $r_{gov,DY2}\,\bar{d}^2$, both absorbed into the shift)

The premium is exactly zero at the debt target, so the steady state is unchanged; along the transition path, debt above (or below) the target raises the sovereign rate convexly — about 7 basis points at $D/Y = 0.9$ and 22 basis points at $D/Y = 1.0$. This is the [Schmitt-Grohé and Uribe (2003)](https://www.nber.org/system/files/working_papers/w9270/w9270.pdf) debt-elastic premium in convex form, following the fiscal-limits literature ([Bi 2012](https://www.sciencedirect.com/science/article/abs/pii/S0014292111001085); [Ghosh et al. 2013](https://www.nber.org/system/files/working_papers/w16782/w16782.pdf)) and emerging-market spread-to-debt empirics of roughly 4–8 basis points per percentage point of debt ([Jaramillo and Weber 2012](https://www.imf.org/external/pubs/ft/wp/2012/wp12198.pdf)). It also matches South Africa's own experience: spreads were broadly stable while gross debt climbed through the 60s percent-of-GDP range, then blew out in 2020 when debt jumped toward 70% amid the downgrade to sub-investment grade — and compressed again (2.93 to 2.26 percentage points during 2025) as consolidation took hold. The value of $r_{gov,DY2} = 0.04$ matches OG-PHL's calibration of the same channel.

### Aggregate transfers

Aggregate transfers to households are set as a share of GDP with the parameter $\alpha_T = 0.037$. They are cash transfers only: South Africa's social grants, including the Social Relief of Distress grant, cost 3.6% of GDP in 2025/26 ([Budget Review 2026](https://www.treasury.gov.za/documents/National%20Budget/2026/review/FullBR.pdf), Table 5.7). The old-age grant (1.5% of GDP) is part of this total; South Africa has no contributory state pension, so the model's pension system is switched off (all PIA rates zero) and the grant is a transfer.

### Public investment

Government investment is set with $\alpha_I = 0.0242$, general government gross fixed capital formation in 2025 (2.42% of GDP, [SARB Quarterly Bulletin](https://www.resbank.co.za/), Table S-127). Public corporations (Eskom, Transnet and others, another 1.56% of GDP) are market producers in the national accounts and their capital is part of the private stock $K$ below. Public capital does not enter production in this calibration ($\gamma_g = 0$).

### Government expenditures

Government consumption is set with $\alpha_G$. In the steady state OG-Core solves for it from the budget identity below, and it lands at 18.5% of GDP, against general government final consumption of 18.9% in 2025 (SARB QB S-116). Along the transition $\alpha_G$ is a short path, `[0.1896, 0.1866, 0.1851]`, so the primary balance follows the Treasury's consolidated plan in the first two years (+1.3% of GDP in 2025/26, +1.6% in 2026/27) and then holds at the debt-stabilising level. From 2027/28 the Treasury plans surpluses (2.0–3.0%) above the stabilising one, to bring debt down; the model does not chase them, because the steady state holds debt at the target rather than below it.

## Fiscal consistency

The spending shares, the revenue the tax system raises and the debt target `debt_ratio_ss` are not free of one another. For debt to hold at $\bar{d} = 0.765$ in the steady state, the government must run a primary surplus $pb^\* = \tfrac{r_{gov}-g}{1+g}\,\bar{d}$, so primary spending, including public investment and transfers, must equal revenue minus $pb^\*$:

```{math}
\alpha_G + \alpha_T + \alpha_I = \text{revenue}/Y - pb^\*
```

With $r_{gov} = 3.7\%$ and $g = 1.38\%$, $pb^\* = 1.75\%$ of GDP. Revenue is 26.4% of GDP, so primary spending is 24.7%: 18.5% government consumption, 3.7% transfers and 2.4% public investment. The Treasury's consolidated primary surplus is 1.3% in 2025/26 and is planned to rise above 1.75% from 2027/28, so the steady state's fiscal stance is one the Treasury plans to pass within three years.

South Africa's debt is not a solvency or default question for this purpose — it is ~90% rand-denominated, long-maturity and domestically financed, rated sub-investment-grade-but-*stabilising* (BB/Ba2, all three agencies on positive outlook in 2025–26), and the IMF assesses debt-distress risk as moderate with no restructuring risk. The stable-debt steady state is therefore a modelling device for the sustainable long run, consistent with the National Treasury's own plan, which has debt at 76.6% of GDP in 2029/30, not a claim that default is off the table or that current policy already stabilises debt (the IMF's more cautious baseline has debt still rising absent a sustained ~3% primary surplus).

### Steady-state validation

The steady state of the packaged parameters, solved by `examples/run_og_zaf.py` (ogcore 0.22.0, offline), against South African data. Targets and sources are in `ogzaf/targets.py`; `examples/tune_ss.py` prints the same table.

| Moment | Model | Data | Source |
|---|---|---|---|
| Personal income tax / GDP | 10.13% | 10.15% | MTBPS 2026, 2025/26 |
| Company tax + dividends tax / GDP | 5.02% | 5.02% | MTBPS 2026 |
| Consumption taxes, net of SACU / GDP | 8.91% | 8.92% | MTBPS 2026, BR 2026 |
| Payroll levies (SDL + UIF) / GDP | 0.67% | 0.67% | MTBPS 2026, UIF |
| Property taxes / GDP | 1.60% | 1.61% | Stats SA P9114, BR 2026 |
| Estate duty + donations tax / GDP | 0.08% | 0.08% | BR 2026 |
| Real rate on government debt | 3.71% | 3.7% | MTBPS 2026, deflated |
| Debt / GDP | 0.765 | 0.765 (2029/30 path) | MTBPS 2026 |
| Foreign share of debt | 0.25 | 0.25 | BR 2026 |
| Government consumption / GDP | 18.5% | 18.9% | SARB QB 2025 |
| Transfers / GDP | 3.7% | 3.6% | BR 2026 |
| Public investment / GDP | 2.42% | 2.42% | SARB QB 2025 |
| Private capital / GDP | 2.79 | 2.80 | PWT 11.0, SARB |
| Real return on capital | 11.0% | 12.0% | PWT 11.0 `irr` |
| Foreign-owned share of capital | 21.2% | 8–20.5% (gross) | SARB IIP |
| Household wealth / GDP | 2.77 | 2.97 | SARB QB S-142 |
| Private consumption / GDP (purchaser prices) | 67.2% | 65.8% | SARB QB 2025 |
| Total investment / GDP | 17.6% | 13.9% (2025); 18% (2010–2019) | SARB QB; PWT 11.0 `csh_i` |

Three rows are not targeted. Investment is above 2025's depressed level and close to the 2010s average. The trade surplus is 5.7% of GDP against 1.9% in 2025: in the model it pays the return on foreign-owned capital, while South Africa also earns income on foreign assets of 135% of GDP, which OG-Core's single foreign-capital stock cannot hold. Household wealth is 0.2 times GDP below the SARB balance sheet.

### Fiscal path validation

`examples/validate_fiscal_path.py` scores the first years of the baseline transition against the Treasury's numbers.

```{figure} ./images/fiscal_path.png
---
height: 300px
name: FigFiscalPath
---
Baseline transition against gross loan debt (MTBPS 2026 Tables 1.3 and 3.8), the consolidated primary balance (MTBPS 2026 Tables 1.2, 3.6 and 4.1), tax revenue (MTBPS 2026 Table 3.1, bridged to the model's concept) and public investment (SARB QB S-127).
```

The model starts from the measured debt ratio, 77.0% of GDP, and its primary surplus follows the Treasury's in the first two years (1.05% and 1.33% against 1.3% and 1.6%). Debt then falls, to 73.7% in 2029 against the Treasury's 77.5%, and to a low of 61% around 2045 before it returns to the 76.5% target. The cause is growth: model GDP grows 3.3% a year in 2026–2029 against the Treasury's 1.2–2.0%, because the model employs the whole growing adult population (see Economic Assumptions above). With growth 1.5 points too high, the debt ratio falls about 1.1 points a year faster. Revenue (26.2% of GDP against 26.6% in 2025/26) and public investment match. Consumption is smooth from the first year; there is no start-up jump.

The model's first-year household wealth is 2.61 times GDP, set by the steady-state wealth profile and the 2025 population. The data are 2.93 (capital side: private capital net of foreign ownership plus domestically held debt) and 2.97 (SARB household balance sheet). ogcore 0.22.0 can target this level (`initial_wealth_ratio`); it is not set in this calibration.
