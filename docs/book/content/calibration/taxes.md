(Chap_Tax)=
# Taxes in OG-ZAF

The government is not an optimizing agent in `OG-ZAF`. The government levies taxes on household income, corporate income, and value added. With these resources, the government provides transfers to households, spends resources on public goods, and makes rule-based adjustments to stabilize the economy in the long-run. The government can run budget deficits or surpluses in a given year and must, therefore, be able to accumulate debt or savings.  The spending and debt parameters are discussed in Chapter {ref}`Chap_MacroCalib`.  Taxes are discussed in this chapter.


## Personal income taxes
The government sector influences households through two terms in the household budget constraint {eq}`EqHHBC`---government transfers $TR_{t}$ and through the total tax liability function $T_{s,t}$, which can be decomposed into the effective tax rate times total income. In this chapter, we detail the household tax component of government activity $T_{s,t}$ in `OG-ZAF`.

```{math}
:label: EqHHBC
  c_{j,s,t} + b_{j,s+1,t+1} &= (1 + r_{hh,t})b_{j,s,t} + w_t e_{j,s} n_{j,s,t} + \\
  &\quad\quad\zeta_{j,s}\frac{BQ_t}{\lambda_j\omega_{s,t}} + \eta_{j,s,t}\frac{TR_{t}}{\lambda_j\omega_{s,t}} + ubi_{j,s,t} - T_{s,t}  \\
  &\quad\forall j,t\quad\text{and}\quad s\geq E+1 \quad\text{where}\quad b_{j,E+1,t}=0\quad\forall j,t
```

The total tax function, $T_{s,t}$, is a function of personal income taxes, payroll taxes, taxes on bequests and wealth taxes. Each is set as an effective rate: what South Africa collects divided by the base the model taxes.

**Property taxes.** Municipal property rates raised R101.5bn in the year to June 2025, 1.36% of GDP ([Stats SA P9114](https://www.statssa.gov.za/publications/P9114/P91142025.pdf)); transfer duty and the securities transfer tax add 0.25% (Budget Review 2026 annexure). OG-Core's wealth tax is the nearest instrument. We set `h_wealth = 1`, `m_wealth = 1`, so the rate is almost flat above small holdings and lower at the bottom, as the rates exemption on low-value homes makes it, and tune `p_wealth = 0.0078` so the tax collects 1.61% of GDP.

**Estate duty.** Estate duty and donations tax raised R5.9bn, 0.08% of GDP (Budget Review 2026 annexure). The model's bequests are much larger than taxable estates, so the effective rate is small: `tau_bq = 0.0078` collects 0.08% of GDP. It is not the 20% statutory estate duty rate, which applied to the model's bequests would collect about 2% of GDP.

South Africa has a genuinely progressive personal income tax — statutory marginal rates rise from 18% to 45%, with a tax threshold (R95,750 in 2025/26) plus a primary rebate that exempt low earners entirely ([SARS, Rates of Tax for Individuals 2025/26](https://www.sars.gov.za/tax-rates/income-tax/rates-of-tax-for-individuals/)). Modelling it as a single flat rate applied to every household — the previous calibration used a 22% effective / 31% labour-marginal / 25% capital-marginal *linear* tax — misstates incidence and, more importantly, imposes a spurious labour-supply and saving wedge on the low earners who in reality fall below the tax threshold and remit no income tax (about a third of South African employment is informal, and PIT reaches only the formal, above-threshold minority).

We therefore use OG-Core's Gouveia-Strauss (GS) progressive tax function (`tax_func_type = "GS"`), under which the effective rate on total income $y$ rises smoothly from zero at low income toward an asymptote $\phi_0$ at high income:

```{math}
:label: eqn:gs
  \tau^{etr}(y) = \phi_0\,\frac{y - \left(y^{-\phi_1} + \phi_2\right)^{-1/\phi_1}}{y}
```

with the marginal rate rising to the same asymptote $\phi_0$. We anchor $\phi_0 = 0.464$ to South Africa's statutory top marginal rate (45%, lifted slightly so PIT hits its revenue target), fit the curvature parameter $\phi_1 = 1.393$ to the SARS 2025/26 effective-rate schedule (the bracket schedule net of the primary rebate), and tune the scale parameter $\phi_2 = 1.656\times10^{-8}$ so that personal income tax collects **10.15% of GDP** (R790.6bn in 2025/26, [MTBPS 2026](https://www.treasury.gov.za/MediumTermBudget/Medium%20Term%20Budget%20Policy%20Statement%20Full%20document.pdf) Table 3.1) in the model steady state — collected entirely from the above-threshold half of earners (see below) — matching South Africa's actual PIT take ([National Treasury Budget Review 2026](https://www.treasury.gov.za/documents/National%20Budget/2026/review/Chapter%204.pdf) / SARS). The resulting effective rate rises from ~0 near the tax threshold to about 21% at R500,000 and 38% at R2,000,000 — tracking the SARS schedule net of rebates (20% and 36% at those incomes) and sitting slightly above it at mid-to-high incomes, the price of matching aggregate collections that in reality include capital-income taxes and fiscal drag the model books through this same function — with the marginal rate ramping toward the 45% statutory top.

We chose GS over the smoother Heathcote-Storesletten-Violante (HSV) form specifically because **GS floors the effective rate at zero**: below South Africa's tax threshold, earners pay exactly nothing. This is both faithful to the statutory reality and numerically essential — HSV's mildly *negative* effective rate at the bottom (a small implicit subsidy) drained early-transition revenue and, interacting with the debt-elastic sovereign premium, made the baseline transition diverge (documented in {ref}`Chap_MacroCalib`).

We use a progressive function rather than the graded tax-*noncompliance* device that some sibling models use for informality (e.g. OG-ETH, where ~85% of employment is informal and PIT is a thin, near-flat tax). In South Africa low earners pay ~0 personal income tax **by statute** (the threshold and rebate), not through noncompliance, and the PIT is both progressive and a large, well-collected share of GDP — so a progressive function is the faithful representation and avoids double-counting the below-threshold population as evaders.

**Who is in the PIT net.** The calibration also models the extensive margin explicitly: `income_tax_filer` is set to 0 for the bottom two lifetime-income groups (the bottom 50% of earners) and 1 elsewhere. The National Treasury estimates ~8.3 million individuals with taxable income above the tax threshold in 2026/27 ([Budget Review 2026, Table 4.5](https://www.treasury.gov.za/documents/National%20Budget/2026/review/Chapter%204.pdf)), against 16.8 million employed ([Stats SA QLFS Q1 2026](https://www.statssa.gov.za/publications/P0211/P02111stQuarter2026.pdf)) — almost exactly half of earners inside the PIT net, matching the model's bottom-two-group population share ($\lambda_1 + \lambda_2 = 0.50$). Membership is keyed to being *outside the PIT net* (below-threshold and informal workers), not to SARS return-submission counts: most formal employees pay via PAYE withholding without ever filing a return. Setting `income_tax_filer = 0` zeroes both the average and the marginal income-tax rate for those groups, so the model imposes no labour-supply or saving wedge on workers the PIT does not reach; $\phi_2$ is tuned so the five paying groups deliver the full 10.1% of GDP. One caveat: ogcore's transition-path revenue accounting applies year-0 filer values to the whole path, so a *time-varying* formalization reform would need an upstream ogcore fix — the constant filer used here is unaffected.

## Corporate income taxes

`OG-ZAF` uses the top statutory rate of 27% for the corporate income tax rate. Because informal and exempt firms mean corporate collections fall short of the statutory rate applied to the whole capital stock, the effective corporate tax is scaled by `adjustment_factor_for_cit_receipts` (together with `c_corp_share_of_assets`). We set this factor to 0.738, so that corporate income tax collects **5.02% of GDP** in the steady state: CIT of R345.4bn plus dividends tax of R45.9bn in 2025/26 (MTBPS 2026, Table 3.1). Dividends tax is a second layer of tax on distributed company profits; the corporate tax is the model's only instrument that falls on that base.

## Consumption / indirect taxes

South Africa's statutory VAT rate is 15%, but OG-Core's `tau_c` is a tax on the consumption good and so captures **all** taxes on consumption: VAT (6.43% of GDP in 2025/26), the general fuel levy (1.22%), excise (0.93%), customs (1.05%) and other indirect taxes (0.24%), 9.87% of GDP in total (MTBPS 2026 Table 3.1; Budget Review 2026 Table 4.1). We net out the 0.95% of GDP paid to the other Southern African Customs Union members, which the government collects but does not keep. The effective `tau_c = 0.153` collects the remaining 8.92% of GDP in the steady state.

## Payroll taxes

South Africa's payroll levies are the Skills Development Levy (1% of payroll; R26.0bn in 2025/26, MTBPS 2026) and Unemployment Insurance Fund contributions (1% employer plus 1% employee, capped; R26.7bn, UIF Annual Report 2025/26), together 0.67% of GDP. They are modelled explicitly with `tau_payroll`, a flat tax on labour income, which in ogcore 0.22.0 is separate from the pension system (switched off for South Africa). The effective rate is collections over the model's labour income, $0.0067/0.528 = 0.0127$. UIF benefit payments are not in $\alpha_T$, which holds social grants only.
