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

The total tax function, $T_{s,t}$, is a function of personal income taxes, taxes on bequests, and wealth taxes. Wealth and bequest taxes are set to zero in `OG-ZAF` (`tau_bq = 0`); an earlier calibration inadvertently carried a 20% bequest tax (`tau_bq = 0.2`), which has been removed — South Africa's estate duty raises a negligible ~0.1% of GDP, and the stray parameter was silently collecting ~3.9% of GDP and masking a fiscal inconsistency (see {ref}`Chap_MacroCalib`).

South Africa has a genuinely progressive personal income tax — statutory marginal rates rise from 18% to 45%, with a tax threshold (R95,750 in 2025/26) plus a primary rebate that exempt low earners entirely ([SARS, Rates of Tax for Individuals 2025/26](https://www.sars.gov.za/tax-rates/income-tax/rates-of-tax-for-individuals/)). Modelling it as a single flat rate applied to every household — the previous calibration used a 22% effective / 31% labour-marginal / 25% capital-marginal *linear* tax — misstates incidence and, more importantly, imposes a spurious labour-supply and saving wedge on the low earners who in reality fall below the tax threshold and remit no income tax (about a third of South African employment is informal, and PIT reaches only the formal, above-threshold minority).

We therefore use OG-Core's Gouveia-Strauss (GS) progressive tax function (`tax_func_type = "GS"`), under which the effective rate on total income $y$ rises smoothly from zero at low income toward an asymptote $\phi_0$ at high income:

```{math}
:label: eqn:gs
  \tau^{etr}(y) = \phi_0\,\frac{y - \left(y^{-\phi_1} + \phi_2\right)^{-1/\phi_1}}{y}
```

with the marginal rate rising to the same asymptote $\phi_0$. We anchor $\phi_0 = 0.464$ to South Africa's statutory top marginal rate (45%, lifted slightly so PIT hits its revenue target), and fit the curvature parameters $(\phi_1, \phi_2) = (1.393,\ 1.43\times10^{-8})$ to the SARS 2025/26 effective-rate schedule (the bracket schedule net of the primary rebate). The level is set so that personal income tax collects **10.1% of GDP** in the model steady state, matching South Africa's actual PIT take ([National Treasury Budget Review 2026](https://www.treasury.gov.za/documents/National%20Budget/2026/review/Chapter%204.pdf) / SARS). The resulting effective rate rises from ~0 near the tax threshold to about 20% at R500,000 and 36% at R2,000,000, closely tracking the SARS schedule, with the marginal rate ramping toward the 45% statutory top.

We chose GS over the smoother Heathcote-Storesletten-Violante (HSV) form specifically because **GS floors the effective rate at zero**: below South Africa's tax threshold, earners pay exactly nothing. This is both faithful to the statutory reality and numerically essential — HSV's mildly *negative* effective rate at the bottom (a small implicit subsidy) drained early-transition revenue and, interacting with the debt-elastic sovereign premium, made the baseline transition diverge (documented in {ref}`Chap_MacroCalib`).

We use a progressive function rather than the graded tax-*noncompliance* device that some sibling models use for informality (e.g. OG-ETH, where ~85% of employment is informal and PIT is a thin, near-flat tax). In South Africa low earners pay ~0 personal income tax **by statute** (the threshold and rebate), not through noncompliance, and the PIT is both progressive and a large, well-collected share of GDP — so a progressive function is the faithful representation and avoids double-counting the below-threshold population as evaders.

## Corporate income taxes

`OG-ZAF` uses the top statutory rate of 27% for the corporate income tax rate. Because informal and exempt firms mean corporate collections fall short of the statutory rate applied to the whole capital stock, the effective corporate tax is scaled by `adjustment_factor_for_cit_receipts` (together with `c_corp_share_of_assets`). We set this factor to 0.80, so that corporate income tax collects **4.5% of GDP** in the steady state, matching South Africa's actual CIT take ([National Treasury Budget Review 2026](https://www.treasury.gov.za/documents/National%20Budget/2026/review/Chapter%204.pdf)).

## Consumption / indirect taxes

South Africa's statutory VAT rate is 15% (the proposed 2025 increase was withdrawn), but OG-Core's `tau_c` is a tax on the consumption good and should therefore capture **all** taxes that fall on consumption — VAT *plus* the general fuel levy (~1.25% of GDP), specific and ad valorem excise (~0.9%), and customs duties (~1.0%) — which together raise about 10% of GDP. Measured against household final consumption, that is an effective rate near 16–18%, so we set an *effective* `tau_c = 0.18`, which reproduces ~9.5% of GDP in consumption-tax revenue in the steady state. (An earlier version set `tau_c = 0.10` on VAT alone; capturing VAT only under-collected and, with the correct income tax, left the government budget inconsistent with the debt target — see {ref}`Chap_MacroCalib`.)

## Payroll taxes

South Africa's payroll levies — UIF (2%, capped) and the Skills Development Levy (1%, with a small-employer exemption) — are small and, unlike a broad social-security contribution, are not a major revenue instrument; their coverage-weighted effective rate is roughly 2%. Because the payroll tax interacts with the modelled pension system (setting it also changes pension accounting), we leave `tau_payroll = 0` in this calibration and flag it as a candidate refinement.
