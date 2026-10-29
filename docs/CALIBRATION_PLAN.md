# OG-ZAF calibration plan: moments, levers and dashboards

Step zero of the calibration (og-country-calibration skill), written from the
equations ogcore 0.22.0 evaluates (`firm.get_r`, `aggregates.get_K_splits`,
`fiscal.get_r_gov`, `fiscal.get_G_ss`) on `feature/sa-calibration`; statuses after the
upgrade round. Status of each lever: **S** sourced from data,
**T** tuned to the named moment, **D** deliberately defaulted with a reason,
**U** unexamined (inherited from OG-USA or an earlier port; must end as S, T
or D).

## Two dashboards

1. **Steady state** (`examples/tune_ss.py`, targets in `ogzaf/targets.py`).
   Fiscal rows first, then external, production, households.
2. **First year and near-term path** (baseline transition,
   `examples/validate_fiscal_path.py`): 2025 revenue by instrument and private
   `K/Y` (the tax dials and initial wealth are tuned here, because 2025 is
   what is measured); debt, primary balance, revenue and public investment
   against actuals and the National Treasury's medium-term path; growth of
   output per hour against the baseline productivity path.

## Moment x lever table

| Moment | Model closed form | Levers (status) | Shares levers with |
|---|---|---|---|
| `K/Y` (private) | `(1-tau_b)·gamma / (r + delta - tau_b·delta_tau - itc·delta)`; `r` from the household Euler through `r_p` | `gamma` S (ILOSTAT), `delta` S (0.0405, private CFC / private K), `tau_b = cit_rate·c_corp_share·adj_factor` S/T, `delta_tau` D (0.04, tax depreciation), `beta` T (0.8706, to private K/Y 2.80), `sigma` D, `g_y` S, `debt_ratio_ss` S, `r_gov_shift` T, `zeta_K` S, `world_int_rate` S | `I/Y`, `C/Y`, `K_f/K`, `r` |
| `K_f/K` | `K_f = zeta_K·(K_open - B + D_d)`, `K_d = B - D_d` | `zeta_K` S (Chinn-Ito 0.16; lands on the gross IIP share, 21%), `world_int_rate` S, everything that moves `B` | `K/Y`, `NX/Y`, `C/Y` |
| `D_f/D` | `zeta_D` share of new issuance; SS `D_f = zeta_D·D` | `zeta_D` S = `initial_foreign_debt_ratio` S (stock = flow keeps the share stationary) | `NX/Y` |
| `r_gov` | `max(scale·r - shift + DY·D/Y + DY2·(D/Y)^2, floor)` | `r_gov_shift` T to the effective real rate, `r_gov_scale` D (LMWW slope), `r_gov_DY`/`DY2` D (centered at `debt_ratio_ss`), `r_gov_floor` D (0, real rates positive) | `pb*`, `G/Y` |
| `G/Y` (solved) | `G = rev + new_borrowing + aid - (pensions + TR + debt_service + UBI + I_g)` | every revenue line, `alpha_T` S, `alpha_I` S (0.0242), `debt_ratio_ss` S, `r_gov`, `g_y`, `g_n_ss`; `alpha_G` (input) only on the transition | everything fiscal |
| PIT/Y | GS ETR on total income, scaled by `factor` | `phi0` S (45% top rate, lifted to 0.464), `phi1` S (SARS schedule), `phi2` T, `income_tax_filer` S, `mean_income_data` S | payroll |
| payroll/Y | `tau_payroll·w·e·n` | `tau_payroll` S (0.0127, SDL + UIF) | PIT |
| CIT/Y | `tau_b·(Y - wL - delta_tau·K)`-type base | `cit_rate` S (27%), `adjustment_factor_for_cit_receipts` T, `c_corp_share_of_assets` D (0.7, absorbed by the adjustment factor) | `K/Y` |
| consumption tax/Y | `tau_c·C` | `tau_c` T (all indirect taxes) | `C/Y` |
| property tax/Y | `p_wealth·B` (flat, `h = 1`, `m` small) | `p_wealth` T (0.0073, property taxes 1.61% of GDP in 2025) | saving, `K/Y` |
| bequest tax/Y | `tau_bq·BQ` | `tau_bq` T (0.0100, estate duty 0.08% of GDP in 2025) | `chi_b` |
| pensions/Y | US-style formula with PIA rates | D: all PIA rates 0. South Africa has no contributory state pension; the old-age grant is a cash transfer in `alpha_T`; GEPF is a funded employer scheme (private saving) | `G/Y`, saving |
| `TR/Y` | `alpha_T·Y` | `alpha_T` S (IMF GFS social benefits; check cash-only) | `G/Y`, `C/Y` |
| `I_g/Y` | `alpha_I·Y` | `alpha_I` S | `G/Y`, `(I+I_g)/Y` |
| `(I+I_g)/Y` | `(e^{g_y}(1+g_n)-1+delta)·K/Y + I_g/Y` | `delta` S, `g_y` S, `g_n_ss` (demographic window), `K/Y` levers | `C/Y` |
| `C/Y` | residual: `Y - I - I_g - G - NX` | none of its own: decompose the gap | all |
| `NX/Y` | residual of foreign flows | `zeta_K`, `zeta_D`, `world_int_rate`; in the SS compare only its sign with `K_f` | `C/Y` |
| `wL/Y` | `1 - gamma - gamma_g` at `epsilon = 1` | not a free check at `epsilon = 1` | — |
| `g_n_ss` | growth of frozen `final_data_year` rates | demographic window S (UN data to 2098) | `pb*`, `I/Y` |
| earnings `e` | hand-coded WID x NTA (`income.get_e_orig`) | D: documented, not re-derived here. Carries unemployment: NTA labour income per person and WID shares include the jobless; 2005 employment rates by age are close to 2024's (ILO) | PIT, wealth |
| `chi_n` | OG-USA profile | D: borrowed (document). Not used for unemployment, which would make joblessness a choice | hours |
| output per hour, 2026-2050 | `Z_t` path, labour from demographics | `Z` T (`ogzaf/productivity.py`: average of our rule and the capped OECD EO117 baseline; `Z_ss = 1`) | GDP growth, debt path |
| household wealth / GDP, 2025 | `initial_wealth_ratio·Y_0` | `initial_wealth_ratio` T (2.79, to private `K/Y` 2.80 in 2025) | first-year `r`, revenue |
| PIT/Y along the path | GS `phi2_t = phi2_2025·lambda_t^-phi1` | brackets indexed to model average income (D, neutral indexation) | PIT/Y |
| debt path after 2029/30 | budget rule from `tG1` | `tG1` D (5: after the MTBPS window), `rho_G` D (0.1), `alpha_G` S/T (Treasury primary balance 2025/26-2026/27, then stabilising) | `G/Y` |

## Tuning order

1. Sourced structurals: `delta`, `alpha_I` (then re-solve `alpha_G`), `alpha_T`
   cash check, demographic window, debt and `start_year` check.
2. Tax dials against their own revenue lines: `phi2`, CIT adjustment, `tau_c`,
   `tau_payroll`, `p_wealth`, `tau_bq`.
3. `r_gov_shift` against the effective real rate on debt.
4. `beta` against private `K/Y`.
5. `zeta_K` against gross `K_f/K`, or a written reason if the world-rate sign
   check fails.
6. Re-tune 2 if 4 or 5 moved the bases.
7. Transition, jointly: the `Z` path to the baseline output per hour,
   `initial_wealth_ratio` to private `K/Y` in 2025, the tax dials of step 2
   to 2025 collections, and the bracket index to model average income.
   Then re-check the steady state (`r_gov`, `K/Y`, `K_f/K`) and the fiscal
   path panel.
