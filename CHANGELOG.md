# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Added an **eight-industry (M=8) calibration** in its own file, leaving the single-industry default untouched. `ogzaf/create_multisector_calibration.py` builds `ogzaf/ogzaf_default_parameters_multisector.json` from the 2019 South African SAM (SASAM) and the QLFS employment series. The file is self-sufficient — the single-industry calibration with the multi-industry parameters merged in — so it loads standalone in one step like the single-industry default (see `examples/run_og_zaf_multiple_industry.py`); regenerate it whenever the single-industry base is recalibrated. The industries are Agriculture, Mining, Electricity, Water and Waste, Construction, Trade/Transport/Accommodation, Services and Manufacturing — Electricity is its own industry so energy policy is directly analysable, and Manufacturing is placed last (OG-Core's numeraire and sole investment-good producer). Per-industry capital shares `gamma` and TFP `Z` come from the SASAM (TFP as a Solow residual using QLFS employment as the physical labour input); `alpha_c` and the `io_matrix` (the domestic value-added content of each consumption good, via make/use Leontief algebra derived from the SASAM) come from household consumption and the input-output structure. The multi-industry representation is kept consistent with the single-industry one: `gamma` is rescaled to the same value-added-weighted mean (0.47164), the `Z` level is set by a Hicks-neutral rescale so the model's `factor` matches, and `chi_n`/`chi_b` are converted for the multi-good composite-consumption units. All economy-wide and fiscal parameters are inherited from the base. New construction tools live in `ogzaf/input_output.py` (`get_gamma`, `get_Z`, `get_employment`, `get_io_matrix_value_added`, and a household-based `get_alpha_c`). See the firms calibration chapter.
- Made the personal income tax **progressive**, replacing the flat linear rates (22% effective / 31% labour / 25% capital applied identically to every household) with OG-Core's Gouveia-Strauss progressive tax function (`tax_func_type = "GS"`, `etr/mtrx/mtry_params = [0.464, 1.393, 1.43e-8]`). The asymptote `phi0` is anchored to South Africa's 45% statutory top marginal rate and the curvature fit to the SARS 2025/26 schedule; the level is set so PIT collects 10.1% of GDP. This removes the spurious labour-supply and saving wedge the flat rate imposed on the low earners who fall below South Africa's tax threshold (~a third of employment is informal and pays no PIT). We use GS rather than the smoother HSV form because GS **floors the effective rate at zero** — faithful to the statutory reality (below-threshold earners pay exactly nothing) and numerically essential (HSV's negative bottom-end rate drained transition revenue and destabilised the model). Distinct from the graded tax-noncompliance device sibling models use for higher-informality economies. See the taxes calibration chapter.
- Enabled a debt-elastic sovereign premium — the crowding-out-via-risk channel OG-Core's defaults leave off (`r_gov_DY = r_gov_DY2 = 0`). It is a *centered* convex form, `r_gov_DY2 * (D/Y - 0.765)^2` (`r_gov_DY2 = 0.04`, `r_gov_DY = -0.0612`), flat at the 0.765 debt target and steepening only as debt moves away, so the premium is exactly zero at the target and the steady state is unchanged. Matches South African experience (stable spreads through the 60s-percent debt range, a 2020 blowout, compression again in 2025) and mirrors OG-PHL. See the macro calibration chapter.

### Changed

- Recalibrated the open-economy and debt block to South African official data. World interest rate `world_int_rate_annual` 0.04 -> 0.063 (4% global risk-free + the National Treasury's 2.26pp sovereign risk premium, 2026 Budget Review Ch. 7). Capital openness `zeta_K` 0.9 -> 0.16 (normalized Chinn-Ito index for South Africa). Initial debt `initial_debt_ratio` -> 0.789 and steady-state target `debt_ratio_ss` -> 0.765 (Budget Review: gross loan debt stabilises at 78.9% in 2025/26, declining to 76.5% by 2028/29 — National Treasury national-government debt, not the wider World Bank QPSD perimeter). `initial_foreign_debt_ratio`/`zeta_D` -> 0.25 (25% foreign participation in the domestic bond market).
- Recalibrated indirect and corporate taxes to effective South-African rates: `tau_c` 0.15 -> 0.18, capturing **all** consumption/indirect taxes (VAT + fuel levy + excise + customs, ~9.5% of GDP), not VAT alone; `adjustment_factor_for_cit_receipts` -> 0.80 so CIT collects 4.5% of GDP. Payroll `tau_payroll` left at 0 (UIF/SDL are small and entangled with the modelled pension system; flagged as a candidate refinement).
- Set `g_y_annual` 0.006 -> 0.014, a forward-looking long-run productivity growth. On the balanced path GDP grows at `g_y + g_n`, so with the model's `g_n ≈ 0.42%` this reproduces the ~1.8% medium-term growth the IMF and National Treasury assume — the growth path the 0.765 debt-stabilisation anchor is built on. The stagnant realized 0.6% would be internally inconsistent with a stabilising debt target (it is the IMF's pessimistic, debt-drifts-up scenario).
- Re-anchored the `r_gov` level to South Africa's actual borrowing cost. Kept the LMW slope (`r_gov_scale = 0.24485`, the pass-through elasticity) but re-anchored the level `mu_d` (0.0338 -> 0.0254) so the steady-state `r_gov` equals SA's effective real rate on the debt stock (~3.7% = the Budget Review's 7.1% nominal debt-service/gross-debt, deflated), rather than the LMW cross-country nominal-USD intercept that over-predicted it by ~0.6pp. `r_gov_shift` -> -0.0488 (the re-anchored level plus the premium recentering). Together with the `g_y` change this brings the model's interest-growth differential in line with South Africa's, so the debt target no longer demands an implausibly austere primary balance.
- Set `alpha_G` 0.233 -> 0.19, pinning government consumption to the level consistent with the steady-state debt target (primary spending = revenue minus the debt-stabilising primary surplus). This sits just below SA's actual ~0.21 because the steady state is the sustainable, post-consolidation state; see the new fiscal-consistency section of the macro chapter.
- Froze the curated macro parameters against live-API clobber: `get_macro_params` now refreshes only `gamma` (ILOSTAT) and `alpha_T` (IMF GFS). `g_y_annual`, `alpha_G`, the debt block, and the `r_gov` wedge are curated packaged values and are no longer pulled/recomputed (a refresh would restore a raw data figure or the LMW intercept and break the calibration's internal consistency).
- Retuned the packaged steady-state initial guesses (`initial_guess_r_SS` -> 0.0482, `initial_guess_TR_SS` -> 0.0489, `initial_guess_factor_SS` -> 124684). The calibrated steady state matches South African data instrument by instrument — PIT 10.1% of GDP, CIT 4.5%, consumption tax 9.5%, debt-to-GDP exactly 0.765, foreign debt share 0.25, `r_gov` 3.7% real, growth 1.8% — with `factor` within 0.7% of the mean-income anchor. The full example (baseline + CIT reform, transition paths) runs end to end.

### Fixed

- Removed a spurious 20% bequest tax (`tau_bq` 0.2 -> 0) that was silently collecting ~3.9% of GDP — South Africa's estate duty is negligible (~0.1% of GDP), and the model docs already stated bequest tax was zero (a doc/JSON drift). Together with the previously over-collecting flat PIT, this stray revenue had masked a structural spending-above-revenue inconsistency in the fiscal block, which the corrected taxes exposed and the recalibration above reconciles.

## [0.2.0] - 2026-06-03 12:00:00

### Changed

- Migrated the project from conda to uv. Install with `uv sync --extra dev`; `pyproject.toml` is the single source of truth for dependencies and `uv.lock` pins exact versions.
- CI uses `astral-sh/setup-uv`, and ruff replaces black for formatting and linting (`check_format.yml` -> `check_ruff.yml`).
- Updated the README, `AGENTS.md`, and the Makefile to the uv workflow.

### Removed

- `setup.py`, `environment.yml`, `pytest.ini`, and `MANIFEST.in` (their settings moved into `pyproject.toml`).

## [0.1.1] - 2026-04-20 13:00:00

### Added

- Documentation updates, [PR #113](https://github.com/EAPD-DRB/OG-ZAF/pull/106)

## [0.1.0] - 2026-04-14 22:00:00

### Added

- Adds `use_api` flag to `Calibration` object to gate API use [PR #106](https://github.com/EAPD-DRB/OG-ZAF/pull/106)
- Fixes broken IMF data link in `macro_params.py` [PR #111](https://github.com/EAPD-DRB/OG-ZAF/pull/111)
- Fixed typos in tags in the `README.md` and documentation `intro.md` and other documentation updates [PR #107](https://github.com/EAPD-DRB/OG-ZAF/pull/107), [PR #108](https://github.com/EAPD-DRB/OG-ZAF/pull/108), [PR #109](https://github.com/EAPD-DRB/OG-ZAF/pull/109)

## [0.0.11] - 2026-03-31 22:00:00

### Added

- Adds `AGENTS.md` and `CLAUDE.md` from [PR #105](https://github.com/EAPD-DRB/OG-ZAF/pull/105)
- Fixed typos in tags in the `README.md` and documentation `intro.md`

## [0.0.10] - 2026-03-30 22:00:00

### Added

- Updates the `jupyter-book` package pin to be `jupyter-book<2.0.0` in `environment.yml`, `deploy_docs.yml`, and `docs_check.yml`. The newer packages of `jupyter-book>=2.0.0` have no support for many Sphinx functions that we use in our Jupyter Book (see [this issue](https://github.com/jupyter-book/mystmd/issues/1259)).
- Updates the Copyright year to 2026 in `_config.yml`
- Removes the UBI chapter from `_toc.yml`
- Adds a function to `utils.rst`
- Updates the metadata in `earnings.md` and `exogenous_parameters.md`
- Small typographical updates in `demographics.md`

## [0.0.9] - 2025-08-15 21:00:00

### Added

- Updates for Python 3.13 compatibility
- Removes the deprecated `initial_guess_w_SS` parameter from the default parameters file

## [0.0.8] - 2025-04-28 12:00:00

### Added

- Updates `environment.yml` and `setup.py` to pin to version `marshmallow<4.0.0`

## [0.0.7] - 2025-03-12 4:00:00

### Added

- Updated calibration of the `io_matrix` to use the [2019 SAM file from UNU Wider](https://www.wider.unu.edu/sites/default/files/Publications/Technical-note/PDF/tn2023-1-2019-SAM-South-Africa-occupational-capital-stock-detail.pdf)
- Updated values for `alpha_T` and `alpha_G`
- Various updates to documentation and unit tests.

## [0.0.6] - 2025-03-03 1:00:00

### Added

- Updated scripts in `calibrate.py` and `macro_params.py` with boolean `update_from_api`.

## [0.0.5] - 2025-02-11 23:00:00

### Added

- Updated Python 3.12 in GH Actions
- Replaced miniforge and mambaforge with miniconda and "latest" in `deploy_docs.yml` and `docs_check.yml`
- Added `pip install setuptools` to `publish_to_pypi.yml` GH Action
- Updated Python 3.11 and 3.12 in `README.md`
- Update documentation with "UN Tutorial" section


## [0.0.4] - 2024-12-07 12:00:00

### Added

- Tests on Python 3.12
- Updates symbol for local currency units in `constants.py`


## [0.0.3] - 2024-07-26 12:00:00

### Added

- Updates the `.gitignore` to ignore any `un_api_token.txt` files
- Adds a reference to `OGZAF_references.bib`
- Changes all references to OG-USA in the documentation to OG-ZAF
- Updates `demographics.md` documentation to include better South Africa images, although we still need to finish the last three population images


## [0.0.2] - 2024-06-18 12:00:00

### Added

- Updates to `Calibration` to work with OG-Core 0.11.10
- Removal of unused lines of code throughout the package


## [0.0.1] - 2023-10-30 8:00:00

### Added

- Adds CI tests `check_format.yml`, `build_and_test.yml`, `docs_build.yml`, `deploy_docs.yml`, `publish_to_pypi.yml`
- Adds `/tests/` directory with test files in it
- Added `Makefile`
- Updated `environment.yml` and `setup.py`
- Updated `pyproject.toml`
- Updates the `README.md`

## [0.0.0] - 2022-10-15 12:00:00

### Added

- This version is a pre-release alpha. The example run script OG-ZAF/examples/run_og_zaf.py runs, but the model is not currently calibrated to represent the South African economy and population.


[0.2.0]: https://github.com/EAPD-DRB/OG-ZAF/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/EAPD-DRB/OG-ZAF/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/EAPD-DRB/OG-ZAF/compare/v0.0.11...v0.1.0
[0.0.11]: https://github.com/EAPD-DRB/OG-ZAF/compare/v0.0.10...v0.0.11
[0.0.10]: https://github.com/EAPD-DRB/OG-ZAF/compare/v0.0.9...v0.0.10
[0.0.9]: https://github.com/EAPD-DRB/OG-ZAF/compare/v0.0.8...v0.0.9
[0.0.8]: https://github.com/EAPD-DRB/OG-ZAF/compare/v0.0.7...v0.0.8
[0.0.7]: https://github.com/EAPD-DRB/OG-ZAF/compare/v0.0.6...v0.0.7
[0.0.6]: https://github.com/EAPD-DRB/OG-ZAF/compare/v0.0.5...v0.0.6
[0.0.5]: https://github.com/EAPD-DRB/OG-ZAF/compare/v0.0.4...v0.0.5
[0.0.4]: https://github.com/EAPD-DRB/OG-ZAF/compare/v0.0.3...v0.0.4
[0.0.3]: https://github.com/EAPD-DRB/OG-ZAF/compare/v0.0.2...v0.0.3
[0.0.2]: https://github.com/EAPD-DRB/OG-ZAF/compare/v0.0.1...v0.0.2
[0.0.1]: https://github.com/EAPD-DRB/OG-ZAF/compare/v0.0.0...v0.0.1
