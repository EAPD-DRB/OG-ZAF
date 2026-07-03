# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Enabled a debt-elastic sovereign premium in `ogzaf_default_parameters.json`, the crowding-out-via-risk channel that OG-Core's defaults leave off (`r_gov_DY = r_gov_DY2 = 0`). It is a *centered* convex form, `r_gov_DY2 * (D/Y - 0.765)^2`, flat at the 0.765 debt target and steepening only as debt moves away — `r_gov_DY2 = 0.04`, `r_gov_DY = -0.0612`, with `r_gov_shift` recentered from -0.0338 to -0.0572 so the premium is exactly zero at the target and the steady state is unchanged. This matches South African experience (stable spreads through the 60s percent-of-GDP debt range, a blowout in 2020, compression again in 2025 as consolidation took hold) and mirrors the same channel in OG-PHL. See the macro calibration chapter for lineage and sources.

### Changed

- Recalibrated the open-economy and debt block to South African official data, in `ogzaf_default_parameters.json`. World interest rate `world_int_rate_annual` 0.04 -> 0.063: a 4% global risk-free rate plus the National Treasury's own 2.26-percentage-point sovereign risk premium (2026 Budget Review, Chapter 7). Capital openness `zeta_K` 0.2 -> 0.16, pinned to the normalized Chinn-Ito index for South Africa (0.1626, 2022 update). Initial debt `initial_debt_ratio` 0.842 -> 0.789 and steady-state debt target `debt_ratio_ss` 0.84 -> 0.765, from the same Budget Review (gross loan debt stabilizes at 78.9% of GDP in 2025/26 and declines to 76.5% by 2028/29) — the National Treasury's national-government gross loan debt replaces the World Bank QPSD general-government series, which measures a wider perimeter. `initial_foreign_debt_ratio` and `zeta_D` stay at 0.25, now cited to the Budget Review's 25% foreign participation in the domestic bond market (2025).
- Macro parameters with documented point-in-time sources are no longer clobbered by API pulls: `get_macro_params` no longer refreshes the debt block (`initial_debt_ratio`, `initial_foreign_debt_ratio`, `zeta_D`) from the World Bank QPSD, nor recomputes the `r_gov` wedge (`r_gov_scale`, `r_gov_shift`) — the Li-Magud-Werner-Witte inversion is deterministic, and a recompute would silently undo the sovereign-premium recentering. Live pulls remain for `g_y_annual` (World Bank), `gamma` (ILOSTAT), and `alpha_T`/`alpha_G` (IMF GFS), whose documented sources are those APIs.
- Retuned the packaged steady-state initial guesses to the recalibrated economy (`initial_guess_r_SS` 0.04 -> 0.048506, `initial_guess_TR_SS` 0.042 -> 0.051713, `initial_guess_factor_SS` 6766 -> 114805.11). The old factor seed pointed at a steady state 17x away; from the retuned guesses the steady state solves in seconds and the full example (baseline + CIT-reform transition paths) completes in about 8 minutes. The recalibrated baseline it encodes: r = 0.0485, r_gov = 0.0456, debt-to-GDP exactly at the 0.765 target, foreign share of government debt 0.25, and a mildly negative foreign capital share (-4.4%), consistent with South Africa's positive net international investment position.

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
