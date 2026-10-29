# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Requires ogcore >= 0.22.0. `tau_payroll` and `alpha_db` are T x J arrays; the transition uses Anderson acceleration with `nu = 0.2`.
- Recalibrated the single-industry baseline to South African data. Each value is an effective rate or a measured ratio; sources are in the calibration chapters and `ogzaf/targets.py`.
  - Taxes: progressive Gouveia-Strauss income tax (`tax_func_type` linear -> GS, `etr_params` 0.22 -> [0.464, 1.39288, 1.2558e-8] in 2025) collecting 10.15% of GDP in 2025, with `phi2` indexed to model average income so brackets move with incomes (rising to 1.716e-8); bottom two income groups outside the income tax net (`income_tax_filer`). `tau_c` 0.15 -> 0.1586 (all consumption taxes net of SACU payments, 8.92% of GDP). `adjustment_factor_for_cit_receipts` 0.3 -> 0.7385 (CIT plus dividends tax, 5.02%). `tau_payroll` 0 -> 0.012681 (SDL + UIF, 0.67%). Property taxes `p_wealth` 0 -> 0.0073 with `h_wealth` 0.1 -> 1 (municipal rates, transfer duty and securities transfer tax, 1.61%). `tau_bq` 0.2 -> 0.00995 (estate duty and donations tax, 0.08%). Every tax rate is tuned to 2025 collections.
  - Production and households: `delta_annual` 0.05 -> 0.0405 (private consumption of fixed capital over private capital). `beta_annual` 0.96 -> 0.8706, set so private K/Y = 2.80 (PWT 11.0 less government capital); the implied real return of 11.0% is close to PWT's 12.0%.
  - Growth: `g_y_annual` 0.0059 -> 0.014, PWT output per worker growth 1994-2019 and 2000-2019. `Z` 1 -> a path from 1.083 in 2025 to 1 in the steady state (`ogzaf/productivity.py`): output per hour grows 0.4% a year in 2026 and about 1.1% in the 2030s, the average of a rule built on South Africa's recent growth and cross-country persistence (Pritchett and Summers 2014) and the OECD long-term baseline capped at 1.44%.
  - Open economy: `world_int_rate_annual` 0.04 -> 0.063 (4% plus the Treasury's 2.26-point sovereign premium). `zeta_K` 0.9 -> 0.16 (Chinn-Ito); the steady state's foreign-owned capital share, 21%, matches the gross IIP measure. `zeta_D` and `initial_foreign_debt_ratio` 0.162 -> 0.25 (non-resident share of domestic bonds).
  - Debt: `initial_debt_ratio` 0.842 -> 0.770 (gross loan debt at the end of 2024/25, MTBPS 2026). `debt_ratio_ss` 1.2 -> 0.765 (Treasury path, 2029/30). `r_gov_shift` -0.0338 -> -0.0334 so the real rate on debt is the Treasury's effective 3.7%, with a debt-elastic premium centred on the target (`r_gov_DY` 0 -> -0.0612, `r_gov_DY2` 0 -> 0.04).
  - Spending: `alpha_I` 0.0003 -> 0.0242 (general government investment). `alpha_G` 0.233 -> [0.1896, 0.1866, 0.1900]: the first two years on the Treasury's primary balance path, then the debt-stabilising level. `tG1` 20 -> 5: the budget rule takes over in 2030, after the MTBPS window.
  - Initial wealth: `initial_wealth_ratio` unset -> 2.7907, so private capital is 2.80 times GDP in 2025.
- Demographics use UN data through 2098 (`DEMOG_FINAL_DATA_YEAR`), so long-run population growth `g_n_ss` is 0.42% -> -0.03%, and fertility and mortality differ by lifetime income group with South Africa's measured gradients.
- `get_macro_params` refreshes only `gamma` and `alpha_T` on a live update; the curated values above are not overwritten.

### Added

- `ogzaf/targets.py`, `examples/tune_ss.py`: the steady-state dashboard and tuning script.
- `examples/validate_fiscal_path.py`: scores the baseline transition against the Treasury's debt, primary balance, revenue and public investment, the 2025 revenue targets, and the productivity path.
- `ogzaf/productivity.py`: the baseline productivity path, its sources and the function that builds `Z`.
- `ogzaf/warm_start.py`, `examples/save_warm_start.py` and a packaged seed: the steady state starts from the last calibrated solution.
- `python -m ogzaf.update_demographics`: regenerates only the demographic parameters.

### Fixed

- Fixed the installation instructions. Removed the stale PyPI install path from the `README.md` (on older Python versions `pip install ogzaf` silently resolves a years-old release) and replaced it with two supported paths written as per-platform copy-paste blocks: the OG model family's universal installer (`install.sh --repo og-zaf`, from PSLmodels/OG-Core) and a manual install (install `uv`, clone, `uv run python examples/run_og_zaf.py`), plus a "What happens" note covering the environment auto-creation, the expected runtime, the output location, and the UN API token prompt. Updated the contributor guide from the conda-era workflow (Anaconda install, `conda env create` against the `environment.yml` removed in 0.2.0) to the `uv` workflow, corrected the test command to this repository's markers (`pytest -m "not local"`), and fixed stale `master`-branch and OG-USA references. Updated the UN tutorial's getting-started page and 3-period exercise solutions from the same conda instructions to `uv sync --extra dev` and `uv run` (including the Jupyter notebook steps), and fixed the getting-started page describing OG-ZAF as a model of "Philippine fiscal policy" (an OG-PHL copy-paste leftover).

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
