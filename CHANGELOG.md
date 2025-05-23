# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


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


[0.0.8]: https://github.com/EAPD-DRB/OG-ZAF/compare/v0.0.7...v0.0.8
[0.0.7]: https://github.com/EAPD-DRB/OG-ZAF/compare/v0.0.6...v0.0.7
[0.0.6]: https://github.com/EAPD-DRB/OG-ZAF/compare/v0.0.5...v0.0.6
[0.0.5]: https://github.com/EAPD-DRB/OG-ZAF/compare/v0.0.4...v0.0.5
[0.0.4]: https://github.com/EAPD-DRB/OG-ZAF/compare/v0.0.3...v0.0.4
[0.0.3]: https://github.com/EAPD-DRB/OG-ZAF/compare/v0.0.2...v0.0.3
[0.0.2]: https://github.com/EAPD-DRB/OG-ZAF/compare/v0.0.1...v0.0.2
[0.0.1]: https://github.com/EAPD-DRB/OG-ZAF/compare/v0.0.0...v0.0.1
