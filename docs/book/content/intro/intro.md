(Chap_Intro)=
# OG-ZAF

| | |
| --- | --- |
| Org | [![United Nations DESA](https://img.shields.io/badge/United%20Nations%20DESA-blue)](https://www.un.org/en/desa) [![PSL cataloged](https://img.shields.io/badge/PSL-cataloged-a0a0a0.svg)](https://www.PSLmodels.org) [![OS License: CC0-1.0](https://img.shields.io/badge/OS%20License-CC0%201.0-yellow)](https://github.com/EAPD-DRB/OG-ZAF/blob/main/LICENSE) |
| Package | [![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-31111/) [![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3129/) [![PyPI Latest Release](https://img.shields.io/pypi/v/ogzaf.svg)](https://pypi.org/project/ogzaf/) [![PyPI Downloads](https://img.shields.io/pypi/dm/ogzaf.svg?label=PyPI%20downloads)](https://pypi.org/project/ogzaf/) |
| Testing | ![example event parameter](https://github.com/EAPD-DRB/OG-ZAF/actions/workflows/build_and_test.yml/badge.svg?branch=main) ![example event parameter](https://github.com/EAPD-DRB/OG-ZAF/actions/workflows/deploy_docs.yml/badge.svg?branch=main) ![example event parameter](https://github.com/EAPD-DRB/OG-ZAF/actions/workflows/check_format.yml/badge.svg?branch=main) [![Codecov](https://codecov.io/gh/EAPD-DRB/OG-ZAF/branch/main/graph/badge.svg)](https://codecov.io/gh/EAPD-DRB/OG-ZAF) |

[`OG-ZAF`](https://github.com/EAPD-DRB/OG-ZAF) is a package that provides code and data to calibrate an overlapping-generations (OG) model to the economy of South Africa (ZAF), the code of which is hosted on GitHub at https://github.com/EAPD-DRB/OG-ZAF. `OG-ZAF` uses as a dependency the [`OG-Core`](https://pslmodels.github.io/OG-Core/) package, which contains the core theory and logic of a general OG model. The `OG-ZAF` calibration package and the `OG-Core` theory and logic make the model that allows for dynamic general equilibrium analysis of federal fiscal policy in South Africa. The model output focuses on changes in macroeconomic aggregates (GDP, investment, consumption), wages, interest rates, and the stream of tax revenues over time. This documentation of the `OG-ZAF` package contains the following major sections, which are regularly updated.

* Contributing to `OG-ZAF`
* `OG-ZAF` API
* Calibration
* References
* Citations of `OG-ZAF`


(Sec_CoreMaintainers)=
## Core Maintainers

Marcelo LaFleur (GitHub handle [@SeaCelo](https://github.com/SeaCelo)) and [Richard W. Evans](https://sites.google.com/site/rickecon/) (GitHub handle [@rickecon](https://github.com/rickecon)) are the core maintainers of `OG-ZAF`. If you have questions about or contributions to the model or repository, please submit a GitHub "Issue" described in the {ref}`Sec_GitHubIssue` subsection or "Pull Request" as described in the {ref}`Sec_GitHubPR` subsection of the {ref}`Sec_Workflow` section of the `OG-ZAF` {ref}`Chap_ContribGuide`.


(Sec_Disclaimer)=
## Disclaimer

The model is continuously under development. Users will be notified through [closed PR threads](https://github.com/EAPD-DRB/OG-ZAF/pulls?q=is%3Apr+is%3Aclosed) and through the [release notes](https://github.com/EAPD-DRB/OG-ZAF/releases) what changes have been implemented. The package will have released versions, which will be checked against existing code prior to release. Stay tuned for an upcoming release!


(Sec_CitingOGZAF)=
## Citing OG-ZAF

`OG-ZAF` (Version #.#.#)[Source code], https://github.com/EAPD-DRB/OG-ZAF
