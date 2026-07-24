# OG-ZAF

| | |
| --- | --- |
| Org | [![United Nations DESA](https://img.shields.io/badge/United%20Nations%20DESA-blue)](https://www.un.org/en/desa) [![PSL cataloged](https://img.shields.io/badge/PSL-cataloged-a0a0a0.svg)](https://www.PSLmodels.org) [![OS License: CC0-1.0](https://img.shields.io/badge/OS%20License-CC0%201.0-yellow)](https://github.com/EAPD-DRB/OG-ZAF/blob/main/LICENSE) [![Jupyter Book Badge](https://raw.githubusercontent.com/jupyter-book/jupyter-book/next/docs/media/images/badge.svg)](https://eapd-drb.github.io/OG-ZAF/) |
| Package | [![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3129/) [![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/release/python-31312/) [![PyPI Latest Release](https://img.shields.io/pypi/v/ogzaf.svg)](https://pypi.org/project/ogzaf/) [![PyPI Downloads](https://img.shields.io/pypi/dm/ogzaf.svg?label=PyPI%20downloads)](https://pypi.org/project/ogzaf/) [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff) |
| Testing | ![example event parameter](https://github.com/EAPD-DRB/OG-ZAF/actions/workflows/build_and_test.yml/badge.svg?branch=main) ![example event parameter](https://github.com/EAPD-DRB/OG-ZAF/actions/workflows/deploy_docs.yml/badge.svg?branch=main) ![example event parameter](https://github.com/EAPD-DRB/OG-ZAF/actions/workflows/check_ruff.yml/badge.svg?branch=main) [![Codecov](https://codecov.io/gh/EAPD-DRB/OG-ZAF/branch/main/graph/badge.svg)](https://codecov.io/gh/EAPD-DRB/OG-ZAF) |

OG-ZAF is an overlapping-generations (OG) model that allows for dynamic general equilibrium analysis of demographics and fiscal policy for South Africa. OG-ZAF is built on the OG-Core framework. The model output includes changes in macroeconomic aggregates (GDP, investment, consumption), demographics, worker productivity, wages, interest rates, and the stream of tax revenues over time. Regularly updated documentation of the model theory--its output, and solution method--and the Python API is available at https://pslmodels.github.io/OG-Core and documentation of the specific South African calibration of the model is available at https://eapd-drb.github.io/OG-ZAF.


## Using and contributing to OG-ZAF

Install and run OG-ZAF by cloning the GitHub repository and installing the `ogzaf` package with its dependencies using [`uv`](https://docs.astral.sh/uv/), as detailed below. (Installing the `ogzaf` package from [PyPI](https://pypi.org/project/ogzaf/) with `pip` is no longer the recommended path: on older Python versions `pip` silently resolves a years-old release of the model, and the PyPI route does not pin the `ogcore` version the repository is tested against. The `uv` workflow installs the exact tested versions of every dependency, including a compatible Python interpreter.)

### Installation

There are two ways to install OG-ZAF: the easy way, using the OG model family's universal installer, and a manual install that runs the same steps one command at a time. Both start from a terminal, put the model in a new OG-ZAF folder inside your current directory (a freshly opened terminal starts in your home folder), and need git.

#### Before you start: git

On a Mac where you have never used git before, run this line first and accept the dialog that appears:

```
xcode-select --install
```

On Windows, install Git first (skip this if you already have it):

```
winget install --id Git.Git -e --source winget
```

#### The easy way: the universal installer

The OG model family has a [universal installer](https://github.com/PSLmodels/OG-Core/blob/master/scripts/QUICK_INSTALL.md) that installs the uv tool, downloads the model, builds its environment, and verifies it — for OG-ZAF or any of its sibling country models.

On macOS and Linux, paste:

```
curl -fsSL https://raw.githubusercontent.com/PSLmodels/OG-Core/master/scripts/install.sh -o og-install.sh
bash og-install.sh --repo og-zaf --yes
```

On Windows (PowerShell), paste:

```
$f = "$env:TEMP\og-install.ps1"; irm https://raw.githubusercontent.com/PSLmodels/OG-Core/master/scripts/install.ps1 -OutFile $f; powershell -ExecutionPolicy Bypass -File $f -Repo og-zaf -Yes
```

When the installer finishes, run the example. On macOS and Linux, paste:

```
source $HOME/.local/bin/env
cd OG-ZAF
uv run python examples/run_og_zaf.py
```

On Windows, open a new PowerShell window (so the just-installed tools are found) and paste:

```
cd OG-ZAF
uv run python examples/run_og_zaf.py
```

#### Manual install

The same setup as individual commands. On macOS and Linux, paste:

```
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
git clone https://github.com/EAPD-DRB/OG-ZAF.git
cd OG-ZAF
uv run python examples/run_og_zaf.py
```

The second line makes the just-installed uv available in the current terminal; from your next terminal session it is available automatically.

On Windows (PowerShell), install uv:

```
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then open a new PowerShell window (so the just-installed tools are found) and paste:

```
git clone https://github.com/EAPD-DRB/OG-ZAF.git
cd OG-ZAF
uv run python examples/run_og_zaf.py
```

#### What happens

The first `uv run` creates the project environment if it does not exist yet — downloading a compatible Python interpreter if needed and installing the exact locked dependencies — and then runs the example. Early in the run you may be asked for a UN API token; just press return, and the model reads the same population data from a public mirror and continues. A full baseline-plus-reform run takes from 35 minutes to more than two hours (well under an hour on a recent machine); when it finishes, its plots and tables are saved under `./examples/OG-ZAF-Example/` (see the list of outputs below).

### Installing for development or contributing

* Fork this repository and clone your fork to a directory on your computer.
* From the terminal, navigate to the cloned directory and run `uv sync --extra dev` to create a local `.venv` and install OG-ZAF with its development dependencies (`uv` downloads a compatible Python if you don't already have one).
* For docs/Jupyter Book work, also run `uv sync --extra dev --extra docs`.
* Run commands in the environment with `uv run <command>`, or activate it first with `source .venv/bin/activate` (macOS/Linux) or `.\.venv\Scripts\Activate.ps1` (Windows).

### Run an example of the model

* From the repository root, run the model with an example reform: `uv run python examples/run_og_zaf.py`.
* You can adjust the `./examples/run_og_zaf.py` by modifying model parameters specified in the dictionary passed to the `p.update_specifications()` calls.
* Model outputs will be saved in the following files:
  * `./examples/OG-ZAF-Example/OG-ZAF_example_plots`
    * This folder will contain a number of plots generated from OG-Core to help you visualize the output from your run
  * `./examples/OG-ZAF-Example/OG-ZAF_example_output.csv`
    * This is a summary of the percentage changes in macro variables over the first ten years and in the steady-state.
  * `./examples/OG-ZAF-Example/OUTPUT_BASELINE/model_params.pkl`
    * Model parameters used in the baseline run
    * See [`ogcore.execute.py`](https://github.com/PSLmodels/OG-Core/blob/master/ogcore/execute.py) for items in the dictionary object in this pickle file
  * `./examples/OG-ZAF-Example/OUTPUT_BASELINE/SS/SS_vars.pkl`
    * Outputs from the model steady state solution under the baseline policy
    * See [`ogcore.SS.py`](https://github.com/PSLmodels/OG-Core/blob/master/ogcore/SS.py) for what is in the dictionary object in this pickle file
  * `./examples/OG-ZAF-Example/OUTPUT_BASELINE/TPI/TPI_vars.pkl`
    * Outputs from the model timepath solution under the baseline policy
    * See [`ogcore.TPI.py`](https://github.com/PSLmodels/OG-Core/blob/master/ogcore/TPI.py) for what is in the dictionary object in this pickle file
  * An analogous set of files in the `./examples/OG-ZAF-Example/OUTPUT_REFORM` directory, which represent objects from the simulation of the reform policy

Note that, depending on your machine, a full model run (solving for the full time path equilibrium for the baseline and reform policies) can take from 35 minutes to more than two hours of compute time.

If you run into errors running the example script, please open a new issue in the OG-ZAF repo with a description of the issue and any relevant tracebacks you receive.

Once the package is installed, one can adjust parameters in the OG-Core `Specifications` object using the `Calibration` class as follows:

```
from ogcore.parameters import Specifications
from ogzaf.calibrate import Calibration
p = Specifications()
c = Calibration(p, update_from_api=True)
updated_params = c.get_dict()
p.update_specifications(updated_params)
```

## Disclaimer
The `OG-ZAF/examples/run_og_zaf.py` script is kept up to date to run with the `main` branch of this repo.

## Core Maintainers

The core maintainers of the OG-ZAF repository are:

* Marcelo LaFleur (GitHub handle: [@SeaCelo](https://github.com/SeaCelo)), Senior Economist, Department of Economic and Social Affairs (DESA), United Nations
* [Richard W. Evans](https://sites.google.com/site/rickecon/) (GitHub handle: [@rickecon](https://github.com/rickecon)), Senior Economist, Abundance Institute; President, Open Research Group, Inc.
* [Jason DeBacker](https://jasondebacker.com) (GitHub handle: [@jdebacker](https://github.com/jdebacker)), Associate Professor, University of South Carolina; Vice President of Research, Open Research Group, Inc.

## Citing OG-ZAF

OG-ZAF (Version #.#.#)[Source code], https://github.com/EAPD-DRB/OG-ZAF.
