[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-3916/)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3108/)
![example event parameter](https://github.com/EAPD-DRB/OG-ZAF/actions/workflows/check_black.yml/badge.svg?branch=main)

# OG-ZAF
OG-ZAF is an overlapping-generations (OG) model that allows for dynamic general equilibrium analysis of fiscal policy for South Africa. OG-ZAF is built on the OG-Core framework. The model output includes changes in macroeconomic aggregates (GDP, investment, consumption), wages, interest rates, and the stream of tax revenues over time. Regularly updated documentation of the model theory--its output, and solution method--and the Python API is available at https://pslmodels.github.io/OG-Core and documentation of the specific South African calibration of the model will be available soon.


## Using and contributing to OG-ZAF

* If you are installing on a Mac computer, install XCode Tools. In Terminal: `xcode-select —install`
* Download and install the appropriate [Anaconda distribution](https://www.anaconda.com/products/distribution#Downloads) of Python. Select the correct version for you platform (Windows, Intel Mac, or M1 Mac).
* In Terminal:
  * Make sure the `conda` package manager is up-to-date: `conda update conda`.
  * Make sure the Anaconda distribution of Python is up-to-date: `conda update anaconda`.
* Fork this repository and clone your fork of this repository to a directory on your computer.
* From the terminal (or Anaconda command prompt), navigate to the directory to which you cloned this repository and run `conda env create -f environment.yml`. The process of creating the `ogzaf-dev` conda environment should not take more than five minutes.
* Then, `conda activate ogzaf-dev`
* Then install by `pip install -e .`
### Run an example of the model
* Navigate to `./examples`
* Run the model with an example reform from terminal/command prompt by typing `python run_og_zaf.py`
* You can adjust the `./examples/run_og_zaf.py` by modifying model parameters specified in the dictionary passed to the `p.update_specifications()` calls.
* Model outputs will be saved in the following files:
  * `./examples/OG-ZAF_example_plots`
    * This folder will contain a number of plots generated from OG-Core to help you visualize the output from your run
  * `./examples/ogzaf_example_output.csv`
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
  * An analogous set of files in the `./examples/OUTPUT_REFORM` directory, which represent objects from the simulation of the reform policy

Note that, depending on your machine, a full model run (solving for the full time path equilibrium for the baseline and reform policies) can take from 35 minutes to more than two hours of compute time.

If you run into errors running the example script, please open a new issue in the OG-ZAF repo with a description of the issue and any relevant tracebacks you receive.

Once the package is installed, one can adjust parameters in the OG-Core `Specifications` object using the `Calibration` class as follows:

```
from ogcore.parameters import Specifications
from ogzaf.calibrate import Calibration
p = Specifications()
c = Calibration(p)
updated_params = c.get_dict()
p.update_specifications({'initial_debt_ratio': updated_params['initial_debt_ratio']})
```

## Disclaimer
The organization of this repository will be changing rapidly, but the `OG-ZAF/examples/run_og_zaf.py` script will be kept up to date to run with the master branch of this repo.

## Core Maintainers

The core maintainers of the OG-ZAF repository are:

* Marcelo LaFleur (GitHub handle: [@SeaCelo](https://github.com/SeaCelo)), Senior Economist, Department of Economic and Social Affairs (DESA), United Nations
* [Richard W. Evans](https://sites.google.com/site/rickecon/) (GitHub handle: [@rickecon](https://github.com/rickecon)), Senior Research Fellow and Director of Open Policy, Center for Growth and Opportunity at Utah State University; President, Open Research Group, Inc.

## Citing OG-ZAF

OG-ZAF (Version #.#.#)[Source code], https://github.com/EAPD-DRB/OG-ZAF
