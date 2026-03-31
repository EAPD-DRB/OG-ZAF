## Project: OG-ZAF

OG-ZAF is a South Africa country calibration of the OG-Core overlapping-generations model of demographics and fiscal policy.

## Environment

- Conda environment: `ogzaf-dev`

## Python formatting

- Run `black` on all touched `.py` files before staging and pushing.
- Do not run `black` on non-Python files (e.g. README.md will fail to parse).
- Re-run tests after formatting to confirm nothing broke.
- Format command: `conda run -n ogzaf-dev python -m black <files>`

## Testing

- Full suite: `conda run -n ogzaf-dev python -m pytest tests/ -q`
- Targeted: `conda run -n ogzaf-dev python -m pytest tests/test_calibrate.py tests/test_input_output.py tests/test_macro_params.py -q`

## Repo conventions

- The packaged JSON default parameters are the standard baseline input for offline/default runs.
- Calibration-related changes can affect macro parameters, demographics, earnings distribution, and industry I/O behavior.
- Changes in calibration or data-source behavior should be validated with targeted tests and, where feasible, the relevant example flows.
