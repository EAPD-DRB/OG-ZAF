"""
Build the OG-ZAF multi-industry (M=8, I=5) calibration and write it to the
packaged ``ogzaf_default_parameters_multisector.json``.

Calibration is a rare event. Run this ONLY to (re)generate the packaged
multi-industry parameters from the underlying data (the 2019 SASAM and the
QLFS employment series):

    uv run python -m ogzaf.create_multisector_calibration

The model itself loads the JSON; it does not call these functions at run time.
The construction tools live in ``ogzaf.input_output`` (``get_alpha_c``,
``get_io_matrix_value_added``, ``get_gamma``, ``get_Z``, ``get_employment``);
this module assembles their output, with the documented production choices,
merges it onto the single-industry base, and serializes the result.

The written file is **self-sufficient**: it is the single-industry
``ogzaf_default_parameters.json`` with the multi-industry parameters below
merged in, so a user loads it standalone (one ``update_specifications`` call)
exactly like the single-industry default -- it is not an overlay that must be
layered on the base at run time. The trade-off is that the economy-wide South
African values (demographics, preferences, the fiscal block, open-economy
dials, ``g_y``, the progressive PIT, the debt targets, and the solver seeds)
are *copied* from the base, so **regenerate this file (the command above)
whenever the single-industry base is recalibrated** -- otherwise the two
representations silently describe different economies.

Multi-industry parameters merged over the base:
  * ``M``, ``I``   - 8 production industries, 5 consumption goods
  * ``alpha_c``    - household consumption shares (SASAM)
  * ``io_matrix``  - 5x8 domestic value-added content (SASAM, Leontief)
  * ``gamma``      - per-industry private capital share: the SAM's total
                     capital shares rescaled to TOTAL_CAPITAL_SHARE, then
                     PUBLIC_CAPITAL_SHARE (0.0 for ZAF) carved out
  * ``Z``          - per-industry TFP, the Solow residual (Manufacturing = 1)
  * ``epsilon``    - 1.0 (Cobb-Douglas; OG-Core default)
  * ``gamma_g``    - PUBLIC_CAPITAL_SHARE for every industry
  * ``c_min``      - 0.0 (no subsistence floor)
  * ``cit_rate``   - 0.27 (statutory corporate income tax, as in the single
                     industry; combined with the base
                     adjustment_factor_for_cit_receipts it hits CIT ~4.5% GDP)
  * ``tau_c``      - 0.18 (effective indirect-tax rate, as in the single
                     industry)
  * ``chi_b``, ``chi_n`` - the base utility weights converted for the
                     multi-good composite-consumption units: scaled by
                     k**(sigma-1) with k = prod(alpha_c**-alpha_c), the units
                     constant OG-Core's unnormalized composite price index
                     picks up when I > 1 (see build_multisector_params)
  * ``nu``         - 0.2 TPI dampening

The multi-industry steady state solves cold from the base's shared solver seeds
(``initial_guess_*``), which this file inherits unchanged from the base.
"""

import json
import os

import numpy as np

from ogzaf import input_output as io
from ogzaf.constants import PUBLIC_CAPITAL_SHARE, TOTAL_CAPITAL_SHARE

CUR_DIR = os.path.dirname(os.path.realpath(__file__))
MULTISECTOR_PARAMS_PATH = os.path.join(
    CUR_DIR, "ogzaf_default_parameters_multisector.json"
)

# TPI dampening. The multi-industry price update is more delicate than the
# single-industry model's; 0.2 keeps the iteration comfortably inside the unit
# circle.
TPI_NU = 0.2

# Statutory corporate income tax and effective indirect-tax rates, taken from
# the single-industry calibration (PR #135) so the two representations raise
# the same revenue. cit_rate combines with the base
# adjustment_factor_for_cit_receipts; tau_c is the all-indirect effective rate.
CIT_RATE = 0.27
TAU_C = 0.18

# Common Hicks-neutral scale on the industry TFP vector. get_Z normalizes the
# numeraire (Manufacturing) to 1, which fixes the relative TFPs but leaves the
# income LEVEL a free normalization; that convention lands the eight-industry
# economy's effective aggregate TFP ~16% above the single-industry model's, so
# its solved ``factor`` (mean_income_data / model_mean_income) comes out ~23%
# below the single's 124,684. Because ``factor`` scales the incomes fed to the
# progressive PIT, the two representations must share it. The single-industry
# model owns the income level (it anchors factor to the SARS mean-income data),
# so we rescale the whole Z vector by this Hicks-neutral constant -- relative
# prices, sector shares, gamma dispersion and TFP ratios are all invariant;
# only the level moves -- calibrated (log-log root find on solved SS factor,
# beta=1.81) so the multi-industry factor matches the single's to ~1%.
Z_LEVEL_SCALE = 0.865


def build_multisector_params():
    """Compute the multi-industry parameter overlay from the packaged data.

    Returns:
        params (dict): an OG-Core ``update_specifications``-format overlay.
    """
    alpha_c = [float(v) for v in io.get_alpha_c().values()]
    io_df = io.get_io_matrix_value_added()
    n_cons, n_ind = io_df.shape
    # Per-industry capital shares: rescale the SAM's total shares to the
    # economy-wide total, then subtract public capital's share to leave the
    # private capital share gamma_m. For ZAF PUBLIC_CAPITAL_SHARE = 0, so the
    # private share equals the total.
    gamma_total = io.get_gamma(target_avg=TOTAL_CAPITAL_SHARE)
    gamma = {k: v - PUBLIC_CAPITAL_SHARE for k, v in gamma_total.items()}
    # Relative TFP from the Solow residual (numeraire = 1), then a common
    # Hicks-neutral level rescale so the income level (factor) matches the
    # single-industry model (see Z_LEVEL_SCALE).
    Z = {
        k: v * Z_LEVEL_SCALE
        for k, v in io.get_Z(gamma=gamma, gamma_g=PUBLIC_CAPITAL_SHARE).items()
    }
    # OG-Core's composite-consumption price index is unnormalized
    # (p_tilde = prod(((1+tau_c) p_i / alpha_i)**alpha_i) in
    # aggregates.get_ptilde), so I=1 -> I=5 shrinks composite-consumption
    # units by k = prod(alpha_i**-alpha_i) while chi_n and chi_b stay fixed
    # numbers set in the single-industry units. Every consumption term in the
    # household FOCs enters as MU(c)/p_tilde and scales by k**(sigma-1) under
    # that units change; the chi_b bequest term (assets, numeraire units) and
    # the chi_n disutility term do not. Scaling both weights by k**(sigma-1)
    # restores the FOCs at the same real allocation, keeping multi-industry
    # households behaviorally identical to the single-industry baseline.
    alpha_arr = np.array(alpha_c)
    k_units = float(np.prod(alpha_arr**-alpha_arr))
    with open(os.path.join(CUR_DIR, "ogzaf_default_parameters.json")) as f:
        base = json.load(f)
    chi_scale = k_units ** (base["sigma"] - 1.0)
    chi_b = [float(v) * chi_scale for v in base["chi_b"]]
    chi_n = [float(v) * chi_scale for v in base["chi_n"]]
    return {
        "M": int(n_ind),
        "I": int(n_cons),
        "alpha_c": alpha_c,
        "io_matrix": io_df.values.tolist(),
        "c_min": [0.0] * n_cons,
        "gamma": [float(v) for v in gamma.values()],
        "epsilon": [1.0] * n_ind,
        "gamma_g": [PUBLIC_CAPITAL_SHARE] * n_ind,
        "Z": [[float(v) for v in Z.values()]],
        "cit_rate": [[CIT_RATE]],
        "tau_c": [[TAU_C]],
        "chi_b": chi_b,
        "chi_n": chi_n,
        "nu": TPI_NU,
    }


def main():
    """Merge the multi-industry parameters onto the single-industry base and
    write the self-sufficient multisector default file."""
    overrides = build_multisector_params()
    with open(os.path.join(CUR_DIR, "ogzaf_default_parameters.json")) as f:
        base = json.load(f)
    params = {**base, **overrides}  # base first, multi-industry values win
    with open(MULTISECTOR_PARAMS_PATH, "w") as f:
        json.dump(params, f, indent=2)
        f.write("\n")
    print(f"Wrote {MULTISECTOR_PARAMS_PATH}")
    print(f"  self-sufficient: {len(params)} keys (base + multi-industry)")
    print(f"  M = {params['M']}, I = {params['I']}")
    print(f"  alpha_c = {np.round(params['alpha_c'], 4).tolist()}")
    print(f"  gamma   = {np.round(params['gamma'], 4).tolist()}")
    print(f"  Z       = {np.round(params['Z'][0], 4).tolist()}")
    return params


if __name__ == "__main__":
    main()
