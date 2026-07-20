"""Tests for the multi-industry parameter overlay.

The multisector JSON is an overlay: it carries ONLY the parameters the
multi-industry calibration changes, and must be loaded AFTER the
single-industry base (two update_specifications calls — see
examples/run_og_zaf_multiple_industry.py). Loaded alone, every omitted
parameter would silently fall back to OG-Core defaults. These tests pin
that contract.
"""

import json
from importlib.resources import files

import numpy as np
from ogcore.parameters import Specifications

# The complete set of parameters the multi-industry calibration changes.
# If a key is added here it must be a genuine multi-industry delta; anything
# economy-wide belongs in the single-industry base and is inherited at load.
OVERLAY_KEYS = {
    "M",
    "I",
    "alpha_c",
    "io_matrix",
    "c_min",
    "gamma",
    "epsilon",
    "gamma_g",
    "Z",
    "cit_rate",
    "tau_c",
    "chi_b",
    "chi_n",
    "nu",
}


def load_json(name):
    content = files("ogzaf").joinpath(name).read_text(encoding="utf-8")
    return json.loads(content)


def test_overlay_contains_only_multisector_keys():
    overlay = load_json("ogzaf_default_parameters_multisector.json")
    assert set(overlay.keys()) == OVERLAY_KEYS


def test_two_step_load():
    p = Specifications()
    p.update_specifications(load_json("ogzaf_default_parameters.json"))
    p.update_specifications(
        load_json("ogzaf_default_parameters_multisector.json")
    )
    # multi-industry dimensions applied
    assert p.M == 8
    assert p.I == 5
    assert len(p.gamma) == 8
    assert np.isclose(np.sum(p.alpha_c), 1.0)
    # base-only values survive the overlay (inherited, not clobbered):
    # the curated PIT block from the single-industry calibration
    assert np.allclose(
        p.income_tax_filer[0], [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    )
    assert p.tax_func_type == "GS"
