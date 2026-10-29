"""
Regenerate the demographic objects and the earnings matrix in the packaged
JSON, leaving every other parameter untouched.

``update_baseline.py`` reruns the whole online calibration and rewrites the
file, which would also refresh the live macro parameters over curated values.
Use this script when only the demographics change (a new UN vintage, a new
data window, a new ``start_year``):

    python -m ogzaf.update_demographics

Without a UN Data Portal token ogcore reads the archived copy of the same UN
data from EAPD-DRB/Population-Data.
"""

import json
from importlib.resources import files

import numpy as np
from ogcore.parameters import Specifications

from ogzaf.calibrate import get_demographics_and_e

PARAMS_FILE = "ogzaf_default_parameters.json"


def _to_json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def main():
    path = files("ogzaf").joinpath(PARAMS_FILE)
    params = json.loads(path.read_text(encoding="utf-8"))
    p = Specifications(baseline=True)
    p.update_specifications(params)

    demog, e = get_demographics_and_e(p)
    updates = dict(demog, e=e)
    changed = [k for k in updates if k in params]
    for key, value in updates.items():
        params[key] = _to_json(value)

    with open(str(path), "w", encoding="utf-8") as f:
        f.write(json.dumps(params, indent=4) + "\n")
    print(f"Updated {len(updates)} keys in {path}: {sorted(updates)}")
    new_keys = sorted(set(updates) - set(changed))
    if new_keys:
        print(f"Keys added (not previously in the file): {new_keys}")


if __name__ == "__main__":
    main()
