"""
Steady-state tuning script for OG-ZAF calibration work.

Loads the packaged JSON plus an optional overrides file, solves the steady
state with the same worker pool as examples/run_og_zaf.py, and prints the
calibration dashboard: each moment, model against the data target in
ogzaf/targets.py.

It sets dials and nothing more. Fold settled values into the packaged JSON,
commit, re-install, and take every reported number from run_og_zaf.py.

    python examples/tune_ss.py [overrides.json] [--label NAME] [--cold]
"""

import argparse
import importlib.resources
import json
import multiprocessing
import os
import time

import numpy as np
from distributed import Client
from ogcore.execute import runner
from ogcore.parameters import Specifications
from ogcore.utils import safe_read_pickle

from ogzaf import warm_start
from ogzaf.targets import TARGETS, ss_moments


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("overrides", nargs="?", default=None)
    ap.add_argument("--label", default="tune")
    ap.add_argument(
        "--cold", action="store_true", help="ignore the warm-start seed"
    )
    args = ap.parse_args()

    num_workers = min(multiprocessing.cpu_count(), 7)
    client = Client(
        n_workers=num_workers, processes=True, threads_per_worker=1
    )

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    out = os.path.join(cur_dir, "OG-ZAF-Tune", args.label)
    p = Specifications(
        baseline=True,
        num_workers=num_workers,
        baseline_dir=out,
        output_base=out,
    )
    with importlib.resources.open_text(
        "ogzaf", "ogzaf_default_parameters.json"
    ) as file:
        p.update_specifications(json.load(file))
    overrides = {}
    if args.overrides:
        with open(args.overrides) as f:
            overrides = json.load(f)
        p.update_specifications(overrides)
    print(f"label {args.label}; overrides {json.dumps(overrides)}")

    warm = False if args.cold else warm_start.enable()
    print(f"warm start: {warm}")
    t0 = time.time()
    runner(p, time_path=False, client=client)
    secs = time.time() - t0
    client.close()

    ss = safe_read_pickle(os.path.join(out, "SS", "SS_vars.pkl"))
    model = ss_moments(ss)
    rc = float(np.max(np.abs(ss["resource_constraint_error"])))
    print(
        f"\nSS solved in {secs:.0f} s; max |RC error| {rc:.2e}; "
        f"r {float(ss['r']):.4f}; r_gov {float(ss['r_gov']):.4f}; "
        f"factor {float(ss['factor']):.5g}"
    )
    print(f"{'moment':18s} {'model':>8s} {'data':>8s} {'gap':>8s}  source")
    for k, (target, src) in TARGETS.items():
        v = model.get(k, np.nan)
        print(f"{k:18s} {v:8.4f} {target:8.4f} {v - target:+8.4f}  {src}")
    extra = sorted(set(model) - set(TARGETS))
    for k in extra:
        print(f"{k:18s} {model[k]:8.4f}")
    with open(os.path.join(out, "dashboard.json"), "w") as f:
        json.dump(
            {
                "model": model,
                "rc": rc,
                "seconds": secs,
                "warm": warm,
                "overrides": overrides,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
