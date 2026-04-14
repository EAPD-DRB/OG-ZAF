import multiprocessing
from distributed import Client
import os
import json
import importlib.resources
from ogzaf.calibrate import Calibration
from ogcore.parameters import Specifications
from ogcore.utils import params_to_json


def main():
    # Define parameters to use for multiprocessing
    num_workers = min(multiprocessing.cpu_count(), 7)
    client = Client(n_workers=num_workers, threads_per_worker=1)
    print("Number of workers = ", num_workers)

    # Directories to save data
    CUR_DIR = os.path.dirname(os.path.realpath(__file__))

    # Set up baseline parameterization
    p = Specifications(
        baseline=True,
        num_workers=num_workers,
    )
    # Update parameters for baseline from default json file
    with importlib.resources.open_text(
        "ogzaf", "ogzaf_default_parameters.json"
    ) as file:
        defaults = json.load(file)
    p.update_specifications(defaults)
    c = Calibration(
        p,
        update_from_api=True,
        client=client,
    )
    d = c.get_dict()
    # update format, keys for some params
    d["g_y_annual"] = d.pop("g_y")
    d["gamma"] = [d["gamma"]]
    d["alpha_T"] = [d["alpha_T"]]
    d["alpha_G"] = [d["alpha_G"]]
    d["alpha_I"] = [d["alpha_I"]]
    d["r_gov_scale"] = [d["r_gov_scale"]]
    d["r_gov_shift"] = [d["r_gov_shift"]]
    d.pop("taxcalc_version", None)
    # update parameters
    p.update_specifications(d)
    # save to json file
    params_to_json(p, os.path.join(CUR_DIR, "ogzaf_default_parameters.json"))


if __name__ == "__main__":
    main()