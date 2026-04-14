import multiprocessing
from distributed import Client
import os
import json
from importlib.resources import files
from ogzaf.calibrate import Calibration
from ogcore.parameters import Specifications
from ogcore.utils import params_to_json


def main():
    # Directories to save data
    CUR_DIR = os.path.dirname(os.path.realpath(__file__))

    # Set up baseline parameterization
    p = Specifications(baseline=True)
    # Update parameters for baseline from default json file
    content = (
        files("ogzaf")
        .joinpath("ogzaf_default_parameters.json")
        .read_text(encoding="utf-8")
    )
    defaults = json.loads(content)
    p.update_specifications(defaults)
    c = Calibration(
        p,
        update_from_api=True,
    )
    d = c.get_dict()
    # update parameters
    p.update_specifications(d)
    # save to json file
    params_to_json(p, os.path.join(CUR_DIR, "ogzaf_default_parameters.json"))


if __name__ == "__main__":
    main()
