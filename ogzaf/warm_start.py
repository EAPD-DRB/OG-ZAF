"""
Warm-start the steady-state solve from a previously solved state.

OG-Core cold-starts every steady-state solve from constants (household savings
and labour at fixed values, the outer unknowns from the ``initial_guess_*``
parameters). That seed can sit far from the solution, and the single
constants OG-Core accepts for ``b`` and ``n`` cannot carry the shape of a
solved distribution. This module seeds the solver from a state that has
already solved: the household matrices and every outer unknown together, so
they are mutually consistent, with ``factor`` passed directly.

``b`` and ``n`` are in model units, so a seed stays valid across modest
parameter moves. Regenerate it with ``python examples/save_warm_start.py``
after any large recalibration.

Ported from OG-KOR's ogkor/warm_start.py (itself from OG-JPN).
"""

import os
import pickle

import numpy as np

import ogcore.SS as SS
from ogcore import firm

CUR_PATH = os.path.abspath(os.path.dirname(__file__))
SEED_PATH = os.path.join(CUR_PATH, "data", "ss_warm_start.pkl")


def save_seed(ss_vars, path=SEED_PATH):
    """
    Store the parts of a solved steady state needed to warm-start another.

    Args:
        ss_vars (dict): a solved ``SS_vars.pkl``
        path (str): where to write the seed

    Returns:
        str: the path written
    """

    def scalar(x):
        return float(np.asarray(x).sum()) if np.ndim(x) else float(x)

    seed = {
        "b": np.asarray(ss_vars["b_sp1"], dtype=float),
        "n": np.asarray(ss_vars["n"], dtype=float),
        "r": scalar(ss_vars["r"]),
        "r_p": scalar(ss_vars["r_p"]),
        "TR": scalar(ss_vars["TR"]),
        "factor": scalar(ss_vars["factor"]),
        "BQ": np.asarray(ss_vars["BQ"], dtype=float).ravel(),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(seed, fh)
    return path


def enable(path=SEED_PATH):
    """
    Patch ``SS.SS_initial_guesses`` to start from the stored seed.

    Call once before solving. Does nothing if no seed is stored, so a fresh
    checkout still runs (slowly) rather than failing.

    Args:
        path (str): seed written by :func:`save_seed`

    Returns:
        bool: True if a warm start is now active
    """
    if not os.path.exists(path):
        return False
    with open(path, "rb") as fh:
        seed = pickle.load(fh)

    original = SS.SS_initial_guesses

    def warm(p, b_val=None, n_val=None, r_tr_scalars=[1.0, 1.0]):
        if seed["b"].shape != (p.S, p.J) or seed["BQ"].size != p.J:
            # Seed built for different model dimensions; fall back rather than
            # hand the solver a wrongly-shaped guess.
            return original(p, b_val, n_val, r_tr_scalars)

        # Scale by the retry-ladder factors so ogcore's DEV_FACTOR_LIST can
        # still explore around the warm point if the first attempt fails.
        r_g = r_tr_scalars[0] * seed["r"]
        r_p_g = r_tr_scalars[0] * seed["r_p"]
        TR_g = r_tr_scalars[1] * seed["TR"]
        Y_g = TR_g / p.alpha_T[-1]
        BQ_g = seed["BQ"] * r_tr_scalars[1]
        BQ_items = [BQ_g.sum()] if p.use_zeta else list(BQ_g)
        w_g = firm.get_w_from_r(r_g, p, "SS")

        guesses = (
            [r_p_g, r_g, w_g] + list(np.ones(p.M)) + [Y_g] + BQ_items + [TR_g]
        )
        if p.baseline:
            # Passed directly: as a parameter it can exceed the 500,000 cap.
            guesses = guesses + [seed["factor"]]
        return guesses, seed["b"].copy(), seed["n"].copy()

    SS.SS_initial_guesses = warm
    return True
