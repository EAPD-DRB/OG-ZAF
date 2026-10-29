"""Tests for the steady-state warm start in ogzaf/warm_start.py."""

import json
import pickle
from importlib.resources import files

import numpy as np
import ogcore.SS as SS
import pytest
from ogcore.parameters import Specifications

from ogzaf import warm_start


@pytest.fixture(scope="module")
def p():
    params = json.loads(
        files("ogzaf")
        .joinpath("ogzaf_default_parameters.json")
        .read_text(encoding="utf-8")
    )
    spec = Specifications(baseline=True)
    spec.update_specifications(params)
    return spec


def fake_ss_vars(S=80, J=7):
    return {
        "b_sp1": np.full((S, J), 0.5),
        "n": np.full((S, J), 0.3),
        "r": 0.11,
        "r_p": 0.1,
        "TR": np.array([0.03]),
        "factor": 150000.0,
        "BQ": np.full(J, 0.01),
    }


def test_save_seed_round_trip(tmp_path):
    path = warm_start.save_seed(fake_ss_vars(), tmp_path / "seed.pkl")
    with open(path, "rb") as fh:
        seed = pickle.load(fh)
    assert seed["b"].shape == (80, 7)
    assert seed["TR"] == pytest.approx(0.03)
    assert seed["factor"] == pytest.approx(150000.0)


def test_enable_without_seed_does_nothing(tmp_path, monkeypatch):
    original = SS.SS_initial_guesses
    monkeypatch.setattr(SS, "SS_initial_guesses", original)
    assert not warm_start.enable(tmp_path / "missing.pkl")
    assert SS.SS_initial_guesses is original


def test_packaged_seed_gives_guesses(p, monkeypatch):
    monkeypatch.setattr(SS, "SS_initial_guesses", SS.SS_initial_guesses)
    assert warm_start.enable()
    guesses, b, n = SS.SS_initial_guesses(p)
    assert b.shape == (p.S, p.J)
    assert n.shape == (p.S, p.J)
    # r_p, r, w, one price per industry, Y, BQ, TR and factor
    n_bq = 1 if p.use_zeta else p.J
    assert len(guesses) == 3 + p.M + 1 + n_bq + 1 + 1
    assert np.all(np.isfinite(guesses))


def test_seed_of_other_dimensions_falls_back(p, tmp_path, monkeypatch):
    calls = []

    def original(p, b_val=None, n_val=None, r_tr_scalars=[1.0, 1.0]):
        calls.append(r_tr_scalars)
        return "cold start"

    monkeypatch.setattr(SS, "SS_initial_guesses", original)
    path = warm_start.save_seed(fake_ss_vars(S=10, J=2), tmp_path / "s.pkl")
    assert warm_start.enable(path)
    assert SS.SS_initial_guesses(p) == "cold start"
    assert calls == [[1.0, 1.0]]
