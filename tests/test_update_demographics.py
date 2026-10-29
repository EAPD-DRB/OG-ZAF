"""Tests for ogzaf/update_demographics.py, without downloading any data."""

import json
import shutil
from importlib.resources import files

import numpy as np

from ogzaf import update_demographics


def test_to_json():
    assert update_demographics._to_json(np.array([1.0, 2.0])) == [1.0, 2.0]
    assert update_demographics._to_json(np.float64(0.5)) == 0.5
    assert isinstance(update_demographics._to_json(np.int64(3)), int)
    assert update_demographics._to_json("text") == "text"


def test_main_updates_only_demographics(tmp_path, monkeypatch):
    name = update_demographics.PARAMS_FILE
    shutil.copy(files("ogzaf").joinpath(name), tmp_path / name)
    before = json.loads((tmp_path / name).read_text(encoding="utf-8"))

    def fake_demographics(p):
        demog = {"g_n_ss": np.float64(0.001), "new_key": np.array([1, 2])}
        return demog, np.ones((p.S, p.J))

    monkeypatch.setattr(update_demographics, "files", lambda _: tmp_path)
    monkeypatch.setattr(
        update_demographics, "get_demographics_and_e", fake_demographics
    )
    update_demographics.main()

    after = json.loads((tmp_path / name).read_text(encoding="utf-8"))
    assert after["g_n_ss"] == 0.001
    assert after["new_key"] == [1, 2]
    assert np.shape(after["e"]) == (80, 7)
    unchanged = set(before) - {"g_n_ss", "e"}
    assert all(after[k] == before[k] for k in unchanged)
