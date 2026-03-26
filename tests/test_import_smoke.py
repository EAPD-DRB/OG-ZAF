"""
Import smoke tests for installed package usage.
"""


def test_import_smoke():
    import ogzaf
    from ogzaf import macro_params
    from ogzaf.calibrate import Calibration

    assert ogzaf is not None
    assert macro_params is not None
    assert Calibration is not None
