"""
Tests of calibrate.py module — offline and partial-failure behavior
"""

import warnings
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from ogzaf.calibrate import Calibration


def _make_mock_p(I=1, M=1):
    """Create a minimal mock Specifications object."""
    p = MagicMock()
    p.I = I
    p.M = M
    p.E = 20
    p.S = 80
    p.T = 160
    p.start_year = 2025
    p.lambdas = np.array(
        [0.25, 0.25, 0.0625, 0.0625, 0.0625, 0.0625, 0.25]
    )
    return p


class TestOfflineMode:
    """Tests for update_from_api=False (the default)."""

    def test_single_sector_returns_identity_values(self):
        """Single-sector offline: get_dict() returns alpha_c=[1.0] and io_matrix=[[1.0]]."""
        p = _make_mock_p(I=1, M=1)
        c = Calibration(p, update_from_api=False)

        d = c.get_dict()
        assert "alpha_c" in d
        assert "io_matrix" in d
        np.testing.assert_array_equal(d["alpha_c"], np.array([1.0]))
        np.testing.assert_array_equal(d["io_matrix"], np.array([[1.0]]))

    def test_single_sector_no_demographics(self):
        """Offline mode should not include demographics or e."""
        p = _make_mock_p(I=1, M=1)
        c = Calibration(p, update_from_api=False)

        d = c.get_dict()
        assert "e" not in d
        assert "omega" not in d
        assert "omega_SS" not in d
        assert "g_n_ss" not in d

    def test_single_sector_no_macro(self):
        """Offline mode should not include macro params."""
        p = _make_mock_p(I=1, M=1)
        c = Calibration(p, update_from_api=False)

        d = c.get_dict()
        assert "g_y_annual" not in d
        assert "initial_debt_ratio" not in d

    def test_multi_sector_omits_alpha_c_and_io_matrix(self):
        """Multi-sector offline: alpha_c and io_matrix are None, omitted from get_dict()."""
        p = _make_mock_p(I=5, M=4)
        c = Calibration(p, update_from_api=False)

        d = c.get_dict()
        assert "alpha_c" not in d
        assert "io_matrix" not in d
        assert c.alpha_c is None
        assert c.io_matrix is None

    def test_get_dict_returns_empty_for_multi_sector(self):
        """Multi-sector offline: get_dict() returns completely empty dict."""
        p = _make_mock_p(I=5, M=4)
        c = Calibration(p, update_from_api=False)

        assert c.get_dict() == {}

    @patch("ogzaf.calibrate.macro_params")
    @patch("ogzaf.calibrate.io")
    def test_no_external_calls(self, mock_io, mock_macro):
        """Offline mode should not call any external-facing functions."""
        p = _make_mock_p(I=5, M=4)
        Calibration(p, update_from_api=False)

        mock_macro.get_macro_params.assert_not_called()
        mock_io.get_alpha_c.assert_not_called()
        mock_io.get_io_matrix.assert_not_called()


class TestOnlinePartialFailure:
    """Tests for update_from_api=True with mocked partial failures."""

    @patch("ogzaf.calibrate.macro_params")
    def test_macro_failure_warns_and_omits(self, mock_macro):
        """When macro update fails, warn and omit from get_dict()."""
        mock_macro.get_macro_params.side_effect = RuntimeError("API down")

        p = _make_mock_p(I=1, M=1)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # Patch demographics to avoid actual API call
            with patch("ogcore.demographics.get_pop_objs") as mock_demog:
                mock_demog.side_effect = RuntimeError("skip")
                c = Calibration(p, update_from_api=True)

        # Macro warning should have been emitted
        macro_warnings = [
            x for x in w if "Macro params" in str(x.message)
        ]
        assert len(macro_warnings) == 1

        d = c.get_dict()
        assert "g_y_annual" not in d
        assert "initial_debt_ratio" not in d

    @patch("ogzaf.calibrate.io")
    @patch("ogzaf.calibrate.macro_params")
    def test_sam_failure_warns_and_omits(self, mock_macro, mock_io):
        """When SAM fetch fails, alpha_c and io_matrix are omitted."""
        mock_macro.get_macro_params.return_value = {"g_y_annual": 0.01}
        mock_io.get_alpha_c.side_effect = RuntimeError("SAM unavailable")
        mock_io.get_io_matrix.side_effect = RuntimeError("SAM unavailable")

        p = _make_mock_p(I=5, M=4)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with patch("ogcore.demographics.get_pop_objs") as mock_demog:
                mock_demog.side_effect = RuntimeError("skip")
                c = Calibration(p, update_from_api=True)

        alpha_warnings = [
            x for x in w if "alpha_c" in str(x.message)
        ]
        io_warnings = [
            x for x in w if "io_matrix" in str(x.message)
        ]
        assert len(alpha_warnings) == 1
        assert len(io_warnings) == 1

        d = c.get_dict()
        assert "alpha_c" not in d
        assert "io_matrix" not in d
        # Macro should still be present
        assert d["g_y_annual"] == 0.01

    @patch("ogzaf.calibrate.macro_params")
    def test_demographics_failure_warns_and_omits(self, mock_macro):
        """When demographics fails, e and demographic_params are omitted."""
        mock_macro.get_macro_params.return_value = {"g_y_annual": 0.01}

        p = _make_mock_p(I=1, M=1)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with patch("ogcore.demographics.get_pop_objs") as mock_demog:
                mock_demog.side_effect = RuntimeError("UN API down")
                c = Calibration(p, update_from_api=True)

        demo_warnings = [
            x for x in w if "Demographics" in str(x.message)
        ]
        assert len(demo_warnings) == 1

        d = c.get_dict()
        assert "e" not in d
        assert "omega" not in d
        assert "omega_SS" not in d
        # Macro and single-sector defaults should still be present
        assert d["g_y_annual"] == 0.01
        assert "alpha_c" in d
        assert "io_matrix" in d
