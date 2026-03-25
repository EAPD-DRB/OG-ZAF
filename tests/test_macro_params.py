"""
Tests of macro_params.py module
"""

import sys
import types

import requests
import pytest

from ogzaf import macro_params


class MockResponse:
    """
    Minimal mock response for requests.get().
    """

    def __init__(self, *, json_data=None, text="", status_code=200):
        self._json_data = json_data
        self.text = text
        self.status_code = status_code

    def json(self):
        if isinstance(self._json_data, Exception):
            raise self._json_data
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(
                f"HTTP {self.status_code} returned from mocked request"
            )


def _wb_payload(observations):
    return [
        {
            "page": 1,
            "pages": 1,
            "per_page": "10000",
            "total": len(observations),
        },
        [
            {
                "date": date,
                "value": value,
                "indicator": {"id": "mock-indicator"},
            }
            for date, value in observations
        ],
    ]


def _mock_requests_get(monkeypatch, wb_payloads, ilo_text=None):
    def fake_get(url, params=None, headers=None):
        if "worldbank.org" in url:
            indicator_code = url.rstrip("/").split("/")[-1]
            return MockResponse(json_data=wb_payloads[indicator_code])
        if "rplumber.ilo.org" in url:
            return MockResponse(
                text=ilo_text or "time,obs_value\n2024,40\n2023,39\n"
            )
        raise AssertionError(f"Unexpected URL requested in test: {url}")

    monkeypatch.setattr(macro_params.requests, "get", fake_get)


def _mock_statsmodels(monkeypatch, params=None):
    fake_statsmodels = types.ModuleType("statsmodels")
    fake_api = types.ModuleType("statsmodels.api")

    def add_constant(values):
        return values

    class OLS:
        def __init__(self, endog, exog):
            self.endog = endog
            self.exog = exog

        def fit(self):
            result = types.SimpleNamespace()
            result.params = params or [1.0, 0.5]
            return result

    fake_api.add_constant = add_constant
    fake_api.OLS = OLS
    fake_statsmodels.api = fake_api
    monkeypatch.setitem(sys.modules, "statsmodels", fake_statsmodels)
    monkeypatch.setitem(sys.modules, "statsmodels.api", fake_api)


def test_fetch_wb_data_annual_success(monkeypatch):
    _mock_requests_get(
        monkeypatch,
        {
            "NY.GDP.PCAP.KD": _wb_payload(
                [("2022", 64.0), ("2024", 100.0), ("2023", 80.0)]
            )
        },
    )

    data = macro_params._fetch_wb_data(
        {"GDP per capita (constant 2015 US$)": "NY.GDP.PCAP.KD"},
        "ZAF",
        2022,
        2024,
        source=2,
    )

    assert list(data.index) == ["2024", "2023", "2022"]
    assert list(data.columns) == ["GDP per capita (constant 2015 US$)"]
    assert data.loc["2024", "GDP per capita (constant 2015 US$)"] == 100.0


def test_fetch_wb_data_quarterly_success(monkeypatch):
    _mock_requests_get(
        monkeypatch,
        {
            "DP.DOD.DECT.CR.GG.Z1": _wb_payload(
                [("2024Q2", 48.0), ("2024Q4", 50.0), ("2024Q3", 49.0)]
            )
        },
    )

    data = macro_params._fetch_wb_data(
        {"Gross PSD Gen Gov - percentage of GDP": "DP.DOD.DECT.CR.GG.Z1"},
        "ZAF",
        2024,
        2024,
        source=20,
    )

    assert list(data.index) == ["2024Q4", "2024Q3", "2024Q2"]
    assert list(data.columns) == ["Gross PSD Gen Gov - percentage of GDP"]
    assert data.loc["2024Q4", "Gross PSD Gen Gov - percentage of GDP"] == 50.0


def test_fetch_wb_data_empty_payload_raises_value_error(monkeypatch):
    _mock_requests_get(
        monkeypatch,
        {"NY.GDP.PCAP.KD": [{"page": 1, "pages": 1, "total": 0}, []]},
    )

    with pytest.raises(ValueError, match="Empty or malformed World Bank"):
        macro_params._fetch_wb_data(
            {"GDP per capita (constant 2015 US$)": "NY.GDP.PCAP.KD"},
            "ZAF",
            2022,
            2024,
            source=2,
        )


def test_get_macro_params_update_from_api_false_returns_empty_dict():
    test_dict = macro_params.get_macro_params(update_from_api=False)

    assert isinstance(test_dict, dict)
    assert test_dict == {}


def test_get_macro_params_update_from_api_true(monkeypatch):
    _mock_statsmodels(monkeypatch)
    _mock_requests_get(
        monkeypatch,
        {
            "NY.GDP.PCAP.KD": _wb_payload(
                [("2022", 64.0), ("2024", 100.0), ("2023", 80.0)]
            ),
            "NY.GDP.MKTP.KD": _wb_payload(
                [("2022", 640.0), ("2024", 1000.0), ("2023", 800.0)]
            ),
            "NY.GDP.MKTP.CD": _wb_payload(
                [("2022", 700.0), ("2024", 1100.0), ("2023", 900.0)]
            ),
            "NE.CON.GOVT.CD": _wb_payload(
                [("2022", 200.0), ("2024", 250.0), ("2023", 225.0)]
            ),
            "DP.DOD.DECD.CR.PS.CD": _wb_payload(
                [("2024Q4", 60.0), ("2024Q3", 58.0), ("2024Q2", 57.0)]
            ),
            "DP.DOD.DECX.CR.PS.CD": _wb_payload(
                [("2024Q4", 40.0), ("2024Q3", 42.0), ("2024Q2", 43.0)]
            ),
            "DP.DOD.DECT.CR.GG.Z1": _wb_payload(
                [("2024Q4", 50.0), ("2024Q3", 49.0), ("2024Q2", 48.0)]
            ),
        },
        ilo_text="time,obs_value\n2024,40\n2023,39\n",
    )

    test_dict = macro_params.get_macro_params(update_from_api=True)

    assert isinstance(test_dict, dict)
    assert sorted(test_dict.keys()) == sorted(
        [
            "r_gov_shift",
            "r_gov_scale",
            "alpha_T",
            "alpha_G",
            "initial_debt_ratio",
            "g_y_annual",
            "gamma",
            "zeta_D",
            "initial_foreign_debt_ratio",
        ]
    )
    assert test_dict["initial_debt_ratio"] == 0.5
    assert test_dict["initial_foreign_debt_ratio"] == 0.4
    assert test_dict["zeta_D"] == [0.4]
    assert test_dict["g_y_annual"] == pytest.approx(0.25)
    assert test_dict["gamma"] == [0.6]
    assert test_dict["r_gov_shift"] == [-0.01]
    assert test_dict["r_gov_scale"] == [0.5]


def test_get_macro_params_uses_last_valid_quarter(monkeypatch):
    _mock_statsmodels(monkeypatch)
    _mock_requests_get(
        monkeypatch,
        {
            "NY.GDP.PCAP.KD": _wb_payload(
                [("2022", 64.0), ("2024", 100.0), ("2023", 80.0)]
            ),
            "NY.GDP.MKTP.KD": _wb_payload(
                [("2022", 640.0), ("2024", 1000.0), ("2023", 800.0)]
            ),
            "NY.GDP.MKTP.CD": _wb_payload(
                [("2022", 700.0), ("2024", 1100.0), ("2023", 900.0)]
            ),
            "NE.CON.GOVT.CD": _wb_payload(
                [("2022", 200.0), ("2024", 250.0), ("2023", 225.0)]
            ),
            "DP.DOD.DECD.CR.PS.CD": _wb_payload(
                [("2024Q4", None), ("2024Q3", 60.0), ("2024Q2", None)]
            ),
            "DP.DOD.DECX.CR.PS.CD": _wb_payload(
                [("2024Q4", None), ("2024Q3", 40.0), ("2024Q2", None)]
            ),
            "DP.DOD.DECT.CR.GG.Z1": _wb_payload(
                [("2024Q4", None), ("2024Q3", 50.0), ("2024Q2", None)]
            ),
        },
        ilo_text="time,obs_value\n2024,40\n2023,39\n",
    )

    test_dict = macro_params.get_macro_params(update_from_api=True)

    assert test_dict["initial_debt_ratio"] == 0.5
    assert test_dict["initial_foreign_debt_ratio"] == 0.4
    assert test_dict["zeta_D"] == [0.4]
