"""
The baseline path of total factor productivity (Z) and where it comes from.

South Africa's productivity has fallen since 2008: in PWT 11.0, TFP
(``rtfpna``) fell 1.0% a year over 2008-2019 and real GDP per person engaged
grew 0.25% a year, against 2.4% over 1994-2008. The model's long-run growth
rate of labour productivity, g_y = 1.44%, is the 1994-2019 average. Z is the
bridge: it lies below that trend now and closes the gap over the transition.

The path of output per hour the model is fitted to is the average of two
baseline estimates, year by year:

1. Our rule. Growth starts from South Africa's recent trend (real GDP growth
   of 1.1% a year, the 2022-2025 average, IMF WEO October 2026) and the
   productivity shortfall then shrinks by 11.35% a year. That is the
   cross-country persistence of growth rates: about 0.3 of a country's gap to
   the world mean carries into the next decade (Pritchett and Summers 2014,
   NBER WP 20573, Table 1, adjacent-decade regression coefficients
   0.21-0.38), and 0.3 ** (1/10) = 0.8865.
2. The OECD long-term baseline (Economic Outlook 117, June 2025,
   DF_EO_LTB): potential output per potential worker grows 0.7% a year in
   2025-2030 and 1.9% in 2030-2040. We interpolate between the period
   midpoints and cap it at our long-run 1.44%; the OECD's higher later rates
   rest on convergence toward the frontier, which we do not assume.

Forecasts by the IMF, the SARB and the National Treasury are not used: they
assume reforms (energy, logistics) take hold, and the IMF's five-year-ahead
growth forecasts for South Africa overshot every outturn from 2008 to 2019,
by 2.1 percentage points a year on average (WEO historical forecasts).

Model labour input is set by the demographics, at today's employment rates
by age (they are in the earnings matrix), so only productivity is fitted.

``DRAGS`` holds the fitted fall in log Z below its trend in each year from
2026 to 2050. After 2050 the drag keeps shrinking at the rule's 0.8865 a
year, and Z is normalised to 1 in the steady state, so the long run is
unchanged. ``z_path`` turns the drags into the packaged ``Z`` parameter.
"""

import numpy as np

LONG_RUN_GROWTH = 0.0144
RULE_PERSISTENCE = 0.8865
N_YEARS = 60

# Growth of output per hour, 2026-2050. RULE_PATH is the model's own path
# under our rule (Z falling 1.56% below trend in 2026, the drag shrinking at
# RULE_PERSISTENCE a year); OECD_PATH is the capped OECD baseline;
# TARGET_PATH is their average, the path the baseline is fitted to.
RULE_PATH = [
    0.0018, 0.00206, 0.00256, 0.00325, 0.00415, 0.00515, 0.00611, 0.00701,
    0.00783, 0.00863, 0.00929, 0.00958, 0.00938, 0.00879, 0.00815, 0.00768,
    0.00716, 0.00778, 0.00875, 0.00892, 0.00967, 0.01, 0.01041, 0.01088,
    0.01129,
]  # fmt: skip
OECD_PATH = [
    min(max(0.007 + 0.012 * (y - 2028) / 7.5, 0.007), LONG_RUN_GROWTH)
    for y in range(2026, 2051)
]
TARGET_PATH = [0.5 * (a + b) for a, b in zip(RULE_PATH, OECD_PATH)]

# fall in log Z below trend, 2026-2050, fitted so the baseline's output per
# hour follows TARGET_PATH (examples/validate_fiscal_path.py checks it)
DRAGS = [
    0.011637, 0.010657, 0.009383, 0.007926, 0.006679, 0.005087, 0.004157,
    0.003542, 0.003381, 0.003232, 0.003036, 0.002691, 0.002157, 0.001496,
    0.000881, 0.000413, -0.000038, 0.000116, 0.000448, 0.000358, 0.000176,
    0.000136, 0.000137, 0.000177, 0.000196,
]  # fmt: skip


def z_path(drags=None, n_years=N_YEARS, persistence=RULE_PERSISTENCE):
    """
    The Z parameter: one value per year from the start year, steady-state 1.

    Args:
        drags (list): fall in log Z in each year after the start year
        n_years (int): years until Z reaches its steady-state value
        persistence (float): yearly decay of the drag after the last fitted
            year

    Returns:
        Z (list): [[Z_0], [Z_1], ..., [1.0]], length n_years + 1
    """
    d = list(DRAGS if drags is None else drags)
    d += [d[-1] * persistence**k for k in range(1, n_years - len(d) + 1)]
    d = np.array([0.0] + d)
    return [
        [round(float(np.exp(d[t + 1 :].sum())), 6)] for t in range(n_years + 1)
    ]
