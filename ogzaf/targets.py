"""
Data targets for the OG-ZAF steady-state dashboard, and the function that
computes the matching model moments from a solved ``SS_vars.pkl``.

Every target carries its source. Ratios are to GDP (Stats SA, rebased 2021
vintage) unless stated. Fiscal rows first, then external, production and
households.
"""

import numpy as np

# moment: (target, source)
TARGETS = {
    # Fiscal: revenue by instrument (FY2025/26 outcome, GDP R7,787bn)
    "PIT/Y": (0.1015, "PIT R790.6bn, MTBPS 2026 Table 3.1"),
    "CIT/Y": (
        0.0502,
        "CIT R345.4bn + dividends tax R45.9bn, MTBPS 2026 Table 3.1",
    ),
    "payroll/Y": (
        0.0067,
        "Skills Development Levy R26.0bn (MTBPS 2026) + UIF "
        "contributions R26.7bn (UIF Annual Report 2025/26)",
    ),
    "cons tax/Y": (
        0.0892,
        "VAT + fuel levy + excise + customs + other indirect taxes "
        "(9.87%) less SACU payments (0.95%), MTBPS 2026 / BR 2026",
    ),
    "wealth tax/Y": (
        0.0161,
        "municipal property rates R101.5bn (Stats SA P9114, 2024/25) "
        "+ transfer duty and securities transfer tax (BR 2026)",
    ),
    "bequest tax/Y": (
        0.0008,
        "estate duty R4.8bn + donations tax R1.2bn, BR 2026 annexure",
    ),
    "r_gov": (
        0.037,
        "debt service / average gross loan debt 7.08% nominal (MTBPS "
        "2026) less ~3.4% expected inflation",
    ),
    "D/Y": (0.765, "debt_ratio_ss: MTBPS 2026 gross loan debt 2029/30"),
    "D_f/D": (0.25, "non-resident share of domestic bonds, BR 2026"),
    # Spending
    "TR/Y": (0.036, "social grants incl. SRD, BR 2026 Table 5.7"),
    "I_g/Y": (0.0242, "general government GFCF 2025, SARB QB S-127"),
    # External
    "K_f/K": (
        0.205,
        "upper bound: inward DI equity 22.7% + portfolio equity 34.8% "
        "of GDP (SARB IIP end-2025) / private K/Y; DI equity alone 0.08",
    ),
    # Production
    "K/Y": (
        2.80,
        "private capital: PWT 11.0 cn/cgdpo 3.82 (2023) less the "
        "general government share 26.7% (SARB QB E-5)",
    ),
    "r": (0.12, "PWT 11.0 real internal rate of return, 2023"),
}


def ss_moments(ss, p):
    """Model side of every dashboard row from a solved SS_vars dict."""
    Y = float(np.squeeze(ss["Y"]))

    def sc(key):
        return float(np.sum(ss[key]))

    C = sc("C")
    nx = Y - C - sc("I_total") - sc("I_g") - sc("G")
    m = {
        "PIT/Y": sc("iit_revenue") / Y,
        "payroll/Y": sc("payroll_tax_revenue") / Y,
        "CIT/Y": sc("business_tax_revenue") / Y,
        "cons tax/Y": sc("cons_tax_revenue") / Y,
        "wealth tax/Y": sc("wealth_tax_revenue") / Y,
        "bequest tax/Y": sc("bequest_tax_revenue") / Y,
        "total tax/Y": sc("total_tax_revenue") / Y,
        "r_gov": sc("r_gov"),
        "D/Y": sc("D") / Y,
        "D_f/D": sc("D_f") / sc("D") if sc("D") else np.nan,
        "G/Y": sc("G") / Y,
        "TR/Y": sc("TR") / Y,
        "I_g/Y": sc("I_g") / Y,
        "pensions/Y": sc("agg_pension_outlays") / Y,
        "K_f/K": sc("K_f") / sc("K"),
        "NX/Y": nx / Y,
        "K/Y": sc("K") / Y,
        "(I+I_g)/Y": (sc("I_total") + sc("I_g")) / Y,
        "C/Y purch.": (C + sc("cons_tax_revenue")) / Y,
        "B/Y": sc("B") / Y,
        "BQ/Y": sc("BQ") / Y,
        "r": sc("r"),
        "factor": sc("factor"),
    }
    return m
