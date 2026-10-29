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
    # Fiscal: revenue by instrument
    "PIT/Y": (0.101, "SARS/National Treasury 2025/26: PIT R786.2bn"),
    "CIT/Y": (0.045, "SARS/National Treasury 2025/26: CIT R346.5bn"),
    "cons tax/Y": (
        0.095,
        "VAT + fuel levy + excise + customs, Budget Review 2026 Ch. 4",
    ),
    "bequest tax/Y": (0.001, "estate duty + donations tax, SARS"),
    "r_gov": (
        0.037,
        "debt service / gross loan debt 7.1% nominal less ~3.4% "
        "inflation (Budget Review 2026 Ch. 7)",
    ),
    "D/Y": (0.765, "debt_ratio_ss: Budget Review 2026 Table 7.7, 2028/29"),
    "D_f/D": (0.25, "non-resident share of domestic bonds (Treasury)"),
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
