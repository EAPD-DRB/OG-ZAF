"""
Score the baseline transition against South Africa's fiscal accounts.

The steady state is decades away and its debt ratio is a policy anchor we
chose. The first years of the transition start from measured values and can
be checked against what the National Treasury reports and plans: gross loan
debt, the primary balance, tax revenue and public investment.

Run after examples/run_og_zaf.py, from the same checkout:

    python examples/validate_fiscal_path.py

Timing. Model period t is calendar year START_YEAR + t, which we read as
fiscal year (START_YEAR + t)/(START_YEAR + t + 1). Model debt in period t is
the stock at the start of that year, so it is scored against the Treasury's
end-of-year stock for the fiscal year before.

Sources (all % of GDP):
- Gross loan debt, end of fiscal year: MTBPS 2026 Tables 1.3 and 3.8.
- Consolidated primary balance: MTBPS 2026 Tables 1.2, 3.6 and 4.1
  (budget balance plus interest); 2024/25 from Budget Review 2026.
- Tax revenue: MTBPS 2026 Table 3.1, gross tax 25.82% in 2025/26, bridged to
  the model's concept: less SACU payments (0.95), plus UIF contributions
  (0.34) and municipal property rates (1.36).
- Public investment: general government GFCF 2.42% in 2025, SARB QB S-127.

It also prints the first year's revenue by instrument against the targets in
ogzaf/targets.py, and the growth of output per hour against the baseline
path in ogzaf/productivity.py.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ogcore.utils import safe_read_pickle

from ogzaf.productivity import TARGET_PATH
from ogzaf.targets import TARGETS

START_YEAR = 2025
HORIZON = 12

# end-of-fiscal-year stock, keyed by the calendar year the fiscal year starts
DEBT = {
    2024: 0.770,
    2025: 0.784,
    2026: 0.782,
    2027: 0.779,
    2028: 0.775,
    2029: 0.766,
}
PRIMARY_BALANCE = {
    2025: 0.013,
    2026: 0.016,
    2027: 0.020,
    2028: 0.025,
    2029: 0.030,
}
REVENUE = {2025: 0.2582 - 0.0095 + 0.0034 + 0.0136}
PUBLIC_INVESTMENT = {2025: 0.0242}


def _agg(x):
    x = np.asarray(x)
    return x.sum(axis=tuple(range(1, x.ndim))) if x.ndim > 1 else x


FIRST_YEAR = {
    "PIT/Y": "iit_revenue",
    "CIT/Y": "business_tax_revenue",
    "payroll/Y": "payroll_tax_revenue",
    "cons tax/Y": "cons_tax_revenue",
    "wealth tax/Y": "wealth_tax_revenue",
    "bequest tax/Y": "bequest_tax_revenue",
    "K/Y": "K",
}


def growth_paths(tpi, p, n):
    """Real GDP growth and growth of output per hour, years 1..n-1."""
    Y = _agg(tpi["Y"])[:n]
    gn = np.asarray(p.g_n)[:n]
    om = np.asarray(p.omega)[:n]
    hours = (om * np.asarray(tpi["n"])[:n]).sum(axis=(1, 2)) / om.sum(
        axis=(1, 2)
    )
    gy = np.exp(p.g_y) - 1
    g = Y[1:] / Y[:-1] * (1 + gy) * (1 + gn[1:]) - 1
    gh = hours[1:] / hours[:-1] * (1 + gn[1:]) - 1
    return g, (1 + g) / (1 + gh) - 1


def model_paths(base_dir):
    tpi = safe_read_pickle(os.path.join(base_dir, "TPI", "TPI_vars.pkl"))
    p = safe_read_pickle(os.path.join(base_dir, "model_params.pkl"))
    Y = _agg(tpi["Y"])[:HORIZON]
    g, yph = growth_paths(tpi, p, HORIZON + 1)

    def ratio(key):
        return _agg(tpi[key])[:HORIZON] / Y

    outlays = ratio("G") + ratio("TR") + ratio("I_g")
    outlays = outlays + ratio("agg_pension_outlays")
    if "UBI" in tpi:
        outlays = outlays + ratio("UBI")
    rev = ratio("total_tax_revenue")
    return {
        "D/Y": ratio("D"),
        "primary balance": rev - outlays,
        "revenue/Y": rev,
        "I_g/Y": ratio("I_g"),
        "G/Y": ratio("G"),
        "C/Y": ratio("C"),
        "B/Y": ratio("B"),
        "r_gov": _agg(tpi["r_gov"])[:HORIZON],
        "GDP growth": np.r_[np.nan, g[: HORIZON - 1]],
        "Y/hour growth": np.r_[np.nan, yph[: HORIZON - 1]],
        "first year": {
            k: float(_agg(tpi[v])[0] / Y[0]) for k, v in FIRST_YEAR.items()
        },
    }


def data_paths():
    years = range(START_YEAR, START_YEAR + HORIZON)
    return {
        "D/Y": [DEBT.get(y - 1, np.nan) for y in years],
        "primary balance": [PRIMARY_BALANCE.get(y, np.nan) for y in years],
        "revenue/Y": [REVENUE.get(y, np.nan) for y in years],
        "I_g/Y": [PUBLIC_INVESTMENT.get(y, np.nan) for y in years],
        "Y/hour growth": [np.nan]
        + [TARGET_PATH[t] for t in range(HORIZON - 1)],
    }


def main():
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    save_dir = os.path.join(cur_dir, "OG-ZAF-Example")
    m = model_paths(os.path.join(save_dir, "OUTPUT_BASELINE"))
    d = data_paths()
    years = [START_YEAR + t for t in range(HORIZON)]

    print(f"First year ({START_YEAR}): model against targets")
    for k, v in m["first year"].items():
        print(f"  {k:14s}{v:9.4f}{TARGETS[k][0]:9.4f}")
    print("\nFiscal path: model against the National Treasury")
    for key in (
        "D/Y",
        "primary balance",
        "revenue/Y",
        "I_g/Y",
        "Y/hour growth",
    ):
        print(f"\n{key}\n  {'year':6s}{'model':>9s}{'data':>9s}{'gap':>9s}")
        for t, y in enumerate(years):
            dv = d[key][t]
            gap = "" if np.isnan(dv) else f"{m[key][t] - dv:+9.4f}"
            dat = "" if np.isnan(dv) else f"{dv:9.4f}"
            print(f"  {y:<6d}{m[key][t]:9.4f}{dat:>9s}{gap:>9s}")
    print("\nUnscored paths (first years)")
    for key in ("GDP growth", "G/Y", "C/Y", "B/Y", "r_gov"):
        vals = " ".join(f"{v:.4f}" for v in m[key][:6])
        print(f"  {key:6s} {vals}")
    for key in ("D/Y", "primary balance", "C/Y"):
        lo, hi = int(np.argmin(m[key])), int(np.argmax(m[key]))
        print(
            f"  {key}: min {m[key][lo]:.4f} in {years[lo]}, "
            f"max {m[key][hi]:.4f} in {years[hi]}"
        )

    panels = [
        ("D/Y", "Gross loan debt / GDP (start of year)"),
        ("primary balance", "Primary balance / GDP"),
        ("revenue/Y", "Tax revenue / GDP"),
        ("I_g/Y", "Public investment / GDP"),
        ("Y/hour growth", "Growth of output per hour"),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.4))
    x = np.arange(HORIZON)
    for ax, (key, title) in zip(axes, panels):
        ax.plot(x, m[key], "-o", ms=4, label="OG-ZAF baseline")
        lab = "baseline path" if key == "Y/hour growth" else "Treasury/SARB"
        ax.plot(x, d[key], "o", ms=7, color="#d62728", label=lab)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(x[::2])
        ax.set_xticklabels(years[::2], fontsize=9)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    out = os.path.join(save_dir, "fiscal_path.png")
    fig.savefig(out, dpi=150)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
