"""
Build the calibration scorecard: the figures and tables in
docs/book/content/calibration/scorecard.md.

Run after examples/run_og_zaf.py, from the same checkout:

    python examples/calibration_scorecard.py

It reads the baseline and reform output of the example run, writes the
figures to docs/book/content/calibration/images/scorecard_*.png, and
rewrites each generated table in scorecard.md between its
``<!-- scorecard:NAME -->`` and ``<!-- /scorecard:NAME -->`` markers. The
prose around the tables is edited by hand.

Data sources are those of examples/validate_fiscal_path.py (the Treasury's
fiscal plan), ogzaf/targets.py (steady-state and first-year targets),
ogzaf/productivity.py (the baseline productivity path) and
docs/book/content/calibration/data/imf_weo_growth_forecasts.csv (IMF World
Economic Outlook growth forecasts for South Africa made five years ahead,
from the WEO historical forecasts database, against outturns from the
current WEO).
"""

import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from ogcore.utils import safe_read_pickle

from ogzaf.productivity import OECD_PATH, RULE_PATH, TARGET_PATH
from ogzaf.targets import TARGETS, ss_moments
from validate_fiscal_path import (
    DEBT,
    PRIMARY_BALANCE,
    PUBLIC_INVESTMENT,
    REVENUE,
    START_YEAR,
    _agg,
    growth_paths,
)

CUR_DIR = os.path.dirname(os.path.realpath(__file__))
EXAMPLE_DIR = os.path.join(CUR_DIR, "OG-ZAF-Example")
DOCS_DIR = os.path.join(
    CUR_DIR, "..", "docs", "book", "content", "calibration"
)
IMAGE_DIR = os.path.join(DOCS_DIR, "images")
PAGE = os.path.join(DOCS_DIR, "scorecard.md")
IMF_FILE = os.path.join(DOCS_DIR, "data", "imf_weo_growth_forecasts.csv")

FISCAL_YEARS = 12  # 2025-2036
GROWTH_YEARS = 25  # 2026-2050
LONG_YEARS = 60  # 2025-2084

MODEL, DATA, THIRD, FOURTH = "#1f5fbf", "#c2410c", "#0f8a6a", "#a16207"
MUTED = "#9ca3af"
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
    }
)
PCT = FuncFormatter(lambda v, _: f"{100 * v:.1f}%")

LABELS = {
    "PIT/Y": "Personal income tax / GDP",
    "CIT/Y": "Company tax (incl. dividends tax) / GDP",
    "payroll/Y": "Payroll taxes (SDL and UIF) / GDP",
    "cons tax/Y": "Consumption taxes, net of SACU / GDP",
    "wealth tax/Y": "Property and transfer taxes / GDP",
    "bequest tax/Y": "Estate and donations tax / GDP",
    "total tax/Y": "Total tax revenue / GDP",
    "K/Y": "Private capital / GDP",
    "I_g/Y": "Public investment / GDP",
    "r_gov": "Real interest rate on government debt",
    "primary balance": "Primary balance / GDP",
    "D/Y": "Gross loan debt / GDP",
    "D_f/D": "Foreign-held share of government debt",
    "TR/Y": "Social transfers / GDP",
    "K_f/K": "Foreign-owned share of private capital",
    "r": "Real return on private capital",
    "G/Y": "Government consumption / GDP",
    "NX/Y": "Net exports / GDP",
    "(I+I_g)/Y": "Total investment / GDP",
    "C/Y purch.": "Household consumption / GDP",
    "B/Y": "Household wealth / GDP",
    "total tax/Y ss": "Total tax revenue / GDP",
}
RATIOS = {"K/Y", "B/Y"}


def fmt(key, v):
    """Format a value as a ratio or a percentage."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    return f"{v:.2f}" if key in RATIOS else f"{100 * v:.2f}%"


def md_table(header, rows, align):
    """A Markdown table; align is a string of 'l' and 'r'."""
    sep = ["---:" if a == "r" else "---" for a in align]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(sep) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def load(name):
    base = os.path.join(EXAMPLE_DIR, name)
    tpi = safe_read_pickle(os.path.join(base, "TPI", "TPI_vars.pkl"))
    ss = safe_read_pickle(os.path.join(base, "SS", "SS_vars.pkl"))
    p = safe_read_pickle(os.path.join(base, "model_params.pkl"))
    return tpi, ss, p


def paths(tpi, p):
    """Ratios to GDP and rates on the baseline path."""
    Y = _agg(tpi["Y"])

    def ratio(key):
        return _agg(tpi[key]) / Y

    outlays = ratio("G") + ratio("TR") + ratio("I_g")
    outlays = outlays + ratio("agg_pension_outlays")
    if "UBI" in tpi:
        outlays = outlays + ratio("UBI")
    rev = ratio("total_tax_revenue")
    g, yph = growth_paths(tpi, p, GROWTH_YEARS + 1)
    hours = (1 + g) / (1 + yph) - 1
    return {
        "D/Y": ratio("D"),
        "primary balance": rev - outlays,
        "revenue/Y": rev,
        "G/Y": ratio("G"),
        "TR/Y": ratio("TR"),
        "I_g/Y": ratio("I_g"),
        "K/Y": ratio("K"),
        "C/Y": ratio("C"),
        "I/Y": ratio("I_total"),
        "r": _agg(tpi["r"]),
        "r_gov": _agg(tpi["r_gov"]),
        "GDP growth": g,
        "Y/hour growth": yph,
        "hours growth": hours,
        "Z": np.asarray(p.Z).ravel(),
    }


def first_year_table(tpi, m):
    Y0 = _agg(tpi["Y"])[0]
    rows = []
    keys = [
        ("PIT/Y", "iit_revenue"),
        ("CIT/Y", "business_tax_revenue"),
        ("payroll/Y", "payroll_tax_revenue"),
        ("cons tax/Y", "cons_tax_revenue"),
        ("wealth tax/Y", "wealth_tax_revenue"),
        ("bequest tax/Y", "bequest_tax_revenue"),
    ]
    for key, var in keys:
        v = _agg(tpi[var])[0] / Y0
        rows.append((key, v, TARGETS[key][0], TARGETS[key][1]))
    rows += [
        (
            "total tax/Y",
            m["revenue/Y"][0],
            REVENUE[START_YEAR],
            "MTBPS 2026 gross tax, less SACU payments, plus UIF and "
            "municipal rates; includes 0.12 points of minor taxes the "
            "model does not carry",
        ),
        ("K/Y", m["K/Y"][0], TARGETS["K/Y"][0], TARGETS["K/Y"][1]),
        (
            "I_g/Y",
            m["I_g/Y"][0],
            PUBLIC_INVESTMENT[START_YEAR],
            TARGETS["I_g/Y"][1],
        ),
        ("r_gov", m["r_gov"][0], TARGETS["r_gov"][0], TARGETS["r_gov"][1]),
        (
            "primary balance",
            m["primary balance"][0],
            PRIMARY_BALANCE[START_YEAR],
            "MTBPS 2026, 2025/26",
        ),
        (
            "D/Y",
            m["D/Y"][0],
            DEBT[START_YEAR - 1],
            "Gross loan debt at the end of 2024/25, MTBPS 2026",
        ),
    ]
    out = []
    for key, mv, dv, src in rows:
        gap = mv - dv
        gap = round(gap if key in RATIOS else 100 * gap, 2) + 0.0
        gap_s = f"{gap:+.2f}"
        out.append((LABELS[key], fmt(key, mv), fmt(key, dv), gap_s, src))
    return md_table(
        ["Measure", "Model", "Published", "Difference", "Source"],
        out,
        "lrrrl",
    )


def fiscal_table(m):
    rows = []
    for t in range(FISCAL_YEARS):
        y = START_YEAR + t

        def pc(v):
            return "" if v is None else f"{100 * v:.1f}"

        rows.append(
            (
                y,
                pc(m["D/Y"][t]),
                pc(DEBT.get(y - 1)),
                pc(m["primary balance"][t]),
                pc(PRIMARY_BALANCE.get(y)),
                pc(m["revenue/Y"][t]),
                pc(m["G/Y"][t]),
                pc(m["TR/Y"][t]),
                pc(m["I_g/Y"][t]),
                f"{100 * m['r_gov'][t]:.2f}",
            )
        )
    return md_table(
        [
            "Year",
            "Debt, model",
            "Debt, Treasury",
            "Primary balance, model",
            "Primary balance, Treasury",
            "Tax revenue",
            "Government consumption",
            "Social transfers",
            "Public investment",
            "Real rate on debt",
        ],
        rows,
        "lrrrrrrrrr",
    )


def steady_state_table(mom):
    rows = []
    for key, (target, src) in TARGETS.items():
        rows.append(
            (
                LABELS[key],
                fmt(key, mom[key]),
                fmt(key, target),
                src.replace("debt_ratio_ss: ", ""),
            )
        )
    for key in ("G/Y", "NX/Y", "(I+I_g)/Y", "C/Y purch.", "B/Y"):
        rows.append((LABELS[key], fmt(key, mom[key]), "", "not targeted"))
    rows.append(
        (
            LABELS["total tax/Y"],
            fmt("total tax/Y", mom["total tax/Y"]),
            "",
            "not targeted",
        )
    )
    return md_table(["Measure", "Model", "Target", "Source"], rows, "lrrl")


def growth_table(m):
    def avg(series, a, b):
        i, j = a - START_YEAR - 1, b - START_YEAR
        return f"{100 * np.mean(series[i:j]):.2f}%"

    rows = []
    for a, b in ((2026, 2030), (2031, 2040), (2041, 2050)):
        rows.append(
            (
                f"{a}–{b}",
                avg(m["GDP growth"], a, b),
                avg(m["Y/hour growth"], a, b),
                avg(TARGET_PATH, a, b),
                avg(RULE_PATH, a, b),
                avg(OECD_PATH, a, b),
                avg(m["hours growth"], a, b),
            )
        )
    return md_table(
        [
            "Period",
            "GDP",
            "Output per hour, model",
            "Baseline path",
            "Estimate 1: recent growth",
            "Estimate 2: OECD",
            "Hours worked",
        ],
        rows,
        "lrrrrrr",
    )


def reform_table():
    df = pd.read_csv(
        os.path.join(EXAMPLE_DIR, "OG-ZAF_example_output.csv"),
        index_col=0,
    )
    rows = []
    for _, r in df.iterrows():
        name = re.sub(r"\s*\(\$.*\$\)", "", r["Variable"])
        name = {"Capital Stock": "Capital", "Labor": "Labour"}.get(name, name)
        rows.append(
            (
                name,
                f"{r['2025']:+.2f}",
                f"{r['2030']:+.2f}",
                f"{r['2025-2034']:+.2f}",
                f"{r['SS']:+.2f}",
            )
        )
    return md_table(
        ["Variable", "2025", "2030", "Average 2025–2034", "Long run"],
        rows,
        "lrrrr",
    )


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(IMAGE_DIR, f"scorecard_{name}.png"), dpi=150)
    plt.close(fig)


def fig_fiscal(m):
    years = np.arange(START_YEAR, START_YEAR + FISCAL_YEARS)
    n = FISCAL_YEARS
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.5))
    a = ax[0, 0]
    a.plot(years, m["D/Y"][:n], "-o", ms=3, color=MODEL, label="Model")
    a.plot(
        years,
        [DEBT.get(y - 1, np.nan) for y in years],
        "o",
        ms=7,
        color=DATA,
        label="Treasury plan (MTBPS 2026)",
    )
    a.set_title("Government debt / GDP, start of year")
    a = ax[0, 1]
    a.plot(
        years, m["primary balance"][:n], "-o", ms=3, color=MODEL, label="Model"
    )
    a.plot(
        years,
        [PRIMARY_BALANCE.get(y, np.nan) for y in years],
        "o",
        ms=7,
        color=DATA,
        label="Treasury plan (MTBPS 2026)",
    )
    a.set_title("Primary balance / GDP")
    a = ax[1, 0]
    a.plot(years, m["revenue/Y"][:n], "-o", ms=3, color=MODEL)
    a.plot(years, m["G/Y"][:n], "-o", ms=3, color=THIRD)
    a.set_title("Tax revenue and government consumption / GDP")
    a.legend(["Tax revenue", "Government consumption"], loc="center right")
    a = ax[1, 1]
    a.plot(years, m["r"][:n], color=MODEL, label="Return on private capital")
    a.plot(years, m["r_gov"][:n], color=DATA, label="Rate on government debt")
    a.set_ylim(0, None)
    a.set_title("Real interest rates")
    for a in ax.ravel():
        a.yaxis.set_major_formatter(PCT)
        if not a.get_legend():
            a.legend()
    save(fig, "fiscal_path")


def fig_steady_state(mom):
    keys = list(TARGETS)
    rel = [mom[k] / TARGETS[k][0] - 1 for k in keys]
    colors = [
        THIRD if abs(v) <= 0.05 else FOURTH if abs(v) <= 0.15 else DATA
        for v in rel
    ]
    fig, a = plt.subplots(figsize=(9, 5.5))
    ypos = np.arange(len(keys))[::-1]
    a.barh(ypos, rel, color=colors)
    a.set_yticks(ypos, [LABELS[k] for k in keys])
    a.axvline(0, color="k", lw=0.8)
    lim = max(0.2, 1.2 * max(abs(v) for v in rel))
    a.set_xlim(-lim, lim)
    a.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.0%}"))
    for y, v in zip(ypos, rel):
        a.text(
            v + (0.004 if v >= 0 else -0.004),
            y,
            f"{v:+.1%}",
            va="center",
            ha="left" if v >= 0 else "right",
            fontsize=8,
        )
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=c) for c in (THIRD, FOURTH, DATA)
    ]
    a.legend(
        handles,
        ["Within 5% of target", "Within 15%", "Further"],
        loc="lower right",
    )
    a.set_title("Long run: model relative to target")
    save(fig, "steady_state")


def fig_productivity(m, g_y):
    years = np.arange(START_YEAR + 1, START_YEAR + 1 + GROWTH_YEARS)
    long_run = np.exp(g_y) - 1
    fig, a = plt.subplots(figsize=(11, 5))
    a.plot(years, TARGET_PATH, color=MUTED, lw=7, alpha=0.5)
    a.plot(years, RULE_PATH, "--", color=THIRD)
    a.plot(years, OECD_PATH, "--", color=FOURTH)
    a.plot(years, m["Y/hour growth"][:GROWTH_YEARS], "o", ms=4, color=MODEL)
    a.axhline(long_run, color="k", lw=0.8, ls=":")
    a.set_ylim(0, None)
    a.yaxis.set_major_formatter(PCT)
    a.legend(
        [
            "Baseline path: average of the two estimates",
            "Estimate 1: recent growth, recovering at the speed seen "
            "across countries",
            "Estimate 2: OECD long-term baseline, capped at 1.44%",
            "Model",
            f"Long-run growth rate, {100 * long_run:.1f}% a year (output per "
            "worker, 1994–2019, PWT 11.0)",
        ],
        loc="lower right",
    )
    a.set_title("Growth in output per hour worked, % a year")
    save(fig, "productivity")


def fig_growth_sources(m):
    years = np.arange(START_YEAR + 1, START_YEAR + 1 + GROWTH_YEARS)
    yph = m["Y/hour growth"][:GROWTH_YEARS]
    hours = m["hours growth"][:GROWTH_YEARS]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    a = ax[0]
    a.stackplot(
        years,
        yph,
        hours,
        colors=[MODEL, THIRD],
        alpha=0.45,
        labels=["Output per hour", "Hours worked"],
    )
    a.plot(years, m["GDP growth"][:GROWTH_YEARS], color="k", label="GDP")
    a.yaxis.set_major_formatter(PCT)
    a.legend(loc="lower right")
    a.set_title("GDP growth by source, % a year")
    a = ax[1]
    zyears = np.arange(START_YEAR, START_YEAR + 41)
    a.plot(zyears, m["Z"][:41], color=MODEL, label="Total factor productivity")
    a.axhline(1, color="k", lw=0.8, ls=":", label="Long-run trend")
    a.legend()
    a.set_title("Total factor productivity relative to trend")
    save(fig, "growth_sources")


def fig_imf(imf):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    a = ax[0]
    a.plot(imf.year, imf.forecast, "-o", ms=3, color=DATA)
    a.plot(imf.year, imf.actual, "-o", ms=3, color=MODEL)
    a.legend(["IMF forecast made five years earlier", "Actual growth"])
    a.set_title("Real GDP growth, %")
    a = ax[1]
    err = imf.forecast - imf.actual
    covid = imf.year.isin([2020, 2021])
    colors = np.where(covid, MUTED, np.where(err > 0, DATA, MODEL))
    a.bar(imf.year, err, color=colors)
    a.axhline(0, color="k", lw=0.8)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=c) for c in (DATA, MODEL, MUTED)
    ]
    a.legend(handles, ["Forecast too high", "Forecast too low", "Covid years"])
    a.set_title("Forecast minus actual, percentage points")
    save(fig, "imf")


def fig_long_paths(m):
    years = np.arange(START_YEAR, START_YEAR + LONG_YEARS)
    n = LONG_YEARS
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    panels = [
        ("K/Y", "Private capital / GDP", False),
        ("D/Y", "Government debt / GDP", True),
        ("I/Y", "Private investment / GDP", True),
        ("C/Y", "Household consumption / GDP, before taxes", True),
    ]
    for a, (key, title, pct) in zip(ax.ravel(), panels):
        a.plot(years, m[key][:n], color=MODEL)
        a.set_title(title)
        if pct:
            a.yaxis.set_major_formatter(PCT)
    save(fig, "long_paths")


def fig_solver(tpi):
    rc = np.abs(np.asarray(tpi["resource_constraint_error"]).ravel())
    es = np.abs(np.asarray(tpi["euler_savings"])).max(axis=(1, 2))
    en = np.abs(np.asarray(tpi["euler_labor_leisure"])).max(axis=(1, 2))
    years = np.arange(START_YEAR, START_YEAR + len(rc))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].semilogy(years, rc + 1e-16, color=MODEL)
    ax[0].set_title("Output not accounted for by spending, share of output")
    ax[1].semilogy(years, es, color=DATA, label="Saving decisions")
    ax[1].semilogy(years, en, color=THIRD, label="Work decisions")
    ax[1].legend()
    ax[1].set_title("Largest error in household decisions")
    save(fig, "solver")


def fig_reform():
    df = pd.read_csv(
        os.path.join(EXAMPLE_DIR, "OG-ZAF_example_output.csv"),
        index_col=0,
    ).set_index("Variable")
    years = [str(y) for y in range(START_YEAR, START_YEAR + 10)]
    fig, a = plt.subplots(figsize=(9, 4.2))
    names = ["GDP", "Consumption", "Capital Stock", "Labor"]
    labels = ["GDP", "Consumption", "Capital", "Labour"]
    for name, lab, c in zip(names, labels, [MODEL, DATA, THIRD, FOURTH]):
        row = [i for i in df.index if i.startswith(name)][0]
        a.plot(
            [int(y) for y in years],
            df.loc[row, years],
            "-o",
            ms=3,
            color=c,
            label=lab,
        )
    a.axhline(0, color="k", lw=0.8)
    a.legend()
    a.set_title("Higher company tax rate: change from the baseline, %")
    save(fig, "reform")


def write_tables(tables):
    with open(PAGE) as f:
        text = f.read()
    for name, table in tables.items():
        pattern = re.compile(
            rf"(<!-- scorecard:{name} -->\n).*?(<!-- /scorecard:{name} -->)",
            re.S,
        )
        if not pattern.search(text):
            raise ValueError(f"no markers for table {name} in {PAGE}")
        text = pattern.sub(
            lambda mt: mt.group(1) + table + "\n" + mt.group(2), text
        )
    with open(PAGE, "w") as f:
        f.write(text)


def main():
    tpi, ss, p = load("OUTPUT_BASELINE")
    m = paths(tpi, p)
    mom = ss_moments(ss)
    imf = pd.read_csv(IMF_FILE)

    fig_fiscal(m)
    fig_steady_state(mom)
    fig_productivity(m, p.g_y)
    fig_growth_sources(m)
    fig_imf(imf)
    fig_long_paths(m)
    fig_solver(tpi)
    fig_reform()

    write_tables(
        {
            "first_year": first_year_table(tpi, m),
            "fiscal_path": fiscal_table(m),
            "steady_state": steady_state_table(mom),
            "growth": growth_table(m),
            "reform": reform_table(),
        }
    )

    rc = np.abs(np.asarray(tpi["resource_constraint_error"]).ravel())
    recent = imf[(imf.year >= 2013) & ~imf.year.isin([2020, 2021])]
    err = recent.forecast - recent.actual
    print(f"largest accounting error to 2100: {rc[:76].max():.1e}")
    print(f"largest accounting error, any year: {rc.max():.1e}")
    print(
        f"IMF forecasts 2013-2024 ex Covid: {(err > 0).sum()} of "
        f"{len(err)} too high, by {err.mean():.1f} points on average"
    )
    print(f"wrote figures to {IMAGE_DIR} and tables to {PAGE}")


if __name__ == "__main__":
    main()
