"""
Multi-industry calibration objects from the South African SAM (SASAM).

The data source is the 2019 South African Social Accounting Matrix (SASAM)
distributed by UNU-WIDER (technical note 2023-1), read from the EAPD-DRB
``SAM-files`` mirror. The functions here aggregate its 61 activities and 108
commodities to the model's M = 8 industries and I = 5 consumption goods using
the concordances in ``ogzaf.constants`` (``PROD_DICT``, ``CONS_DICT``).

The construction functions (``get_alpha_c``, ``get_io_matrix_value_added``,
``get_gamma``, ``get_Z``, ``get_employment``) are assembled into the packaged
``ogzaf_default_parameters_multisector.json`` by
``ogzaf.create_multisector_calibration``; the model loads that JSON and does
not call these at run time.
"""

import numpy as np
import pandas as pd

from ogzaf.utils import is_connected
from ogzaf.constants import (
    CAPITAL_OUTPUT_RATIO,
    CONS_DICT,
    PROD_DICT,
)

# SASAM factor rows: four labour-by-education rows and one
# gross-operating-surplus (capital) row. The SASAM has no separate
# mixed-income row, so -- unlike the Brazilian table -- no Gollin split of
# self-employment income is needed; gos is booked wholesale to capital.
LABOR_FACTORS = ["flab-p", "flab-m", "flab-s", "flab-t"]
CAPITAL_FACTORS = ["gos"]
# Rest-of-world account: commodity imports are the payment from a commodity
# column to this row.
ROW_ACCOUNT = "row"

# Employment by industry, thousands of persons, Statistics South Africa
# Quarterly Labour Force Survey, 2019 annual average (total ~16,340k matches
# the 2019 QLFS employed total). Measured independently of the SAM's factor
# payments so it serves as the physical labour input L_m in the Solow residual
# get_Z. The QLFS "Electricity, gas and water" aggregate (138.5k) is split
# between
# Electricity and Water and Waste 52.7/47.3 by the SASAM's labour-compensation
# shares of those activities -- waste and sanitation are labour-intensive while
# power generation is capital-intensive, so a headcount split by labour
# compensation is more defensible than an output split.
EMPLOYMENT = {
    "Agriculture": 861.0,
    "Mining": 411.8,
    "Electricity": 73.1,
    "Water and Waste": 65.4,
    "Construction": 1347.8,
    "Trade Transport and Accommodation": 4356.3,
    "Services": 7465.5,
    "Manufacturing": 1762.3,
}

"""
Read in Social Accounting Matrix (SAM) file
This is the most recent SAM for 2019 available from the following
page as a downloadable zip folder from UNU WIDER:
https://www.wider.unu.edu/sites/default/files/Publications/Technical-note/tn2023-1-2019-SASAM-for-distribution.zip
"""
# Read in SAM file
storage_options = {"User-Agent": "Mozilla/5.0"}
SAM_path = "https://raw.githubusercontent.com/EAPD-DRB/SAM-files/main/Data/ZAF/tn2023-1-2019-SASAM-for-distribution.xlsx"


def read_SAM():
    if is_connected():
        try:
            SAM = pd.read_excel(
                SAM_path,
                # Can alternatively use sheet_name="SASM 2019 61Ind4Occ"
                sheet_name="SASAM 2019 61Ind 4Educ",
                skiprows=3,
                index_col=0,
                storage_options=storage_options,
            )
            print("Successfully read SAM from Github repository.")
        except Exception as e:
            print(f"Failed to read from the GitHub repository: {e}")
            SAM = None
        # If both attempts fail, SAM will be None
        if SAM is None:
            print("Failed to read SAM from both sources.")
    else:  # pragma: no cover
        SAM = None
        print("No internet connection. SAM cannot be read.")
    return SAM


def get_alpha_c(sam=None, cons_dict=CONS_DICT):
    """
    Calibrate the alpha_c vector: the shares of household expenditure on
    each of the I consumption goods.

    Household final consumption is the payment from the household columns to
    the commodity rows (purchaser prices, the budget households actually
    allocate). The share is over total household consumption, so it needs no
    normalization to GDP.

    Args:
        sam (pd.DataFrame): SAM file
        cons_dict (dict): Dictionary of consumption categories

    Returns:
        alpha_c (dict): Dictionary of shares of household expenditures
    """
    if sam is None:
        sam = read_SAM()
    if sam is None:
        raise RuntimeError("SAM data is unavailable. Cannot compute alpha_c.")
    hh_cols = [c for c in sam.columns if str(c).startswith("hhd")]
    alpha_c = {}
    overall_sum = 0.0
    for key, value in cons_dict.items():
        category_total = (
            sam.loc[sam.index.isin(value), hh_cols].values.astype(float).sum()
        )
        alpha_c[key] = category_total
        overall_sum += category_total
    for key in cons_dict.keys():
        alpha_c[key] = alpha_c[key] / overall_sum

    return alpha_c


def get_io_matrix(sam=None, cons_dict=CONS_DICT, prod_dict=PROD_DICT):
    """
    Calibrate the io_matrix array.  This array relates the share of each
    production category in each consumption category

    Note: this is the naive direct-mapping version, retained for the live
    ``Calibration(update_from_api=True)`` path and its unit test. The packaged
    multi-industry calibration is built from the domestic value-added content
    instead -- see ``get_io_matrix_value_added``.

    Args:
        sam (pd.DataFrame): SAM file
        cons_dict (dict): Dictionary of consumption categories
        prod_dict (dict): Dictionary of production categories

    Returns:
        io_df (pd.DataFrame): Dataframe of io_matrix
    """
    if sam is None:
        sam = read_SAM()
    if sam is None:
        raise RuntimeError(
            "SAM data is unavailable. Cannot compute io_matrix."
        )
    # Create initial matrix as dataframe of 0's to fill in
    io_dict = {}
    for key in prod_dict.keys():
        io_dict[key] = np.zeros(len(cons_dict.keys()))
    io_df = pd.DataFrame(io_dict, index=cons_dict.keys())
    # Fill in the matrix
    # Note, each cell in the SAM represents a payment from the columns
    # account to the row account
    # (see https://www.un.org/en/development/desa/policy/capacity/presentations/manila/6_sam_mams_philippines.pdf)
    # We are thus going to take the consumption categories from rows and
    # the production categories from columns
    for ck, cv in cons_dict.items():
        for pk, pv in prod_dict.items():
            io_df.loc[io_df.index == ck, pk] = sam.loc[
                sam.index.isin(cv), pv
            ].values.sum()
    # change from levels to share (where each row sums to one)
    io_df = io_df.div(io_df.sum(axis=1), axis=0)

    return io_df


def get_io_matrix_value_added(
    sam=None, cons_dict=CONS_DICT, prod_dict=PROD_DICT
):
    """
    Calibrate the ``io_matrix``: the domestic value-added content of each
    consumption good, by producing industry (rows sum to one).

    OG-Core's production side has no intermediate inputs, so the object it
    needs is the value added embodied in final consumption: of one rand spent
    on consumption good i, how much is value added by each industry once the
    whole domestic supply chain is traced. The SASAM separates activities from
    commodities, so we build this with standard industry-technology make/use
    algebra derived from the SAM itself:

      * make ``V`` (activity x commodity): each activity's domestic output of
        each commodity; column sums are commodity output ``g``, row sums are
        activity gross output ``q``.
      * market-share matrix ``D = V / g`` allocates a commodity's domestic
        supply to producing activities.
      * commodity input coefficients ``B = U / q`` from the use matrix ``U``
        (commodity x activity intermediate purchases).
      * industry-by-industry Leontief inverse ``L = (I - D @ B)^-1`` traces the
        supply chain, and value added per unit of gross output ``v = VA / q``
        converts activity output into value added.

    Household consumption of each commodity, scaled to its domestic-supply
    share (imports carry no domestic value added), weights the commodities
    within each consumption good; the row normalization drops the imported
    content.

    Args:
        sam (pd.DataFrame): SAM file
        cons_dict (dict): consumption categories (order sets the I dimension)
        prod_dict (dict): production industries (order sets the M dimension)

    Returns:
        io_df (pd.DataFrame): I x M ``io_matrix`` (rows sum to one)
    """
    if sam is None:
        sam = read_SAM()
    if sam is None:
        raise RuntimeError(
            "SAM data is unavailable. Cannot compute io_matrix."
        )
    acts = [a for v in prod_dict.values() for a in v]
    comms = [c for v in cons_dict.values() for c in v]
    hh_cols = [c for c in sam.columns if str(c).startswith("hhd")]

    V = sam.loc[acts, comms].astype(float).values  # make: activity x commodity
    U = sam.loc[comms, acts].astype(float).values  # use : commodity x activity
    g = V.sum(axis=0)  # commodity domestic output
    q = V.sum(axis=1)  # activity gross output
    VA = (
        sam.loc[sam.index.isin(LABOR_FACTORS + CAPITAL_FACTORS), acts]
        .astype(float)
        .sum(axis=0)
        .reindex(acts)
        .values
    )
    v = np.divide(VA, q, out=np.zeros_like(VA), where=q > 0)
    D = np.divide(V, g[None, :], out=np.zeros_like(V), where=g[None, :] > 0)
    B = np.divide(U, q[None, :], out=np.zeros_like(U), where=q[None, :] > 0)
    L = np.linalg.inv(np.eye(len(acts)) - D @ B)

    hh_cons = (
        sam.loc[comms, hh_cols].astype(float).sum(axis=1).reindex(comms).values
    )
    imports = sam.loc[ROW_ACCOUNT, comms].astype(float).values
    supply = g + np.maximum(imports, 0.0)
    dom_share = np.divide(g, supply, out=np.ones_like(g), where=supply > 0)
    f_all = hh_cons * dom_share

    comm_idx = {c: i for i, c in enumerate(comms)}
    act_to_ind = {a: ind for ind, al in prod_dict.items() for a in al}
    cats, slugs = list(cons_dict), list(prod_dict)
    io = np.zeros((len(cats), len(slugs)))
    for ci, cat in enumerate(cats):
        f = np.zeros(len(comms))
        for c in cons_dict[cat]:
            f[comm_idx[c]] = f_all[comm_idx[c]]
        va_by_activity = v * (L @ (D @ f))
        for ai, a in enumerate(acts):
            io[ci, slugs.index(act_to_ind[a])] += va_by_activity[ai]
    io = io / io.sum(axis=1, keepdims=True)
    return pd.DataFrame(io, index=cats, columns=slugs)


def get_gamma(sam=None, prod_dict=PROD_DICT, target_avg=None):
    """
    Calibrate ``gamma``, the capital share of factor income by industry.

    The SASAM reports one capital row (gross operating surplus) and four
    labour rows; each industry's capital share is operating surplus over total
    factor income. These raw shares are *total* capital shares (self-employed
    income is embedded in operating surplus), so ``target_avg`` applies a
    multiplicative rescale that sets the value-added-weighted mean to the
    calibration target (the single-industry economy-wide capital share) while
    preserving the cross-industry pattern the SAM identifies.

    Args:
        sam (pd.DataFrame): SAM file
        prod_dict (dict): production industries
        target_avg (float | None): rescale so the weighted mean equals this

    Returns:
        gamma (dict): capital share of factor income, keyed by industry
    """
    if sam is None:
        sam = read_SAM()
    if sam is None:
        raise RuntimeError("SAM data is unavailable. Cannot compute gamma.")
    va, cap = {}, {}
    for ind, acts in prod_dict.items():
        lab = (
            sam.loc[sam.index.isin(LABOR_FACTORS), acts]
            .astype(float)
            .values.sum()
        )
        c = (
            sam.loc[sam.index.isin(CAPITAL_FACTORS), acts]
            .astype(float)
            .values.sum()
        )
        va[ind] = lab + c
        cap[ind] = c
    gamma = {k: (cap[k] / va[k] if va[k] > 0 else 0.0) for k in prod_dict}
    if target_avg is not None:
        weights = np.array([va[k] for k in prod_dict])
        current = float(
            np.average([gamma[k] for k in prod_dict], weights=weights)
        )
        gamma = {k: gamma[k] * (target_avg / current) for k in prod_dict}
    return gamma


def get_employment(prod_dict=PROD_DICT):
    """
    Employment (thousands of persons) by industry, from the packaged QLFS
    ``EMPLOYMENT`` series (see its provenance note).

    Args:
        prod_dict (dict): production industries

    Returns:
        employment (dict): employment, keyed by industry
    """
    return {s: float(EMPLOYMENT[s]) for s in prod_dict}


def get_Z(
    sam=None,
    prod_dict=PROD_DICT,
    gamma=None,
    gamma_g=0.0,
    employment=None,
    capital_output_ratio=CAPITAL_OUTPUT_RATIO,
):
    """
    Construct industry total factor productivity ``Z_m`` as the Solow
    residual of OG-Core's per-industry production function,

        Z_m = Y_m / (K_m**gamma_m * K_g**gamma_g * L_m**(1-gamma_m-gamma_g)),

    normalized so the numeraire industry (the last in ``prod_dict``,
    Manufacturing) has Z = 1. Y_m is industry value added from the SAM; L_m is
    the QLFS head count (a physical labour measure independent of the SAM's
    factor payments); K_m allocates the national capital stock
    (``capital_output_ratio`` times total value added) across industries by
    their share of capital income, the distribution implied by capital
    mobility at a common return. Public capital is one common stock, so with
    the numeraire normalization only ``gamma_g``'s effect on the labour
    exponent survives and the stock itself is never needed.

    Args:
        sam (pd.DataFrame): SAM file
        prod_dict (dict): production industries
        gamma (dict | None): capital share by industry; pass the same
            (rescaled) gamma the model uses
        gamma_g (float): public capital's output share
        employment (dict | None): employment by industry; packaged QLFS series
            when None
        capital_output_ratio (float): national K/Y anchoring capital

    Returns:
        Z (dict): TFP by industry (numeraire = 1.0)
    """
    if sam is None:
        sam = read_SAM()
    if sam is None:
        raise RuntimeError("SAM data is unavailable. Cannot compute Z.")
    slugs = list(prod_dict)
    va, cap = {}, {}
    for ind, acts in prod_dict.items():
        va[ind] = (
            sam.loc[sam.index.isin(LABOR_FACTORS + CAPITAL_FACTORS), acts]
            .astype(float)
            .values.sum()
        )
        cap[ind] = (
            sam.loc[sam.index.isin(CAPITAL_FACTORS), acts]
            .astype(float)
            .values.sum()
        )
    if employment is None:
        employment = get_employment(prod_dict)
    if gamma is None:
        gamma = get_gamma(sam, prod_dict)
    Y = np.array([va[s] for s in slugs])
    cap_income = np.array([cap[s] for s in slugs])
    L = np.array([employment[s] for s in slugs])
    g = np.array([gamma[s] for s in slugs])
    K = capital_output_ratio * Y.sum() * (cap_income / cap_income.sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        Z = Y / (K**g * L ** (1.0 - g - gamma_g))
    Z = Z / Z[slugs.index(slugs[-1])]  # numeraire (last industry) = 1
    return {s: float(z) for s, z in zip(slugs, Z)}
