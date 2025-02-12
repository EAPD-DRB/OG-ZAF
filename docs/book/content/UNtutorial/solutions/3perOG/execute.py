"""
------------------------------------------------------------------------
This program runs the steady state solver as well as the time path
iteration solution for the model with 3-period lived agents and
exogenous labor from Chapter 5 of the OG textbook.

This Python script imports the following module(s):
    SS.py
    TPI.py

This Python script calls the following function(s):
    ss.feasible()
    ss.get_SS()
    ss.get_K
    tpi.get_TPI()
------------------------------------------------------------------------
"""

# Import packages
import numpy as np
import SS as ss
import TPI as tpi

"""
------------------------------------------------------------------------
Declare parameters
------------------------------------------------------------------------
S            = integer in [3,80], number of periods an individual lives
beta_annual  = scalar in (0,1), discount factor for one year
beta         = scalar in (0,1), discount factor for each model period
sigma        = scalar > 0, coefficient of relative risk aversion
nvec         = [S,] vector, exogenous labor supply n_{s,t}
L            = scalar > 0, exogenous aggregate labor
A            = scalar > 0, total factor productivity parameter in firms'
               production function
alpha        = scalar in (0,1), capital share of income
delta_annual = scalar in [0,1], one-year depreciation rate of capital
delta        = scalar in [0,1], model-period depreciation rate of
               capital
SS_tol       = scalar > 0, tolerance level for steady-state fsolve
SS_graphs    = boolean, =True if want graphs of steady-state objects
T            = integer > S, number of time periods until steady state
TPI_solve    = boolean, =True if want to solve TPI after solving SS
TPI_tol      = scalar > 0, tolerance level for fsolve's in TPI
maxiter_TPI  = integer >= 1, Maximum number of iterations for TPI
mindist_TPI  = scalar > 0, Convergence criterion for TPI
xi           = scalar in (0,1], TPI path updating parameter
TPI_graphs   = Boolean, =True if want graphs of TPI objects
EulDiff      = Boolean, =True if want difference version of Euler errors
               beta*(1+r)*u'(c2) - u'(c1), =False if want ratio version
               [beta*(1+r)*u'(c2)]/[u'(c1)] - 1
------------------------------------------------------------------------
"""
# Household parameters
S = int(3)
beta_annual = 0.96
beta = beta_annual**20
sigma = 3.0
nvec = np.array([1.0, 1.0, 0.2])
L = nvec.sum()
# Firm parameters
A = 1.0
alpha = 0.35
delta_annual = 0.05
delta = 1 - ((1 - delta_annual) ** 20)
# SS parameters
SS_tol = 1e-13
SS_graphs = True
# TPI parameters
T = int(round(6 * S))
TPI_solve = True
TPI_tol = 1e-13
maxiter_TPI = 200
mindist_TPI = 1e-13
xi = 0.99
TPI_graphs = True
# Overall parameters
EulDiff = False

"""
------------------------------------------------------------------------
Check feasibility
------------------------------------------------------------------------
f_params    = length 4 tuple, (nvec, A, alpha, delta)
bvec_guess1 = (2,) vector, guess for steady-state bvec (b1, b2)
b_cnstr     = (2,) Boolean vector, =True if b_s causes negative
              consumption c_s <= 0 or negative aggregate capital stock
              K <= 0
c_cnstr     = (3,) Boolean vector, =True for elements of negative
              consumption c_s <= 0
K_cnstr     = Boolean, =True if K <= 0
bvec_guess2 = (2,) vector, guess for steady-state bvec (b1, b2)
bvec_guess3 = (2,) vector, guess for steady-state bvec (b1, b2)

------------------------------------------------------------------------
"""
f_params = (nvec, A, alpha, delta)

bvec_guess1 = np.array([1.0, 1.2])
b_cnstr, c_cnstr, K_cnstr = ss.feasible(f_params, bvec_guess1)
print("bvec_guess1", bvec_guess1)
print("c_cnstr", c_cnstr)
print("K_cnstr", K_cnstr)

bvec_guess2 = np.array([0.06, -0.001])
b_cnstr, c_cnstr, K_cnstr = ss.feasible(f_params, bvec_guess2)
print("bvec_guess2", bvec_guess2)
print("c_cnstr", c_cnstr)
print("K_cnstr", K_cnstr)

bvec_guess3 = np.array([0.1, 0.1])
b_cnstr, c_cnstr, K_cnstr = ss.feasible(f_params, bvec_guess3)
print("bvec_guess3", bvec_guess3)
print("c_cnstr", c_cnstr)
print("K_cnstr", K_cnstr)

"""
------------------------------------------------------------------------
Run the steady-state solution
------------------------------------------------------------------------
bvec_guess = (2,) vector, initial guess for steady-state bvec (b1, b2)
f_params   = length 4 tuple, (nvec, A, alpha, delta)
b_cnstr    = (2,) Boolean vector, =True if b_s causes negative
             consumption c_s <= 0 or negative aggregate capital stock
             K <= 0
c_cnstr    = (3,) Boolean vector, =True for elements of negative
             consumption c_s <= 0
K_cnstr    = Boolean, =True if K <= 0
ss_params  = length 9 tuple,
             (beta, sigma, nvec, L, A, alpha, delta, SS_tol, EulDiff)
ss_output  = length 10 dictionary, {b_ss, c_ss, w_ss, r_ss, K_ss, Y_ss,
             C_ss, EulErr_ss, RCerr_ss, ss_time}
------------------------------------------------------------------------
"""
print("BEGIN EQUILIBRIUM STEADY-STATE COMPUTATION")
bvec_guess = np.array([0.1, 0.1])
f_params = (nvec, A, alpha, delta)
b_cnstr, c_cnstr, K_cnstr = ss.feasible(f_params, bvec_guess)
if not K_cnstr and not c_cnstr.max():
    ss_params = (beta, sigma, nvec, L, A, alpha, delta, SS_tol, EulDiff)
    ss_output = ss.get_SS(ss_params, bvec_guess, SS_graphs)

    beta2 = 0.55
    ss_params2 = (beta2, sigma, nvec, L, A, alpha, delta, SS_tol, EulDiff)
    ss_output2 = ss.get_SS(ss_params2, bvec_guess, False)
    print(
        "c2_ss = ",
        ss_output2["c_ss"],
        ", diff is ",
        ss_output2["c_ss"] - ss_output["c_ss"],
    )
    print(
        "b2_ss = ",
        ss_output2["b_ss"],
        ", diff is ",
        ss_output2["b_ss"] - ss_output["b_ss"],
    )
    print(
        "w2_ss = ",
        ss_output2["w_ss"],
        ", diff is ",
        ss_output2["w_ss"] - ss_output["w_ss"],
    )
    print(
        "r2_ss = ",
        ss_output2["r_ss"],
        ", diff is ",
        ss_output2["r_ss"] - ss_output["r_ss"],
    )
    print(
        "K2_ss = ",
        ss_output2["K_ss"],
        ", diff is ",
        ss_output2["K_ss"] - ss_output["K_ss"],
    )
    print(
        "Y2_ss = ",
        ss_output2["Y_ss"],
        ", diff is ",
        ss_output2["Y_ss"] - ss_output["Y_ss"],
    )
    print(
        "C2_ss = ",
        ss_output2["C_ss"],
        ", diff is ",
        ss_output2["C_ss"] - ss_output["C_ss"],
    )
    print(
        "EulErr2_ss = ",
        ss_output2["EulErr_ss"],
        ", diff is ",
        ss_output2["EulErr_ss"] - ss_output["EulErr_ss"],
    )
    print(
        "RCerr2_ss = ",
        ss_output2["RCerr_ss"],
        ", diff is ",
        ss_output2["RCerr_ss"] - ss_output["RCerr_ss"],
    )
else:
    print("Initial guess for SS bvec does not satisfy K>0 or c_s>0.")

"""
------------------------------------------------------------------------
Run the time path iteration (TPI) solution
------------------------------------------------------------------------
b_ss         = (2,) vector, steady-state savings distribution
K_ss         = scalar > 0, steady-state aggregate capital stock
C_ss         = scalar > 0, steady-state aggregate consumption
bvec1        = (2,) vector, initial period savings distribution
K1           = scalar, initial period aggregate capital stock
K_constr_tp1 = Boolean, =True if K1 <= 0
tpi_params   = length 17 tuple, (S, T, beta, sigma, nvec, L, A, alpha,
               delta, b_ss, K_ss, C_ss, maxiter_TPI, mindist_TPI, xi,
               TPI_tol, EulDiff)
tpi_output   = length 10 dictionary, {bpath, cpath, wpath, rpath, Kpath,
               Ypath, Cpath, EulErrPath, RCerrPath, tpi_time}
------------------------------------------------------------------------
"""
if TPI_solve:
    print("BEGIN EQUILIBRIUM TIME PATH COMPUTATION")
    b_ss = ss_output["b_ss"]
    K_ss = ss_output["K_ss"]
    C_ss = ss_output["C_ss"]
    bvec1 = np.array([0.8 * b_ss[0], 1.1 * b_ss[1]])
    # Make sure init. period distribution is feasible in terms of K
    K1, K_constr_tpi1 = ss.get_K(bvec1)
    if K_constr_tpi1:
        print(
            "Initial savings distribution is not feasible because "
            + "K1<=0. Some element(s) of bvec1 must increase."
        )
    else:
        tpi_params = (
            S,
            T,
            beta,
            sigma,
            nvec,
            L,
            A,
            alpha,
            delta,
            b_ss,
            K_ss,
            C_ss,
            maxiter_TPI,
            mindist_TPI,
            xi,
            TPI_tol,
            EulDiff,
        )
        tpi_output = tpi.get_TPI(tpi_params, bvec1, TPI_graphs)
