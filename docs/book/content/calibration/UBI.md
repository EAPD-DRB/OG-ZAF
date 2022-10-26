(Chap_UBI)=
# Universal Basic Income (UBI)

[TODO: This section is far along but needs to be updated.]

We have included the modeling of a universal basic income (UBI) policy directly in the theory and code for [`OG-Core`] on which dependency the `OG-USA` is based. UBI shows up in the household budget constraint {eq}`EqHHBC`, and is described in the [Budget Constraint](https://pslmodels.github.io/OG-Core/content/theory/households.html#budget-constraint) section of the Households chapter of the `OG-Core` documentation. We calculate the time series of a UBI matrix $ubi_{j,s,t}$ representing the UBI transfer to every household with head of household age $s$, lifetime income group $j$, in period $t$. We calculate the time series of this matrix from five parameters and some household composition data that we impose upon the existing demographics of `OG-USA`.


(SecUBIcalc)=
## Calculating UBI

  We calculate the time series of UBI household transfers in model units $ubi_{j,s,t)}$ and the time series of total UBI expenditures in model units $UBI_t$ from five parameters described in the [`ogusa_default_parameters.json`](https://github.com/PSLmodels/OG-USA/blob/master/ogusa/ogusa_default_parameters.json) file (`ubi_growthadj`, `ubi_nom_017`, `ubi_nom_1864`, `ubi_nom_65p`, and `ubi_nom_max`) interfaced with the `OG-USA` demographic dynamics over lifetime income groups $j$ and ages $s$, and multiplied by household composition matrices from the `Calibrate` class of the `OG-USA/ogusa/calibrate.py` module in the repository.

  From the [OG-USA](https://github.com/PSLmodels/OG-USA) repository, we have four $S\times J$ matrices `ubi_num_017_mat`$_{j,s}$, `ubi_num_1864_mat`$_{j,s}$, and `ubi_num_65p_mat`$_{j,s}$ representing the number of children under age 0-17, number of adults ages 18-64, and the number of seniors age 65 and over, respectively, by lifetime ability group $j$ and age $s$ of head of household. Because our demographic age data match up well with head-of-household data from other datasets, we do not have to adjust the values in these matrices.[^HOH_age_dist_note]

  Now we can solve for the dollar-valued (as opposed to model-unit-valued) UBI transfer to each household in the first period $ubi^{\$}_{j,s,t=0}$ in the following way. Let the parameter `ubi_nom_017` be the dollar value of the UBI transfer to each household per dependent child age 17 and under. Let the parameter `ubi_nom_1864` be the dollar value of the UBI transfer to each household per adult between the ages of 18 and 64. Let `ubi_nom_65p` be the dollar value of UBI transfer to each household per senior 65 and over. And let `ubi_nom_max` be the maximum UBI benefit per household.

  ```{math}
  :label: EqUBIubi_dol_jst0
    \begin{split}
      ubi^{\$}_{j,s,t=0} = \min\Bigl(&\texttt{ubi_nom_max}, \\
      &\texttt{ubi_nom_017} * \texttt{ubi_num_017_mat}_{j,s} + \\
      &\texttt{ubi_nom_1864} * \texttt{ubi_num_1864_mat}_{j,s} + \\
      &\texttt{ubi_nom_65p} * \texttt{ubi_num_65p_mat}_{j,s}\Bigr) \quad\forall j,s
    \end{split}
  ```

  The rest of the time periods of the household UBI transfer and the respective steady-states are determined by whether the UBI is growth adjusted or not as given in the `ubi_growthadj` Boolean parameter. The following two sections cover these two cases.


(SecUBI_NonGrowthAdj)=
## UBI specification not adjusted for economic growth

  A non-growth adjusted UBI (`ubi_growthadj = False`) is one in which the initial nonstationary dollar-valued $t=0$ UBI matrix $ubi^{\$}_{j,s,t=0}$ does not grow, while the economy's long-run growth rate is $g_y$ for the most common parameterization where the long-run growth rate is positive $g_y>0$.

  ```{math}
  :label: EqUBIubi_dol_NonGrwAdj_jst
    ubi^{\$}_{j,s,t} = ubi^{\$}_{j,s,t=0} \quad\forall j,s,t
  ```

  As described in the [OG-Core chapter on stationarization](https://pslmodels.github.io/OG-Core/content/theory/stationarization.html), the stationarized UBI transfer to each household $\hat{ubi}_{j,s,t}$ is the nonstationary transfer divided by the growth rate since the initial period. When the long-run economic growth rate is positive $g_y>0$ and the UBI specification is not growth-adjusted the steady-state stationary UBI household transfer is zero $\overline{ubi}_{j,s}=0$ for all lifetime income groups $j$ and ages $s$ as time periods $t$ go to infinity. However, to simplify, we assume in this case that the stationarized steady-state UBI transfer matrix to households is the stationarized value of that matrix in period $T$.

  ```{math}
  :label: EqUBIubi_mod_NonGrwAdj_SS
    \overline{ubi}_{j,s} = ubi_{j,s,t=T} \quad\forall j,s
  ```

  Note that in non-growth-adjusted case, if $g_y<0$, then the stationary value of $\hat{ubi}_{j,s,t}$ is going to infinity as $t$ goes to infinity. Therefore, a UBI specification must be growth adjusted for any assumed negative long run growth $g_y<0$.[^GrowthAdj_note]


(SecUBI_GrowthAdj)=
## UBI specification adjusted for economic growth

  Put description of growth-adjusted specification here.


(SecUBIfootnotes)=
## Footnotes

[^HOH_age_dist_note]: DeBacker and Evans compared the `OG-USA` age demographics $\hat{\omega}_{s,t}$ with the respective age demographics in Tax Policy Center's microsimulation model and in [Tax-Calculator](https://github.com/PSLmodels/Tax-Calculator)'s microsimulation model. The latter two microsimulation models' age demographics are based on head of household tax filer age distributions, whereas `OG-USA`'s demographics are based on the population age distribution.

[^GrowthAdj_note]: We impose this requirement of `ubi_growthadj = False` when `g_y_annual < 0` in the [`ogusa_default_parameters.json`](https://github.com/PSLmodels/OG-USA/blob/master/ogusa/ogusa_default_parameters.json) "validators" specification of the parameter.
