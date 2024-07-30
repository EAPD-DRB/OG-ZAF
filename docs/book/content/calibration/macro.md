(Chap_MacroCalib)=
# Calibration of Macroeconomic Parameters

## Economic Assumptions

As the default rate of labor augmenting technological change, $g_y$, we use a value of 0%.  The average annual growth rate in GDP per capita in South Africa between 2007 and 2023 is 0.0% per year.

## Open Economy Parameters

### Foreign holding of government debt in the initial period

The path of foreign holding of domestic debt is endogenous, but the initial period stock of debt held by foreign investors is exogenous.  We set this parameter, `initial_foreign_debt_ratio` to 0.26, consistent with data from the World Bank WDI.

### Foreign purchases of newly issued debt

We set $\zeta_D = 0.23$.  This is the average share of foreign purchases of newly issued government debt found from the World Bank WDI.

### Foreign holdings of excess capital

We set $\zeta_K = 0.9$. Note, this parameter is harder to pin down from the data as foreign purchases on "excess" capital demand is not typically directly measured or reported.  A value of 0.9 implies a high degree of openness to international capital flows.

## Government Debt, Spending and Transfers

### Government Debt

The path of government debt is endogenous.  But the initial value is exogenous.  To avoid converting between model units and dollars, we calibrate the initial debt to GDP ratio, rather than the dollar value of the debt.  This is the model parameter $\alpha_D$.  We compute this from the ratio of publicly held debt outstanding to GDP.  Based on 2023 values, this gives us a ratio of 0.59.

### Aggregate transfers

Aggregate (non-Social Security) transfers to households are set as a share of GDP with the parameter $\alpha_T$. We exclude Social Security from transfers since it is modeled specifically. With this definition, the share of transfers to GDP in 2015 is 0.04 according to [IMF data](https://data.imf.org/?sk=b052f0f0-c166-43b6-84fa-47cccae3e219&hide_uv=1).

### Government expenditures

Government spending on goods and services are also set as a share of GDP with the parameter $\alpha_G$. We define government spending as:
    <center>Government Spending = Total Outlays - Transfers - Net Interest on Debt - Social Security</center>
With this definition, the share of government expenditure to GDP is 0.267 based on [data from the IMF](https://data.imf.org/?sk=b052f0f0-c166-43b6-84fa-47cccae3e219&hide_uv=1).
