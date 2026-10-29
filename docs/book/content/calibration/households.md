(Chap_HouseholdCalib)=
# Calibration of Household Preference Parameters

## Behavioral Assumptions

### Elasticity of labor supply

As discussed in the [OG-Core household theory documentation](https://pslmodels.github.io/OG-Core/content/theory/households.html), we use the elliptical disutility of labor function developed by {cite}`EvansPhillips:2017`.  We then fit the parameters of the elliptical utility function to match the marginal disutility from a constant Frisch elasticity function.  `OG-ZAF` users enter the constant Frisch elasticity as a parameter.  {cite}`Peterman:2016` finds a range of Frisch elasticities estimated from microeconomic and macroeconomic data.  These range from 0 to 4.  Peterman makes the case that in lifecycle models without an extensive margin for employment the  Frisch elasticity should be higher. For `OG-ZAF` we take a default value of 0.4 from {cite}`Altonji:1986`.

### Intertemporal elasticity of substitution

The default value for the intertemporal elasticity of substitution, $\sigma$, is taken from {cite}`ABMW:1999`.  We set $\sigma=1.5$.

### Rate of time preference

The rate of time preference is calibrated, not borrowed. With the capital share, depreciation and tax rates set from data, $\beta$ moves the return on capital and so the capital-output ratio. We set $\beta = 0.8706$ (annual, the same for every lifetime-income group), the value at which the steady-state private capital-output ratio equals 2.80 (see the firms chapter). This is lower than the 0.96 common in models of high-saving economies, and it matches South Africa's low domestic saving: gross national saving in 2025 was about 13.4% of GDP (gross capital formation of 13.9% less a current account deficit of 0.5%, SARB Quarterly Bulletin Tables S-116 and S-159). The implied real return on capital is 11.0%, against PWT's measured 12.0%.
