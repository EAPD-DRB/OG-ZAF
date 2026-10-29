(Chap_FirmCalib)=
# Calibration of Firms Parameters

## Aggregate Production Function and Capital Accumulation

The [OG-Core firm theory documentation](https://pslmodels.github.io/OG-Core/content/theory/firms.html) outlines the constant returns to scale, constant elasticity of substitution production function of the representative firm.  This function has two parameters; the elasticity of substitution and capital's share of output.

The production function is given as:

```{math}
:label: EqFirmsCESprodfun
  \begin{split}
    Y_{m,t} &= F(K_{m,t}, K_{g,m,t}, L_{m,t}) \\
    &\equiv Z_{m,t}\biggl[(\gamma_m)^\frac{1}{\varepsilon_m}(K_{m,t})^\frac{\varepsilon_m-1}{\varepsilon_m} + (\gamma_{g,m})^\frac{1}{\varepsilon_m}(K_{g,m,t})^\frac{\varepsilon_m-1}{\varepsilon_m} + \\
    &\quad\quad\quad\quad\quad(1-\gamma_m-\gamma_{g,m})^\frac{1}{\varepsilon_m}(e^{g_y t}L_{m,t})^\frac{\varepsilon_m-1}{\varepsilon_m}\biggr]^\frac{\varepsilon_m}{\varepsilon_m-1} \quad\forall m,t
  \end{split}
```

  This production function has the following parameters:
  * $\varepsilon_m$ is the elasticity of substitution between capital, labor, and infrastructure in sector $m$.
  * $\gamma_m$ is the share of capital in sector $m$.
  * $\gamma_{g,m}$ is the share of government capital in sector $m$.
  * $Z_{m,t}$ is the total factor productivity in sector $m$ at time $t$.

### Elasticity of substitution

`OG-ZAF`'s default parameterization has an elasticity of substitution of $\varepsilon=1.0$, which implies a Cobb-Douglas production function.

### Factor shares of output

In the default calibration, we set infrastructure's share of output to $\gamma_{g,m}=0.0$ for all sectors.  This parameter is hard to identify from national accounts data and would entail an empirical study to tease out the relationship between infrastructure and output.

We set $\gamma = 0.4716$, one minus labour's share of output of 0.528 in the [ILOSTAT database](https://ilostat.ilo.org/) (SDG 10.4.1, 2024). The SARB's factor-cost compensation share (51.5% in 2025) and PWT 11.0 (`labsh` 0.543, 2023) bracket it.

### Capital stock and depreciation

OG-Core's capital $K$ is private capital; the government's own capital stock is separate. We target a private capital-output ratio of 2.80: PWT 11.0 gives $K/Y = 3.82$ for South Africa in 2023 (`cn`/`cgdpo`), and the general government holds 26.7% of the fixed capital stock (SARB Quarterly Bulletin, Table E-5), leaving 2.80 for firms and households. Public corporations count as private here, as they do in the national accounts.

The depreciation rate is consumption of fixed capital over that stock, both private: total CFC is 13.65% of GDP in 2025 and the general government's is 2.31% (SARB QB, Tables S-116 and S-136), so private CFC is 11.34% of GDP and `delta_annual` $= 0.1134/2.80 = 0.0405$. The steady state then invests $(g + \delta) K/Y \approx 15\%$ of GDP privately, close to South Africa's 2010–2019 average and above the depressed 11.5% of 2025.

The capital measure matters. PWT's ratio at current national prices is lower, 2.42 in 2023 (1.77 private), because capital goods are relatively cheap in South Africa; the SARB's own replacement-cost stock is lower still, 1.82 times GDP. With $\gamma = 0.47$ those ratios imply a real return on capital of 15–20%. The 2.80 ratio implies 11.0% in the steady state, close to PWT's real internal rate of return for South Africa (12.0% in 2023, 11–13% over 2015–2023), so we use it.

### Total factor productivity

South Africa's productivity has fallen since 2008. In PWT 11.0, TFP fell 1.0% a year over 2008–2019 and real GDP per person engaged grew 0.25% a year, against 2.4% a year over 1994–2008. The Conference Board's Total Economy Database shows the same: TFP growth of −1.0% a year over 2011–2019. Studies trace the fall to the electricity crisis, the logistics state-owned enterprises and weaker state capacity.

The model's long-run growth of labour productivity is $g_y = 1.4\%$ a year (see the macro chapter), so today's economy sits below that trend. The path of TFP, `Z`, bridges the two. It starts above its long-run value and falls toward it as the shortfall closes. The steady state normalises $Z = 1$, so the long run does not depend on the path. The values come from `ogzaf/productivity.py`.

We do not take the path from a forecaster. The IMF, SARB and National Treasury growth paths assume that the energy and logistics reforms succeed, so they are reform scenarios rather than baselines. The IMF's own record shows the bias: every five-year-ahead forecast of South African growth it made from 2008 to 2020 was above the outturn, leaving aside the Covid years 2020–2021, by 2.1 percentage points a year on average ([WEO historical forecasts](https://www.imf.org/external/pubs/ft/weo/data/weohistorical.xlsx)). The baseline is instead the average, year by year, of two estimates of output per hour:

| | 2026–2030 | 2031–2040 | Long run |
|---|---|---|---|
| Our rule | 0.3% | 0.8% | 1.4% |
| OECD long-term baseline (interpolated, capped at 1.44%) | 0.8% | 1.4% | 1.44% |
| Baseline (average) | 0.5% | 1.1% | 1.4% |

- **Our rule** starts from South Africa's recent growth: real GDP grew 1.1% a year over 2022–2025. The shortfall then shrinks at the speed found across countries: about 0.3 of a country's gap to the world's growth rate carries into the next decade ([Pritchett and Summers 2014](https://www.nber.org/papers/w20573), Table 1), so the gap shrinks 11% a year and halves in six years.
- **The OECD long-term baseline** ([Economic Outlook 117, June 2025](https://sdmx.oecd.org/public/rest/data/OECD.ECO.MAD,DSD_EO_LTB@DF_EO_LTB,1.0/ZAF)) has potential output per worker growing 0.7% a year in 2025–2030 and 1.9% in 2030–2040, rising to 2.2% later. Its later rates rest on catching up with richer economies, which we do not assume, so we cap it at 1.44%, South Africa's own growth in output per worker over 1994–2019.

TFP is then fitted year by year from 2026 to 2050 so that the baseline's output per hour follows the average path; after 2050 the remaining gap keeps shrinking at the rule's speed. In 2025 TFP stands 8.3% above its long-run level. Output per hour grows 0.4% a year in 2026–2028, 1.1% in the 2030s and 1.4% in the long run. Labour input is set by the demographics at today's employment rates by age, which the earnings matrix carries, so GDP grows faster than output per hour while the working-age population grows: 1.6% in 2026 and about 2.1% a year in the 2030s.

In the case of multiple production sectors, we use {cite}`PRS2020` who identify TFP for various sectors in South Africa.
