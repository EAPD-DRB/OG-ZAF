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

In the case of the single production sector, we can normalize $Z_{m,t}=1.0$.  In the case of multiple production sectors, we use {cite}`PRS2020` who identify TFP for various sectors in South Africa.
