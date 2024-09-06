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

We use a default value of $\gamma =0.40$, which corresponds to one minus labor's share of output, where labor's share of output is found as 0.60 in in the [UN ILOSTAT database](https://rshiny.ilo.org/dataexplorer9/?lang=en).

### Total factor productivity

In the case of the single prodcution sector, we can normalize $Z_{m,t}=1.0$.  In the case of multiple production sectors, we use [Julius Pain, Mpho Rapapali and Daan Steenkamp, "Industry TFP estimates for South Africa", Chapter 5 in Occasional Bulletin of Economic Notes, OBEN/20/02, South African Reserve Bank, Nov. 2020](https://econpapers.repec.org/scripts/redir.pf?u=https%3A%2F%2Fwww.resbank.co.za%2Fcontent%2Fdam%2Fsarb%2Fpublications%2Foccasional-bulletin-of-economic-notes%2F2020%2F10412%2FOBEN%25202002%2520%28Industry%2520TFP%2520estimates%2520for%2520South%2520Africa%29%2520-%2520November%25202020.pdf;h=repec:rbz:oboens:10412) who identify TFP for various sectors in South Africa.