[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/FxAEmrI0)
# Actuarial Theory and Practice A

_"Tell me and I forget. Teach me and I remember. Involve me and I learn." – Benjamin Franklin_

---

### Congrats on completing the [2026 SOA Research Challenge](https://www.soa.org/research/opportunities/2026-student-research-case-study-challenge/)!


> Now it's time to build your own website to showcase your work.  
> Creating a website using GitHub Pages is simple and a great way to present your project.

This page is written in Markdown.
- Click the [assignment link](https://classroom.github.com/a/FxAEmrI0) to accept your assignment.

---

> Be creative! You can embed or link your [data](player_data_salaries_2020.csv), [code](sample-data-clean.ipynb), and [images](ACC.png) here.

More information on GitHub Pages can be found [here](https://pages.github.com/).

![](Actuarial.gif)

## Objective Overview

This document covers the research and analysis performed by the actuarial team of Galaxy General Insurance Company (herein GGIC) to produce an insurance product suite for Cosmic Quarry Mining Corporation (herein CQMC) and their operations across multiple solar systems (Helionis Cluster, Bayesia System, and Oryn Delta). These insurance products target the client's four main operational hazards: Equipment Failure, Cargo Loss, Worker's Compensation, and Business Interruption.

## Data

Our pricing models were trained upon historical claims experience of a similar GGIC product suite, and can be viewed here:
- [Business Interruption Claims](https://www.soa.org/globalassets/assets/files/research/opportunities/2026/student-research-case-study/srcsc-2026-claims-business-interruption.xlsx)
- [Workers Compensation Claims](https://www.soa.org/globalassets/assets/files/research/opportunities/2026/student-research-case-study/srcsc-2026-claims-workers-comp.xlsx)
- [Cargo Loss Claims](https://www.soa.org/globalassets/assets/files/research/opportunities/2026/student-research-case-study/srcsc-2026-claims-cargo.xlsx)
- [Equipment Failure Claims](https://www.soa.org/globalassets/assets/files/research/opportunities/2026/student-research-case-study/srcsc-2026-claims-equipment-failure.xlsx)
- [Interest and Inflation Rates](https://www.soa.org/globalassets/assets/files/research/opportunities/2026/student-research-case-study/srcsc-2026-interest-and-inflation.xlsx)
- [Data Dictionary](https://www.soa.org/globalassets/assets/files/research/opportunities/2026/student-research-case-study/srcsc-2026-data.pdf)

CQMC-specific data was also provided to gain an understanding of their exposure:
- [CQMC Inventory](https://www.soa.org/globalassets/assets/files/research/opportunities/2026/student-research-case-study/srcsc-2026-cosmic-quarry-inventory.xlsx)
- [CQMC Personnel](https://www.soa.org/globalassets/assets/files/research/opportunities/2026/student-research-case-study/srcsc-2026-cosmic-quarry-personnel.xlsx)

## External Libraries

For analysis and modelling, the following libraries were used:
**Python**
```python
import pandas
import numpy
import scipy
import sklearn
import matplotlib
import seaborn
import statsmodels
import warnings
import patsy
```
**R**
```r

```

## Data Assumptions & Limitations

2. Product Designs
Tanvir

3. Model Building
Gilbert

## 4. Capital Modelling

The capital modelling was conducted to quantify the potential financial loss of each product line. The stochastic aggregate loss models are built for each hazard: Equipment Failure, Cargo Loss, Workers’ Compensation, and Business Interruption. 

The aggregate annual loss is modelled through a compound frequency and severity framework, where frequency and severity distributions for each line were estimated using historical claims data. Combining the number of claims in a year and the claim amounts retrieved through 100,000 Monte Carlo simulations, the aggregate loss distributions capture loss statistics including expected loss, variance and extreme tail events. 

#### Methodology

Our capital modelling process involves four key steps:
  1. Cleaning and validating historical claim data.
  2. Estimating parameters of separate frequency and severity distributions for each line of businesses.
  3. Simulating annual aggregate losses using Monte Carlo simulation.
  4. Summarising key metrics including expectation, variance, and Value-at-Risk (VaR).

#### Model Choices

| Line of Business | Frequency Model | Severity Model |
|---|---|---|
| Equipment Failure | Poisson | Gamma |
| Cargo Loss | Negative Binomial | Beta × cargo value |
| Business Interruption | Negative Binomial | Gamma |
| Workers’ Compensation | Poisson | Lognormal |

#### Simulation Logic

In each iteration of the Monte Carlo simulation implemented, the annual claim count is simulated first from the frequency distributions fitted, followed by individual claim severities simulated using fitted severity distributions. Summing up the claim amounts allows us to obtain total annual loss.

A simple example - Equipment Failure aggregate loss simulation

```r
# 1. Inputs
r_cargo <- 0.5370
p_cargo <- 0.6868
a_cargo <- 1.3832
b_cargo <- 14.3850

cargo_values <- cargo_data_raw$cargo_value
cargo_values <- cargo_values[!is.na(cargo_values) & cargo_values > 0]

# 2. Simulate aggregate cargo loss
cargo_loss <- numeric(n_sim)

for(i in 1:n_sim){
  
  n_claims <- rnbinom(1, size = r_cargo, prob = p_cargo)
  
  if(n_claims > 0){
    damage_ratio <- rbeta(n_claims, shape1 = a_cargo, shape2 = b_cargo)
    sampled_values <- sample(cargo_values, size = n_claims, replace = TRUE)    
    losses <- damage_ratio * sampled_values
    cargo_loss[i] <- sum(losses)
  }
}

# 3. Summary table
var95_cargo  <- as.numeric(quantile(cargo_loss, 0.95))
var99_cargo  <- as.numeric(quantile(cargo_loss, 0.99))
var995_cargo <- as.numeric(quantile(cargo_loss, 0.995))
```

This logic was applied to each product line, using their unique fitted distributions listed in the section above.

#### Aggregate Loss Simulation Results

| Metric | Equipment Failure | Cargo Loss | Business Interruption | Workers’ Compensation |
|---|---:|---:|---:|---:|
| Expected annual loss ($) | 8,935 | 1,978,730 | 443,322 | 67.93 |
| Standard deviation ($) | 23,011 | 11,396,840 | 2,958,488 | 943.56 |
| VaR 95% ($) | 61,858 | 6,597,604 | 596,933 | 0 |
| VaR 99% ($) | 103,491 | 57,528,540 | 13,684,392 | 1,672.69 |
| VaR 99.5% ($) | 120,090 | 83,444,880 | 21,261,136 | 3,720.63 |

The aggregate loss distributions are strongly right-skewed across all four products given their extremely large VaR. Cargo Loss and Business Interruption are the main tail-risk drivers due to the material size of their loss, while Equipment Failure and Workers’ Compensation are comparatively stable and manageable.


#### Stress testing

We tested two stressed scenarios in addition to the baseline scenario:
- **Moderate stress:** claim frequency +25%, claim severity +10%
- **Extreme stress:** claim frequency +50%, claim severity +30%

| Product | Baseline VaR 99% | Moderate Stress | Extreme Stress |
|---|---:|---:|---:|
| Equipment Failure | 102,773 | 129,148 | 157,252 |
| Business Interruption | 59,961,210 | 76,315,150 | 94,751,300 |
| Cargo Loss | 13,684,390 | 18,918,430 | 26,962,710 |
| Workers’ Compensation | 1,673 | 2,494 | 3,430 |

Under systematic stress, tail losses rise sharply for all four products, especially for Business Interruption and Cargo Loss. This reveals that these two products require the greatest capital support to comply with the regulatory requirement and internal risk tolerance, which also would benefit most from stop-loss reinsurance protection.

#### Inflation Adjustment

As a final but important step, we accounted for inflation by indexing claim severities by **2.46%** per annum (15-year average inflation rate). This adjustment is relevant to:

  - Repair and replacement for **Equipment Failure**.
  - Value of goods transported for **Cargo Loss**.
  - Cost of disrupted operations for **Business Interruption**.
  - Claim costs over time for **Workers’ Compensation**.

This adjustment increases both expected loss and tail risk of the simulated results.



5. Risk Considerations
Olivia










