# Credit Risk VaR Simulator

A Monte Carlo credit-loss model for a 100-counterparty bond portfolio, built to answer one
question: **how wrong is the normal approximation when you use it to size a tail risk?**

The answer, at the 99.9% level, is that it reports about **60% of the real loss** — and the
simulation is structured to demonstrate exactly that rather than assert it.

## How it works

Each counterparty holds a credit rating across 8 states (AAA down to default) and can migrate
between them over the horizon. A counterparty's migration is driven by a latent creditworthiness
variable built from two pieces: a **systemic** shock from its assigned credit driver, and an
**idiosyncratic** shock unique to it, mixed by the counterparty's beta. The systemic drivers are
correlated with each other, and correlated draws are generated through a Cholesky factor of the
driver correlation matrix. Where that latent variable falls relative to the counterparty's
credit-state boundaries — recovered from its migration probabilities with an inverse normal — sets
its rating at the horizon, and therefore its loss. Defaults are additionally scaled by
`(1 - recovery rate)`.

That machinery is then run at several sampling budgets so they can be compared against each other:

| Label | Scenarios | Purpose |
|---|---|---|
| Out-of-sample | 100,000 | the reference distribution — treated as "true" |
| MC1 | 1,000 systemic × 5 idiosyncratic = 5,000 | in-sample, systemic-light |
| MC2 | 5,000 systemic × 1 idiosyncratic = 5,000 | in-sample, systemic-heavy |
| N0 / N1 / N2 | analytic | normal approximation fitted to the loss moments |

MC1 and MC2 both spend 5,000 scenarios but split them differently between systemic and
idiosyncratic sampling, which is the comparison that shows where sampling budget actually matters.

Two portfolios are evaluated throughout: **Portfolio 1** weighted by counterparty value, and
**Portfolio 2** equally weighted across all 100 counterparties.

## Results

All figures are **one-period credit-migration loss, in dollars, on a 100-counterparty portfolio**,
from 100,000 out-of-sample scenarios. VaR and CVaR are meaningless without their confidence level
and portfolio, so both are carried on every row.

**Portfolio 1 (value-weighted):**

| Method | 99% VaR | 99% CVaR | 99.9% VaR | 99.9% CVaR |
|---|---|---|---|---|
| Out-of-sample (reference) | $37.44M | $45.24M | $54.13M | $61.94M |
| In-sample MC1 (1000×5) | $37.11M | $44.68M | $53.67M | $61.22M |
| In-sample MC2 (5000×1) | $37.26M | $44.90M | $54.39M | $61.90M |
| **Normal approximation** | **$26.41M** | **$29.32M** | **$32.97M** | **$35.35M** |
| KDE (smoothed empirical) | $37.56M | $45.28M | $54.29M | $62.00M |

**Portfolio 2 (equally weighted):**

| Method | 99% VaR | 99% CVaR | 99.9% VaR | 99.9% CVaR |
|---|---|---|---|---|
| Out-of-sample (reference) | $27.87M | $33.88M | $42.47M | $48.21M |
| In-sample MC1 (1000×5) | $27.44M | $33.62M | $41.70M | $47.78M |
| In-sample MC2 (5000×1) | $27.45M | $33.59M | $41.06M | $47.64M |
| **Normal approximation** | **$21.23M** | **$23.40M** | **$26.14M** | **$27.92M** |
| KDE (smoothed empirical) | $27.91M | $33.91M | $42.53M | $48.26M |

### The normal approximation fails, and it fails worse the further into the tail you go

Averaged over 100 trials, as a percentage of the true value:

| | 99% VaR | 99% CVaR | 99.9% VaR | 99.9% CVaR |
|---|---|---|---|---|
| Normal, Portfolio 1 | 69.9% | 64.3% | **60.4%** | **56.6%** |
| Monte Carlo, Portfolio 1 | 99.1–99.5% | 98.7–99.3% | 99.1–100.5% | 98.8–99.9% |

At the 99.9% level the normal model reports 60% of the true VaR and 57% of the true CVaR. A bank
sizing capital off it would hold roughly **$21 million too little** against this portfolio. The
reason is structural rather than a tuning problem: credit loss distributions are heavily
right-skewed — most scenarios see no defaults at all, a few see many correlated ones — and a normal
fitted to the mean and variance of that has no way to represent the shape of the tail it is being
asked about.

Monte Carlo, by contrast, lands within about 1% of the reference at every level.

### Sampling error is measured, not assumed

The in-sample estimators were re-run across **100 independent trials** and the standard deviation of
each estimate recorded, which is what makes the accuracy figures above meaningful:

| Estimator | 99% VaR std | 99.9% VaR std |
|---|---|---|
| MC1 (1000 systemic × 5) | $1.40M | $4.10M |
| MC2 (5000 systemic × 1) | $1.02M | $3.26M |

Two things follow. Sampling error roughly **triples** moving from the 99% to the 99.9% quantile —
deeper tail, fewer scenarios out there to estimate from. And **MC2 beats MC1 at both levels** on the
same budget of 5,000 scenarios: spending the budget on more systemic draws rather than more
idiosyncratic ones per draw gives a tighter estimate, because the systemic factor is what drives
the correlated defaults that populate the tail.

At 99.9%, MC1's standard deviation of $4.10M is about 7.6% of the estimate — so a single run's
99.9% VaR should be read as an estimate with real width, not a point.

## Running it

```bash
pip install -r requirements.txt
jupyter notebook credit_risk_simulation.ipynb
```

No solver and no licence needed — this is NumPy and SciPy throughout.

**The input data is not in this repository.** The notebook expects two files beside it:

- `instrum_data.csv` — one row per counterparty, no header: ID, credit driver index, beta,
  expected recovery rate, value, then 8 credit-state migration probabilities (default → AAA), then
  8 corresponding exposures, then the market return.
- `credit_driver_corr.csv` — the correlation matrix between systemic credit drivers, **tab
  separated**, no header.

The 100,000-scenario generation takes a while, so the notebook caches it to `losses_out.npz` and
reuses that file on later runs if it exists.

**All 12 plots and every result table are committed with the notebook**, so the full study —
loss distributions per portfolio, the KDE-versus-empirical overlay, and the sampling-error tables —
renders on GitHub without running anything.

## What I would do next

**Give the KDE something independent to estimate.** The KDE is fitted to the same 100,000
out-of-sample losses that the empirical quantile is read from, so its near-exact agreement with the
reference ($37.56M against $37.44M) demonstrates that the smoothing is faithful — not that it
generalises. Fitting it on the 5,000 in-sample draws and scoring it against the 100,000-scenario
reference would turn it into a real test of whether smoothing buys you tail accuracy that raw
in-sample quantiles do not.

**Put a confidence interval on the reported VaR.** The 100-trial study already produces the standard
deviation; carrying it through to the headline numbers would present them as the interval estimates
they are.

**Vary the assumptions that the whole model rests on.** The latent variables are Gaussian, the
driver correlations are fixed and estimated outside the model, betas are constant, and recovery
rates are deterministic point estimates. Gaussian latent variables in particular are the assumption
credit models are most often criticised for, precisely because they understate joint extreme
defaults — the same failure mode this study catches the normal approximation making at the portfolio
level. Re-running with a t-copula would test whether the model's own tail is as fat as it should be.

---

*Originally built as a graduate course project at the University of Toronto.*
