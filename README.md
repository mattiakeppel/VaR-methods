# VaR-methods

MATLAB comparison of four 1-day Value-at-Risk models on a small fixed-weight portfolio of stocks, backtested over a stable period (2014–2017) and a crisis period (2018–2021).

## Methodology

VaR at 95% and 99% is estimated daily on a rolling 250-day window of portfolio returns:

- Normal: zero-mean parametric VaR, $-z_p\,\sigma$.
- HS: historical simulation (empirical quantile).
- FHS-GARCH / FHS-GJR: filtered historical simulation. A GARCH(1,1) or GJR-GARCH(1,1) is fitted to the window, standardized residuals are bootstrapped (10,000 paths), and VaR and Expected Shortfall are read from the simulated distribution. Parameters are re-estimated every 5 days.

$$\sigma_t^2 = \omega + \left(\alpha + \gamma\,\mathbb{1}_{\{\varepsilon_{t-1}<0\}}\right)\varepsilon_{t-1}^2 + \beta\,\sigma_{t-1}^2 \qquad (\gamma = 0 \text{ for GARCH})$$

Each regime is backtested with Kupiec (POF), Christoffersen independence (CCI) and conditional coverage (CC) tests at the 5% level. The script also checks the Monte Carlo stability of FHS-GJR for different numbers of simulations.

Beyond the VaR quantiles, each model's full 1-day forecast distribution is evaluated with the probability integral transform

$$u_t = \hat F_{t\mid t-1}(r_t), \qquad z_t = \Phi^{-1}(u_t)$$

using the Normal CDF, or the empirical CDF of the window (HS) or of the simulated returns (FHS). For a correct model, $u_t$ is i.i.d. Uniform(0,1) and $z_t$ is standard Normal.

## Requirements

MATLAB R2020a or later with:

- Econometrics Toolbox
- Financial Toolbox
- Risk Management Toolbox
- Statistics and Machine Learning Toolbox

## Usage

`main.m` reads `portfolio.csv` from the working directory (not included: daily Yahoo Finance close prices, 2006–2026): a `Date` column followed by one price column per stock, in the same order as the weights `W` in the script. The sample portfolio uses `ABI.BR`, `ALV.DE`, `ASML.AS`, `DB1.DE`, `DTE.DE`, `IBE.MC`, `OR.PA`, `SAN.MC`, `SAN.PA`, `SIE.DE`, `TTE.PA`, in that order.

```matlab
main
```

Results are written to `output/`:

- `figures/`: VaR plots per model and backtest summaries per regime (PDF)
- `tables/`: VaR, ES and violation series, PIT values, backtest results, Monte Carlo stability (CSV)
- `log.txt`: console output

The slides, including the PIT and Q–Q diagnostics, are in [`presentation/`](presentation/final_presentation.pdf); see its [README](presentation/README.md) to rebuild them.

## Results

On the sample portfolio (11 European stocks), no model is rejected by any test in the stable regime. In the crisis regime only FHS-GJR passes all three tests at both confidence levels:

| Model | Failures 95% (exp. 51.3) | Rejected by | Failures 99% (exp. 10.3) | Rejected by |
|---|---|---|---|---|
| Normal | 58 | CCI, CC | 26 | POF, CCI, CC |
| HS | 57 | CCI, CC | 15 | CCI, CC |
| FHS-GARCH | 59 | CCI, CC | 19 | POF, CC |
| FHS-GJR | 62 | none | 17 | none |

The crisis PIT scores show the Normal model's thin tails: 15 scores below −3, against about 1.4 expected. HS and FHS-GJR each have 13 returns outside the forecast support ($u_t$ = 0 or 1), against about 8 expected.
