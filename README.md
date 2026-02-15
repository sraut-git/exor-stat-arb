# EXOR Holdco Statistical Arbitrage

A market-neutral statistical arbitrage strategy exploiting the persistent discount of EXOR N.V. (John Elkann's holding company) to its net asset value (NAV). The project constructs a hedged position that isolates discount compression as the sole source of alpha.

# Thesis

EXOR trades at a ~45-55% discount to its NAV, which consists primarily of public equity stakes in Ferrari, Stellantis, CNH Industrial, and Philips. The discount is statistically mean-reverting, and management is actively catalysing compression through share buybacks and asset monetisation.

Trade structure: Long EXOR, short portfolio companies (weighted to achieve market neutrality), profit from discount narrowing.

# Methodology

1. NAV Replication (`nav_model.py`)
- Replicates EXOR's daily NAV from disclosed public holdings (share counts × market prices), adjusted for FX (EUR/USD), private holdings, and net debt
- Calculates daily discount-to-NAV time series
- Reconciles against EXOR's reported NAV (targeting ±10% accuracy)

2. Mean Reversion Analysis (`mean_reversion.py`)
- Augmented Dickey-Fuller test: tests null hypothesis that discount follows a random walk
- Zivot-Andrews test: tests for mean reversion allowing for structural breaks (identifies regime changes e.g. COVID, Ferrari spinoff)
- AR(1) modelling: estimates mean-reversion speed (φ), long-run equilibrium discount (μ), and half-life
- Predictive regression: tests whether current discount level predicts future discount changes (timing signal)
- Regime analysis: compares full-sample vs post-2020 parameters to identify current regime dynamics

# 3. Ornstein-Uhlenbeck Process (`ou_analysis.py`)
- Fits continuous-time OU process to the discount series: `dX = θ(μ - X)dt + σdW`
- Estimates mean-reversion speed (θ), long-run mean (μ), and volatility (σ) via AR(1) regression
- Identifies structural break points and analyses pre/post-break regimes separately
- Generates investment signals based on current discount vs estimated equilibrium

# 4. Hedge Construction & Factor Regression (`hedge_optimisation.py`)
- Single-factor regression: estimates market beta of EXOR and each portfolio company vs Euro Stoxx 50
- Nelder-Mead optimisation: finds optimal short weights across Ferrari, Stellantis, CNH, Philips to minimise hedged portfolio's market beta (targeting β ≈ 0)
- Multi-factor regression: tests hedged portfolio against market, luxury sector, Italian equities, EUR/USD, and VIX to confirm factor neutrality
- Reports beta reduction (typically >95% market beta eliminated)

# 5. Macro Factor Analysis (`macro_regression.py`)
- Regresses quarterly discount changes against 10 macroeconomic factors (rates, term spread, VIX, equities, FX, oil, gold, credit, dollar)
- Identifies statistically significant catalysts for discount compression/widening
- Diagnostic plots: residuals, Q-Q, fitted vs actual

# 6. Buyback Catalyst Study (`buyback_catalyst.py`)
- Event study around EXOR's three major buyback announcements (2023 tender €750M, 2024 buyback €250M, 2025 tender €1B)
- Measures average discount change in ±30 day window around each event
- Buyback event study shows mixed results — no consistent short-term discount compression around announcements, suggesting the compression mechanism operates over longer horizons than the 30-day window tested.

# Key Results
- Discount is statistically mean-reverting in the post-COVID regime (ADF and Zivot-Andrews tests reject unit root)
- AR(1) half-life: approximately 4-6 quarters depending on regime
- Optimised hedge achieves >95% market beta reduction (β ≈ 0 vs Euro Stoxx 50)
- Hedged position is neutral to luxury sector, Italian market, FX, and volatility factors
- Buyback announcements are associated with discount compression

# Tech Stack
- Python: NumPy, pandas, statsmodels, SciPy, scikit-learn, yfinance, matplotlib
- Statistical methods: ADF test, Zivot-Andrews structural break test, AR(1) autoregression, Ornstein-Uhlenbeck process estimation, OLS regression, Nelder-Mead optimisation
- Data: Yahoo Finance (daily prices), EXOR annual reports (holdings, NAV)

# Repository Structure

```
├── README.md
├── nav_model.py              # NAV replication model
├── mean_reversion.py         # ADF, Zivot-Andrews, AR(1), predictive regression
├── ou_analysis.py            # Ornstein-Uhlenbeck process fitting
├── hedge_optimisation.py     # Factor regression & Nelder-Mead hedge optimisation
├── macro_regression.py       # Macro factor analysis
├── buyback_catalyst.py       # Buyback event study
└── requirements.txt          # Dependencies
```

# Usage

```bash
pip install -r requirements.txt
python nav_model.py           # Build NAV and discount time series
python mean_reversion.py      # Run stationarity tests and AR(1) model
python hedge_optimisation.py  # Optimise hedge ratios
```

Note: Requires an internet connection for Yahoo Finance data downloads.

# Disclaimer

This project was built for educational and competition purposes. It does not constitute investment advice. Past statistical relationships may not persist.
