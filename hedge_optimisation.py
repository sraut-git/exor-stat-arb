"""
Hedge Construction & Factor Regression
Optimises short weights via Nelder-Mead to minimise market beta,
then confirms factor neutrality with multi-factor regression.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.api import OLS, add_constant
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

TICKERS = {
    'EXOR': 'EXO.AS',
    'Ferrari': 'RACE.MI',
    'Stellantis': 'STLAM.MI',
    'CNH': 'CNH',
    'Philips': 'PHIA.AS',
}

FACTOR_TICKERS = {
    'Market': '^STOXX50E',
    'Italy': 'FTSEMIB.MI',
    'EUR_USD': 'EURUSD=X',
    'VIX': '^VIX',
}

NAV_WEIGHTS = {'Ferrari': 0.339, 'Stellantis': 0.115, 'CNH': 0.104, 'Philips': 0.141}


def download_returns(start='2009-01-01', end='2026-01-26'):
    all_tickers = {**TICKERS, **FACTOR_TICKERS}
    returns = pd.DataFrame()

    for name, ticker in all_tickers.items():
        try:
            data = yf.download(ticker, start=start, end=end, progress=False)
            if not data.empty:
                close = data['Close'].iloc[:, 0] if isinstance(data['Close'], pd.DataFrame) else data['Close']
                returns[name] = close.pct_change()
        except Exception:
            pass

    returns = returns.dropna()
    print(f"Aligned dataset: {len(returns)} trading days")
    return returns


def hedged_returns(returns, weights):
    h = returns['EXOR'].copy()
    for company, w in weights.items():
        h -= w * returns[company]
    return h


def optimise_hedge(returns):
    """Nelder-Mead minimisation of |market beta| over short weights."""
    def objective(w):
        weights = dict(zip(['Ferrari', 'Stellantis', 'CNH', 'Philips'], w))
        h = hedged_returns(returns, weights)
        model = OLS(h, add_constant(returns['Market'])).fit()
        return abs(model.params.iloc[1])

    x0 = [NAV_WEIGHTS[k] for k in ['Ferrari', 'Stellantis', 'CNH', 'Philips']]
    result = minimize(objective, x0, method='Nelder-Mead')
    opt = dict(zip(['Ferrari', 'Stellantis', 'CNH', 'Philips'], result.x))

    print("=" * 70)
    print("OPTIMAL HEDGE RATIOS")
    print("=" * 70)
    for name, w in opt.items():
        print(f"  {name:<12}: {w:.4f}  (NAV: {NAV_WEIGHTS[name]:.3f})")

    return opt


def single_factor(returns, weights):
    h = hedged_returns(returns, weights)
    returns_copy = returns.copy()
    returns_copy['Hedged'] = h
    X = add_constant(returns_copy['Market'])

    print("\n" + "=" * 70)
    print("MARKET BETAS")
    print("=" * 70)

    betas = {}
    for name in ['EXOR', 'Ferrari', 'Stellantis', 'CNH', 'Philips', 'Hedged']:
        m = OLS(returns_copy[name], X).fit()
        betas[name] = m.params.iloc[1]
        print(f"  {name:<12} beta = {m.params.iloc[1]:.4f}")

    reduction = (1 - abs(betas['Hedged']) / abs(betas['EXOR'])) * 100
    print(f"\n  Reduction: {betas['EXOR']:.4f} -> {betas['Hedged']:.4f} ({reduction:.1f}% reduction)")

    print("\nIDIOSYNCRATIC VOLATILITIES")
    for name in ['EXOR', 'Ferrari', 'Stellantis', 'CNH', 'Philips']:
        m = OLS(returns_copy[name], X).fit()
        resid_vol = np.std(returns_copy[name] - m.params.iloc[1] * returns_copy['Market'])
        print(f"  {name:<12} sigma_idio = {resid_vol:.4f}")

    return betas


def multi_factor(returns, weights):
    h = hedged_returns(returns, weights)
    factors = [c for c in ['Market', 'Italy', 'EUR_USD', 'VIX'] if c in returns.columns]
    X = add_constant(returns[factors].dropna())
    y = h.loc[X.index]

    model = OLS(y, X).fit()

    print("\n" + "=" * 70)
    print("MULTI-FACTOR REGRESSION")
    print("=" * 70)
    print(model.summary())
    return model


if __name__ == "__main__":
    returns = download_returns()
    weights = optimise_hedge(returns)
    single_factor(returns, weights)
    multi_factor(returns, weights)
