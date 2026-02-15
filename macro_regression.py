"""
Macro Factor Analysis
Regresses quarterly discount changes against macroeconomic factors
to identify systematic drivers of compression/widening.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.api import OLS, add_constant
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

df = pd.read_excel("data/NAV Model.xlsx", sheet_name="NAV History")
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').set_index('Date')
df['Discount_Change'] = df['Discount'].diff()

macro_tickers = {
    'Rate_10Y': '^TNX', 'Rate_3M': '^IRX', 'VIX': '^VIX',
    'Equity_EU': '^STOXX50E', 'Equity_US': '^GSPC', 'EUR_USD': 'EURUSD=X',
    'Oil': 'CL=F', 'Gold': 'GC=F', 'Credit': 'HYG', 'Dollar': 'DX-Y.NYB'
}

def to_quarterly(data):
    return data['Close'].resample('QE').last()

for name, ticker in macro_tickers.items():
    try:
        data = yf.download(ticker, start="2009-01-01", end="2026-01-26", progress=False)
        if not data.empty:
            q = to_quarterly(data)
            df[name] = q.reindex(df.index, method='nearest')
    except Exception:
        pass

df['Rate_Change'] = df['Rate_10Y'].diff()
df['Term_Spread'] = df['Rate_10Y'] - df['Rate_3M']
df['Term_Spread_Change'] = df['Term_Spread'].diff()
df['VIX_Change'] = df['VIX'].diff()
df['Equity_EU_Ret'] = df['Equity_EU'].pct_change()
df['Equity_US_Ret'] = df['Equity_US'].pct_change()
df['FX_Change'] = df['EUR_USD'].pct_change()
df['Oil_Ret'] = df['Oil'].pct_change()
df['Gold_Ret'] = df['Gold'].pct_change()
df['Credit_Ret'] = df['Credit'].pct_change()
df['Dollar_Change'] = df['Dollar'].pct_change()

factors = ['Rate_Change', 'Term_Spread_Change', 'VIX_Change', 'Equity_EU_Ret',
           'Equity_US_Ret', 'FX_Change', 'Oil_Ret', 'Gold_Ret', 'Credit_Ret',
           'Dollar_Change']

macro = df[['Discount_Change'] + factors].dropna()
X = add_constant(macro[factors])
y = macro['Discount_Change']

model = OLS(y, X).fit()
print(model.summary())

print("\n" + "=" * 70)
print("SIGNIFICANT FACTORS (p < 0.10)")
print("=" * 70)

sig_factors = []
for col in ['const'] + factors:
    idx = list(X.columns).index(col)
    p = model.pvalues.iloc[idx]
    b = model.params.iloc[idx]
    if p < 0.10:
        print(f"  {col:<25}: beta = {b:7.4f}, p = {p:.3f}")
        sig_factors.append(col)

if not sig_factors:
    print("  None (all p > 0.10)")

print(f"\nR-squared:     {model.rsquared:.3f}")
print(f"Adj R-squared: {model.rsquared_adj:.3f}")
print(f"F-stat p:      {model.f_pvalue:.3f}")

# Diagnostics
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].scatter(model.fittedvalues, model.resid, alpha=0.6)
axes[0, 0].axhline(y=0, color='r', linestyle='--')
axes[0, 0].set_xlabel('Fitted')
axes[0, 0].set_ylabel('Residuals')
axes[0, 0].set_title('Residuals vs Fitted')
axes[0, 0].grid(alpha=0.3)

axes[0, 1].hist(model.resid, bins=20, edgecolor='black', alpha=0.7)
axes[0, 1].set_title('Residual Distribution')
axes[0, 1].grid(alpha=0.3)

stats.probplot(model.resid, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Q-Q Plot')
axes[1, 0].grid(alpha=0.3)

axes[1, 1].scatter(y, model.fittedvalues, alpha=0.6)
axes[1, 1].plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
axes[1, 1].set_xlabel('Actual')
axes[1, 1].set_ylabel('Predicted')
axes[1, 1].set_title('Actual vs Predicted')
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('macro_regression_diagnostics.png', dpi=300)
