"""
Mean Reversion Analysis for EXOR Holdco Discount
Tests stationarity via ADF and Zivot-Andrews, fits AR(1) model,
and runs predictive regression for timing signal.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import adfuller, zivot_andrews
from statsmodels.api import OLS, add_constant
import warnings
warnings.filterwarnings('ignore')


def load_discount_data(filepath, sheet_name="NAV History"):
    df = pd.read_excel(filepath, sheet_name=sheet_name)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index()
    if df['Discount'].max() > 1:
        df['Discount'] = df['Discount'] / 100
    df = df.dropna(subset=['Discount'])
    return df


def descriptive_stats(df):
    d = df['Discount']
    z = (d.iloc[-1] - d.mean()) / d.std()
    print("=" * 60)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 60)
    print(f"Observations:     {len(d)}")
    print(f"Mean discount:    {d.mean():.1%}")
    print(f"Current discount: {d.iloc[-1]:.1%}")
    print(f"Min / Max:        {d.min():.1%} / {d.max():.1%}")
    print(f"Current z-score:  {z:.1f} std devs from mean")
    return z


def fit_ar1(series):
    """
    d(t) = c + phi * d(t-1) + eps
    Long-run mean: mu = c / (1 - phi)
    Half-life: ln(0.5) / ln(phi)
    """
    model = AutoReg(series, lags=1, trend='c').fit()
    c = model.params.iloc[0]
    phi = model.params.iloc[1]
    mu = c / (1 - phi)
    hl = np.log(0.5) / np.log(phi)

    ci = model.conf_int(alpha=0.05)
    phi_lo, phi_hi = ci.iloc[1, 0], ci.iloc[1, 1]
    hl_lo = np.log(0.5) / np.log(phi_lo) if phi_lo > 0 else float('inf')
    hl_hi = np.log(0.5) / np.log(phi_hi) if phi_hi > 0 else float('inf')

    print("\nAR(1) MODEL")
    print(f"  phi:       {phi:.4f}  (95% CI: [{phi_lo:.3f}, {phi_hi:.3f}])")
    print(f"  mu:        {mu:.2%}")
    print(f"  Half-life: {hl:.2f} quarters = {hl/4:.2f} years  (CI: [{hl_lo:.2f}, {hl_hi:.2f}])")

    return {'phi': phi, 'mu': mu, 'half_life': hl, 'model': model}


def adf_test(series, label=""):
    result = adfuller(series, regression='c', maxlag=4)
    print(f"\nADF TEST {label}")
    print(f"  Statistic: {result[0]:.4f}")
    print(f"  P-value:   {result[1]:.4f}")
    for k, v in result[4].items():
        print(f"  {k}: {v:.4f}")

    verdict = "Reject random walk (mean-reverting)" if result[1] < 0.05 else "Cannot reject random walk"
    print(f"  --> {verdict}")
    return {'statistic': result[0], 'p_value': result[1]}


def za_test(df):
    za = zivot_andrews(df['Discount'], trim=0.15, regression='c')
    break_date = df.index[za[4]]

    print(f"\nZIVOT-ANDREWS TEST")
    print(f"  Statistic:  {za[0]:.4f}")
    print(f"  P-value:    {za[1]:.4f}")
    print(f"  Break date: {break_date.strftime('%Y-%m-%d')}")
    for k, v in za[2].items():
        print(f"  {k}: {v:.4f}")

    verdict = f"Mean-reverting with break at {break_date.strftime('%Y-%m')}" if za[1] < 0.05 else "Cannot reject unit root"
    print(f"  --> {verdict}")
    return {'statistic': za[0], 'p_value': za[1], 'break_date': break_date}


def predictive_regression(df):
    """Does current discount predict next period's change? Negative beta = timing signal."""
    tmp = df[['Discount']].copy()
    tmp['Change_Next'] = tmp['Discount'].shift(-1) - tmp['Discount']
    tmp = tmp.dropna()

    model = OLS(tmp['Change_Next'], add_constant(tmp['Discount'])).fit()
    beta = model.params.iloc[1]
    p = model.pvalues.iloc[1]

    print(f"\nPREDICTIVE REGRESSION: delta_d(t+1) ~ d(t)")
    print(f"  Beta:    {beta:.4f}")
    print(f"  P-value: {p:.3f}")
    print(f"  R2:      {model.rsquared:.3f}")

    if beta < 0 and p < 0.05:
        print(f"  --> Wide discounts predict compression ({abs(beta)*100:.2f}ppt per 1ppt wider)")
    elif beta < 0 and p < 0.10:
        print(f"  --> Marginal timing signal (p < 0.10)")
    else:
        print(f"  --> No timing signal")

    return {'beta': beta, 'p_value': p, 'r_squared': model.rsquared}


def time_to_target(current, target, mu, phi):
    if target == mu or current == mu:
        return float('inf')
    return np.log((target - mu) / (current - mu)) / np.log(phi)


def plot_discount(df, ar1):
    mu, phi = ar1['mu'], ar1['phi']

    fitted = np.zeros(len(df))
    fitted[0] = df['Discount'].iloc[0]
    for i in range(1, len(df)):
        fitted[i] = mu + phi * (fitted[i-1] - mu)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.plot(df.index, df['Discount'] * 100, linewidth=2)
    ax1.axhline(y=mu * 100, color='red', linestyle='--', label=f'Mean ({mu:.1%})')
    ax1.set_title('EXOR Discount to NAV')
    ax1.set_ylabel('Discount (%)')
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(df.index, df['Discount'] * 100, label='Actual', linewidth=2, alpha=0.7)
    ax2.plot(df.index, fitted * 100, label=f'AR(1) (phi={phi:.3f})', linestyle='--', color='red')
    ax2.axhline(y=mu * 100, color='black', linestyle=':', label=f'Mean ({mu:.1%})')
    ax2.set_title('Mean Reversion Fit')
    ax2.set_ylabel('Discount (%)')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('mean_reversion_analysis.png', dpi=300)


def regime_comparison(full, recent, adf_f, adf_r, pred_f, pred_r):
    print(f"\n{'=' * 60}")
    print("REGIME COMPARISON")
    print(f"{'=' * 60}")
    print(f"{'Metric':<20} {'Full Sample':<15} {'Post-2020':<15}")
    print("-" * 50)
    print(f"{'ADF p-value':<20} {adf_f['p_value']:<15.3f} {adf_r['p_value']:<15.3f}")
    print(f"{'phi':<20} {full['phi']:<15.3f} {recent['phi']:<15.3f}")
    print(f"{'mu':<20} {full['mu']:<15.2%} {recent['mu']:<15.2%}")
    print(f"{'Half-life (Q)':<20} {full['half_life']:<15.2f} {recent['half_life']:<15.2f}")
    print(f"{'Pred beta':<20} {pred_f['beta']:<15.4f} {pred_r['beta']:<15.4f}")
    print(f"{'Pred p-value':<20} {pred_f['p_value']:<15.3f} {pred_r['p_value']:<15.3f}")


if __name__ == "__main__":
    df = load_discount_data("data/NAV Model.xlsx")
    descriptive_stats(df)

    print("\n" + "#" * 70)
    print("FULL SAMPLE")
    print("#" * 70)
    ar1_full = fit_ar1(df['Discount'])
    adf_full = adf_test(df['Discount'], "(Full)")
    za = za_test(df)
    pred_full = predictive_regression(df)

    print("\n" + "#" * 16)
    print("POST-2020 REGIME")
    print("#" * 16)
    df_recent = df[df.index >= '2020-01-01'].copy()
    ar1_recent = fit_ar1(df_recent['Discount'])
    adf_recent = adf_test(df_recent['Discount'], "(Post-2020)")
    pred_recent = predictive_regression(df_recent)

    current = df['Discount'].iloc[-1]
    print(f"\nTIME TO TARGETS (post-2020 regime, current={current:.1%})")
    for target in [0.50, 0.45, 0.40]:
        if target <= ar1_recent['mu']:
            continue
        q = time_to_target(current, target, ar1_recent['mu'], ar1_recent['phi'])
        print(f"  To {target:.0%}: {q:.1f}Q ({q/4:.1f}yr)")

    regime_comparison(ar1_full, ar1_recent, adf_full, adf_recent, pred_full, pred_recent)
    plot_discount(df, ar1_full)
