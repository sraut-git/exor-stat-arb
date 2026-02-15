"""
Ornstein-Uhlenbeck Process Analysis
Fits continuous-time OU model to EXOR discount series,
identifies structural breaks, and generates investment signals.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

sns.set_style("whitegrid")


def estimate_ou_parameters(discounts, dt=63):
    """
    Estimate OU parameters via AR(1) regression.
    dX = theta(mu - X)dt + sigma*dW
    Maps from discrete AR(1): X(t+1) = a + b*X(t) => theta = -ln(b)/dt
    """
    discounts = np.array(discounts)
    n = len(discounts)
    mu = np.mean(discounts)

    X = discounts[:-1]
    Y = discounts[1:]
    X_mean, Y_mean = np.mean(X), np.mean(Y)

    b = np.sum((X - X_mean) * (Y - Y_mean)) / np.sum((X - X_mean) ** 2)
    a = Y_mean - b * X_mean

    if 0 < b < 1:
        theta = -np.log(b) / dt
        mu_reg = a / (1 - b)
        if abs(mu_reg - mu) < 10:
            mu = mu_reg
    else:
        rho = np.corrcoef(discounts[:-1] - mu, discounts[1:] - mu)[0, 1]
        theta = -np.log(max(rho, 0.01)) / dt

    theta = abs(theta)

    residuals = []
    for i in range(n - 1):
        expected = mu + (discounts[i] - mu) * np.exp(-theta * dt)
        residuals.append(discounts[i + 1] - expected)
    residuals = np.array(residuals)
    sigma = np.std(residuals) / np.sqrt(dt)

    half_life = np.log(2) / theta

    predicted = mu + (discounts[:-1] - mu) * np.exp(-theta * dt)
    ss_res = np.sum((discounts[1:] - predicted) ** 2)
    ss_tot = np.sum((discounts[1:] - np.mean(discounts)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    return {
        'theta': theta, 'mu': mu, 'sigma': sigma,
        'half_life': half_life, 'r_squared': r_squared, 'ar_coef': b
    }


def find_structural_break(df, discount_col):
    discounts = df[discount_col].values
    dates = pd.to_datetime(df.iloc[:, 0])
    n = len(discounts)

    window = min(8, n // 3)
    rolling_mean = pd.Series(discounts).rolling(window=window, center=True).mean()
    mean_changes = abs(rolling_mean.diff())
    break_idx = mean_changes.idxmax()

    mid = n // 2
    first_half = np.mean(discounts[:mid])
    second_half = np.mean(discounts[mid:])

    quarter_changes = abs(pd.Series(discounts).diff())
    max_change_idx = quarter_changes.idxmax()

    suggested = mid if abs(second_half - first_half) > 5 else max_change_idx
    break_date = dates.iloc[suggested]

    print(f"\nSTRUCTURAL BREAK ANALYSIS")
    print(f"  First half mean:  {first_half:.2f}%")
    print(f"  Second half mean: {second_half:.2f}%")
    print(f"  Suggested break:  {break_date.strftime('%Y-%m-%d')}")

    return break_date


def analyse_period(df, discount_col, period_name, dt=63):
    discounts = df[discount_col].values
    dates = df.iloc[:, 0]

    print(f"\n{'=' * 60}")
    print(f"{period_name}")
    print(f"{'=' * 60}")
    print(f"  Range:   {dates.iloc[0]} to {dates.iloc[-1]}")
    print(f"  Points:  {len(discounts)}")
    print(f"  Current: {discounts[-1]:.2f}%")
    print(f"  Mean:    {np.mean(discounts):.2f}%")

    if len(discounts) < 5:
        print("  Insufficient data (need >= 5 points)")
        return None

    params = estimate_ou_parameters(discounts, dt=dt)

    print(f"\n  OU PARAMETERS:")
    print(f"    theta:     {params['theta']:.6f} per day")
    print(f"    mu:        {params['mu']:.2f}%")
    print(f"    sigma:     {params['sigma']:.4f}%")
    print(f"    Half-life: {params['half_life']:.0f} days = {params['half_life']/252:.2f} years")
    print(f"    R-squared: {params['r_squared']:.4f}")

    diff = discounts[-1] - params['mu']
    if abs(diff) < 2:
        signal = "NEAR MEAN (fairly valued)"
    elif diff > 0:
        signal = f"WIDER than mean by {abs(diff):.2f}% (undervalued -> BUY)"
    else:
        signal = f"NARROWER than mean by {abs(diff):.2f}% (overvalued -> AVOID)"
    print(f"\n  SIGNAL: {signal}")

    return params


def plot_post_break(df_post, discount_col, params):
    discounts = df_post[discount_col].values
    dates = df_post.iloc[:, 0]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    ax1 = axes[0, 0]
    ax1.plot(dates, discounts, linewidth=2, marker='o', markersize=6, color='#2563eb')
    ax1.axhline(y=params['mu'], color='#ef4444', linestyle='--', linewidth=2, label=f"Mean={params['mu']:.1f}%")
    ax1.set_title('Post-Break Discount')
    ax1.set_ylabel('Discount (%)')
    ax1.legend()
    ax1.grid(alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    ax2 = axes[0, 1]
    ax2.scatter(discounts[:-1], discounts[1:], alpha=0.6, s=70, color='#2563eb')
    z = np.polyfit(discounts[:-1], discounts[1:], 1)
    x_line = np.linspace(discounts.min(), discounts.max(), 100)
    ax2.plot(x_line, np.poly1d(z)(x_line), "r--", linewidth=2, label=f'beta={z[0]:.3f}')
    ax2.plot([discounts.min(), discounts.max()], [discounts.min(), discounts.max()], 'k:', linewidth=1)
    ax2.set_xlabel('Discount(t)')
    ax2.set_ylabel('Discount(t+1)')
    ax2.set_title('Mean Reversion Dynamics')
    ax2.legend()
    ax2.grid(alpha=0.3)

    predicted = params['mu'] + (discounts[:-1] - params['mu']) * np.exp(-params['theta'] * 63)
    residuals = discounts[1:] - predicted

    ax3 = axes[1, 0]
    ax3.scatter(range(len(residuals)), residuals, alpha=0.6, s=70, color='#2563eb')
    ax3.axhline(y=0, color='r', linestyle='--')
    ax3.set_xlabel('Observation')
    ax3.set_ylabel('Residual (%)')
    ax3.set_title('Model Residuals')
    ax3.grid(alpha=0.3)

    ax4 = axes[1, 1]
    ax4.hist(residuals, bins=min(10, len(residuals)//2), density=True, alpha=0.7, color='#2563eb', edgecolor='black')
    x = np.linspace(residuals.min(), residuals.max(), 100)
    ax4.plot(x, stats.norm.pdf(x, np.mean(residuals), np.std(residuals)), 'r-', linewidth=2)
    ax4.set_xlabel('Residual (%)')
    ax4.set_ylabel('Density')
    ax4.set_title('Residual Distribution')
    ax4.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('exor_ou_analysis.png', dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    file_path = "data/exor_discount.csv"

    try:
        df = pd.read_csv(file_path)

        discount_col = None
        for col in df.columns:
            if 'discount' in col.lower() or 'premium' in col.lower():
                discount_col = col
                break
        if discount_col is None:
            raise ValueError("No discount column found")

        df = df.dropna(subset=[discount_col])
        df[discount_col] = pd.to_numeric(
            df[discount_col].astype(str).str.replace('%', '').str.replace(',', '').str.strip(),
            errors='coerce'
        )
        df = df.dropna(subset=[discount_col])

        date_col = df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors='coerce')
        df = df.sort_values(by=date_col).reset_index(drop=True)

        break_date = find_structural_break(df, discount_col)

        df_pre = df[df[date_col] < break_date].reset_index(drop=True)
        if len(df_pre) >= 5:
            analyse_period(df_pre, discount_col, "PRE-BREAK PERIOD")

        df_post = df[df[date_col] >= break_date].reset_index(drop=True)
        if len(df_post) >= 5:
            params = analyse_period(df_post, discount_col, "POST-BREAK PERIOD (Current Regime)")
            if params:
                plot_post_break(df_post, discount_col, params)

    except FileNotFoundError:
        print(f"Error: '{file_path}' not found")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
