"""
Buyback Catalyst Event Study
Measures EXOR discount change in +/-30 day windows around buyback announcements.
"""

import pandas as pd
import numpy as np

df = pd.read_excel("data/exor_nav_model.xlsx", sheet_name="NAV_Calculations")
df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index('Date')

events = {
    '2023-06-19': '2023 Tender (EUR 750M)',
    '2024-04-15': '2024 Buyback (EUR 250M)',
    '2025-03-26': '2025 Tender (EUR 1B)'
}

results = []

for date, label in events.items():
    event_date = pd.to_datetime(date)
    before = df.loc[event_date - pd.Timedelta(days=30): event_date, 'Discount'].mean()
    after = df.loc[event_date: event_date + pd.Timedelta(days=30), 'Discount'].mean()
    change = after - before
    print(f"{label}: {before:.1%} -> {after:.1%} (Change: {change:+.1%})")
    results.append({'Event': label, 'Before': before, 'After': after, 'Change': change})

avg_impact = np.mean([r['Change'] for r in results])
print(f"\nAverage buyback impact: {avg_impact:+.1%}")
