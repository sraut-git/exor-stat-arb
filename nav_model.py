"""
EXOR NAV Replication Model
Replicates daily NAV from disclosed public holdings, FX rates,
private holdings and net debt. Calculates discount-to-NAV time series.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class ExorNAVModel:
    def __init__(self):
        self.holdings = {
            'Ferrari':    {'ticker': 'RACE',     'shares': 37_768_613,  'base_value': 10_945, 'currency': 'USD'},
            'CNH':        {'ticker': 'CNH',      'shares': 366_927_900, 'base_value': 3_368,  'currency': 'USD'},
            'Stellantis': {'ticker': 'STLAM.MI', 'shares': 449_410_092, 'base_value': 3_726,  'currency': 'EUR'},
            'Philips':    {'ticker': 'PHIA.AS',  'shares': 182_543_970, 'base_value': 4_549,  'currency': 'EUR'},
            'Juventus':   {'ticker': 'JUVE.MI',  'shares': 247_849_342, 'base_value': 657,    'currency': 'EUR'},
            'Clarivate':  {'ticker': 'CLVT',     'shares': 67_294_884,  'base_value': 168,    'currency': 'USD'},
        }

        self.private_holdings = {
            'Private_Company_Stakes': 3_509,
            'Lingotto': 3_193,
            'Others': 2_174,
            'Cash': 3_674,
        }

        self.liabilities = {'Gross_Debt': -3_542, 'Other': -102}
        self.exor_shares = 201_505_066
        self.reference_nav = 32_319
        self.reference_nav_per_share = 160.39

    def download_price_data(self, start_date, end_date):
        price_data = {}

        for company, info in self.holdings.items():
            try:
                stock = yf.download(info['ticker'], start=start_date, end=end_date, progress=False)
                if not stock.empty and 'Close' in stock.columns:
                    if isinstance(stock.columns, pd.MultiIndex):
                        price_data[company] = stock['Close'].iloc[:, 0]
                    else:
                        price_data[company] = stock['Close']
            except Exception:
                pass

        if not price_data:
            raise ValueError("No price data downloaded.")

        df_prices = pd.DataFrame(price_data)

        # EUR/USD rate
        try:
            fx = yf.download('EURUSD=X', start=start_date, end=end_date, progress=False)
            fx_close = fx['Close'].iloc[:, 0] if isinstance(fx['Close'], pd.DataFrame) else fx['Close']
            if fx_close.notna().sum() > 100:
                df_prices['EURUSD'] = fx_close.reindex(df_prices.index).ffill().bfill()
        except Exception:
            df_prices['EURUSD'] = 1.08  # fallback

        # EXOR share price (try multiple exchanges)
        for symbol in ['EXO.AS', 'EXO.MI', 'EXOR.PA']:
            try:
                exor = yf.download(symbol, start=start_date, end=end_date, progress=False)
                if not exor.empty:
                    close = exor['Close'].iloc[:, 0] if isinstance(exor['Close'], pd.DataFrame) else exor['Close']
                    if close.notna().sum() > 50:
                        df_prices['Exor_Price'] = close.reindex(df_prices.index).ffill()
                        break
            except Exception:
                continue

        return df_prices

    def calculate_daily_nav(self, df_prices):
        df_nav = pd.DataFrame(index=df_prices.index)
        eurusd = df_prices.get('EURUSD', pd.Series(1.08, index=df_prices.index))

        for company, info in self.holdings.items():
            if company not in df_prices.columns:
                df_nav[f'{company}_Value'] = info['base_value']
                continue

            value_local = df_prices[company] * info['shares'] / 1_000_000

            if info['currency'] == 'USD':
                df_nav[f'{company}_Value'] = value_local / eurusd
            else:
                df_nav[f'{company}_Value'] = value_local

        private_total = sum(self.private_holdings.values())
        df_nav['Private_Holdings'] = private_total

        public_cols = [c for c in df_nav.columns if c.endswith('_Value')]
        df_nav['Public_Holdings_Total'] = df_nav[public_cols].sum(axis=1)
        df_nav['Gross_Asset_Value'] = df_nav['Public_Holdings_Total'] + private_total
        df_nav['Total_Liabilities'] = sum(self.liabilities.values())
        df_nav['Estimated_NAV_Millions'] = df_nav['Gross_Asset_Value'] + df_nav['Total_Liabilities']
        df_nav['NAV_Per_Share'] = (df_nav['Estimated_NAV_Millions'] * 1_000_000) / self.exor_shares

        if 'Exor_Price' in df_prices.columns and df_prices['Exor_Price'].notna().any():
            df_nav['Exor_Share_Price'] = df_prices['Exor_Price']
            df_nav['Exor_Market_Cap_Millions'] = (df_nav['Exor_Share_Price'] * self.exor_shares) / 1_000_000
            df_nav['Discount_to_NAV'] = (
                (df_nav['Exor_Market_Cap_Millions'] - df_nav['Estimated_NAV_Millions'])
                / df_nav['Estimated_NAV_Millions'] * 100
            )

        return df_nav

    def reconcile_nav(self, df_nav):
        latest = df_nav['Estimated_NAV_Millions'].iloc[-1]
        error = ((latest - self.reference_nav) / self.reference_nav) * 100

        print("\nNAV RECONCILIATION")
        print(f"  Reported:  EUR {self.reference_nav:,.0f}M  (EUR {self.reference_nav_per_share:.2f}/share)")
        print(f"  Estimated: EUR {latest:,.0f}M  (EUR {df_nav['NAV_Per_Share'].iloc[-1]:.2f}/share)")
        print(f"  Error:     {error:+.1f}%")

    def create_visualisations(self, df_nav):
        df_plot = df_nav.copy()
        Q1, Q3 = df_plot['Estimated_NAV_Millions'].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        df_plot = df_plot[(df_plot['Estimated_NAV_Millions'] >= Q1 - 3*IQR) &
                          (df_plot['Estimated_NAV_Millions'] <= Q3 + 3*IQR)]

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('EXOR NAV Replication Model', fontsize=16, fontweight='bold')

        ax1 = axes[0, 0]
        ax1.plot(df_plot.index, df_plot['Estimated_NAV_Millions'], label='NAV', linewidth=2, color='#2E86AB')
        if 'Exor_Market_Cap_Millions' in df_plot.columns:
            ax1.plot(df_plot.index, df_plot['Exor_Market_Cap_Millions'], label='Market Cap', linewidth=2, color='#A23B72')
        ax1.axhline(y=self.reference_nav, color='green', linestyle='--', alpha=0.7, label='Reference')
        ax1.set_title('NAV vs Market Cap')
        ax1.set_ylabel('EUR Millions')
        ax1.legend()
        ax1.grid(alpha=0.3)

        ax2 = axes[0, 1]
        if 'Discount_to_NAV' in df_plot.columns:
            disc = df_plot['Discount_to_NAV'].clip(-60, 20)
            ax2.plot(disc.index, disc, color='#F18F01', linewidth=2)
            ax2.axhline(y=0, color='black', linewidth=1)
            ax2.fill_between(disc.index, disc, 0, where=(disc < 0), alpha=0.3, color='red')
            ax2.set_title('Discount to NAV')
            ax2.set_ylabel('Discount (%)')
            ax2.grid(alpha=0.3)

        ax3 = axes[1, 0]
        nav_ps = df_plot['NAV_Per_Share'].clip(50, 300)
        ax3.plot(nav_ps.index, nav_ps, label='NAV/Share', linewidth=2, color='#2E86AB')
        if 'Exor_Share_Price' in df_plot.columns:
            price = df_plot['Exor_Share_Price'].clip(30, 200)
            ax3.plot(price.index, price, label='Share Price', linewidth=2, color='#A23B72')
        ax3.set_title('NAV per Share vs Price')
        ax3.set_ylabel('EUR')
        ax3.legend()
        ax3.grid(alpha=0.3)

        ax4 = axes[1, 1]
        latest = df_nav.iloc[-1]
        holdings = {}
        for col in df_nav.columns:
            if col.endswith('_Value') and col not in ['Public_Holdings_Total', 'Gross_Asset_Value']:
                holdings[col.replace('_Value', '')] = latest[col]
        holdings['Private'] = latest['Private_Holdings']
        holdings = dict(sorted(holdings.items(), key=lambda x: x[1], reverse=True))
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#BC4B51', '#8B5A3C']
        ax4.pie(holdings.values(), labels=holdings.keys(), autopct='%1.1f%%',
                colors=colors, textprops={'fontsize': 9})
        ax4.set_title(f'Portfolio (EUR {latest["Gross_Asset_Value"]:,.0f}M)')

        plt.tight_layout()
        plt.savefig('exor_nav_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')

    def export_to_excel(self, df_nav, df_prices):
        filename = 'exor_nav_model_output.xlsx'
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            df_nav.to_excel(writer, sheet_name='NAV_Calculations')
            df_prices.to_excel(writer, sheet_name='Price_Data')

    def run(self, start_date='2023-01-01', end_date=None):
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        print("=" * 60)
        print(f"EXOR NAV REPLICATION MODEL ({start_date} to {end_date})")
        print("=" * 60)

        df_prices = self.download_price_data(start_date, end_date)
        df_nav = self.calculate_daily_nav(df_prices)
        self.reconcile_nav(df_nav)
        self.create_visualisations(df_nav)
        self.export_to_excel(df_nav, df_prices)
        return df_nav, df_prices


if __name__ == "__main__":
    model = ExorNAVModel()
    df_nav, df_prices = model.run(start_date='2023-01-01')

    if df_nav is not None:
        latest = df_nav.iloc[-1]
        print(f"\nNAV:      EUR {latest['Estimated_NAV_Millions']:,.0f}M")
        print(f"NAV/Shr:  EUR {latest['NAV_Per_Share']:.2f}")
        if 'Discount_to_NAV' in df_nav.columns:
            print(f"Discount: {latest['Discount_to_NAV']:.1f}%")
