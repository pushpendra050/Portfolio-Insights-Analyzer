"""
Portfolio Insights Analyzer
Analyzes portfolio growth patterns from portfolio_data.csv only
Outputs insights to console, insight.txt file, and PNG graphs
"""

import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import warnings
warnings.filterwarnings('ignore')

# Set style for all plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12

# Create output directory for graphs
GRAPH_DIR = 'graphs'
if not os.path.exists(GRAPH_DIR):
    os.makedirs(GRAPH_DIR)


def format_lakhs(x, pos):
    """Format numbers in lakhs for y-axis (1L = 100000)"""
    if abs(x) >= 100000:
        return f'{x/100000:.2f}L'
    elif abs(x) >= 10000:
        return f'{x/100000:.2f}L'
    elif abs(x) >= 1000:
        return f'{x/100000:.3f}L'
    else:
        return f'{x:.0f}'


class DualOutput:
    """Write to both console and file simultaneously"""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.file = open(filename, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)

    def flush(self):
        self.terminal.flush()
        self.file.flush()

    def close(self):
        self.file.close()


def load_portfolio_data(filepath='portfolio_data.csv'):
    """Load and process monthly portfolio data"""
    df = pd.read_csv(filepath)

    month_map = {'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
                 'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12}

    df['Month_Num'] = df['Month'].map(month_map)
    df['Date'] = pd.to_datetime(df[['Year', 'Month_Num']].assign(Day=1).rename(
        columns={'Month_Num': 'month', 'Year': 'year', 'Day': 'day'}))

    df = df.sort_values('Date').reset_index(drop=True)
    return df


def print_header(title):
    """Print formatted section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


# =============================================================================
# GRAPH GENERATION FUNCTIONS
# =============================================================================

def plot_portfolio_growth(df):
    """Generate portfolio value over time chart"""
    fig, ax = plt.subplots(figsize=(14, 7))

    ax.fill_between(df['Date'], df['Consolidated_portfolio_Value'], alpha=0.3, color='#2E86AB')
    ax.plot(df['Date'], df['Consolidated_portfolio_Value'], linewidth=2, color='#2E86AB', label='Portfolio Value')

    # Add milestones
    milestones = [500000, 1000000, 1500000]
    for milestone in milestones:
        if df['Consolidated_portfolio_Value'].max() >= milestone:
            ax.axhline(y=milestone, color='gray', linestyle='--', alpha=0.5, linewidth=1)
            ax.text(df['Date'].iloc[0], milestone, f' Rs.{milestone/100000:.0f}L',
                   va='bottom', ha='left', fontsize=11, color='gray')

    ax.set_title('Portfolio Value Over Time', fontsize=20, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Portfolio Value (Rs.)')
    ax.yaxis.set_major_formatter(FuncFormatter(format_lakhs))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)

    # Add current value annotation
    current_value = df['Consolidated_portfolio_Value'].iloc[-1]
    ax.annotate(f'Rs.{current_value:,.0f}',
                xy=(df['Date'].iloc[-1], current_value),
                xytext=(10, 10), textcoords='offset points',
                fontsize=13, fontweight='bold', color='#2E86AB')

    plt.tight_layout()
    plt.savefig(f'{GRAPH_DIR}/01_portfolio_growth.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_monthly_changes(df):
    """Generate monthly change bar chart"""
    fig, ax = plt.subplots(figsize=(14, 6))

    colors = ['#28A745' if x >= 0 else '#DC3545' for x in df['Change']]
    bars = ax.bar(df['Date'], df['Change'], color=colors, width=25, alpha=0.8)

    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_title('Monthly Portfolio Changes', fontsize=20, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Change (Rs.)')
    ax.yaxis.set_major_formatter(FuncFormatter(format_lakhs))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(f'{GRAPH_DIR}/02_monthly_changes.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_yearly_growth(df):
    """Generate year-wise portfolio growth chart (bar chart)"""
    yearly_data = df.groupby('Year').agg({
        'Change': 'sum',
        'Consolidated_portfolio_Value': ['first', 'last']
    }).reset_index()
    yearly_data.columns = ['Year', 'Total_Change', 'Start_Value', 'End_Value']

    fig, ax = plt.subplots(figsize=(12, 6))

    # Bar chart for yearly growth
    colors = ['#28A745' if x >= 0 else '#DC3545' for x in yearly_data['Total_Change']]
    bars = ax.bar(yearly_data['Year'].astype(str), yearly_data['Total_Change'], color=colors, alpha=0.8)

    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_title('Year-wise Portfolio Growth', fontsize=20, fontweight='bold')
    ax.set_xlabel('Year')
    ax.set_ylabel('Total Change (Rs.)')
    ax.yaxis.set_major_formatter(FuncFormatter(format_lakhs))

    # Add value labels on bars in Lakhs format
    for bar, val in zip(bars, yearly_data['Total_Change']):
        height = bar.get_height()
        label = f'{val/100000:.2f}L'
        ax.annotate(label,
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3 if height >= 0 else -12),
                    textcoords="offset points",
                    ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{GRAPH_DIR}/03a_yearly_growth.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_yearend_values(df):
    """Generate year-end portfolio values chart (line chart)"""
    yearly_data = df.groupby('Year').agg({
        'Consolidated_portfolio_Value': 'last'
    }).reset_index()
    yearly_data.columns = ['Year', 'End_Value']

    fig, ax = plt.subplots(figsize=(12, 6))

    # Line chart for year-end values
    ax.plot(yearly_data['Year'].astype(str), yearly_data['End_Value'],
             marker='o', linewidth=2, markersize=10, color='#2E86AB')
    ax.fill_between(yearly_data['Year'].astype(str), yearly_data['End_Value'], alpha=0.3, color='#2E86AB')

    ax.set_title('Year-end Portfolio Values', fontsize=20, fontweight='bold')
    ax.set_xlabel('Year')
    ax.set_ylabel('Portfolio Value (Rs.)')
    ax.yaxis.set_major_formatter(FuncFormatter(format_lakhs))

    # Add value labels on points in Lakhs format
    for i, (year, val) in enumerate(zip(yearly_data['Year'], yearly_data['End_Value'])):
        label = f'{val/100000:.2f}L'
        ax.annotate(label,
                    xy=(str(int(year)), val),
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{GRAPH_DIR}/03b_yearend_values.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_drawdown(df):
    """Generate drawdown analysis chart"""
    values = df['Consolidated_portfolio_Value'].values
    running_max = np.maximum.accumulate(values)
    drawdowns = (values - running_max) / running_max * 100

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Portfolio value with running max
    ax1.plot(df['Date'], df['Consolidated_portfolio_Value'], linewidth=2, color='#2E86AB', label='Portfolio Value')
    ax1.plot(df['Date'], running_max, linewidth=1.5, color='#28A745', linestyle='--', label='Peak Value', alpha=0.7)
    ax1.fill_between(df['Date'], df['Consolidated_portfolio_Value'], running_max,
                     where=(df['Consolidated_portfolio_Value'] < running_max),
                     alpha=0.3, color='#DC3545', label='Drawdown')

    ax1.set_title('Portfolio Value vs Peak (Drawdown Visualization)', fontsize=18, fontweight='bold')
    ax1.set_ylabel('Portfolio Value (Rs.)')
    ax1.yaxis.set_major_formatter(FuncFormatter(format_lakhs))
    ax1.legend(loc='upper left')

    # Drawdown percentage
    ax2.fill_between(df['Date'], drawdowns, 0, color='#DC3545', alpha=0.5)
    ax2.plot(df['Date'], drawdowns, linewidth=1, color='#DC3545')
    ax2.axhline(y=0, color='black', linewidth=0.5)

    ax2.set_title('Drawdown Percentage', fontsize=18, fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Drawdown (%)')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)

    # Annotate worst drawdown
    min_dd_idx = np.argmin(drawdowns)
    ax2.annotate(f'Worst: {drawdowns[min_dd_idx]:.1f}%',
                xy=(df['Date'].iloc[min_dd_idx], drawdowns[min_dd_idx]),
                xytext=(20, -20), textcoords='offset points',
                arrowprops=dict(arrowstyle='->', color='black'),
                fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{GRAPH_DIR}/04_drawdown_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_monthly_seasonality(df):
    """Generate monthly seasonality heatmap"""
    month_order = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
                   'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']

    # Create pivot table
    pivot = df.pivot_table(values='Percentage_Change', index='Year', columns='Month', aggfunc='mean')
    pivot = pivot.reindex(columns=month_order)

    fig, ax = plt.subplots(figsize=(14, 8))

    # Heatmap
    im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=-10, vmax=10)

    ax.set_xticks(range(len(month_order)))
    ax.set_xticklabels(month_order)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index.astype(int))
    ax.set_title('Monthly Returns Heatmap (%)', fontsize=20, fontweight='bold')
    ax.set_xlabel('Month')
    ax.set_ylabel('Year')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Return %')

    # Add text annotations
    for i in range(len(pivot.index)):
        for j in range(len(month_order)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                text_color = 'white' if abs(val) > 5 else 'black'
                ax.text(j, i, f'{val:.1f}', ha='center', va='center', fontsize=11, color=text_color)

    plt.tight_layout()
    plt.savefig(f'{GRAPH_DIR}/05a_monthly_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_avg_monthly_returns(df):
    """Generate average monthly returns bar chart"""
    month_order = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
                   'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']

    fig, ax = plt.subplots(figsize=(12, 6))

    # Average returns by month
    monthly_avg = df.groupby('Month')['Percentage_Change'].mean().reindex(month_order)
    colors = ['#28A745' if x >= 0 else '#DC3545' for x in monthly_avg.values]
    bars = ax.bar(month_order, monthly_avg.values, color=colors, alpha=0.8)

    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_title('Average Monthly Returns (%)', fontsize=20, fontweight='bold')
    ax.set_xlabel('Month')
    ax.set_ylabel('Average Return (%)')
    plt.xticks(rotation=45)

    # Add value labels
    for bar, val in zip(bars, monthly_avg.values):
        height = bar.get_height()
        ax.annotate(f'{val:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3 if height >= 0 else -12),
                    textcoords="offset points",
                    ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{GRAPH_DIR}/05b_avg_monthly_returns.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_quarterly_performance(df):
    """Generate quarterly performance chart"""
    df_q = df.copy()
    df_q['Quarter'] = df_q['Date'].dt.quarter
    df_q['YearQuarter'] = df_q['Year'].astype(str) + '-Q' + df_q['Quarter'].astype(str)

    quarterly = df_q.groupby(['Year', 'Quarter']).agg({
        'Change': 'sum',
        'Consolidated_portfolio_Value': 'last'
    }).reset_index()
    quarterly['YearQuarter'] = quarterly['Year'].astype(str) + '-Q' + quarterly['Quarter'].astype(str)

    fig, ax = plt.subplots(figsize=(14, 6))

    colors = ['#28A745' if x >= 0 else '#DC3545' for x in quarterly['Change']]
    bars = ax.bar(quarterly['YearQuarter'], quarterly['Change'], color=colors, alpha=0.8)

    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_title('Quarterly Portfolio Changes', fontsize=20, fontweight='bold')
    ax.set_xlabel('Quarter')
    ax.set_ylabel('Change (Rs.)')
    ax.yaxis.set_major_formatter(FuncFormatter(format_lakhs))
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(f'{GRAPH_DIR}/06_quarterly_performance.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_volatility_distribution(df):
    """Generate volatility and return distribution charts (histogram and rolling volatility)"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Histogram of monthly returns
    n, bins, patches = ax1.hist(df['Percentage_Change'], bins=20, color='#2E86AB', alpha=0.7, edgecolor='white')

    # Color negative bins
    for i, patch in enumerate(patches):
        if bins[i] < 0:
            patch.set_facecolor('#DC3545')

    ax1.axvline(x=0, color='black', linewidth=1)
    ax1.axvline(x=df['Percentage_Change'].mean(), color='#FFC107', linewidth=2, linestyle='--', label=f"Mean: {df['Percentage_Change'].mean():.1f}%")
    ax1.set_title('Distribution of Monthly Returns', fontsize=18, fontweight='bold')
    ax1.set_xlabel('Monthly Return (%)')
    ax1.set_ylabel('Frequency')
    ax1.legend()

    # Rolling volatility (12-month)
    rolling_std = df['Percentage_Change'].rolling(window=12).std()
    ax2.plot(df['Date'], rolling_std, linewidth=2, color='#E74C3C')
    ax2.fill_between(df['Date'], rolling_std, alpha=0.3, color='#E74C3C')
    ax2.set_title('12-Month Rolling Volatility (Std Dev)', fontsize=18, fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Standard Deviation (%)')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    plt.savefig(f'{GRAPH_DIR}/07a_volatility_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_return_distribution_by_year(df):
    """Generate return distribution by year (box plot) - Auto-excludes first year if it has high outliers"""
    fig, ax = plt.subplots(figsize=(12, 6))

    all_years = sorted(df['Year'].unique())
    excluded_year = None
    title_suffix = ""

    # Check if first year has high outliers compared to other years
    if len(all_years) > 2:
        first_year = all_years[0]
        first_year_data = df[df['Year'] == first_year]['Percentage_Change']
        other_years_data = df[df['Year'] != first_year]['Percentage_Change']

        # Calculate statistics
        first_year_std = first_year_data.std()
        first_year_range = first_year_data.max() - first_year_data.min()
        other_years_std = other_years_data.std()
        other_years_range = other_years_data.max() - other_years_data.min()

        # If first year has significantly higher volatility (1.5x threshold), exclude it
        if first_year_std > 1.5 * other_years_std or first_year_range > 1.5 * other_years_range:
            excluded_year = first_year
            title_suffix = f" (Excluding {int(first_year)})"

    # Filter data
    if excluded_year is not None:
        df_filtered = df[df['Year'] != excluded_year]
    else:
        df_filtered = df

    years = sorted(df_filtered['Year'].unique())
    data_by_year = [df_filtered[df_filtered['Year'] == year]['Percentage_Change'].values for year in years]
    bp = ax.boxplot(data_by_year, labels=[str(int(y)) for y in years], patch_artist=True)

    for patch in bp['boxes']:
        patch.set_facecolor('#2E86AB')
        patch.set_alpha(0.7)

    ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
    ax.set_title(f'Return Distribution by Year{title_suffix}', fontsize=20, fontweight='bold')
    ax.set_xlabel('Year')
    ax.set_ylabel('Monthly Return (%)')

    plt.tight_layout()
    plt.savefig(f'{GRAPH_DIR}/07b_return_distribution_by_year.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_positive_negative_months(df):
    """Generate positive vs negative months pie chart"""
    fig, ax = plt.subplots(figsize=(10, 8))

    positive = len(df[df['Percentage_Change'] > 0])
    negative = len(df[df['Percentage_Change'] < 0])
    zero = len(df[df['Percentage_Change'] == 0])

    sizes = [positive, negative, zero] if zero > 0 else [positive, negative]
    labels = [f'Positive\n({positive})', f'Negative\n({negative})', f'Flat\n({zero})'] if zero > 0 else [f'Positive\n({positive})', f'Negative\n({negative})']
    colors_pie = ['#28A745', '#DC3545', '#6C757D'] if zero > 0 else ['#28A745', '#DC3545']
    explode = (0.05, 0.05, 0) if zero > 0 else (0.05, 0.05)

    ax.pie(sizes, explode=explode, labels=labels, colors=colors_pie, autopct='%1.1f%%',
            shadow=True, startangle=90, textprops={'fontsize': 14})
    ax.set_title('Positive vs Negative Months', fontsize=20, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{GRAPH_DIR}/07c_positive_negative_months.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_milestones(df):
    """Generate milestones timeline chart"""
    current_value = df['Consolidated_portfolio_Value'].iloc[-1]
    max_milestone = int((current_value // 500000) + 1) * 500000
    milestones = list(range(500000, max_milestone + 1, 500000))

    fig, ax = plt.subplots(figsize=(14, 7))

    # Plot portfolio value
    ax.plot(df['Date'], df['Consolidated_portfolio_Value'], linewidth=2, color='#2E86AB')
    ax.fill_between(df['Date'], df['Consolidated_portfolio_Value'], alpha=0.2, color='#2E86AB')

    # Add milestone lines and annotations
    milestone_dates = []
    for milestone in milestones:
        ax.axhline(y=milestone, color='#28A745', linestyle='--', alpha=0.5, linewidth=1)

        crossed = df[df['Consolidated_portfolio_Value'] >= milestone]
        if len(crossed) > 0:
            cross_date = crossed['Date'].iloc[0]
            milestone_dates.append((cross_date, milestone))

            ax.scatter([cross_date], [milestone], color='#28A745', s=100, zorder=5, marker='*')
            ax.annotate(f'Rs.{milestone/100000:.0f}L\n{cross_date.strftime("%b %Y")}',
                       xy=(cross_date, milestone),
                       xytext=(-50, 20), textcoords='offset points',
                       fontsize=11, ha='center',
                       arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5))

    ax.set_title('Portfolio Milestones Achievement', fontsize=20, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Portfolio Value (Rs.)')
    ax.yaxis.set_major_formatter(FuncFormatter(format_lakhs))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(f'{GRAPH_DIR}/08_milestones.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_top_months(df):
    """Generate top best and worst months chart"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Top 10 best months
    best = df.nlargest(10, 'Change')[['Month', 'Year', 'Change']].copy()
    best['Label'] = best['Month'] + ' ' + best['Year'].astype(int).astype(str)
    best = best.sort_values('Change', ascending=True)

    ax1.barh(best['Label'], best['Change'], color='#28A745', alpha=0.8)
    ax1.set_title('Top 10 Best Months', fontsize=18, fontweight='bold')
    ax1.set_xlabel('Change (Rs.)')
    ax1.xaxis.set_major_formatter(FuncFormatter(format_lakhs))

    for i, (_, row) in enumerate(best.iterrows()):
        ax1.text(row['Change'] + 1000, i, f"Rs.{row['Change']:,.0f}", va='center', fontsize=11)

    # Top 10 worst months
    worst = df.nsmallest(10, 'Change')[['Month', 'Year', 'Change']].copy()
    worst['Label'] = worst['Month'] + ' ' + worst['Year'].astype(int).astype(str)
    worst = worst.sort_values('Change', ascending=False)

    ax2.barh(worst['Label'], worst['Change'], color='#DC3545', alpha=0.8)
    ax2.set_title('Top 10 Worst Months', fontsize=18, fontweight='bold')
    ax2.set_xlabel('Change (Rs.)')
    ax2.xaxis.set_major_formatter(FuncFormatter(format_lakhs))

    for i, (_, row) in enumerate(worst.iterrows()):
        ax2.text(row['Change'] - 1000, i, f"Rs.{row['Change']:,.0f}", va='center', ha='right', fontsize=11)

    plt.tight_layout()
    plt.savefig(f'{GRAPH_DIR}/09_top_months.png', dpi=300, bbox_inches='tight')
    plt.close()


def generate_all_graphs(df):
    """Generate all graphs"""
    print("\nGenerating graphs...")

    plot_portfolio_growth(df)
    print("  [OK] 01_portfolio_growth.png")

    plot_monthly_changes(df)
    print("  [OK] 02_monthly_changes.png")

    plot_yearly_growth(df)
    print("  [OK] 03a_yearly_growth.png")

    plot_yearend_values(df)
    print("  [OK] 03b_yearend_values.png")

    plot_drawdown(df)
    print("  [OK] 04_drawdown_analysis.png")

    plot_monthly_seasonality(df)
    print("  [OK] 05a_monthly_heatmap.png")

    plot_avg_monthly_returns(df)
    print("  [OK] 05b_avg_monthly_returns.png")

    plot_quarterly_performance(df)
    print("  [OK] 06_quarterly_performance.png")

    plot_volatility_distribution(df)
    print("  [OK] 07a_volatility_analysis.png")

    plot_return_distribution_by_year(df)
    print("  [OK] 07b_return_distribution_by_year.png")

    plot_positive_negative_months(df)
    print("  [OK] 07c_positive_negative_months.png")

    plot_milestones(df)
    print("  [OK] 08_milestones.png")

    plot_top_months(df)
    print("  [OK] 09_top_months.png")

    plot_yoy_month_comparison(df)
    print("  [OK] 10_yoy_month_comparison.png")

    plot_recovery_analysis(df)
    print("  [OK] 11_recovery_periods.png")

    print(f"\nAll graphs saved to '{GRAPH_DIR}/' folder.")


# =============================================================================
# ANALYSIS FUNCTIONS (Original)
# =============================================================================

def yearly_summary(df):
    """Year-wise portfolio summary with best/worst months"""
    print_header("YEAR-WISE PORTFOLIO SUMMARY")

    years = df['Year'].unique()
    results = []

    for year in sorted(years):
        year_data = df[df['Year'] == year]
        start_value = year_data['Consolidated_portfolio_Value'].iloc[0]
        end_value = year_data['Consolidated_portfolio_Value'].iloc[-1]

        # Get previous year end value for YoY calculation
        prev_year_data = df[df['Year'] == year - 1]
        if len(prev_year_data) > 0:
            prev_end = prev_year_data['Consolidated_portfolio_Value'].iloc[-1]
            yoy_growth = end_value - prev_end
        else:
            yoy_growth = end_value - start_value

        # Total change during the year (sum of monthly changes)
        total_change = year_data['Change'].sum()

        # Monthly stats
        avg_monthly_change = year_data['Change'].mean()
        positive_months = len(year_data[year_data['Percentage_Change'] > 0])
        negative_months = len(year_data[year_data['Percentage_Change'] < 0])

        best_month = year_data.loc[year_data['Percentage_Change'].idxmax()]
        worst_month = year_data.loc[year_data['Percentage_Change'].idxmin()]

        results.append({
            'Year': year,
            'Start_Value': start_value,
            'End_Value': end_value,
            'YoY_Growth': yoy_growth,
            'Total_Change': total_change,
            'Avg_Monthly_Change': avg_monthly_change,
            'Positive_Months': positive_months,
            'Negative_Months': negative_months,
            'Best_Month': best_month['Month'],
            'Best_Month_Pct': best_month['Percentage_Change'],
            'Best_Month_Amt': best_month['Change'],
            'Worst_Month': worst_month['Month'],
            'Worst_Month_Pct': worst_month['Percentage_Change'],
            'Worst_Month_Amt': worst_month['Change']
        })

    result_df = pd.DataFrame(results)

    # Print summary table
    print(f"\n{'Year':<6} {'Start Value':>14} {'End Value':>14} {'YoY Growth':>14} {'Avg Monthly':>12} {'+ve/-ve':>8}")
    print("-" * 75)

    for _, row in result_df.iterrows():
        print(f"{int(row['Year']):<6} Rs.{row['Start_Value']:>11,.0f} Rs.{row['End_Value']:>11,.0f} "
              f"Rs.{row['YoY_Growth']:>11,.0f} Rs.{row['Avg_Monthly_Change']:>9,.0f} "
              f"{int(row['Positive_Months']):>3}/{int(row['Negative_Months']):<3}")

    # Best & Worst months per year
    print("\n" + "-" * 75)
    print("BEST & WORST MONTHS PER YEAR:")
    print("-" * 75)
    print(f"{'Year':<6} {'Best Month':<25} {'Worst Month':<25}")
    print("-" * 75)

    for _, row in result_df.iterrows():
        best_str = f"{row['Best_Month']} (+{row['Best_Month_Pct']:.1f}%, Rs.{row['Best_Month_Amt']:+,.0f})"
        worst_str = f"{row['Worst_Month']} ({row['Worst_Month_Pct']:.1f}%, Rs.{row['Worst_Month_Amt']:+,.0f})"
        print(f"{int(row['Year']):<6} {best_str:<25} {worst_str:<25}")

    return result_df


def portfolio_milestones(df):
    """Track portfolio milestones in 5 lakh increments"""
    print_header("PORTFOLIO MILESTONES (Every 5 Lakh)")

    current_value = df['Consolidated_portfolio_Value'].iloc[-1]
    max_milestone = int((current_value // 500000) + 1) * 500000

    milestones = list(range(500000, max_milestone + 1, 500000))

    print(f"\n{'Milestone':>15} {'Date Reached':<12} {'Months from Prev':<18} {'Total Months':<15}")
    print("-" * 65)

    start_date = df['Date'].iloc[0]
    prev_date = start_date
    prev_milestone_reached = False

    for milestone in milestones:
        crossed = df[df['Consolidated_portfolio_Value'] >= milestone]
        if len(crossed) > 0:
            cross_date = crossed['Date'].iloc[0]
            cross_value = crossed['Consolidated_portfolio_Value'].iloc[0]

            total_months = (cross_date.year - start_date.year) * 12 + (cross_date.month - start_date.month)

            if prev_milestone_reached:
                months_from_prev = (cross_date.year - prev_date.year) * 12 + (cross_date.month - prev_date.month)
            else:
                months_from_prev = total_months

            print(f"Rs.{milestone:>12,} {cross_date.strftime('%b %Y'):<12} {months_from_prev:>8} months      {total_months:>8} months")

            prev_date = cross_date
            prev_milestone_reached = True
        else:
            print(f"Rs.{milestone:>12,} {'Not yet reached':<12}")

    # Next milestone projection
    print("\n" + "-" * 65)
    next_milestone = (int(current_value // 500000) + 1) * 500000
    remaining = next_milestone - current_value
    avg_monthly_change = df['Change'].tail(12).mean()

    if avg_monthly_change > 0:
        months_needed = remaining / avg_monthly_change
        print(f"NEXT MILESTONE: Rs.{next_milestone:,}")
        print(f"  Remaining: Rs.{remaining:,.0f}")
        print(f"  Avg Monthly Change (last 12 months): Rs.{avg_monthly_change:,.0f}")
        print(f"  Estimated months to reach: {months_needed:.1f} months")


def portfolio_value_changes(df):
    """Show portfolio value changes over different periods (NOT returns)"""
    print_header("PORTFOLIO VALUE CHANGES OVER TIME")
    print("\nNOTE: These are portfolio VALUE changes, not investment returns.")
    print("      Value changes include new investments + market movements.\n")

    current_value = df['Consolidated_portfolio_Value'].iloc[-1]
    current_date = df['Date'].iloc[-1]

    periods = [
        ('1 Month', 1),
        ('3 Months', 3),
        ('6 Months', 6),
        ('1 Year', 12),
        ('2 Years', 24),
        ('3 Years', 36),
        ('Since Inception', len(df) - 1)
    ]

    print(f"As of: {current_date.strftime('%B %Y')}")
    print(f"Current Portfolio Value: Rs.{current_value:,.2f}")
    print()
    print(f"{'Period':<20} {'Value Then':>14} {'Value Now':>14} {'Change':>14}")
    print("-" * 65)

    for period_name, months in periods:
        if months < len(df):
            idx = len(df) - months - 1
            past_value = df['Consolidated_portfolio_Value'].iloc[idx]
            past_date = df['Date'].iloc[idx]

            change = current_value - past_value

            print(f"{period_name:<20} Rs.{past_value:>11,.0f} Rs.{current_value:>11,.0f} Rs.{change:>11,.0f}")


def drawdown_analysis(df):
    """Analyze drawdowns (peak to trough declines)"""
    print_header("DRAWDOWN ANALYSIS")

    values = df['Consolidated_portfolio_Value'].values

    # Calculate running maximum
    running_max = np.maximum.accumulate(values)
    drawdowns = (values - running_max) / running_max * 100

    df_dd = df.copy()
    df_dd['Drawdown_Pct'] = drawdowns
    df_dd['Running_Max'] = running_max
    df_dd['Drawdown_Amt'] = values - running_max

    # Find worst drawdown periods
    print("\nWORST 5 DRAWDOWN PERIODS:")
    print("-" * 70)
    print(f"{'Date':<12} {'Portfolio Value':>15} {'Peak Value':>15} {'Drawdown':>12} {'Drop':>12}")
    print("-" * 70)

    worst_months = df_dd.nsmallest(5, 'Drawdown_Pct')

    for _, row in worst_months.iterrows():
        print(f"{row['Date'].strftime('%b %Y'):<12} Rs.{row['Consolidated_portfolio_Value']:>13,.0f} "
              f"Rs.{row['Running_Max']:>13,.0f} {row['Drawdown_Pct']:>11.2f}% Rs.{row['Drawdown_Amt']:>10,.0f}")

    # Current status
    current_dd = drawdowns[-1]
    print("\n" + "-" * 70)
    if current_dd < 0:
        print(f"CURRENT STATUS: In drawdown of {current_dd:.2f}% from peak of Rs.{running_max[-1]:,.0f}")
        recovery_needed = ((running_max[-1] / values[-1]) - 1) * 100
        print(f"                Need {recovery_needed:.2f}% gain to recover to previous peak")
    else:
        print(f"CURRENT STATUS: At all-time high of Rs.{values[-1]:,.0f}")


def volatility_analysis(df):
    """Analyze portfolio volatility"""
    print_header("VOLATILITY ANALYSIS")

    pct_changes = df['Percentage_Change'].values

    # Overall statistics
    mean_return = np.mean(pct_changes)
    std_dev = np.std(pct_changes)
    max_gain = np.max(pct_changes)
    max_loss = np.min(pct_changes)

    print(f"\nOVERALL MONTHLY STATISTICS:")
    print("-" * 50)
    print(f"  Average Monthly Change:     {mean_return:>10.2f}%")
    print(f"  Standard Deviation:         {std_dev:>10.2f}%")
    print(f"  Maximum Monthly Gain:       {max_gain:>10.2f}%")
    print(f"  Maximum Monthly Loss:       {max_loss:>10.2f}%")

    # Positive vs Negative months
    positive_months = len(df[df['Percentage_Change'] > 0])
    negative_months = len(df[df['Percentage_Change'] < 0])
    zero_months = len(df[df['Percentage_Change'] == 0])
    total_months = len(df)

    print(f"\nMONTH DISTRIBUTION:")
    print("-" * 50)
    print(f"  Positive Months: {positive_months:>3} ({positive_months/total_months*100:.1f}%)")
    print(f"  Negative Months: {negative_months:>3} ({negative_months/total_months*100:.1f}%)")
    print(f"  Flat Months:     {zero_months:>3} ({zero_months/total_months*100:.1f}%)")

    # Year-wise volatility
    print(f"\nYEAR-WISE VOLATILITY:")
    print("-" * 50)
    print(f"{'Year':<6} {'Avg Change':>12} {'Std Dev':>10} {'Best':>10} {'Worst':>10}")
    print("-" * 50)

    for year in sorted(df['Year'].unique()):
        year_data = df[df['Year'] == year]['Percentage_Change']
        print(f"{year:<6} {year_data.mean():>11.2f}% {year_data.std():>9.2f}% "
              f"{year_data.max():>9.1f}% {year_data.min():>9.1f}%")


def streak_analysis(df):
    """Analyze consecutive positive/negative month streaks"""
    print_header("STREAK ANALYSIS")

    changes = df['Percentage_Change'].values

    # Find streaks
    current_streak = 1
    current_sign = 1 if changes[0] >= 0 else -1

    streaks = []
    streak_start_idx = 0

    for i in range(1, len(changes)):
        sign = 1 if changes[i] >= 0 else -1
        if sign == current_sign:
            current_streak += 1
        else:
            streaks.append({
                'type': 'Positive' if current_sign == 1 else 'Negative',
                'length': current_streak,
                'start_idx': streak_start_idx,
                'end_idx': i - 1
            })
            current_streak = 1
            current_sign = sign
            streak_start_idx = i

    # Add final streak
    streaks.append({
        'type': 'Positive' if current_sign == 1 else 'Negative',
        'length': current_streak,
        'start_idx': streak_start_idx,
        'end_idx': len(changes) - 1
    })

    streak_df = pd.DataFrame(streaks)

    # Longest positive streak
    positive_streaks = streak_df[streak_df['type'] == 'Positive']
    if len(positive_streaks) > 0:
        longest_pos = positive_streaks.loc[positive_streaks['length'].idxmax()]
        start_date = df.iloc[longest_pos['start_idx']]['Date']
        end_date = df.iloc[longest_pos['end_idx']]['Date']

        print(f"\nLONGEST POSITIVE STREAK: {longest_pos['length']} months")
        print(f"  Period: {start_date.strftime('%b %Y')} to {end_date.strftime('%b %Y')}")

    # Longest negative streak
    negative_streaks = streak_df[streak_df['type'] == 'Negative']
    if len(negative_streaks) > 0:
        longest_neg = negative_streaks.loc[negative_streaks['length'].idxmax()]
        start_date = df.iloc[longest_neg['start_idx']]['Date']
        end_date = df.iloc[longest_neg['end_idx']]['Date']

        print(f"\nLONGEST NEGATIVE STREAK: {longest_neg['length']} months")
        print(f"  Period: {start_date.strftime('%b %Y')} to {end_date.strftime('%b %Y')}")

    # Current streak
    current = streaks[-1]
    start_date = df.iloc[current['start_idx']]['Date']
    print(f"\nCURRENT STREAK: {current['length']} {current['type'].lower()} months")
    print(f"  Started: {start_date.strftime('%b %Y')}")


def monthly_seasonality(df):
    """Analyze which calendar months tend to perform best/worst"""
    print_header("MONTHLY SEASONALITY ANALYSIS")

    month_order = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
                   'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']

    monthly_stats = df.groupby('Month').agg({
        'Percentage_Change': ['mean', 'std', 'min', 'max', 'count'],
        'Change': ['mean', 'sum']
    })

    monthly_stats.columns = ['Avg_Pct', 'Std_Pct', 'Min_Pct', 'Max_Pct', 'Count', 'Avg_Change', 'Total_Change']
    monthly_stats['Positive_Count'] = df.groupby('Month')['Percentage_Change'].apply(lambda x: (x > 0).sum())
    monthly_stats['Win_Rate'] = monthly_stats['Positive_Count'] / monthly_stats['Count'] * 100

    monthly_stats = monthly_stats.reindex(month_order)

    print(f"\n{'Month':<6} {'Avg %':>8} {'Win Rate':>10} {'Best':>10} {'Worst':>10} {'Avg Rs.':>12}")
    print("-" * 60)

    for month in month_order:
        if month in monthly_stats.index:
            row = monthly_stats.loc[month]
            print(f"{month:<6} {row['Avg_Pct']:>7.2f}% {row['Win_Rate']:>9.0f}% "
                  f"{row['Max_Pct']:>9.1f}% {row['Min_Pct']:>9.1f}% Rs.{row['Avg_Change']:>10,.0f}")

    # Best and worst months
    best_month = monthly_stats['Avg_Pct'].idxmax()
    worst_month = monthly_stats['Avg_Pct'].idxmin()

    print("\n" + "-" * 60)
    print(f"HISTORICALLY BEST MONTH:  {best_month} (Avg: {monthly_stats.loc[best_month, 'Avg_Pct']:.2f}%)")
    print(f"HISTORICALLY WORST MONTH: {worst_month} (Avg: {monthly_stats.loc[worst_month, 'Avg_Pct']:.2f}%)")


def quarterly_analysis(df):
    """Analyze portfolio by quarters"""
    print_header("QUARTERLY ANALYSIS")

    # Assign quarters
    df_q = df.copy()
    df_q['Quarter'] = df_q['Date'].dt.quarter
    df_q['YearQuarter'] = df_q['Year'].astype(str) + '-Q' + df_q['Quarter'].astype(str)

    # Group by quarter
    quarterly = df_q.groupby(['Year', 'Quarter']).agg({
        'Change': 'sum',
        'Percentage_Change': 'mean',
        'Consolidated_portfolio_Value': ['first', 'last']
    }).reset_index()

    quarterly.columns = ['Year', 'Quarter', 'Total_Change', 'Avg_Monthly_Pct', 'Start_Value', 'End_Value']
    quarterly['Quarter_Return'] = ((quarterly['End_Value'] / quarterly['Start_Value']) - 1) * 100

    print(f"\n{'Year-Q':<10} {'Start Value':>14} {'End Value':>14} {'Change':>14} {'Return':>10}")
    print("-" * 65)

    for _, row in quarterly.iterrows():
        qtr_label = f"{int(row['Year'])}-Q{int(row['Quarter'])}"
        print(f"{qtr_label:<10} Rs.{row['Start_Value']:>11,.0f} Rs.{row['End_Value']:>11,.0f} "
              f"Rs.{row['Total_Change']:>11,.0f} {row['Quarter_Return']:>9.2f}%")

    # Best and worst quarters
    print("\n" + "-" * 65)
    best_q = quarterly.loc[quarterly['Total_Change'].idxmax()]
    worst_q = quarterly.loc[quarterly['Total_Change'].idxmin()]

    print(f"BEST QUARTER:  {int(best_q['Year'])}-Q{int(best_q['Quarter'])} "
          f"(Rs.{best_q['Total_Change']:+,.0f}, {best_q['Quarter_Return']:+.2f}%)")
    print(f"WORST QUARTER: {int(worst_q['Year'])}-Q{int(worst_q['Quarter'])} "
          f"(Rs.{worst_q['Total_Change']:+,.0f}, {worst_q['Quarter_Return']:+.2f}%)")


def top_months_analysis(df):
    """Show top gaining and losing months"""
    print_header("TOP 10 BEST & WORST MONTHS")

    print("\nTOP 10 BEST MONTHS (by absolute gain):")
    print("-" * 55)
    print(f"{'Rank':<6} {'Month':<12} {'Change':>14} {'Percentage':>12}")
    print("-" * 55)

    best = df.nlargest(10, 'Change')[['Month', 'Year', 'Change', 'Percentage_Change']]
    for rank, (_, row) in enumerate(best.iterrows(), 1):
        print(f"{rank:<6} {row['Month']} {int(row['Year']):<6} Rs.{row['Change']:>11,.0f} {row['Percentage_Change']:>11.2f}%")

    print("\n\nTOP 10 WORST MONTHS (by absolute loss):")
    print("-" * 55)
    print(f"{'Rank':<6} {'Month':<12} {'Change':>14} {'Percentage':>12}")
    print("-" * 55)

    worst = df.nsmallest(10, 'Change')[['Month', 'Year', 'Change', 'Percentage_Change']]
    for rank, (_, row) in enumerate(worst.iterrows(), 1):
        print(f"{rank:<6} {row['Month']} {int(row['Year']):<6} Rs.{row['Change']:>11,.0f} {row['Percentage_Change']:>11.2f}%")


def portfolio_summary(df):
    """Overall portfolio summary"""
    print_header("PORTFOLIO SUMMARY")

    first_date = df['Date'].iloc[0]
    last_date = df['Date'].iloc[-1]
    first_value = df['Consolidated_portfolio_Value'].iloc[0]
    last_value = df['Consolidated_portfolio_Value'].iloc[-1]

    total_months = len(df)
    total_years = total_months / 12

    total_change = last_value - first_value

    print(f"\n  Investment Period:    {first_date.strftime('%b %Y')} to {last_date.strftime('%b %Y')}")
    print(f"  Duration:             {total_months} months ({total_years:.1f} years)")
    print(f"  Starting Value:       Rs.{first_value:,.2f}")
    print(f"  Current Value:        Rs.{last_value:,.2f}")
    print(f"  Total Value Change:   Rs.{total_change:,.2f}")
    print(f"  All-Time High:        Rs.{df['Consolidated_portfolio_Value'].max():,.2f}")
    print(f"  All-Time Low:         Rs.{df['Consolidated_portfolio_Value'].min():,.2f}")


def recovery_analysis(df):
    """Analyze recovery periods from drawdowns"""
    print_header("RECOVERY ANALYSIS")

    values = df['Consolidated_portfolio_Value'].values
    dates = df['Date'].values

    # Find drawdown periods and their recoveries
    running_max = np.maximum.accumulate(values)
    drawdowns = (values - running_max) / running_max * 100

    # Identify drawdown periods
    in_drawdown = False
    drawdown_periods = []
    peak_idx = 0
    trough_idx = 0
    trough_value = float('inf')

    for i in range(len(values)):
        if drawdowns[i] < 0:
            if not in_drawdown:
                # Start of new drawdown
                in_drawdown = True
                peak_idx = i - 1 if i > 0 else 0
                trough_idx = i
                trough_value = values[i]
            else:
                # Continue in drawdown, check if new trough
                if values[i] < trough_value:
                    trough_idx = i
                    trough_value = values[i]
        else:
            if in_drawdown:
                # Recovery happened
                recovery_idx = i
                drawdown_periods.append({
                    'peak_idx': peak_idx,
                    'peak_date': pd.Timestamp(dates[peak_idx]),
                    'peak_value': running_max[peak_idx],
                    'trough_idx': trough_idx,
                    'trough_date': pd.Timestamp(dates[trough_idx]),
                    'trough_value': trough_value,
                    'recovery_idx': recovery_idx,
                    'recovery_date': pd.Timestamp(dates[recovery_idx]),
                    'drawdown_pct': ((trough_value - running_max[peak_idx]) / running_max[peak_idx]) * 100,
                    'months_to_trough': trough_idx - peak_idx,
                    'months_to_recover': recovery_idx - trough_idx,
                    'total_months': recovery_idx - peak_idx,
                    'recovered': True
                })
                in_drawdown = False
                trough_value = float('inf')

    # Handle ongoing drawdown
    if in_drawdown:
        drawdown_periods.append({
            'peak_idx': peak_idx,
            'peak_date': pd.Timestamp(dates[peak_idx]),
            'peak_value': running_max[peak_idx],
            'trough_idx': trough_idx,
            'trough_date': pd.Timestamp(dates[trough_idx]),
            'trough_value': trough_value,
            'recovery_idx': None,
            'recovery_date': None,
            'drawdown_pct': ((trough_value - running_max[peak_idx]) / running_max[peak_idx]) * 100,
            'months_to_trough': trough_idx - peak_idx,
            'months_to_recover': None,
            'total_months': None,
            'recovered': False
        })

    # Filter significant drawdowns (> 5%)
    significant_drawdowns = [d for d in drawdown_periods if d['drawdown_pct'] < -5]

    if len(significant_drawdowns) == 0:
        print("\nNo significant drawdowns (>5%) found.")
        return

    print(f"\nSIGNIFICANT DRAWDOWN PERIODS (>5% decline):")
    print("-" * 95)
    print(f"{'Peak Date':<12} {'Trough Date':<12} {'Recovery':<12} {'Drawdown':>10} {'To Trough':>12} {'To Recover':>12}")
    print("-" * 95)

    for d in sorted(significant_drawdowns, key=lambda x: x['drawdown_pct']):
        peak_str = d['peak_date'].strftime('%b %Y')
        trough_str = d['trough_date'].strftime('%b %Y')
        recovery_str = d['recovery_date'].strftime('%b %Y') if d['recovered'] else 'Not yet'
        recover_months = f"{d['months_to_recover']} months" if d['recovered'] else 'Ongoing'

        print(f"{peak_str:<12} {trough_str:<12} {recovery_str:<12} {d['drawdown_pct']:>9.1f}% "
              f"{d['months_to_trough']:>8} months {recover_months:>12}")

    # Summary statistics
    recovered_periods = [d for d in significant_drawdowns if d['recovered']]
    if recovered_periods:
        avg_recovery = np.mean([d['months_to_recover'] for d in recovered_periods])
        max_recovery = max([d['months_to_recover'] for d in recovered_periods])
        print("\n" + "-" * 95)
        print(f"RECOVERY STATISTICS (from significant drawdowns):")
        print(f"  Average time to recover: {avg_recovery:.1f} months")
        print(f"  Longest recovery period: {max_recovery} months")


def milestone_tracking_1L(df):
    """Track portfolio milestones in 1 lakh increments"""
    print_header("PORTFOLIO MILESTONES (Every 1 Lakh)")

    current_value = df['Consolidated_portfolio_Value'].iloc[-1]
    max_milestone = int((current_value // 100000) + 1) * 100000

    milestones = list(range(100000, max_milestone + 1, 100000))

    print(f"\n{'Milestone':>12} {'Date Reached':<15} {'Note':<20}")
    print("-" * 50)

    milestone_data = []
    prev_cross_date = None

    for milestone in milestones:
        crossed = df[df['Consolidated_portfolio_Value'] >= milestone]
        if len(crossed) > 0:
            cross_date = crossed['Date'].iloc[0]

            # Check if same month as previous milestone
            same_month_note = ""
            if prev_cross_date is not None and cross_date == prev_cross_date:
                same_month_note = "(same month)"

            milestone_data.append({
                'milestone': milestone,
                'date': cross_date,
                'same_month': same_month_note
            })

            print(f"Rs.{milestone/100000:>8.0f}L   {cross_date.strftime('%b %Y'):<15} {same_month_note:<20}")

            prev_cross_date = cross_date
        else:
            print(f"Rs.{milestone/100000:>8.0f}L   {'Not yet reached':<15}")

    # Show months where multiple milestones were crossed
    dates_with_multiple = {}
    for m in milestone_data:
        date_str = m['date'].strftime('%b %Y')
        if date_str not in dates_with_multiple:
            dates_with_multiple[date_str] = []
        dates_with_multiple[date_str].append(f"{m['milestone']/100000:.0f}L")

    multi_milestone_months = {k: v for k, v in dates_with_multiple.items() if len(v) > 1}

    if multi_milestone_months:
        print("\n" + "-" * 50)
        print("MONTHS WITH MULTIPLE MILESTONES:")
        for date_str, milestones_list in multi_milestone_months.items():
            print(f"  {date_str}: Crossed {', '.join(milestones_list)}")


def best_worst_periods(df):
    """Analyze best and worst 3-month and 6-month periods"""
    print_header("BEST & WORST MULTI-MONTH PERIODS")

    # Calculate rolling sums
    df_periods = df.copy()
    df_periods['Rolling_3M'] = df_periods['Change'].rolling(window=3).sum()
    df_periods['Rolling_6M'] = df_periods['Change'].rolling(window=6).sum()
    df_periods['Rolling_3M_Pct'] = df_periods['Percentage_Change'].rolling(window=3).sum()
    df_periods['Rolling_6M_Pct'] = df_periods['Percentage_Change'].rolling(window=6).sum()

    # Best 3-month periods
    print("\nTOP 3 BEST 3-MONTH PERIODS:")
    print("-" * 70)
    best_3m = df_periods.nlargest(3, 'Rolling_3M')[['Date', 'Rolling_3M', 'Rolling_3M_Pct']]
    for rank, (idx, row) in enumerate(best_3m.iterrows(), 1):
        end_date = row['Date']
        start_date = df_periods.loc[idx - 2, 'Date'] if idx >= 2 else df_periods['Date'].iloc[0]
        print(f"  {rank}. {start_date.strftime('%b %Y')} - {end_date.strftime('%b %Y')}: "
              f"Rs.{row['Rolling_3M']/100000:.2f}L ({row['Rolling_3M_Pct']:+.1f}%)")

    # Worst 3-month periods
    print("\nTOP 3 WORST 3-MONTH PERIODS:")
    print("-" * 70)
    worst_3m = df_periods.nsmallest(3, 'Rolling_3M')[['Date', 'Rolling_3M', 'Rolling_3M_Pct']]
    for rank, (idx, row) in enumerate(worst_3m.iterrows(), 1):
        end_date = row['Date']
        start_date = df_periods.loc[idx - 2, 'Date'] if idx >= 2 else df_periods['Date'].iloc[0]
        print(f"  {rank}. {start_date.strftime('%b %Y')} - {end_date.strftime('%b %Y')}: "
              f"Rs.{row['Rolling_3M']/100000:.2f}L ({row['Rolling_3M_Pct']:+.1f}%)")

    # Best 6-month periods
    print("\nTOP 3 BEST 6-MONTH PERIODS:")
    print("-" * 70)
    best_6m = df_periods.nlargest(3, 'Rolling_6M')[['Date', 'Rolling_6M', 'Rolling_6M_Pct']]
    for rank, (idx, row) in enumerate(best_6m.iterrows(), 1):
        end_date = row['Date']
        start_date = df_periods.loc[idx - 5, 'Date'] if idx >= 5 else df_periods['Date'].iloc[0]
        print(f"  {rank}. {start_date.strftime('%b %Y')} - {end_date.strftime('%b %Y')}: "
              f"Rs.{row['Rolling_6M']/100000:.2f}L ({row['Rolling_6M_Pct']:+.1f}%)")

    # Worst 6-month periods
    print("\nTOP 3 WORST 6-MONTH PERIODS:")
    print("-" * 70)
    worst_6m = df_periods.nsmallest(3, 'Rolling_6M')[['Date', 'Rolling_6M', 'Rolling_6M_Pct']]
    for rank, (idx, row) in enumerate(worst_6m.iterrows(), 1):
        end_date = row['Date']
        start_date = df_periods.loc[idx - 5, 'Date'] if idx >= 5 else df_periods['Date'].iloc[0]
        print(f"  {rank}. {start_date.strftime('%b %Y')} - {end_date.strftime('%b %Y')}: "
              f"Rs.{row['Rolling_6M']/100000:.2f}L ({row['Rolling_6M_Pct']:+.1f}%)")


def yoy_month_comparison(df):
    """Year-over-Year comparison for each calendar month"""
    print_header("YEAR-OVER-YEAR MONTHLY COMPARISON")

    month_order = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
                   'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']

    # Create pivot table
    pivot = df.pivot_table(values='Percentage_Change', index='Month', columns='Year', aggfunc='mean')
    pivot = pivot.reindex(month_order)

    years = sorted(df['Year'].unique())

    # Print header
    year_headers = ''.join([f'{int(y):>8}' for y in years])
    print(f"\n{'Month':<6} {year_headers} {'Avg':>8} {'Min':>8} {'Max':>8}")
    print("-" * (14 + len(years) * 8 + 24))

    # Print each month's data
    for month in month_order:
        if month in pivot.index:
            row_values = []
            for year in years:
                if year in pivot.columns and not pd.isna(pivot.loc[month, year]):
                    row_values.append(f'{pivot.loc[month, year]:>7.1f}%')
                else:
                    row_values.append(f'{"--":>7} ')

            # Calculate stats for this month
            month_data = df[df['Month'] == month]['Percentage_Change']
            avg_val = month_data.mean()
            min_val = month_data.min()
            max_val = month_data.max()

            values_str = ''.join(row_values)
            print(f"{month:<6} {values_str} {avg_val:>7.1f}% {min_val:>7.1f}% {max_val:>7.1f}%")

    # Summary: Most consistent and most volatile months
    print("\n" + "-" * 70)
    month_stats = df.groupby('Month')['Percentage_Change'].agg(['mean', 'std', 'min', 'max'])
    month_stats = month_stats.reindex(month_order)

    most_consistent = month_stats['std'].idxmin()
    most_volatile = month_stats['std'].idxmax()
    best_avg = month_stats['mean'].idxmax()
    worst_avg = month_stats['mean'].idxmin()

    print(f"MOST CONSISTENT MONTH: {most_consistent} (Std Dev: {month_stats.loc[most_consistent, 'std']:.2f}%)")
    print(f"MOST VOLATILE MONTH:   {most_volatile} (Std Dev: {month_stats.loc[most_volatile, 'std']:.2f}%)")
    print(f"BEST AVERAGE MONTH:    {best_avg} (Avg: {month_stats.loc[best_avg, 'mean']:+.2f}%)")
    print(f"WORST AVERAGE MONTH:   {worst_avg} (Avg: {month_stats.loc[worst_avg, 'mean']:+.2f}%)")


def plot_yoy_month_comparison(df):
    """Generate Year-over-Year monthly comparison chart with avg and range"""
    month_order = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
                   'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']

    # Calculate stats for each month
    month_stats = df.groupby('Month')['Percentage_Change'].agg(['mean', 'min', 'max', 'std'])
    month_stats = month_stats.reindex(month_order)

    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(month_order))

    # Plot average as bars
    colors = ['#28A745' if m >= 0 else '#DC3545' for m in month_stats['mean'].values]
    bars = ax.bar(x, month_stats['mean'].values, color=colors, alpha=0.7, label='Average')

    # Plot min-max range as error bars
    yerr_lower = month_stats['mean'].values - month_stats['min'].values
    yerr_upper = month_stats['max'].values - month_stats['mean'].values
    ax.errorbar(x, month_stats['mean'].values, yerr=[yerr_lower, yerr_upper],
                fmt='none', color='black', capsize=5, capthick=2, label='Min-Max Range')

    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(month_order)
    ax.set_title('Year-over-Year Monthly Comparison (Average with Min-Max Range)', fontsize=20, fontweight='bold')
    ax.set_xlabel('Month')
    ax.set_ylabel('Return (%)')
    ax.legend(loc='upper right')

    # Add value labels
    for i, (bar, avg) in enumerate(zip(bars, month_stats['mean'].values)):
        height = bar.get_height()
        ax.annotate(f'{avg:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3 if height >= 0 else -12),
                    textcoords="offset points",
                    ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{GRAPH_DIR}/10_yoy_month_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_recovery_analysis(df):
    """Generate recovery analysis visualization"""
    values = df['Consolidated_portfolio_Value'].values
    dates = df['Date'].values

    running_max = np.maximum.accumulate(values)
    drawdowns = (values - running_max) / running_max * 100

    # Find recovery periods
    in_drawdown = False
    recovery_periods = []
    peak_idx = 0

    for i in range(len(values)):
        if drawdowns[i] < -5:  # Significant drawdown
            if not in_drawdown:
                in_drawdown = True
                peak_idx = i - 1 if i > 0 else 0
        else:
            if in_drawdown:
                recovery_periods.append((peak_idx, i))
                in_drawdown = False

    fig, ax = plt.subplots(figsize=(14, 7))

    # Plot portfolio value
    ax.plot(df['Date'], df['Consolidated_portfolio_Value'], linewidth=2, color='#2E86AB', label='Portfolio Value')

    # Highlight recovery periods
    for start_idx, end_idx in recovery_periods:
        ax.axvspan(pd.Timestamp(dates[start_idx]), pd.Timestamp(dates[end_idx]),
                   alpha=0.2, color='#FFC107', label='Recovery Period' if start_idx == recovery_periods[0][0] else '')

    ax.set_title('Portfolio Value with Recovery Periods Highlighted', fontsize=20, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Portfolio Value (Rs.)')
    ax.yaxis.set_major_formatter(FuncFormatter(format_lakhs))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)
    ax.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(f'{GRAPH_DIR}/11_recovery_periods.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    # Setup dual output
    output = DualOutput('insight.txt')
    sys.stdout = output

    try:
        print("\n" + "=" * 80)
        print("           PORTFOLIO INSIGHTS ANALYZER")
        print("           Generated on: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        print("=" * 80)

        # Load data
        print("\nLoading portfolio_data.csv...")
        df = load_portfolio_data('portfolio_data.csv')
        print(f"Loaded {len(df)} months of data.")

        # Run all analyses
        portfolio_summary(df)
        yearly_summary(df)
        portfolio_milestones(df)
        milestone_tracking_1L(df)
        portfolio_value_changes(df)
        quarterly_analysis(df)
        monthly_seasonality(df)
        yoy_month_comparison(df)
        top_months_analysis(df)
        best_worst_periods(df)
        volatility_analysis(df)
        streak_analysis(df)
        drawdown_analysis(df)
        recovery_analysis(df)

        # Generate graphs
        generate_all_graphs(df)

        print("\n" + "=" * 80)
        print("  ANALYSIS COMPLETE")
        print("  Text output saved to: insight.txt")
        print(f"  Graphs saved to: {GRAPH_DIR}/")
        print("=" * 80 + "\n")

    finally:
        sys.stdout = output.terminal
        output.close()
        print("Insights have been saved to insight.txt")
        print(f"Graphs have been saved to '{GRAPH_DIR}/' folder")


if __name__ == "__main__":
    main()
