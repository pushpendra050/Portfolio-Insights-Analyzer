# Portfolio Insights Analyzer

A Python tool to analyze your investment portfolio growth patterns and generate comprehensive insights with visualizations.

## Overview

This tool analyzes your monthly portfolio data and generates:
- **Text Report** (`insight.txt`) - Detailed analysis with tables and statistics
- **Graphs** (`graphs/` folder) - 11+ visualizations of your portfolio performance

## Data Source

The `portfolio_data.csv` file should contain your consolidated portfolio values. You can obtain this data from:

- **NSDL CAS** (Consolidated Account Statement) - [https://nsdl.co.in/](https://nsdl.co.in/)
- **CDSL CAS** (Consolidated Account Statement) - [https://www.cdslindia.com/](https://www.cdslindia.com/)

Download your CAS statement and extract the monthly portfolio values to create the CSV file.

## CSV File Format

Your `portfolio_data.csv` should have the following columns:

| Column | Description | Example |
|--------|-------------|---------|
| `Month` | 3-letter month code (uppercase) | JAN, FEB, MAR, ... |
| `Year` | 4-digit year | 2023, 2024 |
| `Consolidated_portfolio_Value` | Total portfolio value at month end | 525000 |
| `Change` | Change from previous month (Rs.) | 15000 |
| `Percentage_Change` | Change from previous month (%) | 2.94 |

**Sample CSV:**
```csv
Month,Year,Consolidated_portfolio_Value,Change,Percentage_Change
OCT,2017,52340,52340,0.0
NOV,2017,58920,6580,12.57
DEC,2017,61450,2530,4.29
```

## Installation

1. Clone this repository
2. Install required packages:
```bash
pip install pandas numpy matplotlib
```

## Usage

1. Place your `portfolio_data.csv` in the same directory as `insights.py`
2. Run the script:
```bash
python insights.py
```
3. Check the outputs:
   - `insight.txt` - Text report with all analyses
   - `graphs/` folder - All generated visualizations

## Generated Analyses

### Text Report Sections
- Portfolio Summary
- Year-wise Portfolio Summary
- Best & Worst Months Per Year
- Portfolio Milestones (5L increments)
- Portfolio Milestones (1L increments with same-month tracking)
- Average Speed / Pace Analysis
- Portfolio Value Changes Over Time
- Quarterly Analysis
- Monthly Seasonality Analysis
- Year-over-Year Monthly Comparison
- Top 10 Best & Worst Months
- Best/Worst 3-month and 6-month Periods
- Volatility Analysis
- Streak Analysis
- Drawdown Analysis
- Recovery Analysis

### Generated Graphs
| File | Description |
|------|-------------|
| `01_portfolio_growth.png` | Portfolio value over time |
| `02_monthly_changes.png` | Monthly change bar chart |
| `03a_yearly_growth.png` | Year-wise portfolio growth |
| `03b_yearend_values.png` | Year-end portfolio values |
| `04_drawdown_analysis.png` | Drawdown visualization |
| `05a_monthly_heatmap.png` | Monthly returns heatmap |
| `05b_avg_monthly_returns.png` | Average monthly returns |
| `06_quarterly_performance.png` | Quarterly changes |
| `07a_volatility_analysis.png` | Return distribution & volatility |
| `07b_return_distribution_by_year.png` | Box plot by year |
| `07c_positive_negative_months.png` | Positive vs negative months pie |
| `08_milestones.png` | Milestone achievement timeline |
| `08b_milestones_1L.png` | 1 Lakh milestones with time taken |
| `09_top_months.png` | Top best & worst months |
| `10_yoy_month_comparison.png` | Year-over-year monthly comparison |
| `11_recovery_periods.png` | Recovery periods visualization |

## Important Notes

- This tool analyzes **portfolio value changes**, not investment returns
- Value changes include: new investments (SIP/lumpsum) + market movements
- For pure return analysis, you would need to separate investments from gains

## Requirements

- Python 3.7+
- pandas
- numpy
- matplotlib

## License

MIT License - Feel free to use and modify as needed.

## Disclaimer

This tool is for personal portfolio tracking and analysis only. It does not provide investment advice. Past performance does not guarantee future results.
