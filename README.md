# Managerial Economics Train Ticket Sales Analysis

This project analyzes train ticket sales data to understand demand, sales trends, and business logic for getting the demand function. The analysis is performed using Python, pandas, matplotlib, and seaborn.

## Features
- Data loading and exploration
- Data cleaning and type conversion
- Data quality checks (missing values, duplicates, unique values, ranges)
- Business logic validation for ticket types (isOneway, isReturn)
- Univariate and bivariate analysis (distribution, correlation, categorical relationships)
- Feature engineering (date-based, interaction, aggregation features)
- Visualization of sales trends, demand, and outliers

## Files
- `main.py`: Main analysis script containing all functions and workflow
- `data.csv`: Input dataset (train ticket sales)
- `requirements.txt`: Python dependencies

## Usage
1. Place your sales data in `data.csv`.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the analysis:
   ```bash
   python main.py
   ```
4. Output plots and analysis will be saved as PNG files in the project directory.

## Key Concepts
- **Culmulative_sales**: Running total of ticket sales, used to analyze sales momentum and demand.
- **num_seats_total**: Target variable representing ticket sales per transaction or day.
- **isOneway / isReturn**: Boolean features indicating ticket type, validated for business logic.
- **Feature Engineering**: Includes date features, interaction terms, and sales momentum.

## Customization
- Modify `main.py` to adjust grouping, filtering, or feature engineering as needed for your business questions.
- Add new visualizations or statistical tests to deepen your analysis.

## License
This project is for educational and analytical purposes. Please update the license as needed for your organization.
