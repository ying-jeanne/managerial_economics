# DBA5101: Estimation of Demand Function for Train Travel

This project estimates the demand function for train travel using econometric methods to address endogeneity in price-quantity relationships. The analysis applies OLS and Two-Stage Least Squares (2SLS) with supply-side instruments to identify causal price effects on train ticket demand.

## Project Overview
**Objective**: Estimate price elasticity of demand for train travel while correcting for simultaneity bias
**Dataset**: 209,697 observations of train ticket sales with pricing, customer, and temporal data
**Methods**: OLS vs 2SLS comparison using supply-side instrumental variables
**Key Finding**: Demand elasticity of -0.96 (2SLS) vs -0.17 (OLS), showing 5.8× endogeneity bias

## Analysis Pipeline
- **Data Preparation**: Cleaning, type conversion, feature engineering
- **Exploratory Analysis**: Correlation patterns, distributional analysis, customer segmentation
- **Econometric Estimation**: OLS and 2SLS demand function estimation
- **Specification Testing**: Linear vs log-log, booking horizon functional forms
- **Instrument Validation**: Supply-side identification strategy
- **Robustness Checks**: Multiple specifications and temporal patterns

## Key Files
- `main.py`: Main econometric analysis pipeline
- `data_explore.py`: Exploratory data analysis and visualization
- `feature_selection.py`: Specification testing and instrument analysis
- `economy.tex`: Academic paper with methodology and results
- `data.csv`: Train ticket sales dataset (209,697 observations)
- `enhanced_data_1.csv`: Processed dataset with engineered features

## Usage
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run complete analysis:
   ```bash
   python main.py
   ```
3. Generate academic report:
   ```bash
   pdflatex economy.tex
   ```

## Key Economic Findings
- **Price Elasticity**: -0.96 (moderately inelastic) after endogeneity correction
- **Endogeneity Bias**: OLS understates price sensitivity by factor of 5.8×
- **Identification**: Supply-side instruments (capacity utilization, route characteristics)
- **Revenue Implications**: Price increases generate net revenue gains given inelastic demand

## Methodology
**Instrumental Variables**: Cumulative sales and train route characteristics
**Specifications**: Log-log specification preferred for distributional properties
**Controls**: Booking horizon, seasonal patterns, customer segments, interaction terms
**Validation**: Theoretical soundness of supply-side identification strategy

## Output Files
- `basic_statistic.png`: Key variables correlation heatmap
- `target_analysis.png`: Price and quantity distributions
- `booking_spec_correlation_heatmap.png`: Booking horizon relationships
- `booking_specifications_comparison.png`: Specification testing results
- `temporal_iv_analysis.png`: Seasonal vs monthly instrument analysis
- `economy.pdf`: Complete academic analysis report

## Academic Context
This analysis was conducted for DBA5101 (Managerial Economics) at National University of Singapore, demonstrating applied econometric methods for demand estimation in transportation markets.

## License
This project is for educational and analytical purposes. Please update the license as needed for your organization.