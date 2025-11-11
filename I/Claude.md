# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Plan & Review

### Before starting work
- Always in plan mode to make a plan
- After get the plan, make sure you write the plan to .claude/tasks/TASK_NAME.md
- The plan should be a detailed implementation plan and the reasoning behid them, as well as tasks broken down.
- If the task require external knowledge or certain package, also research to get latest knowledge (use Task tool for research)
- Don't over plan it, always think MVP.
- Once you write the plan, firstly ask me to review it. Do not continue until I approve the plan.

### While implementing 
- You should update the plan as you work.
- After you complete tasks in the plan, you should update and append detailed descriptions of the change you made, so following tasks can be easily hand over to other engineers.

## Project Overview

This is a managerial economics analysis project focused on train ticket sales data. The main goal is to get the demand function of train tickets.

## Key Commands

### Setup and Dependencies
```bash
pip install -r requirements.txt
```

### Running the Analysis
```bash
python main.py
```
This will run the complete analysis pipeline and generate visualization files (PNG format).

## Architecture and Code Structure

### Core Analysis Pipeline (main.py)
The analysis follows a structured pipeline with these key phases:

1. **Data Loading & Exploration**: `load_and_explore_data()` - Basic dataset inspection
2. **Data Preparation**: `data_preparation_pipeline()` - Comprehensive data cleaning and type conversion
3. **Data Quality Verification**: `verify_data_quality()` - Missing values, duplicates, and range checks
4. **Business Logic Validation**: `is_isoneday_isreturn_hypothese_true()` - Validates ticket type logic
5. **Target Analysis**: `analyse_target()` - Analyzes the target variable with log transformation

### Key Business Logic
- **Target Variable**: `num_seats_total` (transformed to log scale as `log_num_seats_total`)
- **Ticket Types**: 
  - `isOneway=1, isReturn=0`: One-way tickets
  - `isOneway=0, isReturn=0`: Round-trip outbound leg
  - `isOneway=0, isReturn=1`: Round-trip return leg
  - Invalid combinations are cleaned from the dataset
- **Cumulative Sales**: `Culmulative_sales` tracks running totals across transactions

### Data Processing Features
- Automatic date parsing for `Dept_Date` and `Purchase_Date`
- Categorical variable encoding with one-hot encoding for low cardinality
- Advanced feature engineering including days to departure, seasonal features, and price interactions
- Log transformation of the target variable to handle skewness

## Output Files
- `target_analysis.png`: Target variable distribution and log transformation
- `univariate_analysis.png`: Distribution plots for all variables
- `correlation_analysis.png`: Correlation heatmap and scatter plots
- `categorical_correlation_analysis.png`: Categorical variable analysis
- `residual_analysis.png`: Model residuals (when modeling is enabled)

## Data Schema
- Main dataset: `data.csv` (9MB train ticket sales data)
- Target: Number of seats sold per transaction
- Features include ticket prices, dates, customer categories, train numbers, and ticket types
- Enhanced dataset available: `enhanced_data_1.csv` (additional processed features)

## Development Notes
- All analyses use pandas, numpy, matplotlib, seaborn, and scikit-learn
- The codebase emphasizes data quality validation and business logic verification
- Feature selection uses LASSO regularization (currently disabled in main execution)
- Visualization outputs are high-resolution PNG files (200 DPI)