# Demand Function Estimation Implementation Plan

## Project Overview
Transform the current prediction-focused train ticket analysis into proper microeconomic demand function estimation using both OLS and 2SLS approaches to handle the bias-variance tradeoff.

## Current Analysis Issues
- Uses log-transformed target for prediction purposes
- Treats price-quantity relationship as prediction problem
- No consideration of endogeneity between price and quantity demanded
- Missing proper demand elasticity calculation

## Available Data (from logfile analysis)
- **Quantity**: `num_seats_total` (1-66, mean=2.38, 209,697 observations)
- **Price**: `mean_net_ticket_price` (1.28-7855.77, mean=230.12)
- **Potential Instruments**: `Culmulative_sales`, `Train_Number_All` (15 trains), `isNormCabin`
- **Demand Shifters**: temporal (weekends, seasons), customer segments, advance booking days

## Temporal Instrumental Variable Investigation ✅ COMPLETED

### Investigation Results (2025-09-15)
Comprehensive temporal IV analysis completed to determine optimal instrumental variables for 2SLS demand estimation.

**Final Recommendation: Use `departure_season_*` variables as temporal instruments**

#### Statistical Evidence
- **Season F-statistic**: 209.50 (Strong - well above F > 10 threshold)
- **Month F-statistic**: 152.38 (Also strong, but lower)
- **Season demonstrates superior first-stage predictive power** for price prediction

#### Economic Justification
**Seasonal patterns reflect clear supply-side operational factors:**
- **Winter**: Higher heating costs, weather disruptions, holiday operational premiums
- **Summer**: Peak capacity costs, higher operational complexity, vacation season logistics  
- **Spring/Fall**: Standard operational periods with baseline costs

**Instrumental Variable Validity:**
- **Relevance**: Season correlations with price range from -0.039 to +0.050 (adequate strength)
- **Exogeneity**: Seasons affect operational costs and capacity planning (supply-side) but shouldn't directly affect individual purchase decisions (demand-side)
- **Predetermined**: Seasonal patterns cannot be influenced by current demand shocks

#### Implementation Status ✅
- System already correctly uses `departure_season_*` variables in demand estimation (main.py:525, 671)
- Temporal IV investigation generated `temporal_iv_analysis.png` with comprehensive 8-panel analysis
- Documentation updated in `.claude/tasks/temporal_iv_investigation.md`

## Progress Status (Updated 2025-09-14)

### Key Change: Dual Specification Approach
- Keep both `num_seats_total` (levels) and `log_num_seats_total` (log form)  
- Compare linear vs log-linear demand specifications
- Determine which form provides better economic interpretation and fit

### Phase 1: Data Preparation ✅ COMPLETED
1. ✅ **Add statsmodels dependency** to requirements.txt - DONE
2. ✅ **Target analysis approach confirmed**: Keep both normal and log forms for comparison
3. **Preserve existing feature engineering** for demand shifters

## Implementation Tasks

### Phase 2: Dual Specification OLS Estimation
1. **Linear Demand Model**: 
   - `quantity = β₀ + β₁*price + β₂*demand_shifters + ε`
   - Elasticity: `β₁ * (mean_price/mean_quantity)`
2. **Log-Linear Demand Model**: 
   - `log(quantity) = β₀ + β₁*log(price) + β₂*demand_shifters + ε`
   - Elasticity: `β₁` (direct interpretation)
3. **Compare model specifications**: R², economic reasonableness, residual analysis

### Phase 3: Endogeneity Testing  
1. **Test both specifications** for price endogeneity
2. **Durbin-Wu-Hausman tests** for each model
3. **Assess practical significance** of potential bias in both forms

### Phase 4: 2SLS Implementation ✅ INSTRUMENTS VALIDATED
1. **First Stage** (same for both):
   - `price = π₀ + π₁*instruments + π₂*demand_shifters + υ`
   - **Instrument strength verified**: Season F-statistic = 209.50 > 10 ✅
   - **Recommended instruments**: `departure_season_*` (4 categories: Winter, Spring, Summer, Fall)
2. **Second Stage - Linear**: 
   - `quantity = β₀ + β₁*predicted_price + β₂*demand_shifters + ε`
3. **Second Stage - Log-Linear**: 
   - `log(quantity) = β₀ + β₁*log(predicted_price) + β₂*demand_shifters + ε`

### Phase 5: Model Comparison & Selection
1. **Economic Criteria**:
   - Price elasticity reasonableness (negative, magnitude 0.5-2.0)
   - Elasticity consistency across methods
2. **Statistical Criteria**:
   - Model fit (R², AIC)
   - Residual analysis and diagnostics
   - Robustness across specifications
3. **Final Recommendation**: Best specification with economic interpretation

### Phase 6: Integration
1. **Update main analysis pipeline**:
   - Replace prediction-focused modeling section
   - Add dual demand estimation workflow
   - Generate comparative visualizations
2. **Create summary output**: Economic interpretation and model selection rationale

## Technical Specifications

### Variable Roles
- **Y₁ (linear quantity)**: `num_seats_total`
- **Y₂ (log quantity)**: `log_num_seats_total` 
- **X₁ (linear price)**: `mean_net_ticket_price`
- **X₂ (log price)**: `log(mean_net_ticket_price)`
- **Z (instruments)**: `Culmulative_sales`, `Train_Number_All_*`, `isNormCabin`, **`departure_season_*`** (recommended temporal instruments)
- **W (demand shifters)**: `departure_is_weekend`, `departure_season_*`, `Customer_Cat_*`, `days_to_departure`

### Key Functions to Implement
```python
def estimate_linear_ols_demand(df, demand_shifters)
def estimate_loglinear_ols_demand(df, demand_shifters)
def test_endogeneity_dual(linear_ols, loglinear_ols, linear_2sls, loglinear_2sls) 
def estimate_linear_2sls_demand(df, instruments, demand_shifters)
def estimate_loglinear_2sls_demand(df, instruments, demand_shifters)
def calculate_elasticities(results_dict, df)
def compare_demand_specifications(results_dict)
```

### Expected Outputs
1. **Linear vs Log-Linear Comparison Table**
2. **Demand Elasticities**: Both specifications with confidence intervals
3. **Endogeneity Test Results**: For both models
4. **Model Selection Recommendation**: Statistical and economic criteria
5. **Economic Interpretation**: 
   - Price sensitivity analysis
   - Demand shifter effects
   - Seasonal/temporal patterns
   - Method robustness assessment

## Success Criteria
- Economically reasonable price elasticities in both specifications
- Clear statistical basis for model selection
- Strong instrument validation (F > 10) if using 2SLS
- Comprehensive comparison of linear vs log-linear approaches
- Actionable economic insights about train ticket demand

## Risk Mitigation
- If instruments are weak (F < 10), focus on OLS comparison
- If specifications give contradictory results, investigate data issues
- Ensure both models are economically interpretable before final selection
- Document limitations and assumptions clearly