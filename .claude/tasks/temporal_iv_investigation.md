# Temporal Instrumental Variable Investigation Plan

## Objective
Create a comprehensive analysis function to determine whether to use `departure_month` or `departure_season` as instrumental variables for demand function estimation.

## Background
From the demand function estimation plan, we need to select appropriate instruments for the 2SLS estimation. The choice between monthly (12 categories) vs seasonal (4 categories) temporal dummies affects both statistical power and economic interpretation.

## Key Instrumental Variable Criteria
1. **Relevance**: Strong correlation with endogenous variable (price) - F > 10 in first stage
2. **Exogeneity**: No direct correlation with unobserved demand factors
3. **Economic Logic**: Plausible that temporal patterns affect pricing but not demand directly

## Implementation Plan

### 1. Create `investigate_temporal_instruments()` function in main.py
**Function Signature:**
```python
def investigate_temporal_instruments(df):
    """
    Investigate departure_month vs departure_season as instrumental variables.
    Returns analysis results dict + saves visualization files.
    """
```

**Input**: Prepared DataFrame with temporal data (post data_preparation_pipeline)
**Output**: 
- Analysis results dictionary with statistical tests
- Saved visualization file: `temporal_iv_analysis.png`

### 2. Statistical Analysis Components

#### 2.1 Data Preparation (UPDATED)
- **Input**: Processed DataFrame with existing `departure_month_*` dummy variables
- Reconstruct `departure_month` (1-12) from dummy variables for analysis
- Create `departure_season` (Winter/Spring/Summer/Fall) from reconstructed months
- **Integration Point**: Called after `add_features()` in pipeline

#### 2.2 Variation Analysis
- **Monthly Variation**: Calculate coefficient of variation for price/quantity by month
- **Seasonal Variation**: Calculate coefficient of variation for price/quantity by season
- **Comparison**: Assess which provides more useful variation for identification

#### 2.3 First-Stage Strength Testing
- **Month Model**: `price = α + Σβᵢ*month_dummy_i + ε`
- **Season Model**: `price = α + Σγⱼ*season_dummy_j + ε`
- **Metrics**: F-statistics, R², coefficient significance
- **Threshold**: Strong instruments have F > 10

#### 2.4 Correlation Structure Analysis
- Correlation matrix: temporal variables vs price/quantity
- Cross-correlation patterns
- Identification of potential confounders

### 3. Visualization Components (6-panel combined plot)

#### Panel 1: Time Series Trends
- Monthly price and quantity averages over time
- Seasonal overlays to show aggregation patterns
- Trend lines and confidence intervals

#### Panel 2: Distribution Box Plots
- Price distributions by month and season
- Quantity distributions by month and season
- Outlier identification and variance comparison

#### Panel 3: Correlation Heatmap
- All temporal variables vs price/quantity correlations
- Color-coded significance levels
- Instrument relevance visualization

#### Panel 4: First-Stage Regression Comparison
- Fitted vs actual prices for month and season instruments
- R² and F-statistic annotations
- Residual scatter patterns

#### Panel 5: Coefficient Plots
- Estimated temporal dummy coefficients with confidence intervals
- Month vs season coefficient magnitude comparison
- Statistical significance indicators

#### Panel 6: Instrument Validity Diagnostics
- First-stage residuals analysis
- Heteroskedasticity tests
- Instrument strength summary metrics

### 4. Decision Framework

#### 4.1 Primary Criterion: Statistical Strength
- **Strong**: F > 10, R² > 0.05, significant coefficients
- **Moderate**: 5 < F < 10, some significant coefficients
- **Weak**: F < 5, low R², few significant coefficients

#### 4.2 Secondary Criteria
- **Economic Interpretation**: Seasonal pricing makes business sense
- **Variation Capture**: Sufficient price variation for identification
- **Granularity Trade-off**: More categories vs overfitting risk

#### 4.3 Recommendation Logic
```
if month_F > 10 and season_F > 10:
    choose higher F-statistic
elif month_F > 10:
    recommend departure_month
elif season_F > 10:
    recommend departure_season
else:
    recommend alternative instruments
```

### 5. Integration Points

#### 5.1 Main Analysis Pipeline (UPDATED)
- **Integration Point**: Called after `add_features()` in main analysis pipeline
- **Data Flow**: Uses processed DataFrame with departure_month dummies
- Store results for use in demand estimation
- Log recommendation reasoning

#### 5.2 Demand Function Updates
- Update `W (demand shifters)` list based on recommendation
- Modify instrument list in 2SLS estimation
- Document choice in analysis output

#### 5.3 Output Files
- `temporal_iv_analysis.png`: Combined 6-panel visualization
- Analysis log entry with recommendation
- Updated demand function estimation parameters

## Expected Results

### Statistical Output
```
TEMPORAL INSTRUMENTAL VARIABLE INVESTIGATION
Month Instruments - First Stage F-stat: XX.XX, R²: X.XXXX
Season Instruments - First Stage F-stat: XX.XX, R²: X.XXXX
Recommendation: departure_[month/season]
Reason: [statistical justification]
```

### Economic Interpretation
- Seasonal pricing patterns (holidays, weather, school schedules)
- Monthly granularity captures specific business cycles
- Trade-off between statistical power and economic intuition

### Visualization Insights
- Clear visual evidence of temporal pricing patterns
- Instrument strength comparison through fitted values
- Correlation structure supporting exogeneity assumption

## Success Criteria
1. **Clear Recommendation**: Statistical evidence for instrument choice
2. **Strong First Stage**: Chosen instrument has F > 10
3. **Economic Logic**: Selected temporal pattern makes business sense
4. **Visual Evidence**: Charts support statistical conclusions
5. **Integration Ready**: Results feed directly into demand estimation

## Risk Mitigation
- If both instruments are weak (F < 10), explore alternative identification strategies
- If results are contradictory, investigate data quality issues
- Document assumptions and limitations clearly
- Provide robustness checks with both instruments if both are strong

## Implementation Timeline
1. **Phase 1**: Create basic analysis function with statistical tests
2. **Phase 2**: Add comprehensive visualizations
3. **Phase 3**: Integrate with main analysis pipeline
4. **Phase 4**: Validate recommendation through demand estimation results

## Implementation Lessons Learned

### Initial Challenges
1. **Data Pipeline Integration**: Original design expected raw date columns but they were dropped during feature engineering
2. **Timing Issues**: Function needed to be called at the right point in the data processing pipeline
3. **Data Flow Mismatch**: Expected raw datetime data but received processed categorical variables

### Solution Approach
1. **Adaptive Function Design**: Modified function to work with existing `departure_month_*` dummy variables
2. **Dummy Variable Reconstruction**: Reverse-engineered month values from one-hot encoded variables
3. **Pipeline Integration**: Positioned function call after feature engineering completion

### Key Implementation Changes
- **Input Adaptation**: Changed from raw `Dept_Date` to processed dummy variables
- **Month Reconstruction**: Added logic to handle baseline month (month 1 with no dummy)
- **Error Handling**: Enhanced validation for required dummy columns
- **Integration Point**: Moved from pre-processing to post-feature-engineering

### Technical Design Decisions
- **Dummy Variable Approach**: Leverages existing data processing instead of duplicating date conversion
- **Baseline Handling**: Month 1 identified by absence of other month dummies
- **Season Mapping**: Preserved original business logic for seasonal categorization
- **Statistical Framework**: Maintained F-statistic and correlation analysis approach

## Instrumental Variable Selection: Season vs Month Analysis

### Final Recommendation: Use `departure_season` 

Based on comprehensive statistical analysis and economic theory, **departure_season** is the preferred instrumental variable over departure_month for the following reasons:

### 1. Statistical Strength ✅
- **Season F-statistic**: 197.86 (Strong - well above F > 10 threshold)
- **Month F-statistic**: 145.36 (Also strong, but lower)
- **Season has superior first-stage predictive power** for price prediction

### 2. Economic Interpretation ✅ **[Primary Advantage]**
**Seasonal patterns reflect clear supply-side operational factors:**
- **Winter**: Higher heating costs, weather disruptions, holiday operational premiums
- **Summer**: Peak capacity costs, higher operational complexity, vacation season logistics
- **Spring/Fall**: Standard operational periods with baseline costs

**Economic Logic for Exogeneity:**
- **Supply-side reasoning**: Seasons affect operational costs and capacity planning decisions
- **Demand-side exogeneity**: Individual ticket purchase decisions shouldn't systematically vary by season (controlling for price)
- **Predetermined nature**: Seasonal patterns cannot be influenced by current demand shocks

### 3. Instrumental Variable Validity Assessment ✅

**Relevance Condition** (`Corr(Z, price) ≠ 0`):
- **Month correlations with price**: Range from -0.035 to +0.056
- **Season correlations with price**: Range from -0.039 to +0.050
- **Both satisfy relevance**, but season shows stronger F-statistic

**Exogeneity Condition** (`Corr(Z, ε) = 0`):
- **Seasons more plausibly exogenous**: 4 broad categories reflect natural operational cycles
- **Months potentially less exogenous**: 12 specific categories harder to justify as purely supply-driven
- **Exclusion restriction**: Seasons affect quantity only through price, not through direct demand preferences

### 4. Granularity Trade-off Analysis ✅

**Month Dummies (12 categories):**
- ✅ More price variation (CV = 0.0336)
- ❌ Overfitting risk with 12 parameters
- ❌ Harder to justify economic logic for specific months (why January vs February?)
- ❌ Some months may reflect demand-side holidays rather than supply costs

**Season Dummies (4 categories):**
- ✅ Strong statistical power (F = 197.86 > 145.36)
- ✅ Clear economic interpretation aligned with operational cycles
- ✅ Reduces overfitting risk (4 vs 12 parameters)
- ✅ More robust to model specification changes
- ❌ Slightly less variation (CV = 0.0244)

### 5. Correlation Pattern Evidence ✅

**From Analysis Results:**
```
Season-Price Correlations:
  Summer: +0.050 (highest - peak operational costs)
  Winter: +0.019 (moderate - weather premiums)  
  Spring: -0.029 (lower - standard operations)
  Fall: -0.039 (lowest - off-peak operations)

Season-Quantity Correlations:
  Winter: +0.053 (moderate)
  Spring: +0.007 (minimal)
  Summer: -0.030 (negative) 
  Fall: -0.029 (negative)
```

**Interpretation**: Reasonable correlation magnitudes - strong enough for instrument relevance but not so strong as to suggest direct demand effects violating exogeneity.

### 6. Decision Algorithm Logic

The recommendation algorithm chose season based on:
```python
if f_stat_month > f_stat_season and f_stat_month > 10:
    recommendation = "departure_month"
elif f_stat_season > 10:  # ← This condition triggered
    recommendation = "departure_season"  
    reason = "adequate first-stage strength (197.86) with better economic interpretation"
```

**Key Insight**: Even though months provide marginally more price variation, seasons demonstrate **superior first-stage predictive power AND stronger economic justification** for the exclusion restriction required in demand estimation.

### 7. Practical Implementation Benefits ✅

- **Robustness**: Fewer parameters lead to more stable 2SLS estimates
- **Interpretability**: Easier to explain seasonal pricing patterns to stakeholders  
- **Model Parsimony**: Simpler to incorporate into demand models without overfitting
- **Economic Intuition**: Natural business cycles align with operational cost variations

### Conclusion

**departure_season** provides the optimal balance of statistical power and economic validity for instrumental variable estimation in the train ticket demand function. The superior F-statistic combined with clear supply-side economic logic makes it the more robust choice for identifying causal price effects in demand estimation.

## Technical Notes
- Use `statsmodels` for F-statistic calculations
- Leverage existing plotting infrastructure from main.py
- Work with processed categorical variables instead of raw dates
- Save high-resolution plots (200 DPI) matching project standards
- Handle one-hot encoding baseline category (month 1) properly