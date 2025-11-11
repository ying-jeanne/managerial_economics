# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**RecycleRight @ NUS - DBA5101 Group Project 2**

This project rigorously assesses the causal impact of two behavioral interventions on recycling contamination rates at the National University of Singapore using Difference-in-Differences (DiD) methodology.

**Business Problem**: High contamination rates in recycling streams impose downstream costs, reduce the value of recycled materials, and hinder NUS's "zero waste" sustainability goals. The university needs evidence-based behavioral nudges to improve recycling quality.

**Research Questions**:
1. Did shaped openings (Phase 2) cause a statistically significant reduction in contamination rates vs baseline?
2. Did informational banners (Phase 3) cause a further significant reduction beyond shaped openings?
3. Which intervention was more effective, and were there differential effects across material types?

## Experimental Design

**Quasi-Experimental Design** (suitable for Difference-in-Differences analysis):

- **Treatment Group (UTOWN)**: Received sequential interventions
  - Phase 1: Baseline (standard bins)
  - Phase 2: Intervention 1 (shaped openings)
  - Phase 3: Intervention 2 (shaped openings + informational banners)

- **Control Group (ENGINE)**: Standard bins throughout all phases (counterfactual)

**Key Dependent Variables**:
- `PaperContaminant`: Contamination rate (%) for paper bins
- `PlasticContaminant`: Contamination rate (%) for plastic bins
- `CanContaminant`: Contamination rate (%) for can bins

**Independent Variables**:
- `Area`: UTOWN (treatment) vs ENGINE (control)
- `FirstTrialPhase`: Phase 1, 2, or 3
- `Date`: Temporal dimension for time series analysis

## Analysis Pipeline

### 1. Data Loading & Preparation

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Load data
df = pd.read_csv('Project2Data.xls - RecycleRightFirstTrialDataSimpl.csv')

# Convert Excel date format to standard datetime
# Excel dates are stored as days since 1899-12-30
df['Date'] = pd.to_datetime(df['Date'], unit='D', origin='1899-12-30')

# Check for missing values
df.isnull().sum()

# Create dummy variables for regression
df['IsUTOWN'] = (df['Area'] == 'UTOWN').astype(int)
df['PostPhase2'] = (df['FirstTrialPhase'] >= 2).astype(int)
df['PostPhase3'] = (df['FirstTrialPhase'] == 3).astype(int)
```

### 2. Exploratory Data Analysis (EDA)

**Summary Statistics**:
```python
# Summary stats by phase, location, and material
summary = df.groupby(['FirstTrialPhase', 'Area'])[
    ['PaperContaminant', 'PlasticContaminant', 'CanContaminant']
].agg(['mean', 'median', 'std', 'count'])
```

**Critical: Parallel Trends Assumption Check**:
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Time series plot for each contaminant type
for contaminant in ['PaperContaminant', 'PlasticContaminant', 'CanContaminant']:
    daily_avg = df.groupby(['Date', 'Area'])[contaminant].mean().reset_index()

    plt.figure(figsize=(12, 6))
    for area in ['UTOWN', 'ENGINE']:
        data = daily_avg[daily_avg['Area'] == area]
        plt.plot(data['Date'], data[contaminant], label=area, marker='o')

    # Mark phase transitions
    plt.axvline(phase2_start_date, color='red', linestyle='--', label='Phase 2 Start')
    plt.axvline(phase3_start_date, color='orange', linestyle='--', label='Phase 3 Start')

    plt.title(f'{contaminant} Over Time: Parallel Trends Check')
    plt.xlabel('Date')
    plt.ylabel('Contamination Rate (%)')
    plt.legend()
    plt.show()
```

**Purpose**: Visually verify that UTOWN and ENGINE trends were parallel during Phase 1 (baseline). This is fundamental to DiD validity.

### 3. Difference-in-Differences Analysis

**DiD Core Formula**:
```
DiD Effect = (UTOWN_after - UTOWN_before) - (ENGINE_after - ENGINE_before)
```

#### 3a. Phase 2 Analysis (Shaped Openings Effect)

**Hypothesis**:
- H₀: Shaped openings had no effect (δ₁ = 0)
- H₁: Shaped openings reduced contamination (δ₁ < 0)

**Regression Model**:
```python
import statsmodels.formula.api as smf

# Filter to Phase 1 and Phase 2 only
df_phase2 = df[df['FirstTrialPhase'].isin([1, 2])].copy()
df_phase2['PostPhase2'] = (df_phase2['FirstTrialPhase'] == 2).astype(int)

# Run DiD regression for each contaminant
results_phase2 = {}
for contaminant in ['PaperContaminant', 'PlasticContaminant', 'CanContaminant']:
    model = smf.ols(
        f'{contaminant} ~ PostPhase2 + IsUTOWN + PostPhase2:IsUTOWN',
        data=df_phase2
    ).fit()
    results_phase2[contaminant] = model

    # The coefficient of PostPhase2:IsUTOWN is your DiD estimator (δ₁)
    print(f"\n{contaminant} - Phase 2 DiD Analysis:")
    print(model.summary())
    print(f"DiD Effect (δ₁): {model.params['PostPhase2:IsUTOWN']:.4f}")
    print(f"P-value: {model.pvalues['PostPhase2:IsUTOWN']:.4f}")
```

**Interpretation**: The interaction term `PostPhase2:IsUTOWN` (δ₁) represents the causal effect of shaped openings.

#### 3b. Phase 3 Analysis (Informational Banners Effect)

**Hypothesis**:
- H₀: Banners had no additional effect (δ₂ = 0)
- H₁: Banners caused further reduction (δ₂ < 0)

**Regression Model**:
```python
# Filter to Phase 2 and Phase 3 only
df_phase3 = df[df['FirstTrialPhase'].isin([2, 3])].copy()
df_phase3['PostPhase3'] = (df_phase3['FirstTrialPhase'] == 3).astype(int)

# Run DiD regression for each contaminant
results_phase3 = {}
for contaminant in ['PaperContaminant', 'PlasticContaminant', 'CanContaminant']:
    model = smf.ols(
        f'{contaminant} ~ PostPhase3 + IsUTOWN + PostPhase3:IsUTOWN',
        data=df_phase3
    ).fit()
    results_phase3[contaminant] = model

    # The coefficient of PostPhase3:IsUTOWN is your DiD estimator (δ₂)
    print(f"\n{contaminant} - Phase 3 DiD Analysis:")
    print(f"DiD Effect (δ₂): {model.params['PostPhase3:IsUTOWN']:.4f}")
    print(f"P-value: {model.pvalues['PostPhase3:IsUTOWN']:.4f}")
```

**Interpretation**: The interaction term `PostPhase3:IsUTOWN` (δ₂) represents the additional effect of banners beyond shaped openings.

### 4. Results Synthesis

**Create Summary Table**:
```python
results_summary = pd.DataFrame({
    'Material': ['Paper', 'Plastic', 'Cans'] * 2,
    'Intervention': ['Shaped Openings']*3 + ['Informational Banners']*3,
    'DiD Estimate': [
        results_phase2['PaperContaminant'].params['PostPhase2:IsUTOWN'],
        results_phase2['PlasticContaminant'].params['PostPhase2:IsUTOWN'],
        results_phase2['CanContaminant'].params['PostPhase2:IsUTOWN'],
        results_phase3['PaperContaminant'].params['PostPhase3:IsUTOWN'],
        results_phase3['PlasticContaminant'].params['PostPhase3:IsUTOWN'],
        results_phase3['CanContaminant'].params['PostPhase3:IsUTOWN']
    ],
    'Std Error': [...],  # Extract from model.bse
    'P-value': [...],    # Extract from model.pvalues
    'Significant': [...]  # True if p < 0.05
})
```

### 5. Visualization for Executive Summary

```python
# Bar chart showing DiD effects
fig, ax = plt.subplots(figsize=(10, 6))
results_summary.pivot(index='Material', columns='Intervention', values='DiD Estimate').plot(
    kind='bar', ax=ax, color=['#1f77b4', '#ff7f0e']
)
ax.set_ylabel('Change in Contamination Rate (percentage points)')
ax.set_title('Causal Effect of Interventions on Contamination Rates')
ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
ax.legend(title='Intervention')
plt.tight_layout()
plt.savefig('did_effects_summary.png', dpi=300)
```

## Required Python Libraries

```bash
pip install pandas numpy matplotlib seaborn scipy statsmodels xlrd openpyxl jupyter
```

**Core Libraries**:
- `pandas`: Data manipulation and analysis
- `numpy`: Numerical computations
- `statsmodels`: Regression analysis (OLS for DiD)
- `matplotlib`/`seaborn`: Visualization
- `scipy`: Statistical testing (if needed for robustness checks)
- `xlrd`: Reading .xls files

## Running the Analysis

**Start Jupyter Notebook**:
```bash
cd /Users/ying-jeanne/Workspace/managerial/II/notes
jupyter notebook project2.ipynb
```

**Or JupyterLab**:
```bash
jupyter lab
```

## Key Methodological Considerations

1. **Parallel Trends Assumption**: The validity of DiD depends on UTOWN and ENGINE having parallel trends in Phase 1. Always check this visually and discuss in your report.

2. **Statistical Significance**: Use α = 0.05 as the threshold. Report exact p-values and confidence intervals.

3. **Standard Errors**: Consider using robust standard errors if there's evidence of heteroskedasticity.

4. **Multiple Testing**: With 3 materials × 2 interventions = 6 tests, consider Bonferroni correction if being conservative.

5. **Effect Sizes**: Report both statistical significance AND practical significance (magnitude of contamination reduction).

## Report Structure

1. **Executive Summary** (1 paragraph): Clear recommendation with key numbers
2. **Problem Definition**: Business context and research questions
3. **Methodology**: DiD approach, experimental design, parallel trends validation
4. **Results**: Summary table + visualization of DiD effects with significance levels
5. **Discussion**:
   - Interpret findings in plain language
   - Address parallel trends assumption
   - Compare effectiveness across materials
   - Cost-benefit considerations
6. **Recommendations**:
   - Implementation strategy
   - Which intervention to scale
   - Next steps for further research
7. **Appendices**: Detailed regression output, additional charts

## Managerial Recommendations Framework

**Structure**:
1. Clear, actionable recommendation first
2. Justify with strongest DiD results
3. Cost vs. benefit analysis
4. Implementation roadmap
5. Limitations and future research

**Example Template**:
> "We recommend university-wide implementation of shaped-opening bin designs. Our Difference-in-Differences analysis shows that shaped openings caused a statistically significant X% reduction in [material] contamination (p=0.02, 95% CI: [a, b]), after controlling for campus-wide trends. The informational banners, however, showed no significant additional effect (p>0.10) and do not justify their ongoing maintenance costs. We propose a phased rollout starting with high-traffic areas, followed by a 6-month evaluation."

## Key Files

- [project2.ipynb](project2.ipynb): Main analysis notebook
- `Project2Data.xls - RecycleRightFirstTrialDataSimpl.csv`: Field trial data
- [GP2_Objective.pdf](../GP2_Objective.pdf): Project objectives
- [GP2_DataGlossary.pdf](../GP2_DataGlossary.pdf): Variable definitions
- [L6_DID.pdf](../L6_DID.pdf): Difference-in-Differences methodology reference

## Academic Context

DBA5101 (Managerial Economics) at NUS. This project demonstrates applied causal inference using DiD methodology to evaluate real-world behavioral interventions. The analysis must balance technical rigor with clear business communication.
