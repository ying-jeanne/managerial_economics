import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_white
from statsmodels.stats.stattools import durbin_watson
from statsmodels.sandbox.regression.gmm import IV2SLS
from scipy import stats
from data_explore import data_exploration, get_interaction_shifters
from feature_selection import get_season
target = "num_seats_total"

# Load and explore the data
def load_and_explore_data(file_path):
    """Load CSV and do initial exploration"""
    print("="*60)
    print("LOADING AND EXPLORING TRAIN DATA")
    print("="*60)
    
    # Load data
    df = pd.read_csv(file_path)
    
    # Basic info
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\nFirst few rows:")
    print(df.head())
    print("\nData types:")
    print(df.dtypes)
    print("\nBasic statistics:")
    print(df.describe())
    df = fix_data_type(df)
    return df

def fix_data_type(df):
    """Convert date columns to datetime"""
    print("\n" + "="*60)
    print("CONVERTING DATES and Categorical Variables")
    print("="*60)
    
    df['Dept_Date'] = pd.to_datetime(df['Dept_Date'], errors='coerce')
    df['Purchase_Date'] = pd.to_datetime(df['Purchase_Date'], errors='coerce')

    # count number of lines has Dept_Date or Purchase_Date convert to NaT
    mask_dept = df['Dept_Date'].isna()
    mask_purchase = df['Purchase_Date'].isna()
    print(f"Rows with invalid Dept_Date: {mask_dept.sum()}")
    print(f"Rows with invalid Purchase_Date: {mask_purchase.sum()}")

    # remove those rows with invalid dates
    df = df[~(mask_dept | mask_purchase)].copy()

    # Convert categorical variables
    categorical_cols = ['Train_Number_All', 'Customer_Cat']
    df[categorical_cols] = df[categorical_cols].astype('category')

    # Convert boolean variables
    df['isNormCabin'] = df['isNormCabin'].astype('bool')
    df['isReturn'] = df['isReturn'].astype('bool') 
    df['isOneway'] = df['isOneway'].astype('bool')

    print("Date conversion completed!")
    print("\nData types after conversion:", df.dtypes)
    return df

def add_features(df):
    """Perform feature engineering"""
    print("\n" + "="*60)
    print("FEATURE ENGINEERING: Creating New Features")
    print("="*60)
    
    df_features = df.copy()

    # Ensure days_to_departure exists
    if 'days_to_departure' not in df.columns:
        df['Purchase_Date'] = pd.to_datetime(df['Purchase_Date'])
        df['Dept_Date'] = pd.to_datetime(df['Dept_Date'])
        df['days_to_departure'] = (df['Dept_Date'] - df['Purchase_Date']).dt.days
    
    # Always create log term if days_to_departure exists
    if 'days_to_departure' in df.columns:
        min_days = df['days_to_departure'].min()
        if min_days <= 0:
            print(f"⚠️  Found same-day bookings (min: {min_days}), using log(days + 1)")
            df_features['log_days_to_departure'] = np.log1p(df['days_to_departure'])
        else:
            df_features['log_days_to_departure'] = np.log(df['days_to_departure'])
    
    # Proper log transformations for demand estimation
    df_features['log_num_seats_total'] = np.log(df['num_seats_total'])  # Keep log1p for quantity (handles zeros)
    
    # Fix price log transformation - use natural log for proper elasticity interpretation
    min_price = df['mean_net_ticket_price'].min()
    if min_price <= 0:
        print(f"⚠️  Found zero/negative prices (min: {min_price:.2f}), using log(price + 1)")
        df_features['log_price'] = np.log1p(df['mean_net_ticket_price'])  # log(price + 1)
    else:
        df_features['log_price'] = np.log(df['mean_net_ticket_price'])  # standard log(price)

    df_features['departure_season'] = df['Dept_Date'].dt.month.apply(get_season).astype('category')
    df_features['departure_is_weekend'] = (df['Dept_Date'].dt.dayofweek >= 5).astype(bool)

    # Interaction shifters
    df_features['Business_x_Premium'] = (df['Customer_Cat'] == 'A') & ~df['isNormCabin']
    df_features['Individual_x_Premium'] = (df['Customer_Cat'] == 'B') & ~df['isNormCabin']
    df_features['Business_x_NormCabin'] = (df['Customer_Cat'] == 'A') & df['isNormCabin']

    print(f"Created features. New shape: {df_features.shape}")
    df_features.drop(columns=['Dept_Date', 'Purchase_Date'], inplace=True)  # Drop original date columns
    new_features = set(df_features.columns) - set(df.columns)
    print(f"New features: {list(new_features)}")
    
    return df_features

def encode_categorical(df, max_categories=15):
    """Encode categorical variables appropriately"""
    print("\n" + "="*60)
    print("FEATURE ENGINEERING: Encoding Categorical Variables")
    print("="*60)

    df_encoded = df.copy()
    
    # Get categorical columns
    categorical_cols = df.select_dtypes(include=['category']).columns

    for col in categorical_cols:
        n_categories = df[col].nunique()
        if n_categories <= max_categories:
            # One-hot encoding for low cardinality
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df_encoded = pd.concat([df_encoded, dummies], axis=1)
            df_encoded = df_encoded.drop(col, axis=1)
            print(f"{col}: One-hot encoded ({n_categories} categories)")
        else:
            # Handle high cardinality differently
            print(f"{col}: Too many categories ({n_categories}), consider grouping")
            # Option: Keep top N categories, group rest as 'Other'
            top_categories = df[col].value_counts().head(max_categories-1).index
            # Add 'Other' to categories if not present
            if 'Other' not in df_encoded[col].cat.categories:
                df_encoded[col] = df_encoded[col].cat.add_categories(['Other'])
            df_encoded[col] = df_encoded[col].where(~df_encoded[col].isin(top_categories), 'Other')

            # Then one-hot encode
            dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True)
            df_encoded = pd.concat([df_encoded, dummies], axis=1)
            df_encoded = df_encoded.drop(col, axis=1)
    
    print(f"Final encoded shape: {df_encoded.shape}")
    return df_encoded

def estimate_ols_demand(df, specification='linear'):
    """
    Estimate demand function using OLS: quantity = β₀ + β₁*price + β₂*demand_shifters + ε
    
    OLS APPROACH: Treats price as exogenous, includes ALL relevant demand shifters
    
    Demand shifters (W): Variables that directly affect quantity demanded
    - days_to_departure: Linear booking horizon effects
    - departure_is_weekend: Trip type/purpose (leisure vs business)
    - departure_season_*: Seasonal travel demand patterns (holidays, vacations)  
    - Customer_Cat_*: Customer segment effects
    - isNormCabin: Cabin type preference
    - Train_Number_All_*: Train type characteristics (route/service quality)
    - isOneway/isReturn: Trip type characteristics (one-way vs return travel)
    - Interaction terms: Business×Premium, Individual×Premium, etc.
    
    CONTRAST WITH 2SLS: In 2SLS, departure_season_* used as instruments (not shifters)
    """
    print(f"\n{'='*60}")
    print(f"OLS DEMAND ESTIMATION - {specification.upper()}")
    print(f"{'='*60}")
    
    # Prepare demand shifters based on our IV discussion
    base_shifters = ['log_days_to_departure', 'departure_is_weekend', 'isNormCabin', 'Customer_Cat_B']
    season_cols = [col for col in df.columns if 'departure_season_' in col]
    
    # Add train type characteristics as demand shifters
    train_cols = [col for col in df.columns if 'Train_Number_All_' in col]
    
    # Add trip type characteristics as demand shifters  
    trip_type_cols = [col for col in df.columns if col in ['isOneway', 'isReturn']]
    
    # Add interaction shifters for demand heterogeneity
    available_interactions = [col for col in get_interaction_shifters() if col in df.columns]

    # Combine all demand shifters (W variables)
    demand_shifters = base_shifters + season_cols + train_cols + trip_type_cols + available_interactions
    available_shifters = [col for col in demand_shifters if col in df.columns]
    
    print(f"Using {len(available_shifters)} demand shifters:")
    print(f"  • Base shifters: {[s for s in base_shifters if s in df.columns]}")
    print(f"  • Seasonal patterns: {len([s for s in season_cols if s in df.columns])}")
    print(f"  • Train characteristics: {len([s for s in train_cols if s in df.columns])}")
    print(f"  • Trip type shifters: {[s for s in trip_type_cols if s in df.columns]}")
    print(f"  • Interaction shifters: {available_interactions}")
    
    if specification == 'linear':
        y = df['num_seats_total'].astype(float)
        X_vars = ['mean_net_ticket_price'] + available_shifters
        price_var = 'mean_net_ticket_price'
    else:  # loglog
        y = df['log_num_seats_total'].astype(float)
        X_vars = ['log_price'] + available_shifters
        price_var = 'log_price'
    
    X = df[X_vars].copy()

    # Convert all columns to numeric, handling booleans properly
    for col in X.columns:
        if X[col].dtype == 'bool':
            X[col] = X[col].astype(int)  # Convert boolean to 0/1
        else:
            X[col] = X[col].astype(float)  # Ensure float type
    
    # Add constant term
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    
    print(f"Observations: {len(y):,}")
    print(f"R-squared: {model.rsquared:.4f}")
    
    if price_var in model.params:
        price_coef = model.params[price_var]
        price_pvalue = model.pvalues[price_var]
        
        if specification == 'linear':
            elasticity = price_coef * (df['mean_net_ticket_price'].mean() / df['num_seats_total'].mean())
        else:
            elasticity = price_coef
        print(f"\nPrice Elasticity: {elasticity:.4f} (p-value: {price_pvalue:.4e})")
        
        if elasticity < 0:
            print(f"✅ Negative price elasticity")
            print(f"   → {'Inelastic' if abs(elasticity) < 1 else 'Elastic'} demand")
        else:
            print(f"❌ Positive elasticity - may indicate endogeneity")
        return model, elasticity
    else:
        print("❌ Price variable not found")
        return None, None

def test_instrument_strength(df):
    """
    Test strength of instruments for 2SLS first-stage regression
    
    2SLS First Stage: price = π₀ + π₁*instruments + π₂*demand_shifters + υ
    
    DECISION ON departure_season_*:
    Using seasons as INSTRUMENTS (assuming exclusion restriction holds):
    - Relevance: Seasons affect operational costs → prices (F-stat: 209.50)
    - Exogeneity: Seasonal operational cycles are predetermined, affect supply-side costs
    - Exclusion: Assumes seasons affect quantity ONLY through price (not direct demand)
    
    Other instruments (Z): Variables affecting price but exogenous to demand
    - Culmulative_sales: Supply-side capacity constraints
    - Train_Number_All_*: Fixed route characteristics
    
    Demand shifters (W): Control variables included in BOTH stages
    - days_to_departure: Booking horizon preferences  
    - departure_is_weekend: Trip type (leisure vs business)
    - Customer_Cat_*: Customer segment effects
    """
    print(f"\n{'='*60}")
    print("INSTRUMENT STRENGTH TESTING FOR 2SLS")
    print(f"{'='*60}")
    
    # INSTRUMENTS (Z): Variables for identification, only in first stage
    season_cols = [col for col in df.columns if 'departure_season_' in col]  # Main instruments
    supply_instruments = ['Culmulative_sales']  # Supply-side constraints
    train_cols = [col for col in df.columns if 'Train_Number_All_' in col]  # Route characteristics
    
    instruments = season_cols + supply_instruments + train_cols
    available_instruments = [col for col in instruments if col in df.columns]
    
    # DEMAND SHIFTERS (W): Control variables for both stages (consistent with OLS)
    shifters = ['days_to_departure', 'log_days_to_departure', 'departure_is_weekend', 'Customer_Cat_B']
    
    # Add interaction shifters (same as OLS and 2SLS)
    available_interactions = [col for col in get_interaction_shifters() if col in df.columns]
    
    shifters.extend(available_interactions)
    available_shifters = [col for col in shifters if col in df.columns]
    
    print(f"INSTRUMENTS (first stage only): {len(available_instruments)} variables")
    print(f"  • Seasonal instruments: {len([s for s in season_cols if s in df.columns])} (main IV)")
    print(f"  • Supply-side instruments: {len([i for i in supply_instruments if i in df.columns])}")
    print(f"  • Route instruments: {len([t for t in train_cols if t in df.columns])}")
    print(f"DEMAND SHIFTERS (both stages): {len(available_shifters)} variables")
    print(f"  • Base shifters: {[s for s in ['days_to_departure', 'departure_is_weekend'] if s in df.columns]}")
    print(f"  • Interaction shifters: {available_interactions}")
    
    X_first = df[available_instruments + available_shifters].copy()
    y_first = df['mean_net_ticket_price'].astype(float)
    
    # Convert all columns to numeric, handling booleans properly
    for col in X_first.columns:
        if X_first[col].dtype == 'bool':
            X_first[col] = X_first[col].astype(int)
        elif X_first[col].dtype == 'object':
            X_first[col] = pd.to_numeric(X_first[col], errors='coerce')
        else:
            X_first[col] = X_first[col].astype(float)
    
    # Handle NaN values
    if X_first.isnull().any().any() or y_first.isnull().any():
        print(f"⚠️  Found NaN values in instrument test")
        mask = ~(X_first.isnull().any(axis=1) | y_first.isnull())
        X_first = X_first[mask]
        y_first = y_first[mask]
    
    X_first = sm.add_constant(X_first)
    
    try:
        first_stage = sm.OLS(y_first, X_first).fit()
    except Exception as e:
        print(f"❌ Instrument test failed: {str(e)}")
        return {'f_stat': 0, 'strength': 'failed', 'first_stage': None}
    
    instrument_params = [p for p in available_instruments if p in first_stage.params.index]
    if instrument_params:
        f_stat = first_stage.f_test(instrument_params).fvalue
        f_pvalue = first_stage.f_test(instrument_params).pvalue
    else:
        f_stat, f_pvalue = 0, 1.0
    
    print(f"F-statistic: {f_stat:.2f} (p-value: {f_pvalue:.4e})")
    
    strength = "strong" if f_stat > 10 else ("weak" if f_stat > 3 else "very_weak")
    print(f"Instrument strength: {strength}")
    
    return {'f_stat': f_stat, 'strength': strength, 'first_stage': first_stage}

def estimate_2sls_demand(df, specification='linear'):
    """
    Estimate demand function using 2SLS with proper instrumental variables
    
    2SLS Structure:
    First stage:  price = π₀ + π₁*instruments + π₂*demand_shifters + υ
    Second stage: quantity = β₀ + β₁*predicted_price + β₂*demand_shifters + ε
    
    INSTRUMENTS (Z): departure_season_* (recommended), Culmulative_sales, Train_Number_All_*
    DEMAND SHIFTERS (W): days_to_departure, departure_is_weekend, Customer_Cat_*
    """
    print(f"\n{'='*60}")
    print(f"2SLS DEMAND ESTIMATION - {specification.upper()}")
    print(f"{'='*60}")
    
    # INSTRUMENTS (Z): Variables for identification, based on temporal IV investigationnts
    train_cols = [col for col in df.columns if 'Train_Number_All_' in col]
    instruments =  ['Culmulative_sales'] + train_cols
    available_instruments = [col for col in instruments if col in df.columns]
    
    # DEMAND SHIFTERS (W): Same as OLS (consistent approach)
    season_cols = [col for col in df.columns if 'departure_season_' in col]
    base_shifters = season_cols + ['log_days_to_departure', 'departure_is_weekend'] 
    customer_cols = [col for col in df.columns if 'Customer_Cat_' in col]
    
    # Add interaction shifters for demand heterogeneity (same as OLS)
    available_interactions = [col for col in get_interaction_shifters() if col in df.columns]

    demand_shifters = base_shifters + customer_cols + available_interactions
    available_shifters = [col for col in demand_shifters if col in df.columns]
    
    print(f"Using {len(available_instruments)} instruments: Culmulative_sales + {len(train_cols)} route")
    print(f"Using {len(available_shifters)} demand shifters (consistent with OLS)")
    print(f"  • Base shifters: {base_shifters}")
    print(f"  • Customer segments: {len([c for c in customer_cols if c in df.columns])}")
    print(f"  • Interaction shifters: {available_interactions}")
    
    # Prepare variables based on specification
    if specification == 'linear':
        y = df['num_seats_total'].astype(float)
        endog_var = 'mean_net_ticket_price'
    else:  # loglog
        y = df['log_num_seats_total'].astype(float)
        endog_var = 'log_price'
    
    # Separate instruments and demand shifters (critical for proper 2SLS)
    X_instruments = df[available_instruments].copy()  # Z variables (instruments only)
    X_shifters = df[available_shifters].copy()        # W variables (demand shifters)
    X_endog = df[[endog_var]].copy()                  # Endogenous price variable
    
    # Convert all to numeric
    for df_part in [X_instruments, X_shifters, X_endog]:
        for col in df_part.columns:
            if df_part[col].dtype == 'bool':
                df_part[col] = df_part[col].astype(int)
            elif df_part[col].dtype == 'object':
                df_part[col] = pd.to_numeric(df_part[col], errors='coerce')
            else:
                df_part[col] = df_part[col].astype(float)
    
    # Handle NaN values
    combined_data = pd.concat([y, X_instruments, X_shifters, X_endog], axis=1)
    combined_data = combined_data.dropna()
    
    if len(combined_data) == 0:
        print("❌ No valid data after removing NaN values")
        return None, None
    
    n_instruments = len(available_instruments)
    n_shifters = len(available_shifters)
    
    y_clean = combined_data.iloc[:, 0]
    X_instruments_clean = combined_data.iloc[:, 1:1+n_instruments]
    X_shifters_clean = combined_data.iloc[:, 1+n_instruments:1+n_instruments+n_shifters]
    X_endog_clean = combined_data.iloc[:, -1:]
    
    print(f"Observations: {len(y_clean):,}")
    print(f"Instruments (Z): {list(X_instruments_clean.columns)}")
    print(f"Demand shifters (W): {list(X_shifters_clean.columns)}")
    print(f"Endogenous variable: {endog_var}")
    
    try:
        # Correct 2SLS implementation using statsmodels IV2SLS
        # Structure: IV2SLS(endog=y, exog=exogenous_regressors, instrument=all_instruments)
        
        # Exogenous regressors: demand shifters + endogenous price
        X_regressors = pd.concat([X_shifters_clean, X_endog_clean], axis=1)
        X_regressors = sm.add_constant(X_regressors)
        
        # All instruments: demand shifters + instruments (for excluded endogenous variable)
        X_all_instruments = pd.concat([X_shifters_clean, X_instruments_clean], axis=1)
        X_all_instruments = sm.add_constant(X_all_instruments)
        
        model = IV2SLS(y_clean, X_regressors, X_all_instruments).fit()
        
        print(f"R-squared: {model.rsquared:.4f}")
        
        if endog_var in model.params:
            price_coef = model.params[endog_var]
            price_pvalue = model.pvalues[endog_var]
            
            if specification == 'linear':
                mean_price = df['mean_net_ticket_price'].mean()
                mean_quantity = df['num_seats_total'].mean()
                elasticity = price_coef * (mean_price / mean_quantity)
            else:
                elasticity = price_coef
                
            print(f"\nPrice Elasticity: {elasticity:.4f} (p-value: {price_pvalue:.4e})")
            
            if elasticity < 0:
                print(f"✅ Negative price elasticity")
                print(f"   → {'Inelastic' if abs(elasticity) < 1 else 'Elastic'} demand")
            else:
                print(f"❌ Positive elasticity - potential instrument issues")
                
            return model, elasticity
        else:
            print("❌ Price variable not found in results")
            return None, None
            
    except Exception as e:
        print(f"❌ 2SLS estimation failed: {str(e)}")
        return None, None

def test_endogeneity(ols_results, tsls_results, specification='linear'):
    """Perform Durbin-Wu-Hausman test for endogeneity"""
    print(f"\n{'='*60}")
    print(f"ENDOGENEITY TEST - {specification.upper()}")
    print(f"{'='*60}")
    
    if ols_results[0] is None or tsls_results[0] is None:
        print("❌ Cannot perform endogeneity test - missing estimation results")
        return None
    
    ols_model, ols_elasticity = ols_results
    tsls_model, tsls_elasticity = tsls_results
    
    try:
        # Extract price coefficients
        price_var = 'mean_net_ticket_price' if specification == 'linear' else 'log_price'
        
        if price_var not in ols_model.params or price_var not in tsls_model.params:
            print("❌ Price coefficient not found in both models")
            return None
        
        ols_coef = ols_model.params[price_var]
        tsls_coef = tsls_model.params[price_var]
        
        ols_se = ols_model.bse[price_var]
        tsls_se = tsls_model.bse[price_var]
        
        # Calculate Hausman test statistic
        coef_diff = ols_coef - tsls_coef
        var_diff = tsls_se**2 - ols_se**2
        
        if var_diff <= 0:
            print("⚠️  Variance difference is non-positive - Hausman test not applicable")
            hausman_stat = None
            p_value = None
        else:
            hausman_stat = (coef_diff**2) / var_diff
            # Use chi-square distribution with 1 degree of freedom
            p_value = 1 - stats.chi2.cdf(hausman_stat, df=1)
        
        print(f"\nCoefficient comparison:")
        print(f"  OLS:  {ols_coef:.6f} (SE: {ols_se:.6f})")
        print(f"  2SLS: {tsls_coef:.6f} (SE: {tsls_se:.6f})")
        print(f"  Difference: {coef_diff:.6f}")
        
        if hausman_stat is not None:
            print(f"\nHausman test:")
            print(f"  Test statistic: {hausman_stat:.4f}")
            print(f"  P-value: {p_value:.4f}")
            
            if p_value < 0.05:
                print("🔍 REJECT H0: Endogeneity detected (prefer 2SLS)")
                recommendation = "2SLS"
            else:
                print("✅ FAIL TO REJECT H0: No strong evidence of endogeneity (OLS acceptable)")
                recommendation = "OLS"
        else:
            recommendation = "unclear"
        
        # Economic significance test
        elasticity_diff = abs(ols_elasticity - tsls_elasticity) if ols_elasticity and tsls_elasticity else 0
        print(f"\nElasticity comparison:")
        print(f"  OLS elasticity:  {ols_elasticity:.4f}")
        print(f"  2SLS elasticity: {tsls_elasticity:.4f}")
        print(f"  Absolute difference: {elasticity_diff:.4f}")
        
        if elasticity_diff > 0.2:
            print("⚠️  Large elasticity difference - endogeneity may be economically significant")
        
        return {
            'hausman_stat': hausman_stat,
            'p_value': p_value,
            'recommendation': recommendation,
            'ols_coef': ols_coef,
            'tsls_coef': tsls_coef,
            'elasticity_diff': elasticity_diff
        }
        
    except Exception as e:
        print(f"❌ Endogeneity test failed: {str(e)}")
        return None

def compare_demand_specifications(results):
    """Compare and recommend best demand specification"""
    print(f"\n{'='*80}")
    print("📊 MODEL SPECIFICATION COMPARISON")
    print(f"{'='*80}")
    
    comparison_data = []
    
    # Extract results for comparison
    specifications = ['linear', 'loglog']
    methods = ['ols', '2sls']
    
    for spec in specifications:
        for method in methods:
            if method in results and spec in results[method]:
                model_result, elasticity = results[method][spec]
                if model_result and elasticity:
                    r_squared = getattr(model_result, 'rsquared', 0)
                    comparison_data.append({
                        'specification': spec,
                        'method': method,
                        'elasticity': elasticity,
                        'r_squared': r_squared,
                        'economically_reasonable': -3 < elasticity < 0,
                        'magnitude': abs(elasticity)
                    })
    
    if not comparison_data:
        print("❌ No valid results to compare")
        return None
    
    # Display comparison table
    print("\nModel Comparison:")
    print(f"{'Specification':<12} {'Method':<6} {'Elasticity':<10} {'R²':<8} {'Reasonable':<12}")
    print("-" * 60)
    
    for result in comparison_data:
        reasonable = "✅ Yes" if result['economically_reasonable'] else "❌ No"
        print(f"{result['specification']:<12} {result['method'].upper():<6} "
              f"{result['elasticity']:>8.4f} {result['r_squared']:>6.4f}  {reasonable}")
    
    # Find reasonable estimates
    reasonable_results = [r for r in comparison_data if r['economically_reasonable']]
    
    if not reasonable_results:
        print("\n⚠️  No economically reasonable elasticities found")
        return None
    
    # Rank by closeness to unit elasticity (common economic benchmark)
    reasonable_results.sort(key=lambda x: abs(x['magnitude'] - 1.0))
    
    best_result = reasonable_results[0]
    print(f"\n🏆 RECOMMENDED MODEL:")
    print(f"   Specification: {best_result['specification'].upper()}")
    print(f"   Method: {best_result['method'].upper()}")
    print(f"   Elasticity: {best_result['elasticity']:.4f}")
    print(f"   R-squared: {best_result['r_squared']:.4f}")
    
    elasticity_interp = "Inelastic" if best_result['magnitude'] < 1 else "Elastic"
    print(f"   Demand Type: {elasticity_interp}")
    print(f"   Economic Impact: 1% price ↑ → {best_result['magnitude']*100:.1f}% quantity ↓")
    
    # Check for endogeneity recommendation
    if 'endogeneity_test' in results:
        for spec in specifications:
            if spec in results['endogeneity_test'] and results['endogeneity_test'][spec]:
                test_result = results['endogeneity_test'][spec]
                if test_result['recommendation'] == '2SLS':
                    print(f"\n⚠️  Endogeneity test suggests using 2SLS for {spec} specification")
    
    return best_result

def run_demand_analysis(df):
    """Run complete demand function estimation with dual specifications"""
    print(f"\n{'='*80}")
    print("🎯 COMPREHENSIVE DEMAND FUNCTION ESTIMATION")
    print(f"{'='*80}")
    
    results = {}
    
    # Step 1: OLS Estimation for both specifications
    print("\n🔹 OLS ESTIMATION")
    ols_linear = estimate_ols_demand(df, 'linear')
    ols_loglog = estimate_ols_demand(df, 'loglog')
    
    results['ols'] = {
        'linear': ols_linear,
        'loglog': ols_loglog
    }
    
    # Step 2: Instrument Testing
    print("\n🔹 INSTRUMENT TESTING")
    iv_test = test_instrument_strength(df)
    results['instrument_test'] = iv_test
    
    # Step 3: 2SLS Estimation (if instruments are reasonable)
    if iv_test['f_stat'] > 3:  # Proceed if instruments have some strength
        print("\n🔹 2SLS ESTIMATION")
        tsls_linear = estimate_2sls_demand(df, 'linear')
        tsls_loglog = estimate_2sls_demand(df, 'loglog')
        
        results['2sls'] = {
            'linear': tsls_linear,
            'loglog': tsls_loglog
        }
        
        # Step 4: Endogeneity Testing
        print("\n🔹 ENDOGENEITY TESTING")
        endogeneity_results = {}
        
        if ols_linear[0] and tsls_linear[0]:
            endogeneity_linear = test_endogeneity(ols_linear, tsls_linear, 'linear')
            endogeneity_results['linear'] = endogeneity_linear
        
        if ols_loglog[0] and tsls_loglog[0]:
            endogeneity_loglog = test_endogeneity(ols_loglog, tsls_loglog, 'loglog')
            endogeneity_results['loglog'] = endogeneity_loglog
        
        results['endogeneity_test'] = endogeneity_results
    else:
        print(f"\n⚠️  Instruments are weak (F={iv_test['f_stat']:.2f} < 3)")
        print("   Skipping 2SLS estimation - focus on OLS results")
    
    # Step 5: Model Comparison and Recommendation
    best_specification = compare_demand_specifications(results)
    results['best_specification'] = best_specification
    
    # Step 6: Final Summary
    print(f"\n🏁 DEMAND ANALYSIS SUMMARY:")
    print(f"   Dataset: {len(df):,} observations")
    print(f"   Instrument strength: {iv_test['strength']} (F={iv_test['f_stat']:.2f})")
    
    if results['ols']['linear'][1]:
        print(f"   OLS Linear elasticity: {results['ols']['linear'][1]:.4f}")
    if results['ols']['loglog'][1]:
        print(f"   OLS Log-log elasticity: {results['ols']['loglog'][1]:.4f}")
        
    if '2sls' in results:
        if results['2sls']['linear'][1]:
            print(f"   2SLS Linear elasticity: {results['2sls']['linear'][1]:.4f}")
        if results['2sls']['loglog'][1]:
            print(f"   2SLS Log-log elasticity: {results['2sls']['loglog'][1]:.4f}")
    
    if best_specification:
        print(f"\n🎯 FINAL RECOMMENDATION: {best_specification['specification'].upper()} {best_specification['method'].upper()}")
        print(f"   Price elasticity of demand: {best_specification['elasticity']:.4f}")
    
    return results

def main_analysis(file_path):
    """Main function to run all analyses"""
    print("🚂 ANALYZING CUMULATIVE SALES IN TRAIN DATA 🚂\n")
    
    # Step 1: Load and explore data
    df = load_and_explore_data(file_path)

    data_exploration(df)
    # Step 2-6: Data preparation pipeline (includes feature engineering)
    """Full data preparation pipeline"""
    df = add_features(df)
    df = encode_categorical(df)

    # Step 9: DEMAND FUNCTION ESTIMATION
    demand_results = run_demand_analysis(df)

    return df, demand_results

# Run the analysis
if __name__ == "__main__":
    file_path = "data.csv"
    
    try:
        df, demand_results = main_analysis(file_path)
        print("\n✅ Analysis completed successfully!")
        
        # Save results summary
        print("\n📋 FINAL ANALYSIS RESULTS:")
        print(f"Analysis complete with {len(df):,} observations")
        
        # Demand function results
        if 'ols' in demand_results:
            linear_elasticity = demand_results['ols']['linear'][1] if demand_results['ols']['linear'][1] else "N/A"
            loglog_elasticity = demand_results['ols']['loglog'][1] if demand_results['ols']['loglog'][1] else "N/A"
            print(f"\nDemand Elasticities:")
            print(f"  Linear elasticity: {linear_elasticity}")
            print(f"  Log-log elasticity: {loglog_elasticity}")
            
        if 'instrument_test' in demand_results:
            f_stat = demand_results['instrument_test']['f_stat']
            strength = demand_results['instrument_test']['strength']
            print(f"  Instrument strength: {strength} (F={f_stat:.2f})")
        
    except FileNotFoundError:
        print(f"❌ Error: Could not find file '{file_path}'")
        print("Please update the file_path variable with the correct path to your CSV file.")
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")