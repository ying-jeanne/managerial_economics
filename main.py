import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

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

    return df

def fix_data_type(df):
    """Convert date columns to datetime"""
    print("\n" + "="*60)
    print("CONVERTING DATES and Categorical Variables")
    print("="*60)
    
    df['Dept_Date'] = pd.to_datetime(df['Dept_Date'])
    df['Purchase_Date'] = pd.to_datetime(df['Purchase_Date'])

    # Convert categorical variables
    categorical_cols = ['Train_Number_All', 'Customer_Cat']
    df[categorical_cols] = df[categorical_cols].astype('category')

    # Convert boolean variables
    df['isNormCabin'] = df['isNormCabin'].astype('bool')
    df['isReturn'] = df['isReturn'].astype('bool') 
    df['isOneway'] = df['isOneway'].astype('bool')

    # Make a new boolean column for Customer_Cat == 'A' and drop the original column
    df['isCategoryA'] = (df['Customer_Cat'] == 'A').astype('bool')
    df = df.drop(columns=['Customer_Cat'])

    print("Date conversion completed!")
    print("\nData types after conversion:", df.dtypes)
    return df

def verify_data_quality(df):
    """Check for missing values and duplicates"""
    print("\n" + "="*60)
    print("VERIFYING DATA QUALITY")
    print("="*60)

    # Check for missing values
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("No missing values found!")
    else:
        print("Missing values per column:")
        print(missing[missing > 0])
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    if duplicates == 0:
        print("No duplicate rows found!")
    else:
        print(f"\nTotal duplicate rows: {duplicates}")
        df = df.drop_duplicates()
        print("Duplicates removed.")

    # Unique values per column
    print("\nUnique values per column:")
    for col in df.columns:
        if len(df[col].unique()) < 10:
            print(f"{col}: {df[col].unique()} total {len(df[col].unique())} unique values")
        else:
            print(f"{col}: {len(df[col].unique())} unique values")

    # Data range
    for col in df.columns:
        if df[col].dtype == 'category':
            df[col] = df[col].cat.as_ordered()
        print(f"{col} range: {df[col].min()} to {df[col].max()}")
    return df

def is_isoneday_isreturn_hypothese_true(df):
    print("\n" + "="*60)
    print("ANALYZING isOneway and isReturn COLUMNS")
    print("="*60)

    # Hypothese: isOneway = true means one-way trip, when isOneway = false, means it is a round-trip ticket, then isReturn = true means the return leg of the round-trip, and isReturn = false means the outbound leg of the round-trip. The other case (isOneway = true, isReturn = true) should not exist, if they do exist, we will log it and treat it as isOneway = true

    # Get the percentage of the data are correct according to this hypothesis
    valid_oneway = (df['isOneway'] == 1) & (df['isReturn'] == 0)
    valid_roundtrip_outbound = (df['isOneway'] == 0) & (df['isReturn'] == 0)
    valid_roundtrip_return = (df['isOneway'] == 0) & (df['isReturn'] == 1)
    valid_roundtrip = valid_roundtrip_outbound + valid_roundtrip_return

    # Problematic combinations
    both = (df['isOneway'] == 1) & (df['isReturn'] == 1)     # Both (impossible)
    
    # Calculate counts and percentages
    total = len(df)
    
    results = {
        'Valid - Oneway': valid_oneway.sum()/total,
        'Valid - Round Trip': valid_roundtrip.sum()/total,
        'Problematic - Both': both.sum()/total
    }

    roundtrip_ratio = valid_roundtrip_outbound.sum() / valid_roundtrip_return.sum()
    # Check if outbound and return legs are roughly balanced
    print(f"\nRatio of return/outbound: {roundtrip_ratio:.3f}")
    print("(Should be close to 1.0 if your theory is correct)")

    print(f"\n{(valid_oneway.sum() + valid_roundtrip.sum())/total*100:.2f}% of rows are valid combinations under this new hypothesis")
    if (valid_oneway.sum() + valid_roundtrip.sum()) / total > 0.95 and roundtrip_ratio > 0.95 and roundtrip_ratio < 1.05:
        print("✅ isOneway and isReturn mostly make sense under this new hypothesis")
        print("Hypothesis: isOneway = true means one-way trip, isReturn = true means return trip, and both false means outbound leg of a round-trip is likely correct")
        return True
    print("❌ isOneway and isReturn still don't make sense under this new hypothesis")
    print("Hypothesis might be wrong or data quality issue")
    return False

def test_train_specific_hypothesis(df):
    """Test if cumulative sales resets for each train"""
    print("\n" + "="*60)
    print("TESTING TRAIN-SPECIFIC CUMULATIVE HYPOTHESIS")
    print("="*60)
    
    # Check cumulative sales by train
    train_analysis = df.groupby('Train_Number_All').agg({
        'Culmulative_sales': ['min', 'max', 'count'],
        'Purchase_Date': ['min', 'max']
    }).round(2)
    
    train_analysis.columns = ['Cum_Min', 'Cum_Max', 'Transactions', 'First_Date', 'Last_Date']
    
    print("Cumulative sales by train (first 10 trains):")
    print(train_analysis.head(10))
    
    # Check if different trains have overlapping cumulative values
    trains_starting_low = (train_analysis['Cum_Min'] < 1000).sum()
    total_trains = len(train_analysis)
    
    print(f"\nTrains with minimum cumulative < 1000: {trains_starting_low}/{total_trains}")
    
    if trains_starting_low > 1:
        print("❌ NOT train-specific: Multiple trains have low starting cumulative values")
        print("✅ LIKELY overall cumulative across all trains")
    else:
        print("✅ MIGHT BE train-specific cumulative")
    
    return train_analysis

def analyse_target(df):
    """Understand the business problem and target variable"""
    print("\n" + "="*60)
    print("ANALYZING TARGET VARIABLE")
    print("="*60)
    
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(df['num_seats_total'], bins=50, kde=True)
    plt.title('Distribution of num_seats_total (Ticket Sales)')
    plt.xlabel('Number of Seats Sold')
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    df['num_seats_total'].plot(kind='box')
    plt.title('Target outlier')

    # The 1 percentile and 99 percentile of target
    p1 = df['num_seats_total'].quantile(0.01)
    p99 = df['num_seats_total'].quantile(0.99)
    print(f"1st Percentile of num_seats_total: {p1}")
    print(f"99th Percentile of num_seats_total: {p99}")
    plt.savefig('target_analysis.png', dpi=200)
    print("✅ Target analysis completed!")

def univariate_analysis(df):
    """Perform univariate analysis on all variables"""
    print("\n" + "="*60)
    print("UNIVARIATE ANALYSIS")
    print("="*60)

    cols = list(df.columns)
    nrows = (len(cols) + 1) // 2
    ncols = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 20))
    axes = axes.flatten()
    for i, col in enumerate(cols):
        ax = axes[i]
        if df[col].dtype == 'category' or df[col].dtype == 'bool':
            sns.countplot(y=df[col], order=df[col].value_counts().index, ax=ax)
            ax.set_title(f'Count Plot of {col}')
        elif np.issubdtype(df[col].dtype, np.number):
            sns.histplot(df[col], bins=30, kde=True, ax=ax)
            ax.set_title(f'Distribution of {col}')
        elif np.issubdtype(df[col].dtype, np.datetime64):
            # Group by year and month
            ym = df[col].dt.to_period('M').value_counts().sort_index()
            ym.plot(kind='bar', ax=ax)
            ax.set_title(f'Year-Month Distribution of {col}')
        else:
            ax.text(0.5, 0.5, f'Skipped {col}', ha='center', va='center')
            ax.set_title(f'Skipped {col}')
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')
    # Hide any unused subplots
    for j in range(i+1, nrows*ncols):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.savefig('univariate_analysis.png', dpi=200)
    print("✅ Univariate analysis completed and result is saved in univariate_analysis.png!")

def bivariate_analysis(df):
    """Perform bivariate analysis on all variables"""
    print("\n" + "="*60)
    print("BIVARIATE ANALYSIS")
    print("="*60)

    # Numerical & Boolean vs target
    corr_cols = df.select_dtypes(include=['number', 'bool']).columns.tolist()
    cols = [col for col in corr_cols if col != 'num_seats_total']

    # Create a single figure with 3 subplots: 1 for heatmap, 2 for scatter plots
    fig, axes = plt.subplots(nrows=len(cols)+1, ncols=1, figsize=(16, 5*len(corr_cols)))

    # Correlation heatmap (numerical + bool)
    corr_matrix = df[corr_cols].corr()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=axes[0])
    axes[0].set_title("Correlation Heatmap (Numerical + Bool)")

    # Scatter plots
    for i, col in enumerate(cols):
        sns.scatterplot(x=df[col], y=df['num_seats_total'], ax=axes[i+1])
        axes[i+1].set_title(f'Scatter Plot of {col} vs num_seats_total')
        axes[i+1].set_xlabel(col)
        axes[i+1].set_ylabel('num_seats_total')

    plt.tight_layout()
    plt.savefig('correlation_analysis.png', dpi=200)

    # Categorial vs target
    categorial_cols = [col for col in df.columns if df[col].dtype == 'category' and df[col].nunique() <= 20]  # Only those with <=20 unique values
    if len(categorial_cols) == 1:
        fig, ax = plt.subplots(1, 1, figsize=(16, 5))
        df.groupby(categorial_cols[0])['num_seats_total'].mean().plot(kind='bar', ax=ax)
        ax.set_title(f'Bar Plot of num_seats_total by {categorial_cols[0]}')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        fig.savefig('categorical_correlation_analysis.png', dpi=200)
    elif len(categorial_cols) > 1:
        fig, axes = plt.subplots(len(categorial_cols), 1, figsize=(16, 5*len(categorial_cols)))
        for i, col in enumerate(categorial_cols):
            ax = axes[i]
            df.groupby(col)['num_seats_total'].mean().plot(kind='bar', ax=ax)
            ax.set_title(f'Bar Plot of num_seats_total by {col}')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        fig.savefig('categorical_correlation_analysis.png', dpi=200)
    print("✅ Bivariate analysis completed and saved in correlation_analysis.png and categorical_correlation_analysis.png!")

def feature_engineering(df):
    """Perform feature engineering"""
    print("\n" + "="*60)
    print("FEATURE ENGINEERING: Creating New Features")
    print("="*60)
    
    df_features = df.copy()
    
    # Date-based features
    if 'Purchase_Date' in df.columns:
        df_features['purchase_month'] = df['Purchase_Date'].dt.month
        df_features['purchase_day_of_week'] = df['Purchase_Date'].dt.dayofweek
        df_features['is_weekend'] = (df['Purchase_Date'].dt.dayofweek >= 5).astype(int)
    
    if 'Dept_Date' in df.columns and 'Purchase_Date' in df.columns:
        df_features['days_to_departure'] = (df['Dept_Date'] - df['Purchase_Date']).dt.days
        df_features['departure_month'] = df['Dept_Date'].dt.month
        df_features['departure_day_of_week'] = df['Dept_Date'].dt.dayofweek
        df_features['is_departure_weekend'] = (df['Dept_Date'].dt.dayofweek >= 5).astype(int)

    # Interaction features
    df_features['price_x_advance'] = df['mean_net_ticket_price'] * df_features['days_to_departure']

    # Aggregation features
    # df_features['sales_momentum'] = df['Culmulative_sales'].pct_change().fillna(0)
    
    print(f"Created features. New shape: {df_features.shape}")
    new_features = set(df_features.columns) - set(df.columns)
    print(f"New features: {list(new_features)}")

    return df_features

def clean_isoneway_isreturn_properly(df):
    print("=" * 40)
    print("DATA CLEANING REPORT:")
    print("=" * 40)
    
    # Document the issue
    original_count = len(df)
    problematic = df[(df['isOneway'] == 1) & (df['isReturn'] == 1)]
    
    print(f"Original records: {original_count:,}")
    print(f"Problematic records (isOneway=1, isReturn=1): {len(problematic):,}")
    print(f"Percentage to remove: {len(problematic)/original_count*100:.2f}%")
    
    # Remove problematic records
    df_clean = df[~((df['isOneway'] == 1) & (df['isReturn'] == 1))].copy()
    
    print(f"Cleaned records: {len(df_clean):,}")
    print(f"Records removed: {original_count - len(df_clean):,}")
    
    # Verify the cleaning worked
    remaining_problematic = df_clean[(df_clean['isOneway'] == 1) & (df_clean['isReturn'] == 1)]
    print(f"Remaining problematic records: {len(remaining_problematic)}")
    
    # Show the cleaned distribution
    print(f"\nCleaned data distribution:")
    combo_counts = df_clean.groupby(['isOneway', 'isReturn']).size()
    total_clean = len(df_clean)
    
    for (oneway, return_val), count in combo_counts.items():
        pct = (count/total_clean)*100
        if oneway == 0 and return_val == 0:
            desc = "Round-trip tickets, outbound leg"
        elif oneway == 0 and return_val == 1:
            desc = "Round-trip tickets, return leg"
        else:  # oneway == 1 and return_val == 0
            desc = "One-way tickets"
            
        print(f"  {desc}: {count:,} ({pct:.1f}%)")
    
    return df_clean

def encode_categorical(df, max_categories=20):
    """Encode categorical variables appropriately"""
    
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
            df_encoded[col] = df[col].where(df[col].isin(top_categories), 'Other')
            
            # Then one-hot encode
            dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True)
            df_encoded = pd.concat([df_encoded, dummies], axis=1)
            df_encoded = df_encoded.drop(col, axis=1)
    
    print(f"Final encoded shape: {df_encoded.shape}")
    return df_encoded

def main_analysis(file_path):
    """Main function to run all analyses"""
    print("🚂 ANALYZING CUMULATIVE SALES IN TRAIN DATA 🚂\n")
    
    # Step 1: Load and explore data
    df = load_and_explore_data(file_path)

    # Step 2: Convert dates before verifying data quality, it is just better to get dates right first, then just fix all data types
    df = fix_data_type(df)

    # Step 3: Verify data quality
    df = verify_data_quality(df)

    # Step 4: Verify isOneway and isReturn column business logic, and clean data if needed
    if is_isoneday_isreturn_hypothese_true(df):
        df = clean_isoneway_isreturn_properly(df)
    else:
        print("❌ isOneway and isReturn columns do not make sense under the new hypothesis. Please investigate further.")
        return

    # Step 5: Understand the business problem, for getting the demande function, the target is num_seats_total
    analyse_target(df)

    # Step 6: Univariate analysis of all variables
    univariate_analysis(df)

    # Step 7: Bivariate analysis of all variables
    bivariate_analysis(df)


    # Step 8: Feature engineering, create new features if needed
    df = feature_engineering(df)

    # Step 9: Encode categorical variables
    df = encode_categorical(df)

    print(df.describe())
    return

# Run the analysis
if __name__ == "__main__":
    file_path = "data.csv"
    
    try:
        df, daily_sales, train_analysis = main_analysis(file_path)
        print("\n✅ Analysis completed successfully!")
        
        # Optional: Save results
        # daily_sales.to_csv("daily_sales_analysis.csv", index=False)
        # print("📁 Daily analysis saved to 'daily_sales_analysis.csv'")
        
    except FileNotFoundError:
        print(f"❌ Error: Could not find file '{file_path}'")
        print("Please update the file_path variable with the correct path to your CSV file.")
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")