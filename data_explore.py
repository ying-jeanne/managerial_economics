from feature_selection import compare_days_to_departure_specifications, get_season, investigate_season_month_instruments
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def get_interaction_shifters():
    # """Define interaction shifters for feature engineering"""
    return ['Business_x_Premium', 'Individual_x_Premium', 'Business_x_NormCabin']
    # return []

def data_exploration(df):
    df = verify_data_quality(df)

    # Step 2.1: Investigate temporal instruments
    investigate_season_month_instruments(df)

    # Step 2.2: Compare booking specifications (linear vs log vs categorical)
    booking_comparison = compare_days_to_departure_specifications(df)

    basic_statistics(df)

    print(f"the comparation result for booking: {booking_comparison}")

    # Step 5: Understand the target variable for demand function estimation
    analyse_right_skew_variables(df)

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
        if df[col].dtype == 'category':
            df[col] = df[col].cat.as_ordered()
        if df[col].dtype != 'bool':
             print(f"range: {df[col].min()} to {df[col].max()}")
       
    return df

def analyse_right_skew_variables(df):
    """Analyze right skew variable for demand function estimation - keep both linear and log specifications"""
    print("\n" + "="*60)
    print("ANALYZING RIGHT SKEW VARIABLE FOR DEMAND ESTIMATION")
    print("="*60)
    
    plt.figure(figsize=(12, 5))

    # Original quantity distribution
    plt.subplot(1, 2, 1)
    sns.histplot(df['num_seats_total'], bins=50, kde=True, color='gray', edgecolor='black', alpha=0.7)
    plt.title(f'Distribution of num_seats_total (Quantity Demanded)', fontsize=12, fontweight='bold')
    plt.xlabel('Number of Seats Sold (Quantity)', fontsize=11)
    plt.ylabel('Frequency', fontsize=11)
    plt.grid(True, alpha=0.3, axis='y')

    # Create log variables for both quantity and price (keep originals too)
    df['log_num_seats_total'] = np.log1p(df['num_seats_total'])  # Keep existing log quantity
    
    # Safe log transformation for price (handle zero/negative values)
    min_price = df['mean_net_ticket_price'].min()
    if min_price <= 0:
        print(f"⚠️  Found zero/negative prices (min: {min_price:.2f}), using log(price + 1)")
        df['log_price'] = np.log1p(df['mean_net_ticket_price'])  # log(price + 1)
    else:
        df['log_price'] = np.log(df['mean_net_ticket_price'])  # standard log(price)
    
    plt.subplot(1, 2, 2)
    sns.histplot(df['mean_net_ticket_price'], bins=50, kde=True, color='darkgray', edgecolor='black', alpha=0.7)
    plt.title('Distribution of mean_net_ticket_price (Price)', fontsize=12, fontweight='bold')
    plt.xlabel('Mean Net Ticket Price', fontsize=11)
    plt.ylabel('Frequency', fontsize=11)
    plt.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('target_analysis.png', dpi=500, bbox_inches='tight', 
                facecolor='white', edgecolor='none')

    # Summary statistics for both specifications
    print(f"\n📊 SUMMARY STATISTICS FOR DEMAND ESTIMATION")
    print(f"Linear Specification:")
    print(f"  Mean Quantity: {df['num_seats_total'].mean():.2f} seats")
    print(f"  Mean Price: ${df['mean_net_ticket_price'].mean():.2f}")
    print(f"  Price-Quantity Correlation: {df['num_seats_total'].corr(df['mean_net_ticket_price']):.3f}")

    print(f"\nLog-Log Specification:")
    print(f"  Mean Log(Quantity): {df['log_num_seats_total'].mean():.3f}")
    print(f"  Mean Log(Price): {df['log_price'].mean():.3f}")
    print(f"  Log Price-Log Quantity Correlation: {df['log_num_seats_total'].corr(df['log_price']):.3f}")
    
    print("✅ Target analysis for both specifications completed!")
    return df

def basic_statistics(df):
    # first I need to reveals a negative correlation between price and quantity demanded, and business travelers are frequently choose premium cabins and exhibit different booking timing compared to leisure travelers. for this I need maybe a heatmap for correlation and a bar chart for categorical variables, and also show days_to_departure is also correlate with customer category 

    # Create days_to_departure if missing
    if 'days_to_departure' not in df.columns:
        df['days_to_departure'] = (df['Dept_Date'] - df['Purchase_Date']).dt.days
    
    #convert customer_cat to boolean since it is category by one hot encoding
    dummies = pd.get_dummies(df['Customer_Cat'], prefix='Customer_Cat', drop_first=True)
    df = pd.concat([df, dummies], axis=1)
    # Correlation heatmap - focus on key variables (compact horizontal layout)
    key_vars = ['mean_net_ticket_price', 'num_seats_total', 'days_to_departure', 'isNormCabin', 'isOneway'] + list(dummies.columns)
    plt.figure(figsize=(24, 4), dpi=500)  # Higher base DPI
    sns.heatmap(df[key_vars].corr(), annot=True, fmt=".2f", cmap="Greys", center=0, square=False, 
                annot_kws={'size': 12}, linewidths=0.5, linecolor='black')  # Simple grayscale with grid lines
    plt.title("Key Variables Correlation", fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    plt.tight_layout()
    plt.savefig('basic_statistic.png', dpi=500, bbox_inches='tight', 
                facecolor='white', edgecolor='none')  # High DPI, clean background
    plt.close()

    """Simple histogram comparison of customer categories"""
    # Separate by customer category
    business = df[df['Customer_Cat'] == 'A']
    leisure = df[df['Customer_Cat'] == 'B']
    
    # Create simple histogram comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Price distribution by customer category
    ax1.hist(business['mean_net_ticket_price'], bins=30, alpha=0.7, 
             label='Business (A)', color='blue', density=True)
    ax1.hist(leisure['mean_net_ticket_price'], bins=30, alpha=0.7, 
             label='Leisure (B)', color='red', density=True)
    ax1.set_xlabel('Price')
    ax1.set_ylabel('Density')
    ax1.set_title('Price Distribution by Customer Category')
    ax1.legend()
    
    # Quantity distribution by customer category
    ax2.hist(business['num_seats_total'], bins=30, alpha=0.7, 
             label='Business (A)', color='blue', density=True)
    ax2.hist(leisure['num_seats_total'], bins=30, alpha=0.7, 
             label='Leisure (B)', color='red', density=True)
    ax2.set_xlabel('Quantity (Seats)')
    ax2.set_ylabel('Density')
    ax2.set_title('Quantity Distribution by Customer Category')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('customer_comparison.png', dpi=400)
    plt.close()
    
    # Print simple statistics
    print(f"\n📊 CUSTOMER CATEGORY COMPARISON:")
    print(f"Business travelers (A): {len(business):,} observations")
    print(f"Leisure travelers (B): {len(leisure):,} observations")
    print(f"Overall price-quantity correlation: {df['mean_net_ticket_price'].corr(df['num_seats_total']):.3f}")
    print("✅ Customer comparison saved as 'customer_comparison.png'")
