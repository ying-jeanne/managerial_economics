
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score

def plot_correlation_heatmap(df, columns=None, filename="correlation_heatmap.png"):
    """
    Plot and save a correlation heatmap for selected columns.
    """
    if columns is not None:
        corr = df[columns].corr()
    else:
        corr = df.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()
    print(f"✅ Correlation heatmap saved as {filename}")

def compare_days_to_departure_specifications(df):
    """Compare linear, quadratic, and business-quantile categorical booking specifications for demand and price prediction"""

    print(f"\n{'='*60}")
    print("BOOKING SPECIFICATION COMPARISON FOR PRICE PREDICTION")
    print("(Testing Instrument Strength)")
    print(f"{'='*60}")

    # Prepare data
    df_temp = df.copy()
    df_temp['days_to_departure'] = (df_temp['Dept_Date'] - df_temp['Purchase_Date']).dt.days
    df_temp = df_temp[df_temp['days_to_departure'] >= 0]
    
    plot_correlation_heatmap(df_temp, 
                             columns=['mean_net_ticket_price', 'num_seats_total', 'days_to_departure'],
                             filename="booking_spec_correlation_heatmap.png")
    # Correlation between price and quantity
    corr_price_qty = df_temp['mean_net_ticket_price'].corr(df_temp['num_seats_total'])

    # Correlation between days to departure and quantity
    corr_days_qty = df_temp['days_to_departure'].corr(df_temp['num_seats_total'])

    print("Correlation (price, quantity):", corr_price_qty)
    print("Correlation (days_to_departure, quantity):", corr_days_qty)

    # Target variables
    y_demand = df_temp['num_seats_total'].values
    y_price = df_temp['mean_net_ticket_price'].values
    days = df_temp['days_to_departure'].values

    # Create 4-category business-quantile variable based on typical booking behavior
    # Quantiles chosen to reflect business travel patterns: last-minute, rush, standard, early planning
    quantiles = df_temp['days_to_departure'].quantile([0.05, 0.20, 0.90])
    bins = [0, quantiles[0.05], quantiles[0.20], quantiles[0.90], df_temp['days_to_departure'].max()]
    df_temp['booking_type'] = pd.cut(df_temp['days_to_departure'], 
                                   bins=bins,
                                   labels=['Last_Minute', 'Rush', 'Standard', 'Early_Planner'],
                                   include_lowest=True)
    booking_dummies = pd.get_dummies(df_temp['booking_type'], drop_first=True)

    # --- DEMAND (num_seats_total) ---
    # Linear
    X_linear = np.column_stack([np.ones(len(days)), days])
    linear_coef = np.linalg.lstsq(X_linear, y_demand, rcond=None)[0]
    linear_pred = X_linear @ linear_coef
    linear_r2 = r2_score(y_demand, linear_pred)
    # Quadratic
    X_quad = np.column_stack([np.ones(len(days)), days, days**2])
    quad_coef = np.linalg.lstsq(X_quad, y_demand, rcond=None)[0]
    quad_pred = X_quad @ quad_coef
    quad_r2 = r2_score(y_demand, quad_pred)
    # Categorical
    X_cat = np.column_stack([np.ones(len(y_demand)), booking_dummies.values])
    cat_coef = np.linalg.lstsq(X_cat, y_demand, rcond=None)[0]
    cat_pred = X_cat @ cat_coef
    cat_r2 = r2_score(y_demand, cat_pred)

    # --- PRICE (mean_net_ticket_price) ---
    # Linear
    X_linear_p = np.column_stack([np.ones(len(days)), days])
    linear_coef_p = np.linalg.lstsq(X_linear_p, y_price, rcond=None)[0]
    linear_pred_p = X_linear_p @ linear_coef_p
    linear_r2_p = r2_score(y_price, linear_pred_p)
    # Quadratic
    X_quad_p = np.column_stack([np.ones(len(days)), days, days**2])
    quad_coef_p = np.linalg.lstsq(X_quad_p, y_price, rcond=None)[0]
    quad_pred_p = X_quad_p @ quad_coef_p
    quad_r2_p = r2_score(y_price, quad_pred_p)
    # Categorical
    X_cat_p = np.column_stack([np.ones(len(y_price)), booking_dummies.values])
    cat_coef_p = np.linalg.lstsq(X_cat_p, y_price, rcond=None)[0]
    cat_pred_p = X_cat_p @ cat_coef_p
    cat_r2_p = r2_score(y_price, cat_pred_p)

    # Print results
    print(f"\n📊 DEMAND PREDICTION (num_seats_total):")
    print(f"  Linear R²:            {linear_r2:.4f}")
    print(f"  Quadratic R²:         {quad_r2:.4f} (+{quad_r2-linear_r2:.4f})")
    print(f"  Business-Quantile R²: {cat_r2:.4f} (+{cat_r2-linear_r2:.4f})")
    best_demand = 'Quadratic' if quad_r2 > max(linear_r2, cat_r2) else 'Business-Quantile' if cat_r2 > linear_r2 else 'Linear'
    print(f"  → Best specification: {best_demand}")

    print(f"\n📊 INSTRUMENT STRENGTH (PRICE PREDICTION):")
    print(f"  Linear R²:            {linear_r2_p:.4f}")
    print(f"  Quadratic R²:         {quad_r2_p:.4f} (+{quad_r2_p-linear_r2_p:.4f})")
    print(f"  Business-Quantile R²: {cat_r2_p:.4f} (+{cat_r2_p-linear_r2_p:.4f})")
    best_price = 'Quadratic' if quad_r2_p > max(linear_r2_p, cat_r2_p) else 'Business-Quantile' if cat_r2_p > linear_r2_p else 'Linear'
    print(f"  → Best specification: {best_price}")
    print(f"  → Quadratic chosen for capturing non-linear booking behavior")

    # --- Plot for demand prediction ---
    plt.figure(figsize=(15, 5))
    plt.suptitle('Model Performance: Different days_to_departure Specifications for Demand', fontsize=14, fontweight='bold')

    # Panel 1: Model Fits Comparison
    plt.subplot(1, 2, 1)

    # Sample data for cleaner visualization (show only 10% of points)
    sample_size = max(1000, len(days) // 10)
    sample_idx = np.random.choice(len(days), size=min(sample_size, len(days)), replace=False)
    
    # Remove outliers for better visualization (keep 98% of data)
    y_demand_p2, y_demand_p98 = np.percentile(y_demand, [2, 98])
    days_p2, days_p98 = np.percentile(days, [2, 98])

    # Plot sampled data points
    plt.scatter(days[sample_idx], y_demand[sample_idx], alpha=0.4, s=8, color='lightgray', label='Sample Data')
    
    # Plot model predictions
    sort_idx = np.argsort(days)
    plt.plot(days[sort_idx], linear_pred[sort_idx], 'b-', linewidth=3, label=f'Linear (R²={linear_r2:.3f})')
    plt.plot(days[sort_idx], quad_pred[sort_idx], 'r-', linewidth=3, label=f'Quadratic (R²={quad_r2:.3f})')
    
    # Set axis limits to focus on main data distribution
    plt.xlim(max(0, days_p2 - 10), min(days_p98 + 10, days.max()))
    plt.ylim(max(0, y_demand_p2 - 2), y_demand_p98 + 5)

    plt.xlabel('Days to Departure')
    plt.ylabel('Seat Quantity')
    plt.title('Demand Model Fits')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)

    # Panel 2: R² Performance Comparison
    plt.subplot(1, 2, 2)
    models = ['Linear', 'Quadratic', 'Business-Quantile']
    r2_values = [linear_r2, quad_r2, cat_r2]
    colors = ['skyblue', 'lightcoral', 'lightgreen']
    best_idx = np.argmax(r2_values)
    best_colors = [c if i != best_idx else 'gold' for i, c in enumerate(colors)]
    bars = plt.bar(models, r2_values, color=best_colors, alpha=0.8, edgecolor='black')
    for bar, r2 in zip(bars, r2_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{r2:.4f}', ha='center', va='bottom', fontweight='bold')
    plt.ylabel('R² Value')
    plt.title('Model Performance Comparison')
    plt.ylim(0, max(r2_values) * 1.15)
    plt.annotate(f'Best: {models[best_idx]}', xy=(best_idx, r2_values[best_idx]), 
                xytext=(best_idx, r2_values[best_idx] + max(r2_values) * 0.08),
                ha='center', fontsize=10, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    plt.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('booking_specifications_comparison.png', dpi=200, bbox_inches='tight')
    print(f"\n📊 Visualization saved as 'booking_specifications_comparison.png'")

    # --- Plot for price prediction (instrument strength) ---
    plt.figure(figsize=(15, 5))
    plt.suptitle('Model Performance: Different days_to_departure Specifications for Price', fontsize=14, fontweight='bold')

    # Panel 1: Model Fits Comparison
    plt.subplot(1, 2, 1)

    # Remove outliers for better visualization (keep 98% of data)
    y_price_p2, y_price_p98 = np.percentile(y_price, [2, 98])

    # Plot sampled data points (use same sample as demand graph for consistency)
    plt.scatter(days[sample_idx], y_price[sample_idx], alpha=0.4, s=8, color='lightgray', label='Sample Data')
    
    # Plot model predictions
    sort_idx = np.argsort(days)
    plt.plot(days[sort_idx], linear_pred_p[sort_idx], 'b-', linewidth=3, label=f'Linear (R²={linear_r2_p:.3f})')
    plt.plot(days[sort_idx], quad_pred_p[sort_idx], 'r-', linewidth=3, label=f'Quadratic (R²={quad_r2_p:.3f})')
    
    # Set axis limits to focus on main data distribution
    plt.xlim(max(0, days_p2 - 10), min(days_p98 + 10, days.max()))
    plt.ylim(max(0, y_price_p2 - 10), y_price_p98 + 20)

    plt.xlabel('Days to Departure')
    plt.ylabel('Ticket Price')
    plt.title('Price Model Fits')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)

    # Panel 2: R² Performance Comparison
    plt.subplot(1, 2, 2)
    models = ['Linear', 'Quadratic', 'Business-Quantile']
    r2_values = [linear_r2_p, quad_r2_p, cat_r2_p]
    colors = ['skyblue', 'lightcoral', 'lightgreen']
    best_idx = np.argmax(r2_values)
    best_colors = [c if i != best_idx else 'gold' for i, c in enumerate(colors)]
    bars = plt.bar(models, r2_values, color=best_colors, alpha=0.8, edgecolor='black')
    for bar, r2 in zip(bars, r2_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{r2:.4f}', ha='center', va='bottom', fontweight='bold')
    plt.ylabel('R² Value')
    plt.title('Model Performance Comparison')
    plt.ylim(0, max(r2_values) * 1.15)
    plt.annotate(f'Best: {models[best_idx]}', xy=(best_idx, r2_values[best_idx]), 
                xytext=(best_idx, r2_values[best_idx] + max(r2_values) * 0.08),
                ha='center', fontsize=10, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    plt.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('booking_specifications_price_comparison.png', dpi=200, bbox_inches='tight')
    print(f"\n📊 Visualization saved as 'booking_specifications_price_comparison.png'")

    # Return results for both demand and price
    return {
        'demand': {
            'linear_r2': linear_r2,
            'quadratic_r2': quad_r2,
            'behavioral_4cat_r2': cat_r2
        },
        'price': {
            'linear_r2': linear_r2_p,
            'quadratic_r2': quad_r2_p,
            'behavioral_4cat_r2': cat_r2_p
        }
    }

# Create season mapping
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

def investigate_season_month_instruments(df):
    """
    Investigate departure_month vs departure_season as instrumental variables.
    Analyzes variation patterns, instrument strength, and correlation structure.
    Returns analysis results dict + saves visualization file.
    """
    print("\n" + "="*80)
    print("📅 TEMPORAL INSTRUMENTAL VARIABLE INVESTIGATION")
    print("="*80)
    
    # use Dept_Date to create departure_month and departure_season
    if 'Dept_Date' not in df.columns:
        print("❌ 'Dept_Date' column not found in DataFrame.")
        return None
    
    df_temp = df.copy()
    
    # Ensure datetime conversion before using .dt accessor
    if not pd.api.types.is_datetime64_any_dtype(df_temp['Dept_Date']):
        df_temp['Dept_Date'] = pd.to_datetime(df_temp['Dept_Date'])
    
    # Create departure_month from date column
    df_temp['departure_month'] = df_temp['Dept_Date'].dt.month

    # Create season variable
    df_temp['departure_season'] = df_temp['departure_month'].apply(get_season)
    
    print("\n1. TEMPORAL VARIABLE DISTRIBUTIONS")
    print("-" * 50)
    
    # Month distribution
    print("Departure Month Distribution:")
    month_counts = df_temp['departure_month'].value_counts().sort_index()
    for month, count in month_counts.items():
        pct = (count / len(df_temp)) * 100
        print(f"  Month {month:2d}: {count:>6,} tickets ({pct:4.1f}%)")
    
    # Season distribution  
    print("\nDeparture Season Distribution:")
    season_counts = df_temp['departure_season'].value_counts()
    for season, count in season_counts.items():
        pct = (count / len(df_temp)) * 100
        print(f"  {season:>6}: {count:>6,} tickets ({pct:4.1f}%)")
    
    print("\n2. PRICE VARIATION ANALYSIS")
    print("-" * 50)
    
    # Price variation by month
    price_by_month = df_temp.groupby('departure_month')['mean_net_ticket_price'].agg(['mean', 'std', 'count'])
    print("Price Statistics by Month:")
    print(f"{'Month':<6} {'Mean Price':<12} {'Std Dev':<10} {'CV':<8} {'Count':<8}")
    print("-" * 50)
    for month in sorted(price_by_month.index):
        mean_p = price_by_month.loc[month, 'mean']
        std_p = price_by_month.loc[month, 'std']
        count_p = price_by_month.loc[month, 'count']
        cv = std_p / mean_p if mean_p > 0 else 0
        print(f"{month:<6} {mean_p:<12.2f} {std_p:<10.2f} {cv:<8.3f} {count_p:<8}")
    
    # Price variation by season
    price_by_season = df_temp.groupby('departure_season')['mean_net_ticket_price'].agg(['mean', 'std', 'count'])
    print("\nPrice Statistics by Season:")
    print(f"{'Season':<8} {'Mean Price':<12} {'Std Dev':<10} {'CV':<8} {'Count':<8}")
    print("-" * 50)
    for season in ['Winter', 'Spring', 'Summer', 'Fall']:
        if season in price_by_season.index:
            mean_p = price_by_season.loc[season, 'mean']
            std_p = price_by_season.loc[season, 'std']
            count_p = price_by_season.loc[season, 'count']
            cv = std_p / mean_p if mean_p > 0 else 0
            print(f"{season:<8} {mean_p:<12.2f} {std_p:<10.2f} {cv:<8.3f} {count_p:<8}")
    
    print("\n3. QUANTITY VARIATION ANALYSIS")

    print("-" * 50)
    
    # Quantity variation by month
    qty_by_month = df_temp.groupby('departure_month')['num_seats_total'].agg(['mean', 'std', 'count'])
    print("Quantity Statistics by Month:")
    print(f"{'Month':<6} {'Mean Qty':<10} {'Std Dev':<10} {'CV':<8}")
    print("-" * 40)
    for month in sorted(qty_by_month.index):
        mean_q = qty_by_month.loc[month, 'mean']
        std_q = qty_by_month.loc[month, 'std']
        cv = std_q / mean_q if mean_q > 0 else 0
        print(f"{month:<6} {mean_q:<10.2f} {std_q:<10.2f} {cv:<8.3f}")
    
    # Quantity variation by season
    qty_by_season = df_temp.groupby('departure_season')['num_seats_total'].agg(['mean', 'std', 'count'])
    print("\nQuantity Statistics by Season:")
    print(f"{'Season':<8} {'Mean Qty':<10} {'Std Dev':<10} {'CV':<8}")
    print("-" * 40)
    for season in ['Winter', 'Spring', 'Summer', 'Fall']:
        if season in qty_by_season.index:
            mean_q = qty_by_season.loc[season, 'mean']
            std_q = qty_by_season.loc[season, 'std']
            cv = std_q / mean_q if mean_q > 0 else 0
            print(f"{season:<8} {mean_q:<10.2f} {std_q:<10.2f} {cv:<8.3f}")
    
    # Create dummy variables for analysis
    month_dummies = pd.get_dummies(df_temp['departure_month'], prefix='month')
    season_dummies = pd.get_dummies(df_temp['departure_season'], prefix='season')
    
    print("\n4. CORRELATION ANALYSIS")
    print("-" * 50)
    
    # Correlations with price
    print("Correlation with Price (mean_net_ticket_price):")
    month_price_corrs = df_temp[['mean_net_ticket_price']].join(month_dummies).corr()['mean_net_ticket_price'][1:]
    season_price_corrs = df_temp[['mean_net_ticket_price']].join(season_dummies).corr()['mean_net_ticket_price'][1:]
    
    print("  Month correlations:")
    for col, corr in month_price_corrs.items():
        print(f"    {col}: {corr:.4f}")
    
    print("  Season correlations:")
    for col, corr in season_price_corrs.items():
        print(f"    {col}: {corr:.4f}")
    
    # Correlations with quantity
    print("\nCorrelation with Quantity (num_seats_total):")
    month_qty_corrs = df_temp[['num_seats_total']].join(month_dummies).corr()['num_seats_total'][1:]
    season_qty_corrs = df_temp[['num_seats_total']].join(season_dummies).corr()['num_seats_total'][1:]
    
    print("  Month correlations:")
    for col, corr in month_qty_corrs.items():
        print(f"    {col}: {corr:.4f}")
    
    print("  Season correlations:")
    for col, corr in season_qty_corrs.items():
        print(f"    {col}: {corr:.4f}")
    
    print("\n5. INSTRUMENTAL VARIABLE STRENGTH TESTING")
    print("-" * 50)
    
    # First-stage regressions
    y_price = df_temp['mean_net_ticket_price'].astype(float)
    
    # Test month dummies as instruments - ensure proper data types
    X_month = sm.add_constant(month_dummies.astype(float))
    model_month = sm.OLS(y_price, X_month).fit()
    f_stat_month = model_month.fvalue
    r2_month = model_month.rsquared
    
    print(f"Month Instruments - First Stage F-stat: {f_stat_month:.2f}, R²: {r2_month:.4f}")
    
    # Test season dummies as instruments - ensure proper data types
    X_season = sm.add_constant(season_dummies.astype(float))
    model_season = sm.OLS(y_price, X_season).fit()
    f_stat_season = model_season.fvalue
    r2_season = model_season.rsquared
    
    print(f"Season Instruments - First Stage F-stat: {f_stat_season:.2f}, R²: {r2_season:.4f}")
    
    print(f"\nInstrument Strength Assessment:")
    month_strength = 'Strong' if f_stat_month > 10 else 'Weak'
    season_strength = 'Strong' if f_stat_season > 10 else 'Weak'
    print(f"  Month dummies: {month_strength} (F > 10 threshold)")
    print(f"  Season dummies: {season_strength} (F > 10 threshold)")
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    fig.suptitle('Temporal Instrumental Variable Analysis', fontsize=16, fontweight='bold')
    
    # Panel 1: Time series trends
    ax1 = axes[0, 0]
    monthly_avg = df_temp.groupby('departure_month').agg({
        'mean_net_ticket_price': 'mean',
        'num_seats_total': 'mean'
    })
    ax1_twin = ax1.twinx()
    
    line1 = ax1.plot(monthly_avg.index, monthly_avg['mean_net_ticket_price'], 'b-o', label='Price')
    line2 = ax1_twin.plot(monthly_avg.index, monthly_avg['num_seats_total'], 'r-s', label='Quantity')
    
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Price', color='b')
    ax1_twin.set_ylabel('Quantity', color='r')
    ax1.set_title('Monthly Price and Quantity Trends')
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Box plots
    ax2 = axes[0, 1]
    season_order = ['Winter', 'Spring', 'Summer', 'Fall']
    df_temp['departure_season'] = pd.Categorical(df_temp['departure_season'], categories=season_order, ordered=True)
    sns.boxplot(data=df_temp, x='departure_season', y='mean_net_ticket_price', ax=ax2)
    ax2.set_title('Price Distribution by Season')
    ax2.set_xlabel('Season')
    ax2.set_ylabel('Price')
    
    # Panel 3: Month Correlation Matrix
    ax3 = axes[0, 2]
    month_corr_data = pd.concat([
        df_temp[['mean_net_ticket_price', 'num_seats_total']],
        month_dummies
    ], axis=1)
    
    month_corr_matrix = month_corr_data.corr()
    # Only show correlations with price and quantity
    month_corr_subset = month_corr_matrix.loc[
        month_corr_matrix.index.str.startswith('month_'),
        ['mean_net_ticket_price', 'num_seats_total']
    ]
    
    sns.heatmap(month_corr_subset, annot=True, cmap='coolwarm', center=0, ax=ax3, 
                fmt='.3f', cbar_kws={'shrink': 0.8})
    ax3.set_title('Month Correlations\n(Price & Quantity)')
    ax3.set_ylabel('Departure Month')
    
    # Panel 4: Season Correlation Matrix
    ax4 = axes[0, 3]
    season_corr_data = pd.concat([
        df_temp[['mean_net_ticket_price', 'num_seats_total']],
        season_dummies
    ], axis=1)
    
    season_corr_matrix = season_corr_data.corr()
    # Only show correlations with price and quantity
    season_corr_subset = season_corr_matrix.loc[
        season_corr_matrix.index.str.startswith('season_'),
        ['mean_net_ticket_price', 'num_seats_total']
    ]
    
    sns.heatmap(season_corr_subset, annot=True, cmap='coolwarm', center=0, ax=ax4, 
                fmt='.3f', cbar_kws={'shrink': 0.8})
    ax4.set_title('Season Correlations\n(Price & Quantity)')
    ax4.set_ylabel('Departure Season')
    
    # Panel 5: First-stage regression comparison
    ax5 = axes[1, 0]
    fitted_month = model_month.fittedvalues
    fitted_season = model_season.fittedvalues
    
    ax5.scatter(fitted_month, y_price, alpha=0.5, label=f'Month (R²={r2_month:.3f})', s=1)
    ax5.scatter(fitted_season, y_price, alpha=0.5, label=f'Season (R²={r2_season:.3f})', s=1)
    ax5.plot([y_price.min(), y_price.max()], [y_price.min(), y_price.max()], 'k--', alpha=0.8)
    ax5.set_xlabel('Fitted Values')
    ax5.set_ylabel('Actual Price')
    ax5.set_title('First-Stage Regression Fit')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Panel 6: Monthly coefficient plots
    ax6 = axes[1, 1]
    month_coefs = model_month.params[1:]  # Exclude constant
    month_ci = model_month.conf_int().iloc[1:]  # Exclude constant
    
    y_pos = np.arange(len(month_coefs))
    ax6.errorbar(month_coefs.values, y_pos, 
                xerr=[month_coefs.values - month_ci.iloc[:, 0].values,
                      month_ci.iloc[:, 1].values - month_coefs.values],
                fmt='o', capsize=5)
    ax6.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    ax6.set_yticks(y_pos)
    ax6.set_yticklabels([f'Month {i+2}' for i in range(len(month_coefs))])  # Starting from month 2 (month 1 is baseline)
    ax6.set_xlabel('Coefficient')
    ax6.set_title('Monthly Dummy Coefficients')
    ax6.grid(True, alpha=0.3)
    
    # Panel 7: Season coefficient plots
    ax7 = axes[1, 2]
    season_coefs = model_season.params[1:]  # Exclude constant
    season_ci = model_season.conf_int().iloc[1:]  # Exclude constant
    
    y_pos_season = np.arange(len(season_coefs))
    ax7.errorbar(season_coefs.values, y_pos_season, 
                xerr=[season_coefs.values - season_ci.iloc[:, 0].values,
                      season_ci.iloc[:, 1].values - season_coefs.values],
                fmt='s', capsize=5, color='orange')
    ax7.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    ax7.set_yticks(y_pos_season)
    ax7.set_yticklabels(season_coefs.index)
    ax7.set_xlabel('Coefficient')
    ax7.set_title('Season Dummy Coefficients')
    ax7.grid(True, alpha=0.3)
    
    # Panel 8: Instrument strength summary
    ax8 = axes[1, 3]
    instruments = ['Month\nDummies', 'Season\nDummies']
    f_stats = [f_stat_month, f_stat_season]
    colors = ['green' if f > 10 else 'red' for f in f_stats]
    
    bars = ax8.bar(instruments, f_stats, color=colors, alpha=0.7)
    ax8.axhline(y=10, color='orange', linestyle='--', linewidth=2, label='F=10 threshold')
    ax8.set_ylabel('F-statistic')
    ax8.set_title('Instrument Strength Comparison')
    ax8.legend()
    
    # Add F-stat values on bars
    for bar, f_val in zip(bars, f_stats):
        height = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{f_val:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('temporal_iv_analysis.png', dpi=200, bbox_inches='tight')
    print(f"\n📊 Visualization saved as 'temporal_iv_analysis.png'")
    
    print("\n6. RECOMMENDATION")
    print("-" * 50)
    
    # Calculate additional metrics
    month_price_cv_range = (price_by_month['std'] / price_by_month['mean']).std()
    season_price_cv_range = (price_by_season['std'] / price_by_season['mean']).std()
    
    print(f"Price variation patterns:")
    print(f"  Month-level CV variation: {month_price_cv_range:.4f}")
    print(f"  Season-level CV variation: {season_price_cv_range:.4f}")
    print(f"  Month dummies provide: {'More' if month_price_cv_range > season_price_cv_range else 'Less'} variation")
    
    # Make recommendation
    if f_stat_month > f_stat_season and f_stat_month > 10:
        recommendation = "departure_month"
        reason = f"stronger first-stage F-statistic ({f_stat_month:.2f} vs {f_stat_season:.2f})"
    elif f_stat_season > 10:
        recommendation = "departure_season"  
        reason = f"adequate first-stage strength ({f_stat_season:.2f}) with better economic interpretation"
    elif f_stat_month > 5 or f_stat_season > 5:
        recommendation = "departure_month" if f_stat_month > f_stat_season else "departure_season"
        reason = f"moderate strength, choose higher F-statistic"
    else:
        recommendation = "Neither - consider alternative instruments"
        reason = "both show weak first-stage relationships (F < 5)"
    
    print(f"\n🎯 RECOMMENDATION: Use {recommendation}")
    print(f"   Reason: {reason}")
    
    # Return comprehensive analysis results
    return {
        'month_f_stat': f_stat_month,
        'season_f_stat': f_stat_season,
        'month_r2': r2_month,
        'season_r2': r2_season,
        'recommendation': recommendation,
        'reason': reason,
        'month_price_corrs': month_price_corrs.to_dict(),
        'season_price_corrs': season_price_corrs.to_dict(),
        'month_qty_corrs': month_qty_corrs.to_dict(),
        'season_qty_corrs': season_qty_corrs.to_dict(),
        'month_strength': month_strength,
        'season_strength': season_strength,
        'price_by_month': price_by_month.to_dict(),
        'price_by_season': price_by_season.to_dict()
    }
