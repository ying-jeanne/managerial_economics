
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
import statsmodels.api as sm

def plot_correlation_heatmap(df, columns=None, filename="correlation_heatmap.png"):
    """
    Plot and save a correlation heatmap for selected columns.
    """
    if columns is not None:
        corr = df[columns].corr()
    else:
        corr = df.corr()
    plt.figure(figsize=(12, 4))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="Greys", center=0, square=False, 
                annot_kws={'size': 12}, linewidths=0.5, linecolor='black')
    plt.title("Correlation Heatmap", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=500, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()

def compare_days_to_departure_specifications(df):
    """Compare linear, log, and business-quantile categorical booking specifications for demand and price prediction"""

    print(f"\n{'='*60}")
    print("BOOKING SPECIFICATION COMPARISON FOR PRICE PREDICTION")
    print("(Testing Instrument Strength)")
    print(f"{'='*60}")

    # Prepare data
    df_temp = df.copy()
    df_temp['days_to_departure'] = (df_temp['Dept_Date'] - df_temp['Purchase_Date']).dt.days
    df_temp = df_temp[df_temp['days_to_departure'] >= 0]
    
    # Create log transformation for days_to_departure
    min_days = df_temp['days_to_departure'].min()
    if min_days <= 0:
        print(f"⚠️  Found same-day bookings (min: {min_days}), using log(days + 1)")
        df_temp['log_days_to_departure'] = np.log1p(df_temp['days_to_departure'])
    else:
        df_temp['log_days_to_departure'] = np.log(df_temp['days_to_departure'])
    
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
    log_days = df_temp['log_days_to_departure'].values

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
    # Log
    X_log = np.column_stack([np.ones(len(log_days)), log_days])
    log_coef = np.linalg.lstsq(X_log, y_demand, rcond=None)[0]
    log_pred = X_log @ log_coef
    log_r2 = r2_score(y_demand, log_pred)
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
    # Log
    X_log_p = np.column_stack([np.ones(len(log_days)), log_days])
    log_coef_p = np.linalg.lstsq(X_log_p, y_price, rcond=None)[0]
    log_pred_p = X_log_p @ log_coef_p
    log_r2_p = r2_score(y_price, log_pred_p)

    # Categorical
    X_cat_p = np.column_stack([np.ones(len(y_price)), booking_dummies.values])
    cat_coef_p = np.linalg.lstsq(X_cat_p, y_price, rcond=None)[0]
    cat_pred_p = X_cat_p @ cat_coef_p
    cat_r2_p = r2_score(y_price, cat_pred_p)

    # Print results
    print(f"\n📊 DEMAND PREDICTION (num_seats_total):")
    print(f"  Linear R²:            {linear_r2:.4f}")
    print(f"  Log R²:               {log_r2:.4f} (+{log_r2-linear_r2:.4f})")
    print(f"  Business-Quantile R²: {cat_r2:.4f} (+{cat_r2-linear_r2:.4f})")
    best_demand = 'Log' if log_r2 > max(linear_r2, cat_r2) else 'Business-Quantile' if cat_r2 > linear_r2 else 'Linear'
    print(f"  → Best specification: {best_demand}")

    print(f"\n📊 INSTRUMENT STRENGTH (PRICE PREDICTION):")
    print(f"  Linear R²:            {linear_r2_p:.4f}")
    print(f"  Log R²:               {log_r2_p:.4f} (+{log_r2_p-linear_r2_p:.4f})")
    print(f"  Business-Quantile R²: {cat_r2_p:.4f} (+{cat_r2_p-linear_r2_p:.4f})")
    best_price = 'Log' if log_r2_p > max(linear_r2_p, cat_r2_p) else 'Business-Quantile' if cat_r2_p > linear_r2_p else 'Linear'
    print(f"  → Best specification: {best_price}")
    print(f"  → Log chosen for capturing diminishing booking horizon effects")

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
    plt.plot(days[sort_idx], linear_pred[sort_idx], color='black', linewidth=3, label=f'Linear (R²={linear_r2:.3f})')
    plt.plot(days[sort_idx], log_pred[sort_idx], color='gray', linewidth=3, linestyle='--', label=f'Log (R²={log_r2:.3f})')
    
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
    models = ['Linear', 'Log', 'Business-Quantile']
    r2_values = [linear_r2, log_r2, cat_r2]
    colors = ['lightgray', 'gray', 'darkgray']
    best_idx = np.argmax(r2_values)
    best_colors = [c if i != best_idx else 'black' for i, c in enumerate(colors)]
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
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    plt.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('booking_specifications_comparison.png', dpi=400, bbox_inches='tight')
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
    plt.plot(days[sort_idx], linear_pred_p[sort_idx], color='black', linewidth=3, label=f'Linear (R²={linear_r2_p:.3f})')
    plt.plot(days[sort_idx], log_pred_p[sort_idx], color='gray', linewidth=3, linestyle='--', label=f'Log (R²={log_r2_p:.3f})')
    
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
    models = ['Linear', 'Log', 'Business-Quantile']
    r2_values = [linear_r2_p, log_r2_p, cat_r2_p]
    colors = ['lightgray', 'gray', 'darkgray']
    best_idx = np.argmax(r2_values)
    best_colors = [c if i != best_idx else 'black' for i, c in enumerate(colors)]
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
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    plt.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('booking_specifications_price_comparison.png', dpi=400, bbox_inches='tight')
    print(f"\n📊 Visualization saved as 'booking_specifications_price_comparison.png'")

    # Return results for both demand and price
    return {
        'demand': {
            'linear_r2': linear_r2,
            'log_r2': log_r2,
            'behavioral_4cat_r2': cat_r2
        },
        'price': {
            'linear_r2': linear_r2_p,
            'log_r2': log_r2_p,
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
    
    df_temp = df.copy()
    
    # Ensure datetime conversion before using .dt accessor
    if not pd.api.types.is_datetime64_any_dtype(df_temp['Dept_Date']):
        df_temp['Dept_Date'] = pd.to_datetime(df_temp['Dept_Date'])
    
    # Create departure_month from date column
    df_temp['departure_month'] = df_temp['Dept_Date'].dt.month

    # Create season variable
    df_temp['departure_season'] = df_temp['departure_month'].apply(get_season)
    
    # Price variation by month
    price_by_month = df_temp.groupby('departure_month')['mean_net_ticket_price'].agg(['mean', 'std', 'count'])
    
    # Price variation by season
    price_by_season = df_temp.groupby('departure_season')['mean_net_ticket_price'].agg(['mean', 'std', 'count'])
    
    # Quantity variation by month
    qty_by_month = df_temp.groupby('departure_month')['num_seats_total'].agg(['mean', 'std', 'count'])
    
    # Quantity variation by season
    qty_by_season = df_temp.groupby('departure_season')['num_seats_total'].agg(['mean', 'std', 'count'])
    
    # Create dummy variables for analysis
    month_dummies = pd.get_dummies(df_temp['departure_month'], prefix='month')
    season_dummies = pd.get_dummies(df_temp['departure_season'], prefix='season')
    
    # Correlations with price
    print("Correlation with Price (mean_net_ticket_price):")
    month_price_corrs = df_temp[['mean_net_ticket_price']].join(month_dummies).corr()['mean_net_ticket_price'][1:]
    season_price_corrs = df_temp[['mean_net_ticket_price']].join(season_dummies).corr()['mean_net_ticket_price'][1:]
    
    # Correlations with quantity
    print("\nCorrelation with Quantity (num_seats_total):")
    month_qty_corrs = df_temp[['num_seats_total']].join(month_dummies).corr()['num_seats_total'][1:]
    season_qty_corrs = df_temp[['num_seats_total']].join(season_dummies).corr()['num_seats_total'][1:]
    
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
    
    # Create simple temporal patterns visualization
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(18, 12), dpi=150)
    fig.suptitle('Temporal Patterns Analysis', fontsize=16, fontweight='bold')
    
    # Panel 1: Seasonal price patterns (simple bar chart)
    seasonal_price = df_temp.groupby('departure_season')['mean_net_ticket_price'].mean()
    seasonal_price.plot(kind='bar', ax=ax1, color='lightgray', edgecolor='black')
    ax1.set_title('Average Price by Season', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Mean Price', fontsize=12)
    ax1.tick_params(axis='x', rotation=0, labelsize=11)
    ax1.tick_params(axis='y', labelsize=11)
    
    # Panel 2: Seasonal demand patterns
    seasonal_demand = df_temp.groupby('departure_season')['num_seats_total'].mean()
    seasonal_demand.plot(kind='bar', ax=ax2, color='lightgray', edgecolor='black')
    ax2.set_title('Average Demand by Season', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Mean Seats Sold', fontsize=12)
    ax2.tick_params(axis='x', rotation=0, labelsize=11)
    ax2.tick_params(axis='y', labelsize=11)
    
    # Panel 3: Monthly price trends (simplified)
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly_price = df_temp.groupby('departure_month')['mean_net_ticket_price'].mean()
    monthly_price.index = [month_names[i-1] for i in monthly_price.index]
    monthly_price.plot(kind='bar', ax=ax3, color='dimgray', edgecolor='black')
    ax3.set_title('Monthly Price Patterns', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Mean Price', fontsize=12)
    ax3.tick_params(axis='x', rotation=45, labelsize=10)
    ax3.tick_params(axis='y', labelsize=11)
    
    # Panel 4: Booking horizon analysis (if days_to_departure exists)
    if 'days_to_departure' not in df_temp.columns:
        if 'Purchase_Date' in df_temp.columns:
            df_temp['days_to_departure'] = (df_temp['Dept_Date'] - pd.to_datetime(df_temp['Purchase_Date'])).dt.days
    
    # Create booking windows
    df_temp['booking_window'] = pd.cut(df_temp['days_to_departure'], 
                                        bins=[0, 7, 30, 60, float('inf')],
                                        labels=['Last-minute (<7)', 'Optimal (7-30)', 'Advanced (30-60)', 'Early (60+)'])

    horizon_demand = df_temp.groupby('booking_window')['num_seats_total'].mean()
    horizon_demand.plot(kind='bar', ax=ax4, color='dimgray', edgecolor='black')
    ax4.set_title('Demand by Booking Window', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Mean Demand', fontsize=12)
    ax4.tick_params(axis='x', rotation=45, labelsize=10)
    ax4.tick_params(axis='y', labelsize=11)
    
    # Panel 5: Booking horizon analysis for price
    horizon_price = df_temp.groupby('booking_window')['mean_net_ticket_price'].mean()
    horizon_price.plot(kind='bar', ax=ax5, color='dimgray', edgecolor='black')
    ax5.set_title('Price by Booking Window', fontsize=14, fontweight='bold')
    ax5.set_ylabel('Mean Price', fontsize=12)
    ax5.tick_params(axis='x', rotation=45, labelsize=10)
    ax5.tick_params(axis='y', labelsize=11)
    
    instruments = ['Monthly\nDummies', 'Seasonal\nDummies']
    f_stats = [f_stat_month, f_stat_season]
    colors = ['lightgray' if f > 10 else 'dimgray' for f in f_stats]
    
    bars = ax6.bar(instruments, f_stats, color=colors, edgecolor='black', linewidth=1.5)
    ax6.axhline(y=10, color='red', linestyle='--', linewidth=2, label='F=10 threshold')
    ax6.set_ylabel('F-statistic', fontsize=14)
    ax6.set_title('Instrument Strength Comparison', fontsize=16, fontweight='bold')
    ax6.legend(fontsize=12)
    ax6.tick_params(axis='x', labelsize=12)
    ax6.tick_params(axis='y', labelsize=12)

    # Add F-stat values on bars
    for bar, f_val in zip(bars, f_stats):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{f_val:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('temporal_iv_analysis.png', dpi=800, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"\n📊 Visualization saved as 'temporal_iv_analysis.png'")
    
    print("\n6. RECOMMENDATION")
    print("-" * 50)
    
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
