"""
Marketing Mix Model for Insure Co.
==================================
Estimates response curves for each marketing channel to understand
the relationship between spend and profit, including diminishing returns.

Output: Response curves, marginal ROI, saturation analysis, optimal allocation
"""

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit, minimize
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

# Directory where this Python file lives
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / 'insure_co_data'
OUTPUT_DIR = BASE_DIR / 'model_outputs'

# Email cost per lead (from generator config)
EMAIL_CPL = 8.0

# =============================================================================
# RESPONSE CURVE FUNCTIONS
# =============================================================================

def hill_function(spend, K, S, beta):
    """
    Hill saturation function (diminishing returns curve).
    
    Parameters:
    - K: Maximum response (saturation level)
    - S: Half-saturation point (spend at which response = K/2)
    - beta: Shape parameter (steepness of curve)
    
    Returns: Expected response (profit) at given spend level
    """
    return K * (spend ** beta) / (S ** beta + spend ** beta)


def hill_derivative(spend, K, S, beta):
    """
    Derivative of Hill function = marginal response (marginal ROI).
    """
    numerator = K * beta * (spend ** (beta - 1)) * (S ** beta)
    denominator = (S ** beta + spend ** beta) ** 2
    return numerator / denominator


def log_function(spend, a, b):
    """
    Log response function (simpler diminishing returns).
    response = a * log(1 + spend/b)
    """
    return a * np.log(1 + spend / b)


def power_function(spend, a, b):
    """
    Power response function.
    response = a * spend^b (where b < 1 for diminishing returns)
    """
    return a * np.power(spend + 1, b)


# =============================================================================
# DATA LOADING AND PREPARATION
# =============================================================================

def load_and_prepare_data(data_dir):
    """
    Load all data files and prepare weekly aggregated dataset.
    """
    print("=" * 70)
    print("LOADING AND PREPARING DATA")
    print("=" * 70)
    
    # Load data
    leads = pd.read_csv(data_dir / 'leads.csv')
    search_spend = pd.read_csv(data_dir / 'search_daily_spend.csv')
    social_spend = pd.read_csv(data_dir / 'social_daily_spend.csv')
    
    # Convert dates
    leads['lead_date'] = pd.to_datetime(leads['lead_date'])
    leads['sold_date'] = pd.to_datetime(leads['sold_date'])
    search_spend['date'] = pd.to_datetime(search_spend['date'])
    social_spend['date'] = pd.to_datetime(social_spend['date'])
    
    # Calculate profit per sold policy (premium - claims)
    leads['policy_profit'] = leads['total_premium'] - leads['total_claim_amount']
    
    # Add week identifier
    leads['week'] = leads['lead_date'].dt.to_period('W').dt.start_time
    search_spend['week'] = search_spend['date'].dt.to_period('W').dt.start_time
    social_spend['week'] = social_spend['date'].dt.to_period('W').dt.start_time
    
    # Aggregate spend by week
    search_weekly = search_spend.groupby('week').agg({
        'spend': 'sum',
        'leads': 'sum'
    }).reset_index()
    search_weekly.columns = ['week', 'search_spend', 'search_leads']
    
    social_weekly = social_spend.groupby('week').agg({
        'spend': 'sum',
        'leads': 'sum'
    }).reset_index()
    social_weekly.columns = ['week', 'social_spend', 'social_leads']
    
    # Aggregate leads/profit by channel and week
    # We need to track profit from policies that were SOLD, attributed to lead date
    sold_policies = leads[leads['sold_date'].notna()].copy()
    
    channel_weekly_profit = sold_policies.groupby(['week', 'channel']).agg({
        'policy_profit': 'sum',
        'lead_id': 'nunique'
    }).reset_index()
    channel_weekly_profit.columns = ['week', 'channel', 'profit', 'policies_sold']
    
    # Pivot to wide format
    profit_wide = channel_weekly_profit.pivot(index='week', columns='channel', values='profit').reset_index()
    profit_wide.columns = ['week', 'email_profit', 'search_profit', 'social_profit']
    profit_wide = profit_wide.fillna(0)
    
    policies_wide = channel_weekly_profit.pivot(index='week', columns='channel', values='policies_sold').reset_index()
    policies_wide.columns = ['week', 'email_policies', 'search_policies', 'social_policies']
    policies_wide = policies_wide.fillna(0)
    
    # Calculate email spend (leads generated * CPL)
    email_leads_weekly = leads[leads['channel'] == 'email'].groupby('week').agg({
        'lead_id': 'nunique'
    }).reset_index()
    email_leads_weekly.columns = ['week', 'email_leads']
    email_leads_weekly['email_spend'] = email_leads_weekly['email_leads'] * EMAIL_CPL
    
    # Merge everything
    weekly_data = search_weekly.merge(social_weekly, on='week', how='outer')
    weekly_data = weekly_data.merge(email_leads_weekly[['week', 'email_spend', 'email_leads']], on='week', how='outer')
    weekly_data = weekly_data.merge(profit_wide, on='week', how='outer')
    weekly_data = weekly_data.merge(policies_wide, on='week', how='outer')
    weekly_data = weekly_data.fillna(0)
    
    # Sort by week
    weekly_data = weekly_data.sort_values('week').reset_index(drop=True)
    
    print(f"\nWeekly data prepared: {len(weekly_data)} weeks")
    print(f"Date range: {weekly_data['week'].min()} to {weekly_data['week'].max()}")
    
    print("\nWeekly averages:")
    for channel in ['search', 'social', 'email']:
        avg_spend = weekly_data[f'{channel}_spend'].mean()
        avg_profit = weekly_data[f'{channel}_profit'].mean()
        print(f"  {channel}: ${avg_spend:,.0f} spend -> ${avg_profit:,.0f} profit")
    
    return weekly_data, leads


# =============================================================================
# MODEL FITTING
# =============================================================================

def fit_response_curves(weekly_data):
    """
    Fit Hill saturation curves to each channel's spend-profit relationship.
    """
    print("\n" + "=" * 70)
    print("FITTING RESPONSE CURVES")
    print("=" * 70)
    
    channels = ['search', 'social', 'email']
    fitted_params = {}
    fit_stats = {}
    
    for channel in channels:
        spend = weekly_data[f'{channel}_spend'].values
        profit = weekly_data[f'{channel}_profit'].values
        
        # Remove zeros for fitting (log issues)
        mask = (spend > 0) & (profit > 0)
        spend_fit = spend[mask]
        profit_fit = profit[mask]
        
        if len(spend_fit) < 10:
            print(f"\n{channel.upper()}: Insufficient data points")
            continue
        
        # Initial parameter guesses for Hill function
        K_init = profit_fit.max() * 2  # Max response
        S_init = np.median(spend_fit)   # Half-saturation at median spend
        beta_init = 0.8                 # Diminishing returns shape
        
        try:
            # Fit Hill function
            popt_hill, pcov_hill = curve_fit(
                hill_function,
                spend_fit,
                profit_fit,
                p0=[K_init, S_init, beta_init],
                bounds=([0, 0, 0.1], [np.inf, np.inf, 3.0]),
                maxfev=10000
            )
            
            # Calculate R-squared
            pred_hill = hill_function(spend_fit, *popt_hill)
            ss_res = np.sum((profit_fit - pred_hill) ** 2)
            ss_tot = np.sum((profit_fit - profit_fit.mean()) ** 2)
            r2_hill = 1 - (ss_res / ss_tot)
            
            fitted_params[channel] = {
                'K': popt_hill[0],      # Saturation level
                'S': popt_hill[1],      # Half-saturation point
                'beta': popt_hill[2],   # Shape parameter
                'model': 'hill'
            }
            
            fit_stats[channel] = {
                'r2': r2_hill,
                'n_obs': len(spend_fit),
                'spend_range': (spend_fit.min(), spend_fit.max()),
                'profit_range': (profit_fit.min(), profit_fit.max())
            }
            
            print(f"\n{channel.upper()} - Hill Function Fit:")
            print(f"  K (saturation):     ${popt_hill[0]:,.0f} max weekly profit")
            print(f"  S (half-sat point): ${popt_hill[1]:,.0f} spend for 50% of max")
            print(f"  beta (shape):       {popt_hill[2]:.3f}")
            print(f"  R-squared:          {r2_hill:.3f}")
            
        except Exception as e:
            print(f"\n{channel.upper()}: Fitting failed - {e}")
            
            # Fallback to simpler log model
            try:
                popt_log, _ = curve_fit(
                    log_function,
                    spend_fit,
                    profit_fit,
                    p0=[profit_fit.max(), np.median(spend_fit)],
                    maxfev=10000
                )
                
                pred_log = log_function(spend_fit, *popt_log)
                ss_res = np.sum((profit_fit - pred_log) ** 2)
                ss_tot = np.sum((profit_fit - profit_fit.mean()) ** 2)
                r2_log = 1 - (ss_res / ss_tot)
                
                fitted_params[channel] = {
                    'a': popt_log[0],
                    'b': popt_log[1],
                    'model': 'log'
                }
                fit_stats[channel] = {'r2': r2_log, 'n_obs': len(spend_fit)}
                
                print(f"  Fallback to log model, R-squared = {r2_log:.3f}")
                
            except Exception as e2:
                print(f"  Log model also failed: {e2}")
    
    return fitted_params, fit_stats


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def calculate_marginal_roi(fitted_params, spend_levels):
    """
    Calculate marginal ROI (derivative of response curve) at various spend levels.
    """
    results = {}
    
    for channel, params in fitted_params.items():
        if params['model'] == 'hill':
            K, S, beta = params['K'], params['S'], params['beta']
            marginal = hill_derivative(spend_levels, K, S, beta)
            results[channel] = marginal
        elif params['model'] == 'log':
            a, b = params['a'], params['b']
            marginal = a / (b + spend_levels)
            results[channel] = marginal
    
    return results


def find_optimal_allocation(fitted_params, total_budget, min_spend=0, max_iterations=1000):
    """
    Find optimal budget allocation across channels to maximize total profit.
    Uses equal marginal ROI principle.
    """
    channels = list(fitted_params.keys())
    n_channels = len(channels)
    
    def total_profit(allocation):
        """Negative total profit (for minimization)."""
        profit = 0
        for i, channel in enumerate(channels):
            params = fitted_params[channel]
            spend = allocation[i]
            if params['model'] == 'hill':
                profit += hill_function(spend, params['K'], params['S'], params['beta'])
            elif params['model'] == 'log':
                profit += log_function(spend, params['a'], params['b'])
        return -profit  # Negative for minimization
    
    # Constraint: total spend = budget
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - total_budget}
    
    # Bounds: non-negative spend
    bounds = [(min_spend, total_budget) for _ in channels]
    
    # Initial guess: equal split
    x0 = np.array([total_budget / n_channels] * n_channels)
    
    # Optimize
    result = minimize(
        total_profit,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': max_iterations}
    )
    
    optimal_allocation = {channel: result.x[i] for i, channel in enumerate(channels)}
    optimal_profit = -result.fun
    
    return optimal_allocation, optimal_profit


def calculate_saturation_metrics(fitted_params):
    """
    Calculate saturation-related metrics for each channel.
    """
    metrics = {}
    
    for channel, params in fitted_params.items():
        if params['model'] == 'hill':
            K, S, beta = params['K'], params['S'], params['beta']
            
            # Spend level for 50% of max response
            spend_50 = S
            
            # Spend level for 80% of max response
            # 0.8 = x^beta / (S^beta + x^beta) -> x = S * (0.8/0.2)^(1/beta)
            spend_80 = S * (0.8 / 0.2) ** (1 / beta)
            
            # Spend level for 90% of max response
            spend_90 = S * (0.9 / 0.1) ** (1 / beta)
            
            # Marginal ROI at current average spend (approx)
            metrics[channel] = {
                'max_weekly_profit': K,
                'spend_for_50pct': spend_50,
                'spend_for_80pct': spend_80,
                'spend_for_90pct': spend_90,
            }
    
    return metrics


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_response_curves(weekly_data, fitted_params, fit_stats, output_dir):
    """
    Create visualization of response curves for all channels.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    channels = ['search', 'social', 'email']
    colors = {'search': '#2E86AB', 'social': '#A23B72', 'email': '#F18F01'}
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: Response curves with data points
    ax1 = axes[0, 0]
    for channel in channels:
        if channel not in fitted_params:
            continue
            
        spend = weekly_data[f'{channel}_spend'].values
        profit = weekly_data[f'{channel}_profit'].values
        
        # Scatter plot of actual data
        ax1.scatter(spend, profit, alpha=0.3, s=20, color=colors[channel], label=f'{channel} (actual)')
        
        # Fitted curve
        params = fitted_params[channel]
        spend_range = np.linspace(0, spend.max() * 1.2, 200)
        
        if params['model'] == 'hill':
            profit_pred = hill_function(spend_range, params['K'], params['S'], params['beta'])
        else:
            profit_pred = log_function(spend_range, params['a'], params['b'])
        
        r2 = fit_stats[channel]['r2']
        ax1.plot(spend_range, profit_pred, color=colors[channel], linewidth=2, 
                 label=f'{channel} fit (R2={r2:.2f})')
    
    ax1.set_xlabel('Weekly Spend ($)', fontsize=11)
    ax1.set_ylabel('Weekly Profit ($)', fontsize=11)
    ax1.set_title('Response Curves: Spend vs Profit by Channel', fontsize=12, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, None)
    ax1.set_ylim(0, None)
    
    # Plot 2: Marginal ROI curves
    ax2 = axes[0, 1]
    max_spend = max(weekly_data[f'{ch}_spend'].max() for ch in channels) * 1.2
    spend_range = np.linspace(100, max_spend, 200)
    
    marginal_rois = calculate_marginal_roi(fitted_params, spend_range)
    
    for channel in channels:
        if channel in marginal_rois:
            ax2.plot(spend_range, marginal_rois[channel], color=colors[channel], 
                     linewidth=2, label=channel)
    
    ax2.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Break-even (ROI=1)')
    ax2.set_xlabel('Weekly Spend ($)', fontsize=11)
    ax2.set_ylabel('Marginal ROI ($ profit per $ spent)', fontsize=11)
    ax2.set_title('Marginal ROI: Diminishing Returns by Channel', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, None)
    
    # Plot 3: ROI comparison at different spend levels
    ax3 = axes[1, 0]
    spend_points = [500, 1000, 2000, 3000, 4000, 5000]
    
    x_pos = np.arange(len(spend_points))
    width = 0.25
    
    for i, channel in enumerate(channels):
        if channel in marginal_rois:
            params = fitted_params[channel]
            rois = []
            for sp in spend_points:
                if params['model'] == 'hill':
                    roi = hill_derivative(sp, params['K'], params['S'], params['beta'])
                else:
                    roi = params['a'] / (params['b'] + sp)
                rois.append(roi)
            ax3.bar(x_pos + i * width, rois, width, label=channel, color=colors[channel])
    
    ax3.set_xlabel('Weekly Spend Level ($)', fontsize=11)
    ax3.set_ylabel('Marginal ROI', fontsize=11)
    ax3.set_title('Marginal ROI at Different Spend Levels', fontsize=12, fontweight='bold')
    ax3.set_xticks(x_pos + width)
    ax3.set_xticklabels([f'${s:,}' for s in spend_points])
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
    
    # Plot 4: Current vs Optimal allocation
    ax4 = axes[1, 1]
    
    # Calculate current allocation
    current_spend = {ch: weekly_data[f'{ch}_spend'].mean() for ch in channels}
    total_current = sum(current_spend.values())
    
    # Calculate optimal allocation
    optimal_alloc, optimal_profit = find_optimal_allocation(fitted_params, total_current)
    
    x_pos = np.arange(len(channels))
    width = 0.35
    
    current_vals = [current_spend[ch] for ch in channels]
    optimal_vals = [optimal_alloc.get(ch, 0) for ch in channels]
    
    ax4.bar(x_pos - width/2, current_vals, width, label='Current', color='gray', alpha=0.7)
    ax4.bar(x_pos + width/2, optimal_vals, width, label='Optimal', color='green', alpha=0.7)
    
    ax4.set_xlabel('Channel', fontsize=11)
    ax4.set_ylabel('Weekly Spend ($)', fontsize=11)
    ax4.set_title(f'Current vs Optimal Budget Allocation\n(Total: ${total_current:,.0f}/week)', 
                  fontsize=12, fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([ch.title() for ch in channels])
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'response_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved: {output_dir}/response_curves.png")
    
    return fig


def generate_summary_report(weekly_data, fitted_params, fit_stats, output_dir):
    """
    Generate a text summary report of the model results.
    """
    channels = ['search', 'social', 'email']
    
    # Calculate current metrics
    current_spend = {ch: weekly_data[f'{ch}_spend'].sum() for ch in channels}
    current_profit = {ch: weekly_data[f'{ch}_profit'].sum() for ch in channels}
    total_spend = sum(current_spend.values())
    total_profit = sum(current_profit.values())
    
    # Calculate optimal allocation
    weekly_budget = total_spend / len(weekly_data)
    optimal_alloc, optimal_weekly_profit = find_optimal_allocation(fitted_params, weekly_budget)
    
    # Calculate current weekly profit
    current_weekly_profit = 0
    for ch in channels:
        if ch in fitted_params:
            params = fitted_params[ch]
            spend = current_spend[ch] / len(weekly_data)
            if params['model'] == 'hill':
                current_weekly_profit += hill_function(spend, params['K'], params['S'], params['beta'])
    
    # Saturation metrics
    sat_metrics = calculate_saturation_metrics(fitted_params)
    
    report = []
    report.append("=" * 70)
    report.append("MARKETING MIX MODEL - SUMMARY REPORT")
    report.append("=" * 70)
    
    report.append("\n1. MODEL FIT SUMMARY")
    report.append("-" * 50)
    for channel in channels:
        if channel not in fitted_params:
            continue
        params = fitted_params[channel]
        stats = fit_stats[channel]
        
        report.append(f"\n{channel.upper()}")
        if params['model'] == 'hill':
            report.append(f"  Model: Hill Saturation Function")
            report.append(f"  Parameters:")
            report.append(f"    K (max response):     ${params['K']:>12,.0f}/week")
            report.append(f"    S (half-saturation):  ${params['S']:>12,.0f}/week spend")
            report.append(f"    beta (shape):         {params['beta']:>12.3f}")
        report.append(f"  R-squared: {stats['r2']:.3f}")
        report.append(f"  Observations: {stats['n_obs']}")
    
    report.append("\n\n2. CURRENT PERFORMANCE")
    report.append("-" * 50)
    report.append(f"{'Channel':<12} {'Total Spend':>14} {'Total Profit':>14} {'ROI':>10}")
    report.append("-" * 50)
    for ch in channels:
        roi = current_profit[ch] / current_spend[ch] if current_spend[ch] > 0 else 0
        report.append(f"{ch:<12} ${current_spend[ch]:>12,.0f} ${current_profit[ch]:>12,.0f} {roi:>9.1f}x")
    report.append("-" * 50)
    overall_roi = total_profit / total_spend
    report.append(f"{'TOTAL':<12} ${total_spend:>12,.0f} ${total_profit:>12,.0f} {overall_roi:>9.1f}x")
    
    report.append("\n\n3. SATURATION ANALYSIS")
    report.append("-" * 50)
    for ch in channels:
        if ch in sat_metrics:
            m = sat_metrics[ch]
            report.append(f"\n{ch.upper()}")
            report.append(f"  Maximum achievable profit: ${m['max_weekly_profit']:,.0f}/week")
            report.append(f"  Spend for 50% of max:      ${m['spend_for_50pct']:,.0f}/week")
            report.append(f"  Spend for 80% of max:      ${m['spend_for_80pct']:,.0f}/week")
            report.append(f"  Spend for 90% of max:      ${m['spend_for_90pct']:,.0f}/week")
    
    report.append("\n\n4. OPTIMAL BUDGET ALLOCATION")
    report.append("-" * 50)
    report.append(f"Total weekly budget: ${weekly_budget:,.0f}")
    report.append("")
    report.append(f"{'Channel':<12} {'Current':>12} {'Optimal':>12} {'Change':>12}")
    report.append("-" * 50)
    
    for ch in channels:
        current = current_spend[ch] / len(weekly_data)
        optimal = optimal_alloc.get(ch, 0)
        change = optimal - current
        change_str = f"+${change:,.0f}" if change >= 0 else f"-${abs(change):,.0f}"
        report.append(f"{ch:<12} ${current:>10,.0f} ${optimal:>10,.0f} {change_str:>12}")
    
    report.append("")
    report.append(f"Projected profit improvement: ${(optimal_weekly_profit - current_weekly_profit):,.0f}/week")
    report.append(f"                              ${(optimal_weekly_profit - current_weekly_profit) * 52:,.0f}/year")
    
    report.append("\n\n5. MARGINAL ROI AT CURRENT SPEND")
    report.append("-" * 50)
    report.append("(How much profit from the next $1,000 in each channel)")
    report.append("")
    
    for ch in channels:
        if ch in fitted_params:
            params = fitted_params[ch]
            current = current_spend[ch] / len(weekly_data)
            if params['model'] == 'hill':
                marginal = hill_derivative(current, params['K'], params['S'], params['beta'])
            else:
                marginal = params['a'] / (params['b'] + current)
            report.append(f"  {ch}: ${marginal * 1000:,.0f} profit per $1,000 additional spend")
    
    report.append("\n" + "=" * 70)
    
    # Write report
    report_text = "\n".join(report)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'model_report.txt', 'w') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\nSaved: {output_dir}/model_report.txt")
    
    return report_text


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Run the complete marketing mix model pipeline.
    """
    # Load and prepare data
    weekly_data, leads = load_and_prepare_data(DATA_DIR)
    
    # Fit response curves
    fitted_params, fit_stats = fit_response_curves(weekly_data)
    
    # Generate visualizations
    plot_response_curves(weekly_data, fitted_params, fit_stats, OUTPUT_DIR)
    
    # Generate summary report
    generate_summary_report(weekly_data, fitted_params, fit_stats, OUTPUT_DIR)
    
    # Save fitted parameters for later use
    import json
    
    params_export = {}
    for ch, params in fitted_params.items():
        params_export[ch] = {k: float(v) if isinstance(v, (np.floating, float)) else v 
                            for k, v in params.items()}
    
    with open(OUTPUT_DIR / 'fitted_parameters.json', 'w') as f:
        json.dump(params_export, f, indent=2)
    
    print(f"\nSaved: {OUTPUT_DIR}/fitted_parameters.json")
    
    return weekly_data, fitted_params, fit_stats


if __name__ == "__main__":
    weekly_data, fitted_params, fit_stats = main()
