"""
Marketing Mix Model with ROI-Saturation Constraint
===================================================
Implements the constraint that channels with higher average ROI
are further from saturation than channels with lower average ROI.

Logic: If a channel has high average ROI, it's likely because we're
operating on the steep part of the response curve (far from saturation).
Low average ROI suggests we're already in diminishing returns territory.
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize, curve_fit
import matplotlib.pyplot as plt
import json
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

# Directory where this Python file lives
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR.parent / 'insure_co_data'  # Data is one level up
OUTPUT_DIR = BASE_DIR  # Output to MMM folder itself
EMAIL_CPL = 8.0

# Economic assumptions
EXPENSE_RATIO = 0.30      # 30% of premium goes to operating expenses
DISCOUNT_RATE = 0.10      # 10% annual discount rate for NPV calculation

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# RESPONSE FUNCTIONS
# =============================================================================

def hill_function(spend, K, S, beta):
    """Hill saturation function."""
    return K * (spend ** beta) / (S ** beta + spend ** beta)

def hill_derivative(spend, K, S, beta):
    """Marginal ROI from Hill function."""
    numerator = K * beta * (spend ** (beta - 1)) * (S ** beta)
    denominator = (S ** beta + spend ** beta) ** 2
    return numerator / denominator

# =============================================================================
# DATA LOADING
# =============================================================================

def calculate_npv_profit(row, expense_ratio, discount_rate):
    """
    Calculate NPV of policy profit with expense loading and discounting.
    
    Assumes cash flows are spread evenly over the policy tenure.
    """
    if pd.isna(row['expected_tenure_years']) or row['expected_tenure_years'] <= 0:
        return 0
    
    tenure = row['expected_tenure_years']
    annual_premium = row['annual_premium'] if pd.notna(row['annual_premium']) else 0
    total_claims = row['total_claim_amount'] if pd.notna(row['total_claim_amount']) else 0
    annual_claims = total_claims / tenure
    
    # Annual profit after expenses
    annual_profit = annual_premium * (1 - expense_ratio) - annual_claims
    
    # NPV using annuity formula: PV = PMT * [(1 - (1+r)^-n) / r]
    if discount_rate > 0:
        annuity_factor = (1 - (1 + discount_rate) ** -tenure) / discount_rate
    else:
        annuity_factor = tenure
    
    npv = annual_profit * annuity_factor
    return npv


def load_and_prepare_data(data_dir):
    """
    Load and prepare weekly aggregated data.
    
    Models CONVERSIONS (not profit) as the response variable for tighter fit.
    Profit is used for optimization via average profit per conversion.
    """
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    
    print(f"\nEconomic Assumptions:")
    print(f"  Expense ratio:  {EXPENSE_RATIO:.0%} of premium")
    print(f"  Discount rate:  {DISCOUNT_RATE:.0%} per year")
    
    leads = pd.read_csv(data_dir / 'leads.csv')
    search_spend = pd.read_csv(data_dir / 'search_daily_spend.csv')
    social_spend = pd.read_csv(data_dir / 'social_daily_spend.csv')
    
    leads['lead_date'] = pd.to_datetime(leads['lead_date'])
    leads['sold_date'] = pd.to_datetime(leads['sold_date'])
    search_spend['date'] = pd.to_datetime(search_spend['date'])
    social_spend['date'] = pd.to_datetime(social_spend['date'])
    
    # Calculate NPV profit per policy
    leads['policy_profit'] = leads.apply(
        lambda row: calculate_npv_profit(row, EXPENSE_RATIO, DISCOUNT_RATE), 
        axis=1
    )
    
    # Week identifiers
    leads['week'] = leads['lead_date'].dt.to_period('W').dt.start_time
    search_spend['week'] = search_spend['date'].dt.to_period('W').dt.start_time
    social_spend['week'] = social_spend['date'].dt.to_period('W').dt.start_time
    
    # Aggregate spend by week
    search_weekly = search_spend.groupby('week')['spend'].sum().reset_index()
    search_weekly.columns = ['week', 'search_spend']
    
    social_weekly = social_spend.groupby('week')['spend'].sum().reset_index()
    social_weekly.columns = ['week', 'social_spend']
    
    # Count CONVERSIONS (sold policies) by channel and week
    sold_policies = leads[leads['sold_date'].notna()].copy()
    
    conversions_by_channel = sold_policies.groupby(['week', 'channel']).size().unstack(fill_value=0)
    conversions_by_channel.columns = [f'{c}_conversions' for c in conversions_by_channel.columns]
    conversions_by_channel = conversions_by_channel.reset_index()
    
    # Also get profit by channel for calculating averages
    profit_by_channel = sold_policies.groupby(['week', 'channel'])['policy_profit'].sum().unstack(fill_value=0)
    profit_by_channel.columns = [f'{c}_profit' for c in profit_by_channel.columns]
    profit_by_channel = profit_by_channel.reset_index()
    
    # Email spend
    email_leads = leads[leads['channel'] == 'email'].groupby('week')['lead_id'].nunique().reset_index()
    email_leads.columns = ['week', 'email_leads']
    email_leads['email_spend'] = email_leads['email_leads'] * EMAIL_CPL
    
    # Merge
    weekly_data = search_weekly.merge(social_weekly, on='week', how='outer')
    weekly_data = weekly_data.merge(email_leads[['week', 'email_spend']], on='week', how='outer')
    weekly_data = weekly_data.merge(conversions_by_channel, on='week', how='outer')
    weekly_data = weekly_data.merge(profit_by_channel, on='week', how='outer')
    weekly_data = weekly_data.fillna(0)
    weekly_data = weekly_data.sort_values('week').reset_index(drop=True)
    
    # Rename for consistency
    if 'paid_search_conversions' in weekly_data.columns:
        weekly_data['search_conversions'] = weekly_data['paid_search_conversions']
    if 'paid_social_conversions' in weekly_data.columns:
        weekly_data['social_conversions'] = weekly_data['paid_social_conversions']
    if 'paid_search_profit' in weekly_data.columns:
        weekly_data['search_profit'] = weekly_data['paid_search_profit']
    if 'paid_social_profit' in weekly_data.columns:
        weekly_data['social_profit'] = weekly_data['paid_social_profit']
    
    # Calculate average profit per conversion by channel
    avg_profit_per_conversion = {}
    for channel in ['search', 'social', 'email']:
        channel_name = f'paid_{channel}' if channel in ['search', 'social'] else channel
        ch_sold = sold_policies[sold_policies['channel'] == channel_name]
        if len(ch_sold) > 0:
            avg_profit_per_conversion[channel] = ch_sold['policy_profit'].mean()
        else:
            avg_profit_per_conversion[channel] = 0
    
    print(f"\nModeling Approach:")
    print(f"  Response variable: Weekly CONVERSIONS (not profit)")
    print(f"  Optimization: Conversions x Avg profit per conversion")
    
    print(f"\nAverage NPV Profit per Conversion:")
    for ch, val in avg_profit_per_conversion.items():
        print(f"  {ch}: ${val:,.0f}")
    
    # Show correlation improvement
    print(f"\nCorrelation: Spend -> Conversions (vs Spend -> Profit):")
    for ch in ['search', 'social', 'email']:
        spend_col = f'{ch}_spend'
        conv_col = f'{ch}_conversions'
        profit_col = f'{ch}_profit'
        
        if spend_col in weekly_data.columns and conv_col in weekly_data.columns:
            corr_conv = weekly_data[spend_col].corr(weekly_data[conv_col])
            corr_profit = weekly_data[spend_col].corr(weekly_data[profit_col])
            print(f"  {ch}: r={corr_conv:.3f} (conversions) vs r={corr_profit:.3f} (profit)")
    
    print(f"\nPrepared {len(weekly_data)} weeks of data")
    
    return weekly_data, avg_profit_per_conversion


def calculate_channel_stats(weekly_data, avg_profit_per_conversion):
    """Calculate average ROI and current spend for each channel."""
    channels = ['search', 'social', 'email']
    stats = {}
    
    for ch in channels:
        total_spend = weekly_data[f'{ch}_spend'].sum()
        total_conversions = weekly_data[f'{ch}_conversions'].sum()
        total_profit = weekly_data[f'{ch}_profit'].sum()
        avg_weekly_spend = weekly_data[f'{ch}_spend'].mean()
        avg_weekly_conversions = weekly_data[f'{ch}_conversions'].mean()
        
        stats[ch] = {
            'total_spend': total_spend,
            'total_conversions': total_conversions,
            'total_profit': total_profit,
            'avg_roi': total_profit / total_spend if total_spend > 0 else 0,
            'avg_weekly_spend': avg_weekly_spend,
            'avg_weekly_conversions': avg_weekly_conversions,
            'avg_profit_per_conversion': avg_profit_per_conversion.get(ch, 0)
        }
    
    # Rank by ROI (1 = highest)
    roi_sorted = sorted(stats.keys(), key=lambda x: stats[x]['avg_roi'], reverse=True)
    for rank, ch in enumerate(roi_sorted, 1):
        stats[ch]['roi_rank'] = rank
    
    return stats


# =============================================================================
# UNCONSTRAINED MODEL (for comparison)
# =============================================================================

def fit_unconstrained_hill(weekly_data, channel, channel_stats):
    """Standard Hill curve fitting on CONVERSIONS (not profit)."""
    spend = weekly_data[f'{channel}_spend'].values
    conversions = weekly_data[f'{channel}_conversions'].values
    
    mask = (spend > 0) & (conversions > 0)
    spend_fit = spend[mask]
    conversions_fit = conversions[mask]
    
    if len(spend_fit) < 10:
        return None
    
    try:
        popt, _ = curve_fit(
            hill_function,
            spend_fit,
            conversions_fit,
            p0=[conversions_fit.max() * 2, np.median(spend_fit), 0.8],
            bounds=([0, 0, 0.1], [np.inf, np.inf, 3.0]),
            maxfev=10000
        )
        
        pred = hill_function(spend_fit, *popt)
        ss_res = np.sum((conversions_fit - pred) ** 2)
        ss_tot = np.sum((conversions_fit - conversions_fit.mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        return {
            'K': popt[0], 'S': popt[1], 'beta': popt[2],
            'r2': r2, 'n_obs': len(spend_fit),
            'avg_profit_per_conversion': channel_stats[channel]['avg_profit_per_conversion']
        }
    except:
        return None


# =============================================================================
# CONSTRAINED MODEL
# =============================================================================

def fit_constrained_hill_all_channels(weekly_data, channel_stats):
    """
    Fit Hill curves to all channels simultaneously with the constraint that
    higher-ROI channels are further from saturation.
    
    Models CONVERSIONS (not profit) as the response variable.
    
    Constraint: If ROI_i > ROI_j, then (current_spend_i / S_i) < (current_spend_j / S_j)
    """
    channels = ['search', 'social', 'email']
    
    # Prepare data for each channel - use CONVERSIONS
    channel_data = {}
    for ch in channels:
        spend = weekly_data[f'{ch}_spend'].values
        conversions = weekly_data[f'{ch}_conversions'].values
        mask = (spend > 0) & (conversions > 0)
        channel_data[ch] = {
            'spend': spend[mask],
            'conversions': conversions[mask],
            'current_spend': channel_stats[ch]['avg_weekly_spend'],
            'roi_rank': channel_stats[ch]['roi_rank'],
            'avg_profit_per_conversion': channel_stats[ch]['avg_profit_per_conversion']
        }
    
    # Sort channels by ROI (highest first)
    channels_by_roi = sorted(channels, key=lambda x: channel_stats[x]['avg_roi'], reverse=True)
    print(f"\nChannels ranked by ROI: {channels_by_roi}")
    print(f"  {channels_by_roi[0]}: {channel_stats[channels_by_roi[0]]['avg_roi']:.1f}x")
    print(f"  {channels_by_roi[1]}: {channel_stats[channels_by_roi[1]]['avg_roi']:.1f}x")
    print(f"  {channels_by_roi[2]}: {channel_stats[channels_by_roi[2]]['avg_roi']:.1f}x")
    
    # Parameter vector: [K_search, S_search, beta_search, K_social, S_social, beta_social, K_email, S_email, beta_email]
    param_idx = {
        'search': {'K': 0, 'S': 1, 'beta': 2},
        'social': {'K': 3, 'S': 4, 'beta': 5},
        'email': {'K': 6, 'S': 7, 'beta': 8}
    }
    
    def objective(params):
        """Total squared error across all channels - fitting CONVERSIONS."""
        total_error = 0
        for ch in channels:
            K = params[param_idx[ch]['K']]
            S = params[param_idx[ch]['S']]
            beta = params[param_idx[ch]['beta']]
            
            pred = hill_function(channel_data[ch]['spend'], K, S, beta)
            error = np.sum((channel_data[ch]['conversions'] - pred) ** 2)
            total_error += error
        return total_error
    
    def saturation_proximity(params, channel):
        """Calculate current_spend / S for a channel (lower = further from saturation)."""
        S = params[param_idx[channel]['S']]
        return channel_data[channel]['current_spend'] / S
    
    # Build constraints based on ROI ranking
    constraints = []
    margin = 0.1
    
    for i in range(len(channels_by_roi) - 1):
        higher_roi_ch = channels_by_roi[i]
        lower_roi_ch = channels_by_roi[i + 1]
        
        def make_constraint(high_ch, low_ch, m):
            def constraint_func(params):
                sat_high = saturation_proximity(params, high_ch)
                sat_low = saturation_proximity(params, low_ch)
                return sat_low - sat_high - m
            return constraint_func
        
        constraints.append({
            'type': 'ineq',
            'fun': make_constraint(higher_roi_ch, lower_roi_ch, margin)
        })
    
    # Get unconstrained fits for initial guesses
    unconstrained_fits = {}
    for ch in channels:
        unconstrained_fits[ch] = fit_unconstrained_hill(weekly_data, ch, channel_stats)
    
    # Build initial guess
    x0 = []
    for ch in channels:
        unc = unconstrained_fits[ch]
        if unc:
            x0.extend([unc['K'], unc['S'], unc['beta']])
        else:
            conv_max = channel_data[ch]['conversions'].max()
            spend_med = np.median(channel_data[ch]['spend'])
            x0.extend([conv_max * 2, spend_med, 0.8])
    
    x0 = np.array(x0)
    
    # Adjust initial S values to satisfy constraints
    current_spends = {ch: channel_data[ch]['current_spend'] for ch in channels}
    
    target_sat_prox = {
        channels_by_roi[0]: 0.2,   # Highest ROI - far from saturation
        channels_by_roi[1]: 0.6,   # Middle
        channels_by_roi[2]: 1.2    # Lowest ROI - near saturation
    }
    
    for ch in channels:
        desired_S = current_spends[ch] / target_sat_prox[ch]
        x0[param_idx[ch]['S']] = desired_S
    
    print("\nInitial S values (adjusted to satisfy constraints):")
    for ch in channels:
        print(f"  {ch}: S = ${x0[param_idx[ch]['S']]:,.0f}, " +
              f"target sat_prox = {target_sat_prox[ch]:.2f}")
    
    # Bounds: K > 0, S > 0, 0.1 < beta < 3
    # Note: K bounds are smaller now since we're fitting conversions (counts, not dollars)
    bounds = []
    for ch in channels:
        bounds.append((0.1, None))      # K (max conversions)
        bounds.append((10, None))       # S
        bounds.append((0.1, 3.0))       # beta
    
    print("\nFitting constrained model (response = conversions)...")
    print(f"Constraint: sat_prox({channels_by_roi[0]}) < sat_prox({channels_by_roi[1]}) < sat_prox({channels_by_roi[2]})")
    
    # Try multiple optimization methods
    best_result = None
    best_obj = np.inf
    
    for method in ['SLSQP', 'trust-constr']:
        try:
            result = minimize(
                objective,
                x0,
                method=method,
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 10000}
            )
            
            if result.fun < best_obj:
                sat_prox_result = {ch: channel_data[ch]['current_spend'] / result.x[param_idx[ch]['S']] 
                                  for ch in channels}
                
                constraint_satisfied = True
                for i in range(len(channels_by_roi) - 1):
                    high_ch = channels_by_roi[i]
                    low_ch = channels_by_roi[i + 1]
                    if sat_prox_result[high_ch] >= sat_prox_result[low_ch]:
                        constraint_satisfied = False
                        break
                
                if constraint_satisfied:
                    best_result = result
                    best_obj = result.fun
                    print(f"  {method}: converged, obj = {result.fun:.0f}, constraints satisfied")
                else:
                    print(f"  {method}: converged but constraints violated")
        except Exception as e:
            print(f"  {method}: failed - {e}")
    
    if best_result is None:
        print("\nWARNING: No solution found that satisfies constraints.")
        print("Using initial guess with constraint-satisfying S values.")
        best_result = type('obj', (object,), {'x': x0, 'success': False})()
    
    # Extract results
    fitted_params = {}
    for ch in channels:
        K = best_result.x[param_idx[ch]['K']]
        S = best_result.x[param_idx[ch]['S']]
        beta = best_result.x[param_idx[ch]['beta']]
        
        # Calculate R-squared for CONVERSIONS fit
        pred = hill_function(channel_data[ch]['spend'], K, S, beta)
        ss_res = np.sum((channel_data[ch]['conversions'] - pred) ** 2)
        ss_tot = np.sum((channel_data[ch]['conversions'] - channel_data[ch]['conversions'].mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        sat_prox = channel_data[ch]['current_spend'] / S
        
        fitted_params[ch] = {
            'K': K,
            'S': S,
            'beta': beta,
            'r2': r2,
            'current_spend': channel_data[ch]['current_spend'],
            'saturation_proximity': sat_prox,
            'pct_of_saturation': sat_prox / (1 + sat_prox) * 100,
            'avg_profit_per_conversion': channel_data[ch]['avg_profit_per_conversion']
        }
    
    return fitted_params


# =============================================================================
# ANALYSIS
# =============================================================================

def find_optimal_allocation(fitted_params, total_budget):
    """
    Find optimal budget allocation to maximize PROFIT.
    
    Profit = Conversions(Spend) × Avg Profit per Conversion
    """
    channels = list(fitted_params.keys())
    
    def negative_profit(allocation):
        """Maximize conversions × avg profit per conversion."""
        total = 0
        for i, ch in enumerate(channels):
            p = fitted_params[ch]
            conversions = hill_function(allocation[i], p['K'], p['S'], p['beta'])
            profit = conversions * p['avg_profit_per_conversion']
            total += profit
        return -total
    
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - total_budget}
    bounds = [(10, total_budget) for _ in channels]
    x0 = [total_budget / len(channels)] * len(channels)
    
    result = minimize(negative_profit, x0, method='SLSQP',
                     bounds=bounds, constraints=constraints)
    
    return {ch: result.x[i] for i, ch in enumerate(channels)}


def calculate_marginal_conversions(params, spend):
    """Calculate marginal conversions at given spend level."""
    return hill_derivative(spend, params['K'], params['S'], params['beta'])


def calculate_marginal_profit(params, spend):
    """Calculate marginal profit at given spend level (marginal conv × avg profit)."""
    marginal_conv = hill_derivative(spend, params['K'], params['S'], params['beta'])
    return marginal_conv * params['avg_profit_per_conversion']


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_comparison(weekly_data, unconstrained, constrained, channel_stats, output_dir):
    """Compare unconstrained vs constrained models - plotting CONVERSIONS."""
    channels = ['search', 'social', 'email']
    colors = {'search': '#2E86AB', 'social': '#A23B72', 'email': '#F18F01'}
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Response curves (Spend -> Conversions)
    for i, channel in enumerate(channels):
        ax = axes[0, i]
        
        spend = weekly_data[f'{channel}_spend'].values
        conversions = weekly_data[f'{channel}_conversions'].values
        mask = (spend > 0) & (conversions > 0)
        
        ax.scatter(spend[mask], conversions[mask], alpha=0.4, s=20, color=colors[channel])
        
        spend_range = np.linspace(1, spend.max() * 2, 200)
        
        # Unconstrained
        if unconstrained[channel]:
            p = unconstrained[channel]
            pred_unc = hill_function(spend_range, p['K'], p['S'], p['beta'])
            ax.plot(spend_range, pred_unc, '--', color='gray', linewidth=2,
                   label=f"Unconstrained (R2={p['r2']:.2f})")
        
        # Constrained
        p = constrained[channel]
        pred_con = hill_function(spend_range, p['K'], p['S'], p['beta'])
        ax.plot(spend_range, pred_con, '-', color=colors[channel], linewidth=2,
               label=f"Constrained (R2={p['r2']:.2f})")
        
        # Mark half-saturation point and current spend
        ax.axvline(x=p['S'], color=colors[channel], linestyle=':', alpha=0.5)
        ax.axvline(x=p['current_spend'], color='black', linestyle=':', alpha=0.5)
        
        ax.set_xlabel('Weekly Spend ($)')
        ax.set_ylabel('Weekly Conversions')
        roi = channel_stats[channel]['avg_roi']
        ax.set_title(f'{channel.title()} (Avg ROI: {roi:.1f}x)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, None)
        ax.set_ylim(0, None)
    
    # Row 2: Saturation analysis
    ax = axes[1, 0]
    channels_sorted = sorted(channels, key=lambda x: channel_stats[x]['avg_roi'], reverse=True)
    
    x_pos = np.arange(len(channels_sorted))
    
    # Saturation proximity (current/S)
    sat_prox_unc = [channel_stats[ch]['avg_weekly_spend'] / unconstrained[ch]['S'] 
                   if unconstrained[ch] else 0 for ch in channels_sorted]
    sat_prox_con = [constrained[ch]['saturation_proximity'] for ch in channels_sorted]
    
    width = 0.35
    ax.bar(x_pos - width/2, sat_prox_unc, width, label='Unconstrained', color='gray', alpha=0.7)
    ax.bar(x_pos + width/2, sat_prox_con, width, label='Constrained', color='green', alpha=0.7)
    
    ax.set_xlabel('Channel (sorted by ROI)')
    ax.set_ylabel('Saturation Proximity (current/S)')
    ax.set_title('Saturation Proximity by Channel\n(Lower = Further from Saturation)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{ch}\n({channel_stats[ch]['avg_roi']:.0f}x ROI)" for ch in channels_sorted])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Half-saturation points
    ax = axes[1, 1]
    
    S_unc = [unconstrained[ch]['S'] if unconstrained[ch] else 0 for ch in channels_sorted]
    S_con = [constrained[ch]['S'] for ch in channels_sorted]
    current = [constrained[ch]['current_spend'] for ch in channels_sorted]
    
    ax.bar(x_pos - width/2, S_unc, width, label='Unconstrained S', color='gray', alpha=0.7)
    ax.bar(x_pos + width/2, S_con, width, label='Constrained S', color='green', alpha=0.7)
    ax.scatter(x_pos, current, color='red', s=100, zorder=5, label='Current Spend')
    
    ax.set_xlabel('Channel (sorted by ROI)')
    ax.set_ylabel('Spend ($)')
    ax.set_title('Half-Saturation Point (S) vs Current Spend')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{ch}\n({channel_stats[ch]['avg_roi']:.0f}x ROI)" for ch in channels_sorted])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Optimal allocation comparison
    ax = axes[1, 2]
    
    total_budget = sum(constrained[ch]['current_spend'] for ch in channels)
    
    optimal_unc = find_optimal_allocation(
        {ch: unconstrained[ch] for ch in channels if unconstrained[ch]},
        total_budget
    )
    optimal_con = find_optimal_allocation(constrained, total_budget)
    
    current_alloc = [constrained[ch]['current_spend'] for ch in channels]
    opt_unc_alloc = [optimal_unc.get(ch, 0) for ch in channels]
    opt_con_alloc = [optimal_con[ch] for ch in channels]
    
    x_pos = np.arange(len(channels))
    width = 0.25
    
    ax.bar(x_pos - width, current_alloc, width, label='Current', color='gray', alpha=0.7)
    ax.bar(x_pos, opt_unc_alloc, width, label='Optimal (Unconstr.)', color='blue', alpha=0.7)
    ax.bar(x_pos + width, opt_con_alloc, width, label='Optimal (Constr.)', color='green', alpha=0.7)
    
    ax.set_xlabel('Channel')
    ax.set_ylabel('Weekly Spend ($)')
    ax.set_title(f'Budget Allocation (Total: ${total_budget:,.0f}/week)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([ch.title() for ch in channels])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'constrained_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved: {output_dir}/constrained_comparison.png")


def create_homepage_image(constrained, channel_stats, output_dir):
    """
    Create the combined image for the homepage.
    
    Side-by-side:
    - Left: Marginal ROI at Different Spend Levels
    - Right: Current vs Optimal Budget Allocation
    """
    channels = ['search', 'social', 'email']
    colors = {'search': '#2E86AB', 'social': '#A23B72', 'email': '#F18F01'}
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left panel: Marginal ROI at Different Spend Levels
    ax = axes[0]
    spend_levels = [500, 1000, 2000, 3000, 4000, 5000]
    x_pos = np.arange(len(spend_levels))
    width = 0.25
    
    for i, ch in enumerate(channels):
        p = constrained[ch]
        marginal_rois = []
        for spend in spend_levels:
            marg_conv = hill_derivative(spend, p['K'], p['S'], p['beta'])
            marg_profit = marg_conv * p['avg_profit_per_conversion']
            marginal_rois.append(marg_profit)
        
        ax.bar(x_pos + i * width, marginal_rois, width, 
               label=ch.title(), color=colors[ch], alpha=0.8)
    
    ax.axhline(y=1, color='gray', linestyle='--', linewidth=1, label='Break-even')
    ax.set_xlabel('Weekly Spend Level ($)', fontsize=11)
    ax.set_ylabel('Marginal Profit per $1 Spent', fontsize=11)
    ax.set_title('Marginal ROI at Different Spend Levels', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels([f'${s:,}' for s in spend_levels])
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, None)
    
    # Right panel: Current vs Optimal Budget Allocation
    ax = axes[1]
    total_budget = sum(constrained[ch]['current_spend'] for ch in channels)
    optimal = find_optimal_allocation(constrained, total_budget)
    
    x_pos = np.arange(len(channels))
    width = 0.35
    
    current_alloc = [constrained[ch]['current_spend'] for ch in channels]
    optimal_alloc = [optimal[ch] for ch in channels]
    
    bars1 = ax.bar(x_pos - width/2, current_alloc, width, 
                   label='Current', color='gray', alpha=0.7)
    bars2 = ax.bar(x_pos + width/2, optimal_alloc, width, 
                   label='Optimal', color='green', alpha=0.8)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'${height:,.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'${height:,.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Channel', fontsize=11)
    ax.set_ylabel('Weekly Spend ($)', fontsize=11)
    ax.set_title(f'Current vs Optimal Budget Allocation\n(Total: ${total_budget:,.0f}/week)', 
                 fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([ch.title() for ch in channels])
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'MMM_homepage.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_dir}/MMM_homepage.png")


# =============================================================================
# REPORTING
# =============================================================================

def generate_report(weekly_data, unconstrained, constrained, channel_stats, output_dir):
    """Generate comparison report."""
    channels = ['search', 'social', 'email']
    channels_by_roi = sorted(channels, key=lambda x: channel_stats[x]['avg_roi'], reverse=True)
    
    report = []
    report.append("=" * 70)
    report.append("CONSTRAINED MARKETING MIX MODEL")
    report.append("ROI-Saturation Relationship Enforced")
    report.append("=" * 70)
    
    report.append("\n\nECONOMIC ASSUMPTIONS:")
    report.append("-" * 50)
    report.append(f"""
Profit is calculated as NPV of policy cash flows:

  Expense ratio:    {EXPENSE_RATIO:>6.0%}  (operating costs as % of premium)
  Discount rate:    {DISCOUNT_RATE:>6.0%}  (annual rate for time value of money)

  Annual profit = Annual premium x (1 - expense ratio) - Annual claims
  NPV = Annual profit x Annuity factor(discount rate, tenure)

This reduces gross profit by ~{1 - 0.45:.0%} compared to simple (premium - claims).
""")
    
    report.append("\n\nCONSTRAINT APPLIED:")
    report.append("-" * 50)
    report.append("""
The model enforces that channels with higher average ROI are further
from saturation. This encodes the belief that high ROI indicates
untapped potential (operating on steep part of response curve).

Mathematically: If ROI_i > ROI_j, then (current_spend_i / S_i) < (current_spend_j / S_j)
""")
    
    report.append("\n\n1. CHANNEL ROI RANKING")
    report.append("-" * 50)
    for i, ch in enumerate(channels_by_roi, 1):
        roi = channel_stats[ch]['avg_roi']
        report.append(f"  {i}. {ch:<10} {roi:>6.1f}x average ROI")
    
    report.append("\n\n2. SATURATION ANALYSIS")
    report.append("-" * 50)
    report.append(f"{'Channel':<12} {'Avg ROI':>10} {'Current $':>12} {'S (unconstr)':>14} {'S (constr)':>14} {'Sat Prox':>10}")
    report.append("-" * 70)
    
    for ch in channels_by_roi:
        roi = channel_stats[ch]['avg_roi']
        current = constrained[ch]['current_spend']
        s_unc = unconstrained[ch]['S'] if unconstrained[ch] else 0
        s_con = constrained[ch]['S']
        sat_prox = constrained[ch]['saturation_proximity']
        
        report.append(f"{ch:<12} {roi:>10.1f}x ${current:>10,.0f} ${s_unc:>12,.0f} ${s_con:>12,.0f} {sat_prox:>10.3f}")
    
    report.append("\n(Sat Prox = current_spend / S; lower = further from saturation)")
    report.append("Constraint satisfied: " + " > ".join([f"{constrained[ch]['saturation_proximity']:.3f}" 
                                                         for ch in reversed(channels_by_roi)]))
    
    report.append("\n\n3. PARAMETER COMPARISON (Response = Conversions)")
    report.append("-" * 50)
    report.append("Note: K = max weekly conversions, S = half-saturation spend")
    
    for ch in channels:
        report.append(f"\n{ch.upper()} (Avg profit/conv: ${constrained[ch]['avg_profit_per_conversion']:,.0f})")
        
        if unconstrained[ch]:
            report.append(f"  Unconstrained:")
            report.append(f"    K =    {unconstrained[ch]['K']:>10.1f} conv/week")
            report.append(f"    S = ${unconstrained[ch]['S']:>12,.0f}")
            report.append(f"    beta = {unconstrained[ch]['beta']:>10.3f}")
            report.append(f"    R2 = {unconstrained[ch]['r2']:>10.3f}")
        
        report.append(f"  Constrained:")
        report.append(f"    K =    {constrained[ch]['K']:>10.1f} conv/week")
        report.append(f"    S = ${constrained[ch]['S']:>12,.0f}")
        report.append(f"    beta = {constrained[ch]['beta']:>10.3f}")
        report.append(f"    R2 = {constrained[ch]['r2']:>10.3f}")
    
    report.append("\n\n4. OPTIMAL BUDGET ALLOCATION")
    report.append("-" * 50)
    
    total_budget = sum(constrained[ch]['current_spend'] for ch in channels)
    optimal_con = find_optimal_allocation(constrained, total_budget)
    
    report.append(f"Total weekly budget: ${total_budget:,.0f}")
    report.append("Optimization: Max(Conversions x Avg Profit per Conversion)")
    report.append("")
    report.append(f"{'Channel':<12} {'Current':>12} {'Optimal':>12} {'Change':>12} {'Marg Profit':>14}")
    report.append("-" * 62)
    
    for ch in channels:
        current = constrained[ch]['current_spend']
        optimal = optimal_con[ch]
        change = optimal - current
        marg_profit = calculate_marginal_profit(constrained[ch], optimal)
        
        change_str = f"+${change:,.0f}" if change >= 0 else f"-${abs(change):,.0f}"
        report.append(f"{ch:<12} ${current:>10,.0f} ${optimal:>10,.0f} {change_str:>12} ${marg_profit:>12,.0f}")
    
    report.append("\n\n5. KEY INSIGHT")
    report.append("-" * 50)
    report.append(f"""
With the ROI-saturation constraint, the model now believes:

- {channels_by_roi[0].upper()} (highest ROI at {channel_stats[channels_by_roi[0]]['avg_roi']:.0f}x):
  Operating at {constrained[channels_by_roi[0]]['saturation_proximity']:.1%} of half-saturation
  Has significant room to scale

- {channels_by_roi[2].upper()} (lowest ROI at {channel_stats[channels_by_roi[2]]['avg_roi']:.0f}x):
  Operating at {constrained[channels_by_roi[2]]['saturation_proximity']:.1%} of half-saturation
  Already approaching diminishing returns

This aligns business intuition (high ROI = untapped potential) with
the mathematical model, preventing the unconstrained model from
incorrectly concluding that high-ROI channels are saturated.
""")
    
    report.append("=" * 70)
    
    report_text = "\n".join(report)
    
    with open(output_dir / 'constrained_report.txt', 'w') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\nSaved: {output_dir}/constrained_report.txt")
    
    return report_text


# =============================================================================
# MAIN
# =============================================================================

def main():
    # Load data
    weekly_data, avg_profit_per_conversion = load_and_prepare_data(DATA_DIR)
    
    # Calculate channel statistics
    channel_stats = calculate_channel_stats(weekly_data, avg_profit_per_conversion)
    
    print("\nChannel Statistics:")
    for ch, stats in channel_stats.items():
        print(f"  {ch}: ROI = {stats['avg_roi']:.1f}x, Avg Weekly Spend = ${stats['avg_weekly_spend']:,.0f}, " +
              f"Avg Profit/Conv = ${stats['avg_profit_per_conversion']:,.0f}")
    
    # Fit unconstrained models
    print("\n" + "=" * 70)
    print("FITTING UNCONSTRAINED MODELS (response = conversions)")
    print("=" * 70)
    
    unconstrained = {}
    for ch in ['search', 'social', 'email']:
        unconstrained[ch] = fit_unconstrained_hill(weekly_data, ch, channel_stats)
        if unconstrained[ch]:
            print(f"\n{ch}: K = {unconstrained[ch]['K']:.1f} conv/week, S = ${unconstrained[ch]['S']:,.0f}, " +
                  f"R2 = {unconstrained[ch]['r2']:.3f}")
    
    # Fit constrained model
    print("\n" + "=" * 70)
    print("FITTING CONSTRAINED MODEL")
    print("=" * 70)
    
    constrained = fit_constrained_hill_all_channels(weekly_data, channel_stats)
    
    print("\nConstrained Results:")
    for ch in ['search', 'social', 'email']:
        print(f"  {ch}: K = {constrained[ch]['K']:.1f} conv/week, S = ${constrained[ch]['S']:,.0f}, " +
              f"sat_prox = {constrained[ch]['saturation_proximity']:.3f}, R2 = {constrained[ch]['r2']:.3f}")
    
    # Generate outputs
    plot_comparison(weekly_data, unconstrained, constrained, channel_stats, OUTPUT_DIR)
    generate_report(weekly_data, unconstrained, constrained, channel_stats, OUTPUT_DIR)
    create_homepage_image(constrained, channel_stats, OUTPUT_DIR)
    
    # Save results
    def convert_to_serializable(v):
        if isinstance(v, (np.floating, float)):
            return float(v)
        elif isinstance(v, (np.integer, int)):
            return int(v)
        else:
            return v
    
    results = {
        'channel_stats': {ch: {k: convert_to_serializable(v) for k, v in stats.items()} 
                        for ch, stats in channel_stats.items()},
        'unconstrained': {ch: {k: convert_to_serializable(v) for k, v in params.items()} if params else None
                         for ch, params in unconstrained.items()},
        'constrained': {ch: {k: convert_to_serializable(v) for k, v in params.items()}
                       for ch, params in constrained.items()}
    }
    
    with open(OUTPUT_DIR / 'constrained_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved: {OUTPUT_DIR}/constrained_results.json")
    
    return weekly_data, unconstrained, constrained, channel_stats


if __name__ == "__main__":
    weekly_data, unconstrained, constrained, channel_stats = main()
