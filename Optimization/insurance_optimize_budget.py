"""
Budget Optimization Simulation
==============================
Simulates Period 1 (current allocation) vs Period 2 (optimal allocation)
to show the impact of marketing mix optimization.

Author: Michael Belli
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / 'insure_co_data'
MMM_DIR = BASE_DIR.parent / 'MMM'
OUTPUT_DIR = BASE_DIR

# Economic assumptions (must match MMM)
EXPENSE_RATIO = 0.30
DISCOUNT_RATE = 0.10
WEEKS_PER_PERIOD = 12  # 1 year per period (monthly unit)


# =============================================================================
# HILL FUNCTIONS (from MMM)
# =============================================================================

def hill_function(spend, K, S, beta):
    """Hill saturation function."""
    return K * (spend ** beta) / (S ** beta + spend ** beta)


def hill_derivative(spend, K, S, beta):
    """Marginal response from Hill function."""
    numerator = K * beta * (spend ** (beta - 1)) * (S ** beta)
    denominator = (S ** beta + spend ** beta) ** 2
    return numerator / denominator


# =============================================================================
# SIMULATION
# =============================================================================

def load_mmm_results():
    """Load fitted model parameters from MMM."""
    results_path = MMM_DIR / 'constrained_results.json'
    
    if not results_path.exists():
        raise FileNotFoundError(
            f"MMM results not found at {results_path}\n"
            "Please run: python MMM/marketing_mix_model.py first"
        )
    
    with open(results_path) as f:
        results = json.load(f)
    
    return results


def simulate_period(params, allocation, weeks=12):
    """
    Simulate a period (e.g., 1 year) with given budget allocation.
    
    Returns dict with conversions and profit by channel.
    """
    channels = ['search', 'social', 'email']
    results = {'channels': {}, 'totals': {}}
    
    total_conversions = 0
    total_profit = 0
    total_spend = 0
    
    for ch in channels:
        p = params[ch]
        weekly_spend = allocation[ch]
        
        # Predicted weekly conversions from Hill function
        weekly_conversions = hill_function(weekly_spend, p['K'], p['S'], p['beta'])
        
        # Scale to full period
        period_conversions = weekly_conversions * weeks
        period_spend = weekly_spend * weeks
        period_profit = period_conversions * p['avg_profit_per_conversion']
        
        # ROI
        roi = period_profit / period_spend if period_spend > 0 else 0
        
        results['channels'][ch] = {
            'weekly_spend': weekly_spend,
            'period_spend': period_spend,
            'weekly_conversions': weekly_conversions,
            'period_conversions': period_conversions,
            'period_profit': period_profit,
            'roi': roi,
            'avg_profit_per_conversion': p['avg_profit_per_conversion']
        }
        
        total_conversions += period_conversions
        total_profit += period_profit
        total_spend += period_spend
    
    results['totals'] = {
        'spend': total_spend,
        'conversions': total_conversions,
        'profit': total_profit,
        'roi': total_profit / total_spend if total_spend > 0 else 0
    }
    
    return results


def find_optimal_allocation(params, total_budget):
    """Find budget allocation that maximizes profit."""
    from scipy.optimize import minimize
    
    channels = list(params.keys())
    
    def negative_profit(allocation):
        total = 0
        for i, ch in enumerate(channels):
            p = params[ch]
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


def run_optimization_simulation():
    """
    Run the full optimization simulation:
    - Period 1: Current allocation
    - Period 2: Optimal allocation
    """
    print("=" * 60)
    print("BUDGET OPTIMIZATION SIMULATION")
    print("=" * 60)
    
    # Load MMM results
    print("\nLoading MMM results...")
    mmm_results = load_mmm_results()
    params = mmm_results['constrained']
    channel_stats = mmm_results['channel_stats']
    
    # Current allocation (from channel_stats)
    current_allocation = {
        ch: channel_stats[ch]['avg_weekly_spend'] 
        for ch in ['search', 'social', 'email']
    }
    total_budget = sum(current_allocation.values())
    
    print(f"\nTotal monthly budget: ${total_budget:,.0f}")
    print(f"Simulation period: {WEEKS_PER_PERIOD} weeks")
    
    # Find optimal allocation
    optimal_allocation = find_optimal_allocation(params, total_budget)
    
    print("\n" + "-" * 40)
    print("ALLOCATIONS")
    print("-" * 40)
    print(f"{'Channel':<12} {'Current':>12} {'Optimal':>12} {'Change':>12}")
    print("-" * 48)
    for ch in ['search', 'social', 'email']:
        curr = current_allocation[ch]
        opt = optimal_allocation[ch]
        change = opt - curr
        change_str = f"+${change:,.0f}" if change >= 0 else f"-${abs(change):,.0f}"
        print(f"{ch:<12} ${curr:>10,.0f} ${opt:>10,.0f} {change_str:>12}")
    
    # Simulate both periods
    print("\n" + "-" * 40)
    print("SIMULATING PERIODS...")
    print("-" * 40)
    
    period1 = simulate_period(params, current_allocation, WEEKS_PER_PERIOD)
    period2 = simulate_period(params, optimal_allocation, WEEKS_PER_PERIOD)
    
    # Summary
    print("\n" + "=" * 60)
    print("SIMULATION RESULTS")
    print("=" * 60)
    
    print(f"\n{'Metric':<25} {'Period 1':>15} {'Period 2':>15} {'Change':>15}")
    print(f"{'(Current)':<25} {'':>15} {'(Optimal)':>15} {'':>15}")
    print("-" * 70)
    
    # Spend
    print(f"{'Total Spend':<25} ${period1['totals']['spend']:>13,.0f} ${period2['totals']['spend']:>13,.0f} {'$0':>15}")
    
    # Conversions
    conv_change = period2['totals']['conversions'] - period1['totals']['conversions']
    conv_pct = conv_change / period1['totals']['conversions'] * 100
    print(f"{'Total Conversions':<25} {period1['totals']['conversions']:>14,.0f} {period2['totals']['conversions']:>14,.0f} {conv_change:>+14,.0f}")
    
    # Profit
    profit_change = period2['totals']['profit'] - period1['totals']['profit']
    profit_pct = profit_change / period1['totals']['profit'] * 100
    print(f"{'Total Profit':<25} ${period1['totals']['profit']:>13,.0f} ${period2['totals']['profit']:>13,.0f} ${profit_change:>+13,.0f}")
    
    # ROI
    print(f"{'Overall ROI':<25} {period1['totals']['roi']:>14.1f}x {period2['totals']['roi']:>14.1f}x {period2['totals']['roi'] - period1['totals']['roi']:>+14.1f}x")
    
    print("\n" + "-" * 70)
    print(f"{'IMPROVEMENT':<25} {'':>15} {'':>15} {profit_pct:>+14.1f}%")
    
    # Store results
    simulation_results = {
        'period1': period1,
        'period2': period2,
        'current_allocation': current_allocation,
        'optimal_allocation': optimal_allocation,
        'improvement': {
            'conversions_change': conv_change,
            'conversions_pct': conv_pct,
            'profit_change': profit_change,
            'profit_pct': profit_pct
        }
    }
    
    return simulation_results


def create_optimization_charts(results):
    """Create visualization comparing Period 1 vs Period 2."""
    
    channels = ['search', 'social', 'email']
    colors = {'search': '#2E86AB', 'social': '#A23B72', 'email': '#F18F01'}
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Chart 1: Budget Allocation
    ax = axes[0]
    x_pos = np.arange(len(channels))
    width = 0.35
    
    period1_spend = [results['current_allocation'][ch] for ch in channels]
    period2_spend = [results['optimal_allocation'][ch] for ch in channels]
    
    ax.bar(x_pos - width/2, period1_spend, width, label='Period 1 (Current)', color='gray', alpha=0.7)
    ax.bar(x_pos + width/2, period2_spend, width, label='Period 2 (Optimal)', color='green', alpha=0.8)
    
    ax.set_xlabel('Channel')
    ax.set_ylabel('Monthly Spend ($)')
    ax.set_title('Budget Allocation', fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([ch.title() for ch in channels])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Chart 2: Conversions by Channel
    ax = axes[1]
    
    period1_conv = [results['period1']['channels'][ch]['period_conversions'] for ch in channels]
    period2_conv = [results['period2']['channels'][ch]['period_conversions'] for ch in channels]
    
    ax.bar(x_pos - width/2, period1_conv, width, label='Period 1', color='gray', alpha=0.7)
    ax.bar(x_pos + width/2, period2_conv, width, label='Period 2', color='green', alpha=0.8)
    
    ax.set_xlabel('Channel')
    ax.set_ylabel('Annual Conversions')
    ax.set_title('Conversions by Channel', fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([ch.title() for ch in channels])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Chart 3: Profit by Channel
    ax = axes[2]
    
    period1_profit = [results['period1']['channels'][ch]['period_profit'] for ch in channels]
    period2_profit = [results['period2']['channels'][ch]['period_profit'] for ch in channels]
    
    ax.bar(x_pos - width/2, period1_profit, width, label='Period 1', color='gray', alpha=0.7)
    ax.bar(x_pos + width/2, period2_profit, width, label='Period 2', color='green', alpha=0.8)
    
    ax.set_xlabel('Channel')
    ax.set_ylabel('Annual Profit ($)')
    ax.set_title('Profit by Channel', fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([ch.title() for ch in channels])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'optimization_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved: {OUTPUT_DIR}/optimization_comparison.png")
    
    # Homepage image: Single chart showing before/after
    fig, ax = plt.subplots(figsize=(8, 5))
    
    metrics = ['Spend', 'Conversions', 'Profit']
    period1_vals = [
        results['period1']['totals']['spend'],
        results['period1']['totals']['conversions'],
        results['period1']['totals']['profit']
    ]
    period2_vals = [
        results['period2']['totals']['spend'],
        results['period2']['totals']['conversions'],
        results['period2']['totals']['profit']
    ]
    
    # Normalize for display (different scales)
    # Show as indexed to Period 1 = 100
    period1_indexed = [100, 100, 100]
    period2_indexed = [
        period2_vals[0] / period1_vals[0] * 100,
        period2_vals[1] / period1_vals[1] * 100,
        period2_vals[2] / period1_vals[2] * 100
    ]
    
    x_pos = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax.bar(x_pos - width/2, period1_indexed, width, 
                   label='Period 1 (Current)', color='gray', alpha=0.7)
    bars2 = ax.bar(x_pos + width/2, period2_indexed, width, 
                   label='Period 2 (Optimized)', color='green', alpha=0.8)
    
    # Add actual values as labels
    for i, (bar, val) in enumerate(zip(bars1, period1_vals)):
        if i == 1:  # Conversions
            label = f'{val:,.0f}'
        else:
            label = f'${val:,.0f}'
        ax.annotate(label, xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontsize=9)
    
    for i, (bar, val) in enumerate(zip(bars2, period2_vals)):
        if i == 1:  # Conversions
            label = f'{val:,.0f}'
        else:
            label = f'${val:,.0f}'
        ax.annotate(label, xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontsize=9)
    
    ax.axhline(y=100, color='black', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Metric', fontsize=11)
    ax.set_ylabel('Index (Period 1 = 100)', fontsize=11)
    ax.set_title('Optimization Impact: Period 1 vs Period 2\n(Same Total Budget)', 
                 fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metrics)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 130)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'Optimization_homepage.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {OUTPUT_DIR}/Optimization_homepage.png")


def save_results(results):
    """Save simulation results to JSON."""
    
    # Convert numpy types to native Python
    def convert(obj):
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        else:
            return obj
    
    results_clean = convert(results)
    
    with open(OUTPUT_DIR / 'optimization_results.json', 'w') as f:
        json.dump(results_clean, f, indent=2)
    
    print(f"Saved: {OUTPUT_DIR}/optimization_results.json")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    results = run_optimization_simulation()
    create_optimization_charts(results)
    save_results(results)
    
    print("\n" + "=" * 60)
    print("OPTIMIZATION SIMULATION COMPLETE")
    print("=" * 60)
