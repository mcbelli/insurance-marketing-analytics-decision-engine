"""
Business Impact Calculator
==========================
Calculates and visualizes the business impact of marketing optimization.

Shows revenue and profit metrics before/after optimization.

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
OPTIMIZATION_DIR = BASE_DIR.parent / 'Optimization'
OUTPUT_DIR = BASE_DIR

# Revenue assumptions
AVG_ANNUAL_PREMIUM = 3500  # Blended average across products


# =============================================================================
# CALCULATIONS
# =============================================================================

def load_optimization_results():
    """Load results from optimization simulation."""
    results_path = OPTIMIZATION_DIR / 'optimization_results.json'
    
    if not results_path.exists():
        raise FileNotFoundError(
            f"Optimization results not found at {results_path}\n"
            "Please run: python Optimization/optimize_budget.py first"
        )
    
    with open(results_path) as f:
        results = json.load(f)
    
    return results


def calculate_business_impact(opt_results):
    """
    Calculate business impact metrics.
    
    Metrics:
    - Revenue (= conversions × avg premium × avg tenure)
    - Marketing spend
    - Gross margin
    - Net profit (after marketing)
    - Customer acquisition cost (CAC)
    - Return on ad spend (ROAS)
    """
    
    # Extract from optimization results
    p1 = opt_results['period1']['totals']
    p2 = opt_results['period2']['totals']
    
    # Calculate additional metrics
    # Note: 'profit' in opt_results is NPV policy profit (premium - claims - expenses)
    # Revenue here is gross premium collected
    
    # Estimate revenue from conversions
    # Using avg tenure of ~5.5 years
    avg_tenure = 5.5
    
    p1_revenue = p1['conversions'] * AVG_ANNUAL_PREMIUM * avg_tenure
    p2_revenue = p2['conversions'] * AVG_ANNUAL_PREMIUM * avg_tenure
    
    # CAC = Spend / Conversions
    p1_cac = p1['spend'] / p1['conversions'] if p1['conversions'] > 0 else 0
    p2_cac = p2['spend'] / p2['conversions'] if p2['conversions'] > 0 else 0
    
    # ROAS = Revenue / Spend
    p1_roas = p1_revenue / p1['spend'] if p1['spend'] > 0 else 0
    p2_roas = p2_revenue / p2['spend'] if p2['spend'] > 0 else 0
    
    # Profit per customer
    p1_profit_per_cust = p1['profit'] / p1['conversions'] if p1['conversions'] > 0 else 0
    p2_profit_per_cust = p2['profit'] / p2['conversions'] if p2['conversions'] > 0 else 0
    
    impact = {
        'period1': {
            'marketing_spend': p1['spend'],
            'conversions': p1['conversions'],
            'revenue': p1_revenue,
            'policy_profit': p1['profit'],
            'net_profit': p1['profit'] - p1['spend'],
            'cac': p1_cac,
            'roas': p1_roas,
            'profit_per_customer': p1_profit_per_cust,
            'roi': p1['roi']
        },
        'period2': {
            'marketing_spend': p2['spend'],
            'conversions': p2['conversions'],
            'revenue': p2_revenue,
            'policy_profit': p2['profit'],
            'net_profit': p2['profit'] - p2['spend'],
            'cac': p2_cac,
            'roas': p2_roas,
            'profit_per_customer': p2_profit_per_cust,
            'roi': p2['roi']
        },
        'change': {
            'marketing_spend': 0,  # Same budget
            'conversions': p2['conversions'] - p1['conversions'],
            'conversions_pct': (p2['conversions'] - p1['conversions']) / p1['conversions'] * 100,
            'revenue': p2_revenue - p1_revenue,
            'revenue_pct': (p2_revenue - p1_revenue) / p1_revenue * 100,
            'policy_profit': p2['profit'] - p1['profit'],
            'policy_profit_pct': (p2['profit'] - p1['profit']) / p1['profit'] * 100,
            'net_profit': (p2['profit'] - p2['spend']) - (p1['profit'] - p1['spend']),
            'cac': p2_cac - p1_cac,
            'cac_pct': (p2_cac - p1_cac) / p1_cac * 100,
            'roas': p2_roas - p1_roas,
        }
    }
    
    return impact


def print_impact_report(impact):
    """Print formatted business impact report."""
    
    print("\n" + "=" * 70)
    print("BUSINESS IMPACT REPORT")
    print("=" * 70)
    
    print(f"\n{'Metric':<30} {'Period 1':>15} {'Period 2':>15} {'Change':>15}")
    print("-" * 75)
    
    # Marketing spend
    print(f"{'Marketing Spend':<30} ${impact['period1']['marketing_spend']:>13,.0f} ${impact['period2']['marketing_spend']:>13,.0f} {'$0':>15}")
    
    # Conversions
    print(f"{'New Customers':<30} {impact['period1']['conversions']:>14,.0f} {impact['period2']['conversions']:>14,.0f} {impact['change']['conversions']:>+14,.0f}")
    print(f"{'':30} {'':>15} {'':>15} {impact['change']['conversions_pct']:>+13.1f}%")
    
    # Revenue
    print(f"{'Lifetime Revenue':<30} ${impact['period1']['revenue']:>13,.0f} ${impact['period2']['revenue']:>13,.0f} ${impact['change']['revenue']:>+13,.0f}")
    print(f"{'':30} {'':>15} {'':>15} {impact['change']['revenue_pct']:>+13.1f}%")
    
    # Profit
    print(f"{'Policy Profit (NPV)':<30} ${impact['period1']['policy_profit']:>13,.0f} ${impact['period2']['policy_profit']:>13,.0f} ${impact['change']['policy_profit']:>+13,.0f}")
    print(f"{'':30} {'':>15} {'':>15} {impact['change']['policy_profit_pct']:>+13.1f}%")
    
    # Net profit
    print(f"{'Net Profit (after mktg)':<30} ${impact['period1']['net_profit']:>13,.0f} ${impact['period2']['net_profit']:>13,.0f} ${impact['change']['net_profit']:>+13,.0f}")
    
    print("-" * 75)
    
    # Efficiency metrics
    print(f"\n{'EFFICIENCY METRICS':<30}")
    print("-" * 75)
    print(f"{'CAC (Cost per Customer)':<30} ${impact['period1']['cac']:>13,.0f} ${impact['period2']['cac']:>13,.0f} ${impact['change']['cac']:>+13,.0f}")
    print(f"{'ROAS (Revenue/Spend)':<30} {impact['period1']['roas']:>14.1f}x {impact['period2']['roas']:>14.1f}x {impact['change']['roas']:>+14.1f}x")
    print(f"{'Marketing ROI':<30} {impact['period1']['roi']:>14.1f}x {impact['period2']['roi']:>14.1f}x {impact['change']['policy_profit_pct']/100:>+14.2f}x")
    
    print("\n" + "=" * 70)
    print("KEY TAKEAWAY")
    print("=" * 70)
    print(f"""
By reallocating the SAME marketing budget ({impact['period1']['marketing_spend']:,.0f}/year):

  • {impact['change']['conversions']:+,.0f} additional customers ({impact['change']['conversions_pct']:+.1f}%)
  • ${impact['change']['revenue']:+,.0f} additional lifetime revenue
  • ${impact['change']['policy_profit']:+,.0f} additional profit ({impact['change']['policy_profit_pct']:+.1f}%)
  • ${impact['change']['cac']:+,.0f} lower CAC ({impact['change']['cac_pct']:.1f}% improvement)
""")


def create_impact_visualization(impact):
    """Create business impact visualization."""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Revenue & Profit comparison
    ax = axes[0]
    
    metrics = ['Lifetime\nRevenue', 'Policy\nProfit', 'Net Profit\n(after mktg)']
    period1_vals = [
        impact['period1']['revenue'],
        impact['period1']['policy_profit'],
        impact['period1']['net_profit']
    ]
    period2_vals = [
        impact['period2']['revenue'],
        impact['period2']['policy_profit'],
        impact['period2']['net_profit']
    ]
    
    x_pos = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax.bar(x_pos - width/2, [v/1000 for v in period1_vals], width, 
                   label='Period 1 (Current)', color='#E57373', alpha=0.8)
    bars2 = ax.bar(x_pos + width/2, [v/1000 for v in period2_vals], width, 
                   label='Period 2 (Optimized)', color='#81C784', alpha=0.8)
    
    ax.set_xlabel('Metric', fontsize=11)
    ax.set_ylabel("Value ($000's)", fontsize=11)
    ax.set_title('Financial Impact of Optimization', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'${height:,.0f}K',
                        xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=9)
    
    # Right: Efficiency metrics
    ax = axes[1]
    
    metrics = ['CAC\n(lower is better)', 'ROAS', 'Marketing\nROI']
    period1_vals = [
        impact['period1']['cac'],
        impact['period1']['roas'],
        impact['period1']['roi']
    ]
    period2_vals = [
        impact['period2']['cac'],
        impact['period2']['roas'],
        impact['period2']['roi']
    ]
    
    x_pos = np.arange(len(metrics))
    
    bars1 = ax.bar(x_pos - width/2, period1_vals, width, 
                   label='Period 1 (Current)', color='#E57373', alpha=0.8)
    bars2 = ax.bar(x_pos + width/2, period2_vals, width, 
                   label='Period 2 (Optimized)', color='#81C784', alpha=0.8)
    
    ax.set_xlabel('Metric', fontsize=11)
    ax.set_ylabel('Value', fontsize=11)
    ax.set_title('Efficiency Metrics', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        for bar in [bar1, bar2]:
            height = bar.get_height()
            if i == 0:  # CAC
                label = f'${height:,.0f}'
            else:
                label = f'{height:.1f}x'
            ax.annotate(label,
                        xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'business_impact.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved: {OUTPUT_DIR}/business_impact.png")
    
    # Homepage summary image - use separate normalized values
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Show percentage change as the main visual
    categories = ['New\nCustomers', 'Lifetime\nRevenue', 'Profit']
    changes_pct = [
        impact['change']['conversions_pct'],
        impact['change']['revenue_pct'],
        impact['change']['policy_profit_pct']
    ]
    
    x_pos = np.arange(len(categories))
    
    # Color bars based on positive/negative
    colors = ['#81C784' if c > 0 else '#E57373' for c in changes_pct]
    
    bars = ax.bar(x_pos, changes_pct, color=colors, alpha=0.8, width=0.5)
    
    # Add value labels
    for bar, pct in zip(bars, changes_pct):
        height = bar.get_height()
        ax.annotate(f'+{pct:.1f}%',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 5), textcoords='offset points',
                    ha='center', va='bottom', fontsize=14, fontweight='bold',
                    color='darkgreen')
    
    # Add absolute values below x-axis labels
    p1_vals = [
        f'{impact["period1"]["conversions"]:,.0f} → {impact["period2"]["conversions"]:,.0f}',
        f'${impact["period1"]["revenue"]/1e6:.1f}M → ${impact["period2"]["revenue"]/1e6:.1f}M',
        f'${impact["period1"]["policy_profit"]/1e6:.1f}M → ${impact["period2"]["policy_profit"]/1e6:.1f}M'
    ]
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_ylabel('Improvement (%)', fontsize=12)
    ax.set_title('Business Impact of Budget Optimization\n(Same Total Marketing Spend)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{cat}\n{val}' for cat, val in zip(categories, p1_vals)], fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(changes_pct) * 1.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'Business_Impact_homepage.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {OUTPUT_DIR}/Business_Impact_homepage.png")


def save_impact(impact):
    """Save impact results to JSON."""
    with open(OUTPUT_DIR / 'business_impact.json', 'w') as f:
        json.dump(impact, f, indent=2)
    
    print(f"Saved: {OUTPUT_DIR}/business_impact.json")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("BUSINESS IMPACT CALCULATION")
    print("=" * 60)
    
    # Load optimization results
    opt_results = load_optimization_results()
    
    # Calculate impact
    impact = calculate_business_impact(opt_results)
    
    # Print report
    print_impact_report(impact)
    
    # Create visualizations
    create_impact_visualization(impact)
    
    # Save results
    save_impact(impact)
    
    print("\n" + "=" * 60)
    print("BUSINESS IMPACT CALCULATION COMPLETE")
    print("=" * 60)
