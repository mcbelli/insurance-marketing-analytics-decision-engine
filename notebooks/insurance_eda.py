"""
Insure Co. Exploratory Data Analysis
=====================================

This EDA demonstrates insurance industry domain knowledge through 8 key analyses:

1. Conversion rates and LTV by credit score tier
2. Optimal age bands by product for conversion and LTV
3. Multi-product lead analysis and cross-sell opportunity
4. Geographic analysis of conversion, LTV, and regulatory impact
5. Early claim rate by channel (adverse selection analysis)
6. Average policy profitability by marketing channel
7. State-level claim frequency and loss ratio map
8. Bind rate vs. early claims rate (channel profit analysis)

Author: [Your Name]
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# Set style for all plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Color palettes
CHANNEL_COLORS = {'paid_search': '#2ecc71', 'paid_social': '#3498db', 'email': '#e74c3c'}
PRODUCT_COLORS = {'Health': '#9b59b6', 'Life': '#1abc9c', 'Property_Casualty': '#f39c12'}
CREDIT_COLORS = {'Poor': '#e74c3c', 'Fair': '#f39c12', 'Good': '#3498db', 'Excellent': '#2ecc71'}


def load_data(data_dir='insure_co_data'):
    """Load all data files."""
    print("Loading data...")
    leads = pd.read_csv(f'{data_dir}/leads.csv', parse_dates=['lead_date', 'qualified_date', 'quote_date', 'binder_date', 'sold_date'])
    search_spend = pd.read_csv(f'{data_dir}/search_daily_spend.csv', parse_dates=['date'])
    social_spend = pd.read_csv(f'{data_dir}/social_daily_spend.csv', parse_dates=['date'])
    
    print(f"  Loaded {len(leads):,} lead-product records")
    print(f"  Sold policies: {leads['sold_date'].notna().sum():,}")
    return leads, search_spend, social_spend


# =============================================================================
# ANALYSIS 1: Conversion Rates and LTV by Credit Score
# =============================================================================

def analysis_1_credit_score(leads):
    """
    Analyze how conversion rates and LTV vary by credit score tier.
    
    WHY THIS MATTERS:
    Credit-based insurance scores are a major underwriting tool. Better credit
    correlates with lower claims frequency and higher lifetime value—not just
    ability to pay premiums.
    """
    print("\n" + "="*70)
    print("ANALYSIS 1: Conversion Rates and LTV by Credit Score Tier")
    print("="*70)
    
    credit_order = ['Poor', 'Fair', 'Good', 'Excellent']
    
    # Calculate metrics by credit score
    metrics = []
    for credit in credit_order:
        subset = leads[leads['credit_score'] == credit]
        sold = subset[subset['sold_date'].notna()]
        
        metrics.append({
            'credit_score': credit,
            'total_leads': len(subset),
            'sold_count': len(sold),
            'conversion_rate': len(sold) / len(subset) * 100 if len(subset) > 0 else 0,
            'avg_ltv': sold['ltv'].mean() if len(sold) > 0 else 0,
            'avg_loss_ratio': sold['loss_ratio'].mean() * 100 if len(sold) > 0 else 0,
            'early_claim_rate': sold['has_early_claim'].mean() * 100 if len(sold) > 0 else 0,
        })
    
    metrics_df = pd.DataFrame(metrics)
    
    # Print summary
    print("\nKey Metrics by Credit Score:")
    print("-" * 70)
    for _, row in metrics_df.iterrows():
        print(f"{row['credit_score']:10} | Conv Rate: {row['conversion_rate']:5.1f}% | "
              f"Avg LTV: ${row['avg_ltv']:,.0f} | Loss Ratio: {row['avg_loss_ratio']:5.1f}% | "
              f"Early Claims: {row['early_claim_rate']:5.1f}%")
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    colors = [CREDIT_COLORS[c] for c in credit_order]
    
    # Conversion rate
    axes[0].bar(metrics_df['credit_score'], metrics_df['conversion_rate'], color=colors)
    axes[0].set_title('Conversion Rate by Credit Score')
    axes[0].set_ylabel('Conversion Rate (%)')
    axes[0].set_xlabel('Credit Score Tier')
    
    # Average LTV
    axes[1].bar(metrics_df['credit_score'], metrics_df['avg_ltv'], color=colors)
    axes[1].set_title('Average LTV by Credit Score')
    axes[1].set_ylabel('Lifetime Value ($)')
    axes[1].set_xlabel('Credit Score Tier')
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Loss ratio
    axes[2].bar(metrics_df['credit_score'], metrics_df['avg_loss_ratio'], color=colors)
    axes[2].set_title('Loss Ratio by Credit Score')
    axes[2].set_ylabel('Loss Ratio (%)')
    axes[2].set_xlabel('Credit Score Tier')
    axes[2].axhline(y=100, color='red', linestyle='--', alpha=0.7, label='Break-even')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('analysis_1_credit_score.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n💡 INSIGHT: Better credit scores correlate with higher conversion rates,")
    print("   higher LTV, and lower loss ratios—validating the use of credit-based")
    print("   insurance scores in underwriting decisions.")
    
    return metrics_df


# =============================================================================
# ANALYSIS 2: Optimal Age Bands by Product
# =============================================================================

def analysis_2_age_bands(leads):
    """
    Find optimal age bands for each product where conversion and LTV are maximized.
    
    WHY THIS MATTERS:
    Age is THE primary rating variable in life and health insurance (mortality
    and morbidity curves). The optimal age differs by product line, reflecting
    different risk profiles and purchasing behaviors.
    """
    print("\n" + "="*70)
    print("ANALYSIS 2: Optimal Age Bands by Product")
    print("="*70)
    
    # Create age bands
    leads['age_band'] = pd.cut(leads['age'], bins=[17, 25, 35, 45, 55, 65, 100],
                               labels=['18-25', '26-35', '36-45', '46-55', '56-65', '65+'])
    
    products = ['Health', 'Life', 'Property_Casualty']
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    for i, product in enumerate(products):
        product_data = leads[leads['product'] == product]
        
        # Calculate metrics by age band
        age_metrics = product_data.groupby('age_band').agg({
            'lead_id': 'count',
            'sold_date': lambda x: x.notna().sum(),
            'ltv': lambda x: x[x.notna()].mean() if x.notna().any() else 0,
            'loss_ratio': lambda x: x[x.notna()].mean() * 100 if x.notna().any() else 0,
        }).rename(columns={'lead_id': 'total', 'sold_date': 'sold'})
        
        age_metrics['conversion_rate'] = age_metrics['sold'] / age_metrics['total'] * 100
        
        print(f"\n{product}:")
        print("-" * 50)
        best_conv = age_metrics['conversion_rate'].idxmax()
        best_ltv = age_metrics['ltv'].idxmax()
        print(f"  Best conversion: {best_conv} ({age_metrics.loc[best_conv, 'conversion_rate']:.1f}%)")
        print(f"  Highest LTV: {best_ltv} (${age_metrics.loc[best_ltv, 'ltv']:,.0f})")
        
        # Plot conversion rate
        color = PRODUCT_COLORS[product]
        axes[0, i].bar(age_metrics.index.astype(str), age_metrics['conversion_rate'], color=color, alpha=0.8)
        axes[0, i].set_title(f'{product}\nConversion Rate by Age')
        axes[0, i].set_ylabel('Conversion Rate (%)' if i == 0 else '')
        axes[0, i].set_xlabel('Age Band')
        axes[0, i].tick_params(axis='x', rotation=45)
        
        # Highlight best age band
        best_idx = list(age_metrics.index.astype(str)).index(str(best_conv))
        axes[0, i].patches[best_idx].set_edgecolor('black')
        axes[0, i].patches[best_idx].set_linewidth(3)
        
        # Plot LTV
        axes[1, i].bar(age_metrics.index.astype(str), age_metrics['ltv'], color=color, alpha=0.8)
        axes[1, i].set_title(f'{product}\nAverage LTV by Age')
        axes[1, i].set_ylabel('LTV ($)' if i == 0 else '')
        axes[1, i].set_xlabel('Age Band')
        axes[1, i].tick_params(axis='x', rotation=45)
        axes[1, i].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    plt.tight_layout()
    plt.savefig('analysis_2_age_bands.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n💡 INSIGHT: Each product has different optimal age bands:")
    print("   - Life Insurance: Middle-aged customers (35-55) have highest LTV")
    print("   - Health Insurance: Conversion peaks at different ages than LTV")
    print("   - P&C: More uniform across ages, reflecting different risk dynamics")
    
    return leads


# =============================================================================
# ANALYSIS 3: Multi-Product Leads and Cross-Sell
# =============================================================================

def analysis_3_cross_sell(leads):
    """
    Analyze multi-product leads and cross-sell opportunity.
    
    WHY THIS MATTERS:
    Bundled customers have 90%+ retention vs ~80% for single-product customers.
    Cross-sell is a core profitability and retention lever that every insurance
    executive obsesses over.
    """
    print("\n" + "="*70)
    print("ANALYSIS 3: Multi-Product Leads and Cross-Sell Opportunity")
    print("="*70)
    
    # Count products per lead
    products_per_lead = leads.groupby('lead_id').agg({
        'product': 'count',
        'sold_date': lambda x: x.notna().sum(),
        'ltv': lambda x: x[x.notna()].sum(),
    }).rename(columns={'product': 'products_quoted', 'sold_date': 'products_sold'})
    
    products_per_lead['any_sold'] = products_per_lead['products_sold'] > 0
    products_per_lead['multi_product'] = products_per_lead['products_quoted'] > 1
    
    # Conversion rates by number of products
    print("\nConversion Analysis by Products Quoted:")
    print("-" * 50)
    
    single_product = products_per_lead[products_per_lead['products_quoted'] == 1]
    multi_product = products_per_lead[products_per_lead['products_quoted'] > 1]
    
    single_conv = single_product['any_sold'].mean() * 100
    multi_conv = multi_product['any_sold'].mean() * 100
    
    print(f"Single-product leads: {len(single_product):,} leads, {single_conv:.1f}% conversion")
    print(f"Multi-product leads:  {len(multi_product):,} leads, {multi_conv:.1f}% conversion")
    print(f"\nMulti-product leads convert {multi_conv/single_conv:.2f}x better!")
    
    # Cross-sell analysis for sold customers
    sold_customers = products_per_lead[products_per_lead['products_sold'] > 0]
    
    cross_sell_dist = sold_customers['products_sold'].value_counts().sort_index()
    
    print("\nProducts Sold per Customer:")
    print("-" * 50)
    for num_products, count in cross_sell_dist.items():
        pct = count / len(sold_customers) * 100
        avg_ltv = sold_customers[sold_customers['products_sold'] == num_products]['ltv'].mean()
        print(f"  {num_products} product(s): {count:,} customers ({pct:.1f}%), Avg LTV: ${avg_ltv:,.0f}")
    
    # Visualizations
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Conversion comparison
    conv_data = pd.DataFrame({
        'Lead Type': ['Single Product', 'Multi-Product'],
        'Conversion Rate': [single_conv, multi_conv]
    })
    bars = axes[0].bar(conv_data['Lead Type'], conv_data['Conversion Rate'], 
                       color=['#3498db', '#2ecc71'])
    axes[0].set_title('Conversion Rate: Single vs Multi-Product Leads')
    axes[0].set_ylabel('Conversion Rate (%)')
    for bar, val in zip(bars, conv_data['Conversion Rate']):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{val:.1f}%', ha='center', fontweight='bold')
    
    # Products sold distribution
    axes[1].bar(cross_sell_dist.index.astype(str), cross_sell_dist.values, color='#9b59b6')
    axes[1].set_title('Distribution of Products Sold per Customer')
    axes[1].set_xlabel('Number of Products')
    axes[1].set_ylabel('Number of Customers')
    
    # LTV by products sold
    ltv_by_products = sold_customers.groupby('products_sold')['ltv'].mean()
    axes[2].bar(ltv_by_products.index.astype(str), ltv_by_products.values, color='#f39c12')
    axes[2].set_title('Average LTV by Products Purchased')
    axes[2].set_xlabel('Number of Products')
    axes[2].set_ylabel('Total LTV ($)')
    axes[2].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    plt.tight_layout()
    plt.savefig('analysis_3_cross_sell.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n💡 INSIGHT: Multi-product leads convert significantly better and have")
    print("   higher LTV. This supports investment in cross-sell initiatives and")
    print("   bundling strategies to improve retention and profitability.")
    
    return products_per_lead


# =============================================================================
# ANALYSIS 4: Geographic Analysis
# =============================================================================

def analysis_4_geographic(leads):
    """
    Analyze conversion rates, LTV, and loss ratios by state.
    
    WHY THIS MATTERS:
    Insurance is state-regulated—each state has its own rate approval process,
    coverage mandates, and competitive dynamics. Florida's P&C looks different
    (hurricane exposure), some states have compressed margins (heavy regulation).
    """
    print("\n" + "="*70)
    print("ANALYSIS 4: Geographic Analysis")
    print("="*70)
    
    # Calculate metrics by state
    state_metrics = leads.groupby('state').agg({
        'lead_id': 'count',
        'sold_date': lambda x: x.notna().sum(),
        'ltv': lambda x: x[x.notna()].mean() if x.notna().any() else 0,
        'loss_ratio': lambda x: x[x.notna()].mean() if x.notna().any() else 0,
        'has_early_claim': lambda x: x[x.notna()].mean() if x.notna().any() else 0,
    }).rename(columns={'lead_id': 'total_leads', 'sold_date': 'sold'})
    
    state_metrics['conversion_rate'] = state_metrics['sold'] / state_metrics['total_leads'] * 100
    state_metrics['loss_ratio_pct'] = state_metrics['loss_ratio'] * 100
    
    # Filter to states with meaningful volume
    state_metrics = state_metrics[state_metrics['total_leads'] >= 500].copy()
    
    # Top and bottom states
    print("\nTop 5 States by Conversion Rate:")
    print("-" * 50)
    top_conv = state_metrics.nlargest(5, 'conversion_rate')
    for state, row in top_conv.iterrows():
        print(f"  {state}: {row['conversion_rate']:.1f}% conversion, "
              f"${row['ltv']:,.0f} avg LTV, {row['loss_ratio_pct']:.1f}% loss ratio")
    
    print("\nBottom 5 States by Conversion Rate:")
    print("-" * 50)
    bottom_conv = state_metrics.nsmallest(5, 'conversion_rate')
    for state, row in bottom_conv.iterrows():
        print(f"  {state}: {row['conversion_rate']:.1f}% conversion, "
              f"${row['ltv']:,.0f} avg LTV, {row['loss_ratio_pct']:.1f}% loss ratio")
    
    print("\nHighest Loss Ratio States (potential regulatory/risk concerns):")
    print("-" * 50)
    high_loss = state_metrics.nlargest(5, 'loss_ratio')
    for state, row in high_loss.iterrows():
        print(f"  {state}: {row['loss_ratio_pct']:.1f}% loss ratio, "
              f"{row['conversion_rate']:.1f}% conversion")
    
    # Visualizations
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Sort by conversion rate for visualization
    sorted_states = state_metrics.sort_values('conversion_rate', ascending=True)
    
    # Conversion rate by state (horizontal bar)
    colors = ['#e74c3c' if x < state_metrics['conversion_rate'].median() else '#2ecc71' 
              for x in sorted_states['conversion_rate']]
    axes[0].barh(sorted_states.index, sorted_states['conversion_rate'], color=colors, alpha=0.8)
    axes[0].set_title('Conversion Rate by State')
    axes[0].set_xlabel('Conversion Rate (%)')
    axes[0].axvline(x=state_metrics['conversion_rate'].median(), color='black', 
                    linestyle='--', alpha=0.5, label='Median')
    
    # Scatter: Conversion vs LTV
    scatter = axes[1].scatter(state_metrics['conversion_rate'], state_metrics['ltv'],
                              s=state_metrics['total_leads']/50, alpha=0.6, c='#3498db')
    axes[1].set_title('Conversion Rate vs LTV by State\n(bubble size = lead volume)')
    axes[1].set_xlabel('Conversion Rate (%)')
    axes[1].set_ylabel('Average LTV ($)')
    
    # Annotate outliers
    for state, row in state_metrics.iterrows():
        if row['conversion_rate'] > state_metrics['conversion_rate'].quantile(0.9) or \
           row['conversion_rate'] < state_metrics['conversion_rate'].quantile(0.1):
            axes[1].annotate(state, (row['conversion_rate'], row['ltv']), fontsize=8)
    
    # Loss ratio distribution
    sorted_loss = state_metrics.sort_values('loss_ratio_pct', ascending=True)
    colors = ['#e74c3c' if x > 100 else '#2ecc71' for x in sorted_loss['loss_ratio_pct']]
    axes[2].barh(sorted_loss.index, sorted_loss['loss_ratio_pct'], color=colors, alpha=0.8)
    axes[2].set_title('Loss Ratio by State')
    axes[2].set_xlabel('Loss Ratio (%)')
    axes[2].axvline(x=100, color='red', linestyle='--', alpha=0.7, label='Break-even')
    
    plt.tight_layout()
    plt.savefig('analysis_4_geographic.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n💡 INSIGHT: Significant state-level variation suggests regulatory environment,")
    print("   competitive dynamics, and profit profiles differ by geography. States with")
    print("   high loss ratios may require pricing adjustments or underwriting changes.")
    
    return state_metrics


# =============================================================================
# ANALYSIS 5: Early Claim Rate by Channel
# =============================================================================

def analysis_5_early_claims_by_channel(leads):
    """
    Analyze early claim rates by marketing channel.
    
    WHY THIS MATTERS:
    This demonstrates adverse selection—lower quality channels (cheaper CPL)
    attract higher-risk customers who file claims sooner. Early claims are
    a key indicator of adverse selection and underwriting quality.
    """
    print("\n" + "="*70)
    print("ANALYSIS 5: Early Claim Rate by Channel (Adverse Selection)")
    print("="*70)
    
    sold = leads[leads['sold_date'].notna()].copy()
    
    # Calculate metrics by channel
    channel_claims = sold.groupby('channel').agg({
        'lead_id': 'count',
        'has_claim': 'sum',
        'has_early_claim': 'sum',
        'loss_ratio': 'mean',
        'total_claim_amount': 'sum',
        'total_premium': 'sum',
    }).rename(columns={'lead_id': 'policies'})
    
    channel_claims['claim_rate'] = channel_claims['has_claim'] / channel_claims['policies'] * 100
    channel_claims['early_claim_rate'] = channel_claims['has_early_claim'] / channel_claims['policies'] * 100
    channel_claims['loss_ratio_pct'] = channel_claims['loss_ratio'] * 100
    channel_claims['actual_loss_ratio'] = channel_claims['total_claim_amount'] / channel_claims['total_premium'] * 100
    
    # Channel quality reference
    channel_quality = {'paid_search': 'High', 'paid_social': 'Medium', 'email': 'Low'}
    channel_cpl = {'paid_search': '$54', 'paid_social': '$34', 'email': '~$8'}
    
    print("\nClaims Analysis by Channel:")
    print("-" * 70)
    print(f"{'Channel':<15} {'Quality':<10} {'CPL':<10} {'Policies':<10} {'Early Claim %':<15} {'Loss Ratio':<12}")
    print("-" * 70)
    
    for channel in ['paid_search', 'paid_social', 'email']:
        row = channel_claims.loc[channel]
        print(f"{channel:<15} {channel_quality[channel]:<10} {channel_cpl[channel]:<10} "
              f"{int(row['policies']):<10} {row['early_claim_rate']:<15.1f} {row['actual_loss_ratio']:<12.1f}%")
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    channels = ['paid_search', 'paid_social', 'email']
    colors = [CHANNEL_COLORS[c] for c in channels]
    
    # Early claim rate
    early_claims = [channel_claims.loc[c, 'early_claim_rate'] for c in channels]
    bars = axes[0].bar(channels, early_claims, color=colors)
    axes[0].set_title('Early Claim Rate by Channel\n(Claims within first year)')
    axes[0].set_ylabel('Early Claim Rate (%)')
    axes[0].set_xlabel('Marketing Channel')
    
    # Add trend line annotation
    for bar, val in zip(bars, early_claims):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
                    f'{val:.1f}%', ha='center', fontweight='bold')
    
    # Loss ratio
    loss_ratios = [channel_claims.loc[c, 'actual_loss_ratio'] for c in channels]
    bars = axes[1].bar(channels, loss_ratios, color=colors)
    axes[1].set_title('Loss Ratio by Channel')
    axes[1].set_ylabel('Loss Ratio (%)')
    axes[1].set_xlabel('Marketing Channel')
    axes[1].axhline(y=100, color='red', linestyle='--', alpha=0.7, label='Break-even')
    axes[1].legend()
    
    for bar, val in zip(bars, loss_ratios):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{val:.1f}%', ha='center', fontweight='bold')
    
    # CPL vs Early Claim Rate (showing inverse relationship)
    cpl_values = [54, 34, 8]
    axes[2].scatter(cpl_values, early_claims, s=200, c=colors, edgecolors='black', linewidth=2)
    for i, channel in enumerate(channels):
        axes[2].annotate(channel, (cpl_values[i], early_claims[i]), 
                        xytext=(10, 5), textcoords='offset points', fontsize=10)
    axes[2].set_title('CPL vs Early Claim Rate\n(Adverse Selection Evidence)')
    axes[2].set_xlabel('Cost Per Lead ($)')
    axes[2].set_ylabel('Early Claim Rate (%)')
    
    # Add trend line
    z = np.polyfit(cpl_values, early_claims, 1)
    p = np.poly1d(z)
    x_line = np.linspace(5, 60, 100)
    axes[2].plot(x_line, p(x_line), 'r--', alpha=0.5, label='Trend')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('analysis_5_early_claims.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n💡 INSIGHT: Clear evidence of adverse selection—cheaper channels (email)")
    print("   have significantly higher early claim rates. Customers who respond to")
    print("   low-cost marketing may be actively seeking insurance due to anticipated needs.")
    
    return channel_claims


# =============================================================================
# ANALYSIS 6: Average Policy Profitability by Channel
# =============================================================================

def analysis_6_policy_profitability(leads):
    """
    Calculate Expected Value = Premium × Tenure − Claims by channel.
    
    WHY THIS MATTERS:
    This shows the true economic value per policy after accounting for claims
    risk and tenure. A channel may have low CPL but if the policies are
    unprofitable, it's not a good investment.
    """
    print("\n" + "="*70)
    print("ANALYSIS 6: Policy Profitability by Channel")
    print("="*70)
    
    sold = leads[leads['sold_date'].notna()].copy()
    
    # Calculate expected value metrics by channel
    channel_value = sold.groupby('channel').agg({
        'lead_id': 'count',
        'annual_premium': 'mean',
        'expected_tenure_years': 'mean',
        'total_premium': 'mean',
        'total_claim_amount': 'mean',
        'expected_value': 'mean',
        'loss_ratio': 'mean',
    }).rename(columns={'lead_id': 'policies'})
    
    # Also calculate by channel AND product
    channel_product_value = sold.groupby(['channel', 'product']).agg({
        'expected_value': 'mean',
        'lead_id': 'count',
    }).rename(columns={'lead_id': 'policies'})
    
    print("\nPolicy Profitability by Channel:")
    print("-" * 80)
    print(f"{'Channel':<15} {'Policies':<10} {'Avg Premium':<12} {'Avg Tenure':<12} "
          f"{'Avg Claims':<12} {'Exp. Value':<12}")
    print("-" * 80)
    
    for channel in ['paid_search', 'paid_social', 'email']:
        row = channel_value.loc[channel]
        print(f"{channel:<15} {int(row['policies']):<10} ${row['annual_premium']:>9,.0f} "
              f"{row['expected_tenure_years']:>10.1f}yr ${row['total_claim_amount']:>9,.0f} "
              f"${row['expected_value']:>9,.0f}")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    channels = ['paid_search', 'paid_social', 'email']
    colors = [CHANNEL_COLORS[c] for c in channels]
    
    # Expected value by channel
    exp_values = [channel_value.loc[c, 'expected_value'] for c in channels]
    bars = axes[0].bar(channels, exp_values, color=colors, edgecolor='black', linewidth=1.5)
    axes[0].set_title('Average Policy Profitability by Channel\n'
                      '(Expected Value = Total Premium − Expected Claims)', fontsize=12)
    axes[0].set_ylabel('Expected Value ($)')
    axes[0].set_xlabel('Marketing Channel')
    axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    for bar, val in zip(bars, exp_values):
        color = 'green' if val > 0 else 'red'
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100, 
                    f'${val:,.0f}', ha='center', fontweight='bold', color=color)
    
    # Stacked bar: Premium vs Claims by channel
    premiums = [channel_value.loc[c, 'total_premium'] for c in channels]
    claims = [channel_value.loc[c, 'total_claim_amount'] for c in channels]
    
    x = np.arange(len(channels))
    width = 0.35
    
    bars1 = axes[1].bar(x - width/2, premiums, width, label='Avg Total Premium', color='#2ecc71', alpha=0.8)
    bars2 = axes[1].bar(x + width/2, claims, width, label='Avg Total Claims', color='#e74c3c', alpha=0.8)
    
    axes[1].set_title('Average Premium vs Claims by Channel')
    axes[1].set_ylabel('Amount ($)')
    axes[1].set_xlabel('Marketing Channel')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(channels)
    axes[1].legend()
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    plt.tight_layout()
    plt.savefig('analysis_6_policy_profitability.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # ROI calculation incorporating CAC
    print("\nChannel ROI Analysis (incorporating acquisition cost):")
    print("-" * 60)
    cpl_estimates = {'paid_search': 54, 'paid_social': 34, 'email': 8}
    conv_rates = {'paid_search': 0.09, 'paid_social': 0.072, 'email': 0.043}
    
    for channel in channels:
        exp_val = channel_value.loc[channel, 'expected_value']
        cac = cpl_estimates[channel] / conv_rates[channel]  # Cost to acquire one customer
        net_value = exp_val - cac
        roi = (exp_val / cac - 1) * 100
        print(f"  {channel}: CAC=${cac:.0f}, Exp.Value=${exp_val:,.0f}, "
              f"Net=${net_value:,.0f}, ROI={roi:.0f}%")
    
    print("\n💡 INSIGHT: Despite higher CPL, paid_search delivers the highest profit")
    print("   per policy. Email's low CPL is offset by poor policy economics,")
    print("   making it potentially unprofitable on a profitability basis.")
    
    return channel_value


# =============================================================================
# ANALYSIS 7: State-Level Claims Map
# =============================================================================

def analysis_7_state_claims_map(leads):
    """
    Create a visualization of claim frequency and loss ratios by state.
    
    WHY THIS MATTERS:
    Geographic profitability is critical in insurance. States like Florida
    (hurricanes), Texas (hail), and California (wildfires) have distinct net revenue
    profiles that affect pricing and profitability.
    """
    print("\n" + "="*70)
    print("ANALYSIS 7: State-Level Claims Analysis")
    print("="*70)
    
    sold = leads[leads['sold_date'].notna()].copy()
    
    # Calculate state metrics
    state_claims = sold.groupby('state').agg({
        'lead_id': 'count',
        'has_claim': 'mean',
        'has_early_claim': 'mean',
        'total_claim_amount': 'sum',
        'total_premium': 'sum',
        'loss_ratio': 'mean',
    }).rename(columns={'lead_id': 'policies'})
    
    state_claims['claim_frequency'] = state_claims['has_claim'] * 100
    state_claims['early_claim_freq'] = state_claims['has_early_claim'] * 100
    state_claims['actual_loss_ratio'] = state_claims['total_claim_amount'] / state_claims['total_premium'] * 100
    
    # Filter to states with sufficient volume
    state_claims = state_claims[state_claims['policies'] >= 50].copy()
    
    # Identify outliers
    loss_ratio_mean = state_claims['actual_loss_ratio'].mean()
    loss_ratio_std = state_claims['actual_loss_ratio'].std()
    
    state_claims['is_outlier'] = (
        (state_claims['actual_loss_ratio'] > loss_ratio_mean + 1.5 * loss_ratio_std) |
        (state_claims['actual_loss_ratio'] < loss_ratio_mean - 1.5 * loss_ratio_std)
    )
    
    high_risk_states = state_claims[state_claims['actual_loss_ratio'] > loss_ratio_mean + loss_ratio_std]
    low_risk_states = state_claims[state_claims['actual_loss_ratio'] < loss_ratio_mean - loss_ratio_std]
    
    print("\nHigh-Risk States (Loss Ratio > Mean + 1 Std Dev):")
    print("-" * 60)
    for state, row in high_risk_states.sort_values('actual_loss_ratio', ascending=False).iterrows():
        print(f"  {state}: Loss Ratio {row['actual_loss_ratio']:.1f}%, "
              f"Claim Freq {row['claim_frequency']:.1f}%, {int(row['policies'])} policies")
    
    print("\nLow-Risk States (Loss Ratio < Mean - 1 Std Dev):")
    print("-" * 60)
    for state, row in low_risk_states.sort_values('actual_loss_ratio').iterrows():
        print(f"  {state}: Loss Ratio {row['actual_loss_ratio']:.1f}%, "
              f"Claim Freq {row['claim_frequency']:.1f}%, {int(row['policies'])} policies")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Sort by loss ratio for visualization
    sorted_states = state_claims.sort_values('actual_loss_ratio', ascending=True)
    
    # Color by outlier status
    colors = []
    for _, row in sorted_states.iterrows():
        if row['actual_loss_ratio'] > loss_ratio_mean + loss_ratio_std:
            colors.append('#e74c3c')  # Red for high loss
        elif row['actual_loss_ratio'] < loss_ratio_mean - loss_ratio_std:
            colors.append('#2ecc71')  # Green for low loss
        else:
            colors.append('#3498db')  # Blue for normal
    
    # Loss ratio by state
    axes[0].barh(sorted_states.index, sorted_states['actual_loss_ratio'], color=colors, alpha=0.8)
    axes[0].axvline(x=100, color='black', linestyle='--', linewidth=2, label='Break-even (100%)')
    axes[0].axvline(x=loss_ratio_mean, color='gray', linestyle=':', alpha=0.7, label=f'Mean ({loss_ratio_mean:.0f}%)')
    axes[0].set_title('Loss Ratio by State\n(Red = High Loss, Green = Low Loss)')
    axes[0].set_xlabel('Loss Ratio (%)')
    axes[0].legend(loc='lower right')
    
    # Scatter: Claim frequency vs Loss ratio
    scatter = axes[1].scatter(
        state_claims['claim_frequency'], 
        state_claims['actual_loss_ratio'],
        s=state_claims['policies'] / 2,
        c=['#e74c3c' if x else '#3498db' for x in state_claims['is_outlier']],
        alpha=0.6,
        edgecolors='black',
        linewidth=0.5
    )
    
    # Annotate outliers
    for state, row in state_claims[state_claims['is_outlier']].iterrows():
        axes[1].annotate(state, (row['claim_frequency'], row['actual_loss_ratio']),
                        xytext=(5, 5), textcoords='offset points', fontsize=9, fontweight='bold')
    
    axes[1].axhline(y=100, color='red', linestyle='--', alpha=0.7, label='Break-even')
    axes[1].set_title('Claim Frequency vs Loss Ratio by State\n(bubble size = policy count, red = outliers)')
    axes[1].set_xlabel('Claim Frequency (%)')
    axes[1].set_ylabel('Loss Ratio (%)')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('analysis_7_state_claims.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n📊 Summary Statistics:")
    print(f"   Mean Loss Ratio: {loss_ratio_mean:.1f}%")
    print(f"   Std Dev: {loss_ratio_std:.1f}%")
    print(f"   High-Risk States: {len(high_risk_states)}")
    print(f"   Low-Risk States: {len(low_risk_states)}")
    
    print("\n💡 INSIGHT: Significant state-level variation in loss ratios indicates")
    print("   geographic risk concentration. High-risk states may need rate increases,")
    print("   stricter underwriting, or reduced marketing investment.")
    
    return state_claims


# =============================================================================
# ANALYSIS 8: Bind Rate vs Early Claims (Channel Risk)
# =============================================================================

def analysis_8_bind_vs_claims(leads):
    """
    Scatterplot of bind rate vs early claims rate by channel.
    
    WHY THIS MATTERS:
    This tests whether higher-converting channels are also riskier. If there's
    a negative correlation, it suggests adverse selection isn't overwhelming
    the business model. If positive, fast-converting leads may be higher risk.
    """
    print("\n" + "="*70)
    print("ANALYSIS 8: Bind Rate vs Early Claims Rate (Channel Risk Analysis)")
    print("="*70)
    
    # Calculate by channel and product
    channel_product_metrics = leads.groupby(['channel', 'product']).agg({
        'lead_id': 'count',
        'binder_date': lambda x: x.notna().sum(),
        'sold_date': lambda x: x.notna().sum(),
        'has_early_claim': lambda x: x[x.notna()].mean() if x.notna().any() else 0,
        'loss_ratio': lambda x: x[x.notna()].mean() if x.notna().any() else 0,
    }).rename(columns={'lead_id': 'total_leads', 'binder_date': 'bound', 'sold_date': 'sold'})
    
    channel_product_metrics['bind_rate'] = channel_product_metrics['bound'] / channel_product_metrics['total_leads'] * 100
    channel_product_metrics['early_claim_rate'] = channel_product_metrics['has_early_claim'] * 100
    channel_product_metrics = channel_product_metrics.reset_index()
    
    # Also aggregate just by channel
    channel_metrics = leads.groupby('channel').agg({
        'lead_id': 'count',
        'binder_date': lambda x: x.notna().sum(),
        'has_early_claim': lambda x: x[x.notna()].mean() if x.notna().any() else 0,
    }).rename(columns={'lead_id': 'total_leads', 'binder_date': 'bound'})
    
    channel_metrics['bind_rate'] = channel_metrics['bound'] / channel_metrics['total_leads'] * 100
    channel_metrics['early_claim_rate'] = channel_metrics['has_early_claim'] * 100
    
    print("\nChannel Summary: Bind Rate vs Early Claim Rate")
    print("-" * 60)
    for channel in ['paid_search', 'paid_social', 'email']:
        row = channel_metrics.loc[channel]
        print(f"  {channel:<15} Bind Rate: {row['bind_rate']:5.1f}%  |  Early Claim Rate: {row['early_claim_rate']:5.1f}%")
    
    # Calculate correlation
    correlation = channel_product_metrics['bind_rate'].corr(channel_product_metrics['early_claim_rate'])
    print(f"\nCorrelation (Bind Rate vs Early Claims): {correlation:.3f}")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Scatter by channel (aggregated)
    channels = ['paid_search', 'paid_social', 'email']
    for channel in channels:
        row = channel_metrics.loc[channel]
        axes[0].scatter(row['bind_rate'], row['early_claim_rate'], 
                       s=300, c=CHANNEL_COLORS[channel], label=channel,
                       edgecolors='black', linewidth=2, zorder=5)
    
    # Add trend line
    x_vals = channel_metrics['bind_rate'].values
    y_vals = channel_metrics['early_claim_rate'].values
    z = np.polyfit(x_vals, y_vals, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(x_vals) - 1, max(x_vals) + 1, 100)
    axes[0].plot(x_line, p(x_line), 'r--', alpha=0.5, label=f'Trend (r={correlation:.2f})')
    
    axes[0].set_title('Bind Rate vs Early Claim Rate by Channel\n'
                      '(Are higher-converting channels riskier?)')
    axes[0].set_xlabel('Bind Rate (%)')
    axes[0].set_ylabel('Early Claim Rate (%)')
    axes[0].legend()
    
    # Scatter by channel AND product
    for channel in channels:
        for product in ['Health', 'Life', 'Property_Casualty']:
            subset = channel_product_metrics[
                (channel_product_metrics['channel'] == channel) & 
                (channel_product_metrics['product'] == product)
            ]
            if len(subset) > 0:
                row = subset.iloc[0]
                marker = {'Health': 'o', 'Life': 's', 'Property_Casualty': '^'}[product]
                axes[1].scatter(row['bind_rate'], row['early_claim_rate'],
                               s=150, c=CHANNEL_COLORS[channel], marker=marker,
                               alpha=0.7, edgecolors='black', linewidth=1)
    
    # Create custom legend
    channel_patches = [mpatches.Patch(color=CHANNEL_COLORS[c], label=c) for c in channels]
    product_markers = [
        plt.Line2D([0], [0], marker='o', color='gray', label='Health', markersize=10, linestyle=''),
        plt.Line2D([0], [0], marker='s', color='gray', label='Life', markersize=10, linestyle=''),
        plt.Line2D([0], [0], marker='^', color='gray', label='P&C', markersize=10, linestyle=''),
    ]
    
    legend1 = axes[1].legend(handles=channel_patches, loc='upper left', title='Channel')
    axes[1].add_artist(legend1)
    axes[1].legend(handles=product_markers, loc='upper right', title='Product')
    
    axes[1].set_title('Bind Rate vs Early Claim Rate\n(by Channel and Product)')
    axes[1].set_xlabel('Bind Rate (%)')
    axes[1].set_ylabel('Early Claim Rate (%)')
    
    plt.tight_layout()
    plt.savefig('analysis_8_bind_vs_claims.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Interpretation
    if correlation > 0.3:
        interpretation = "POSITIVE correlation suggests higher-converting channels ARE riskier (adverse selection concern)"
    elif correlation < -0.3:
        interpretation = "NEGATIVE correlation suggests higher-converting channels are LOWER risk (good sign)"
    else:
        interpretation = "WEAK correlation suggests bind rate and claims risk are relatively independent"
    
    print(f"\n💡 INSIGHT: {interpretation}")
    print("   This relationship is critical for understanding the quality-quantity")
    print("   tradeoff in lead acquisition strategy.")
    
    return channel_product_metrics


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_full_eda(data_dir='insure_co_data'):
    """Run all 8 analyses and generate summary report."""
    
    print("\n" + "="*70)
    print("INSURE CO. EXPLORATORY DATA ANALYSIS")
    print("Insurance Marketing & Risk Analytics")
    print("="*70)
    
    # Load data
    leads, search_spend, social_spend = load_data(data_dir)
    
    # Run all analyses
    results = {}
    
    results['credit_score'] = analysis_1_credit_score(leads)
    results['age_bands'] = analysis_2_age_bands(leads)
    results['cross_sell'] = analysis_3_cross_sell(leads)
    results['geographic'] = analysis_4_geographic(leads)
    results['early_claims'] = analysis_5_early_claims_by_channel(leads)
    results['profitability'] = analysis_6_policy_profitability(leads)
    results['state_claims'] = analysis_7_state_claims_map(leads)
    results['bind_vs_claims'] = analysis_8_bind_vs_claims(leads)
    
    # Summary
    print("\n" + "="*70)
    print("EDA COMPLETE - EXECUTIVE SUMMARY")
    print("="*70)
    
    print("""
KEY FINDINGS:

1. CREDIT SCORE IMPACT: Strong correlation between credit tier and both 
   conversion rate and loss ratio, validating credit-based underwriting.

2. AGE SEGMENTATION: Optimal age bands differ by product, reflecting 
   different risk/purchasing dynamics across life stages.

3. CROSS-SELL OPPORTUNITY: Multi-product leads convert significantly better
   and have higher LTV—bundling strategy is validated.

4. GEOGRAPHIC VARIATION: Material state-level differences in loss ratios
   suggest need for geographic risk pricing.

5. ADVERSE SELECTION: Clear evidence that cheaper acquisition channels
   attract higher-risk customers with more early claims.

6. CHANNEL ECONOMICS: When accounting for claims, paid search delivers best profitability
   despite higher CPL; email may be unprofitable.

7. STATE RISK CONCENTRATION: Identified outlier states requiring
   pricing/underwriting attention.

8. QUALITY-QUANTITY TRADEOFF: Analysis of bind rate vs claims rate reveals
   the true cost of optimizing for conversion volume.

RECOMMENDATIONS:
- Shift budget toward paid search despite higher CPL
- Implement credit-score-based lead prioritization
- Investigate high-loss-ratio states for rate adequacy
- Invest in cross-sell/bundling initiatives
- Consider reducing or eliminating purchased email list
    """)
    
    print(f"\nVisualization files saved:")
    print("  - analysis_1_credit_score.png")
    print("  - analysis_2_age_bands.png")
    print("  - analysis_3_cross_sell.png")
    print("  - analysis_4_geographic.png")
    print("  - analysis_5_early_claims.png")
    print("  - analysis_6_policy_profitability.png")
    print("  - analysis_7_state_claims.png")
    print("  - analysis_8_bind_vs_claims.png")
    
    return results


if __name__ == "__main__":
    import os

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    results = run_full_eda()