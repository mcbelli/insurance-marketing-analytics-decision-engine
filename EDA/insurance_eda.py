"""
Insurance Marketing EDA
=======================
Eight exploratory analyses of insurance marketing data.

Produces visualizations saved as EDA_*.png in the EDA folder.

Author: Michael Belli
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / 'insure_co_data'
OUTPUT_DIR = BASE_DIR

# Consistent color palette
CHANNEL_COLORS = {'paid_search': '#2E86AB', 'paid_social': '#A23B72', 'email': '#F18F01'}
CHANNEL_LABELS = {'paid_search': 'Paid Search', 'paid_social': 'Paid Social', 'email': 'Email'}
PRODUCT_COLORS = {'Health': '#4CAF50', 'Life': '#2196F3', 'Property_Casualty': '#FF9800'}
CREDIT_ORDER = ['Poor', 'Fair', 'Good', 'Excellent']
CREDIT_COLORS = ['#E57373', '#FFB74D', '#81C784', '#4CAF50']


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(data_dir=None):
    """Load all data files."""
    if data_dir is None:
        data_dir = DATA_DIR
    
    print("Loading data...")
    leads = pd.read_csv(
        data_dir / 'leads.csv',
        parse_dates=['lead_date', 'qualified_date', 'quote_date', 'binder_date', 'sold_date']
    )
    search_spend = pd.read_csv(data_dir / 'search_daily_spend.csv', parse_dates=['date'])
    social_spend = pd.read_csv(data_dir / 'social_daily_spend.csv', parse_dates=['date'])
    
    print(f"  Loaded {len(leads):,} lead-product records")
    print(f"  Sold policies: {leads['sold_date'].notna().sum():,}")
    return leads, search_spend, social_spend


# =============================================================================
# ANALYSIS 1: Credit Score Impact
# =============================================================================

def analysis_1_credit_score(leads):
    """Credit score impact on conversion and loss ratio."""
    print("\n[1/8] Credit Score Impact...")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Conversion rate by credit score
    ax = axes[0]
    credit_conv = leads.groupby('credit_score').agg(
        total=('lead_id', 'count'),
        sold=('sold_date', lambda x: x.notna().sum())
    )
    credit_conv = credit_conv.reindex(CREDIT_ORDER)
    credit_conv['conversion_rate'] = credit_conv['sold'] / credit_conv['total'] * 100
    
    bars = ax.bar(CREDIT_ORDER, credit_conv['conversion_rate'], color=CREDIT_COLORS, alpha=0.8)
    for bar, val in zip(bars, credit_conv['conversion_rate']):
        ax.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=10)
    ax.set_xlabel('Credit Score', fontsize=11)
    ax.set_ylabel('Conversion Rate (%)', fontsize=11)
    ax.set_title('Conversion Rate by Credit Score', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Right: Loss ratio by credit score
    ax = axes[1]
    sold = leads[leads['sold_date'].notna()].copy()
    credit_loss = sold.groupby('credit_score').agg(
        total_premium=('total_premium', 'sum'),
        total_claims=('total_claim_amount', 'sum')
    )
    credit_loss = credit_loss.reindex(CREDIT_ORDER)
    credit_loss['loss_ratio'] = credit_loss['total_claims'] / credit_loss['total_premium'] * 100
    
    bars = ax.bar(CREDIT_ORDER, credit_loss['loss_ratio'], color=CREDIT_COLORS, alpha=0.8)
    for bar, val in zip(bars, credit_loss['loss_ratio']):
        ax.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=10)
    ax.set_xlabel('Credit Score', fontsize=11)
    ax.set_ylabel('Loss Ratio (%)', fontsize=11)
    ax.set_title('Loss Ratio by Credit Score', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'EDA_credit_score.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved EDA_credit_score.png")


# =============================================================================
# ANALYSIS 2: Age Bands by Product
# =============================================================================

def analysis_2_age_bands(leads):
    """Optimal age bands by product."""
    print("\n[2/8] Age Bands by Product...")
    
    sold = leads[leads['sold_date'].notna()].copy()
    
    # Create age bands
    bins = [18, 25, 35, 45, 55, 65, 80]
    labels = ['18-25', '26-35', '36-45', '46-55', '56-65', '66+']
    sold['age_band'] = pd.cut(sold['age'], bins=bins, labels=labels, right=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: LTV by age band and product
    ax = axes[0]
    pivot = sold.groupby(['age_band', 'product'])['ltv'].mean().unstack()
    pivot.plot(kind='bar', ax=ax, color=[PRODUCT_COLORS[p] for p in pivot.columns], alpha=0.8)
    ax.set_xlabel('Age Band', fontsize=11)
    ax.set_ylabel('Average LTV ($)', fontsize=11)
    ax.set_title('Average LTV by Age Band and Product', fontweight='bold')
    ax.legend(title='Product')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticklabels(labels, rotation=0)
    
    # Right: Conversion rate by age band and product
    ax = axes[1]
    conv_data = leads.copy()
    conv_data['age_band'] = pd.cut(conv_data['age'], bins=bins, labels=labels, right=True)
    conv_data['sold'] = conv_data['sold_date'].notna().astype(int)
    
    pivot_conv = conv_data.groupby(['age_band', 'product'])['sold'].mean().unstack() * 100
    pivot_conv.plot(kind='bar', ax=ax, color=[PRODUCT_COLORS[p] for p in pivot_conv.columns], alpha=0.8)
    ax.set_xlabel('Age Band', fontsize=11)
    ax.set_ylabel('Conversion Rate (%)', fontsize=11)
    ax.set_title('Conversion Rate by Age Band and Product', fontweight='bold')
    ax.legend(title='Product')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticklabels(labels, rotation=0)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'EDA_age_bands.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved EDA_age_bands.png")


# =============================================================================
# ANALYSIS 3: Cross-Sell / Multi-Product
# =============================================================================

def analysis_3_cross_sell(leads):
    """Cross-sell and multi-product opportunity."""
    print("\n[3/8] Cross-Sell Opportunity...")
    
    # Count products per lead
    lead_products = leads.groupby('lead_id').agg(
        num_products=('product', 'nunique'),
        num_sold=('sold_date', lambda x: x.notna().sum()),
        total_ltv=('ltv', 'sum'),
        any_sold=('sold_date', lambda x: x.notna().any())
    )
    lead_products['product_category'] = lead_products['num_products'].map(
        {1: 'Single Product', 2: 'Two Products', 3: 'Three Products'}
    )
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Left: Conversion rate by number of products interested
    ax = axes[0]
    conv_by_prods = lead_products.groupby('product_category')['any_sold'].mean() * 100
    conv_by_prods = conv_by_prods.reindex(['Single Product', 'Two Products', 'Three Products'])
    bars = ax.bar(conv_by_prods.index, conv_by_prods.values, 
                  color=['#90CAF9', '#42A5F5', '#1565C0'], alpha=0.8)
    for bar, val in zip(bars, conv_by_prods.values):
        ax.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=10)
    ax.set_ylabel('Conversion Rate (%)', fontsize=11)
    ax.set_title('Conversion Rate by\nProducts Interested', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Center: Average LTV for converted leads
    ax = axes[1]
    sold_leads = lead_products[lead_products['any_sold']]
    ltv_by_prods = sold_leads.groupby('product_category')['total_ltv'].mean()
    ltv_by_prods = ltv_by_prods.reindex(['Single Product', 'Two Products', 'Three Products'])
    bars = ax.bar(ltv_by_prods.index, ltv_by_prods.values, 
                  color=['#90CAF9', '#42A5F5', '#1565C0'], alpha=0.8)
    for bar, val in zip(bars, ltv_by_prods.values):
        ax.annotate(f'${val:,.0f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=10)
    ax.set_ylabel('Average Total LTV ($)', fontsize=11)
    ax.set_title('Average LTV by\nProducts Interested', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Right: Distribution of products per lead
    ax = axes[2]
    prod_dist = lead_products['product_category'].value_counts().reindex(
        ['Single Product', 'Two Products', 'Three Products'])
    ax.pie(prod_dist.values, labels=prod_dist.index, autopct='%1.1f%%',
           colors=['#90CAF9', '#42A5F5', '#1565C0'], startangle=90)
    ax.set_title('Distribution of\nProducts per Lead', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'EDA_cross_sell.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved EDA_cross_sell.png")


# =============================================================================
# ANALYSIS 4: Geographic Variation
# =============================================================================

def analysis_4_geographic(leads):
    """Geographic performance variation."""
    print("\n[4/8] Geographic Variation...")
    
    sold = leads[leads['sold_date'].notna()].copy()
    
    # Aggregate by state
    state_stats = sold.groupby('state').agg(
        policies=('lead_id', 'count'),
        total_premium=('total_premium', 'sum'),
        total_claims=('total_claim_amount', 'sum'),
        avg_ltv=('ltv', 'mean')
    )
    state_stats['loss_ratio'] = state_stats['total_claims'] / state_stats['total_premium']
    
    # Filter to states with enough data
    state_stats = state_stats[state_stats['policies'] >= 20].sort_values('loss_ratio', ascending=False)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Loss ratio by state (top 25)
    ax = axes[0]
    top_states = state_stats.head(25)
    colors = ['#E57373' if lr > state_stats['loss_ratio'].median() else '#81C784' 
              for lr in top_states['loss_ratio']]
    ax.barh(top_states.index, top_states['loss_ratio'] * 100, color=colors, alpha=0.8)
    ax.axvline(x=state_stats['loss_ratio'].median() * 100, color='black', linestyle='--', 
               linewidth=1, label=f'Median: {state_stats["loss_ratio"].median()*100:.1f}%')
    ax.set_xlabel('Loss Ratio (%)', fontsize=11)
    ax.set_ylabel('State', fontsize=11)
    ax.set_title('Loss Ratio by State\n(Highest 25 States)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()
    
    # Right: LTV by state (top 25 by volume)
    ax = axes[1]
    top_vol = state_stats.sort_values('policies', ascending=False).head(25)
    top_vol = top_vol.sort_values('avg_ltv', ascending=True)
    
    ax.barh(top_vol.index, top_vol['avg_ltv'], color='#42A5F5', alpha=0.8)
    ax.axvline(x=state_stats['avg_ltv'].median(), color='black', linestyle='--', 
               linewidth=1, label=f'Median: ${state_stats["avg_ltv"].median():,.0f}')
    ax.set_xlabel('Average LTV ($)', fontsize=11)
    ax.set_ylabel('State', fontsize=11)
    ax.set_title('Average LTV by State\n(Top 25 by Volume)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'EDA_geographic.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved EDA_geographic.png")


# =============================================================================
# ANALYSIS 5: Adverse Selection (Early Claims by Channel)
# =============================================================================

def analysis_5_early_claims(leads, search_spend, social_spend):
    """Adverse selection by marketing channel."""
    print("\n[5/8] Adverse Selection...")
    
    sold = leads[leads['sold_date'].notna()].copy()
    
    # --- Compute CPL by channel (same logic as Analysis 6) ---
    total_search_spend = search_spend['spend'].sum()
    total_social_spend = social_spend['spend'].sum()
    email_leads_count = len(leads[leads['channel'] == 'email']['lead_id'].unique())
    total_email_spend = pd.read_csv(DATA_DIR / 'email_daily_spend.csv')['spend'].sum()

    channel_spend = {
        'paid_search': total_search_spend,
        'paid_social': total_social_spend,
        'email': total_email_spend
    }
    lead_counts = leads.groupby('channel')['lead_id'].nunique()
    cpl = {ch: channel_spend[ch] / lead_counts[ch] for ch in channel_spend if ch in lead_counts}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: CPL by channel
    ax = axes[0]
    cpl_sorted = sorted(cpl.items(), key=lambda x: x[1])
    ch_names = [CHANNEL_LABELS.get(ch, ch) for ch, _ in cpl_sorted]
    ch_vals = [v for _, v in cpl_sorted]
    ch_colors = [CHANNEL_COLORS.get(ch, 'gray') for ch, _ in cpl_sorted]

    bars = ax.barh(ch_names, ch_vals, color=ch_colors, alpha=0.8)
    for bar, val in zip(bars, ch_vals):
        ax.annotate(f'${val:,.0f}', xy=(bar.get_width(), bar.get_y() + bar.get_height()/2),
                    xytext=(3, 0), textcoords='offset points', ha='left', va='center', fontsize=10)
    ax.set_xlabel('Cost per Lead ($)', fontsize=11)
    ax.set_title('Cost per Lead by Channel', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    # Right: Loss ratio by channel (unchanged)
    ax = axes[1]
    channel_loss = sold.groupby('channel').agg(
        total_premium=('total_premium', 'sum'),
        total_claims=('total_claim_amount', 'sum')
    )
    channel_loss['loss_ratio'] = channel_loss['total_claims'] / channel_loss['total_premium'] * 100
    channel_loss = channel_loss.sort_values('loss_ratio', ascending=True)

    colors = [CHANNEL_COLORS.get(ch, 'gray') for ch in channel_loss.index]
    labels = [CHANNEL_LABELS.get(ch, ch) for ch in channel_loss.index]

    bars = ax.barh(labels, channel_loss['loss_ratio'], color=colors, alpha=0.8)
    for bar, val in zip(bars, channel_loss['loss_ratio']):
        ax.annotate(f'{val:.1f}%', xy=(bar.get_width(), bar.get_y() + bar.get_height()/2),
                    xytext=(3, 0), textcoords='offset points', ha='left', va='center', fontsize=10)
    ax.set_xlabel('Loss Ratio (%)', fontsize=11)
    ax.set_title('Loss Ratio by Channel', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'EDA_early_claims.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved EDA_early_claims.png")

# =============================================================================
# ANALYSIS 6: Marketing ROI by Channel
# =============================================================================

def analysis_6_policy_profitability(leads, search_spend, social_spend):
    """Marketing ROI by channel."""
    print("\n[6/8] Marketing ROI by Channel...")
    
    sold = leads[leads['sold_date'].notna()].copy()
    
    # Calculate total spend by channel
    total_search_spend = search_spend['spend'].sum()
    total_social_spend = social_spend['spend'].sum()
    
    # Estimate email spend from CPL ($8) × number of email leads
    email_leads_count = len(leads[leads['channel'] == 'email']['lead_id'].unique())
    total_email_spend = pd.read_csv(DATA_DIR / 'email_daily_spend.csv')['spend'].sum()
    
    channel_spend = {
        'paid_search': total_search_spend,
        'paid_social': total_social_spend,
        'email': total_email_spend
    }
    
    # Profit by channel
    channel_profit = sold.groupby('channel')['expected_value'].sum()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Left: Profit per policy by channel
    ax = axes[0]
    profit_per_policy = sold.groupby('channel')['expected_value'].mean()
    profit_per_policy = profit_per_policy.sort_values(ascending=True)
    
    colors = [CHANNEL_COLORS.get(ch, 'gray') for ch in profit_per_policy.index]
    labels = [CHANNEL_LABELS.get(ch, ch) for ch in profit_per_policy.index]
    
    bars = ax.barh(labels, profit_per_policy.values, color=colors, alpha=0.8)
    for bar, val in zip(bars, profit_per_policy.values):
        ax.annotate(f'${val:,.0f}', xy=(bar.get_width(), bar.get_y() + bar.get_height()/2),
                    xytext=(3, 0), textcoords='offset points', ha='left', va='center', fontsize=10)
    ax.set_xlabel('Average Profit per Policy ($)', fontsize=11)
    ax.set_title('Profit per Policy', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Center: Total marketing spend by channel
    ax = axes[1]
    spend_sorted = sorted(channel_spend.items(), key=lambda x: x[1])
    ch_names = [CHANNEL_LABELS.get(ch, ch) for ch, _ in spend_sorted]
    ch_vals = [v for _, v in spend_sorted]
    ch_colors = [CHANNEL_COLORS.get(ch, 'gray') for ch, _ in spend_sorted]
    
    bars = ax.barh(ch_names, ch_vals, color=ch_colors, alpha=0.8)
    for bar, val in zip(bars, ch_vals):
        ax.annotate(f'${val:,.0f}', xy=(bar.get_width(), bar.get_y() + bar.get_height()/2),
                    xytext=(3, 0), textcoords='offset points', ha='left', va='center', fontsize=10)
    ax.set_xlabel('Total Marketing Spend ($)', fontsize=11)
    ax.set_title('Marketing Spend by Channel', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Right: ROI (profit per marketing dollar)
    ax = axes[2]
    roi_data = {}
    for ch in channel_spend:
        if ch in channel_profit.index:
            roi_data[ch] = channel_profit[ch] / channel_spend[ch]
    
    roi_sorted = sorted(roi_data.items(), key=lambda x: x[1])
    ch_names = [CHANNEL_LABELS.get(ch, ch) for ch, _ in roi_sorted]
    ch_vals = [v for _, v in roi_sorted]
    ch_colors = [CHANNEL_COLORS.get(ch, 'gray') for ch, _ in roi_sorted]
    
    bars = ax.barh(ch_names, ch_vals, color=ch_colors, alpha=0.8)
    for bar, val in zip(bars, ch_vals):
        ax.annotate(f'{val:.1f}x', xy=(bar.get_width(), bar.get_y() + bar.get_height()/2),
                    xytext=(3, 0), textcoords='offset points', ha='left', va='center', fontsize=10)
    ax.set_xlabel('ROI (Profit / Spend)', fontsize=11)
    ax.set_title('Marketing ROI by Channel', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'EDA_policy_profitability.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved EDA_policy_profitability.png")


# =============================================================================
# ANALYSIS 7: State-Level Claims
# =============================================================================

def analysis_7_state_claims(leads):
    """State-level claims and loss ratios."""
    print("\n[7/8] State-Level Claims...")
    
    sold = leads[leads['sold_date'].notna()].copy()
    
    state_stats = sold.groupby('state').agg(
        policies=('lead_id', 'count'),
        total_premium=('total_premium', 'sum'),
        total_claims=('total_claim_amount', 'sum'),
        claim_rate=('has_claim', 'mean')
    )
    state_stats['loss_ratio'] = state_stats['total_claims'] / state_stats['total_premium']
    state_stats = state_stats[state_stats['policies'] >= 15]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Claim rate by state
    ax = axes[0]
    top_claim = state_stats.sort_values('claim_rate', ascending=False).head(20)
    
    colors = ['#E57373' if cr > state_stats['claim_rate'].median() else '#81C784' 
              for cr in top_claim['claim_rate']]
    ax.barh(top_claim.index, top_claim['claim_rate'] * 100, color=colors, alpha=0.8)
    ax.axvline(x=state_stats['claim_rate'].median() * 100, color='black', linestyle='--',
               linewidth=1, label=f'Median: {state_stats["claim_rate"].median()*100:.1f}%')
    ax.set_xlabel('Claim Rate (%)', fontsize=11)
    ax.set_ylabel('State', fontsize=11)
    ax.set_title('Claim Rate by State (Top 20)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()
    
    # Right: Loss ratio distribution
    ax = axes[1]
    ax.hist(state_stats['loss_ratio'] * 100, bins=20, color='#42A5F5', alpha=0.8, edgecolor='white')
    ax.axvline(x=state_stats['loss_ratio'].median() * 100, color='red', linestyle='--',
               linewidth=2, label=f'Median: {state_stats["loss_ratio"].median()*100:.1f}%')
    
    # Highlight high-risk states
    high_risk = state_stats[state_stats['loss_ratio'] > state_stats['loss_ratio'].quantile(0.9)]
    if len(high_risk) > 0:
        ax.axvline(x=state_stats['loss_ratio'].quantile(0.9) * 100, color='orange', linestyle='--',
                   linewidth=1, label=f'90th percentile ({len(high_risk)} states above)')
    
    ax.set_xlabel('Loss Ratio (%)', fontsize=11)
    ax.set_ylabel('Number of States', fontsize=11)
    ax.set_title('Distribution of Loss Ratios Across States', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'EDA_state_claims.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved EDA_state_claims.png")


# =============================================================================
# ANALYSIS 8: Bind Rate vs Claims Rate
# =============================================================================

def analysis_8_bind_vs_claims(leads):
    """Bind rate vs claims rate trade-off."""
    print("\n[8/8] Bind Rate vs Claims Rate...")
    
    # Calculate by channel-product combination
    combos = leads.groupby(['channel', 'product']).agg(
        total=('lead_id', 'count'),
        sold=('sold_date', lambda x: x.notna().sum()),
    ).reset_index()
    
    sold = leads[leads['sold_date'].notna()]
    claims_by_combo = sold.groupby(['channel', 'product']).agg(
        claim_rate=('has_claim', 'mean')
    ).reset_index()
    
    combos = combos.merge(claims_by_combo, on=['channel', 'product'], how='inner')
    combos['bind_rate'] = combos['sold'] / combos['total']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    PRODUCT_MARKERS = {'Health': 'o', 'Life': '^', 'Property_Casualty': 's'}
    for ch in combos['channel'].unique():
        for prod in combos['product'].unique():
            mask = (combos['channel'] == ch) & (combos['product'] == prod)
            if mask.sum() == 0:
                continue
            ax.scatter(
                combos.loc[mask, 'bind_rate'] * 100,
                combos.loc[mask, 'claim_rate'] * 100,
                c=CHANNEL_COLORS.get(ch, 'gray'),
                marker=PRODUCT_MARKERS.get(prod, 'o'),
                s=combos.loc[mask, 'total'] / 10,
                alpha=0.7,
                label=f'{CHANNEL_LABELS.get(ch, ch)} – {prod}',
                edgecolors='white',
                linewidths=0.5
            )


    for ch in combos['channel'].unique():
        mask = combos['channel'] == ch
        ax.scatter(
            combos.loc[mask, 'bind_rate'] * 100,
            combos.loc[mask, 'claim_rate'] * 100,
            c=CHANNEL_COLORS.get(ch, 'gray'),
            s=combos.loc[mask, 'total'] / 10,  # Size by volume
            alpha=0.7,
            label=CHANNEL_LABELS.get(ch, ch),
            edgecolors='white',
            linewidths=0.5
        )
    
    # Add correlation
    corr = combos['bind_rate'].astype(float).corr(combos['claim_rate'].astype(float))
    
    # Trend line
    x_vals = combos['bind_rate'].astype(float).values * 100
    y_vals = combos['claim_rate'].astype(float).values * 100
    z = np.polyfit(x_vals, y_vals, 1)
    p = np.poly1d(z)
    x_range = np.linspace(x_vals.min(), x_vals.max(), 50)
    ax.plot(x_range, p(x_range), '--', color='gray', alpha=0.5, linewidth=1)
    
    ax.set_xlabel('Bind Rate (%)', fontsize=11)
    ax.set_ylabel('Claim Rate (%)', fontsize=11)
    ax.set_title(f'Bind Rate vs Claim Rate (r = {corr:.2f})', fontsize=12, fontweight='bold')
    # Channel legend (color)
    channel_handles = [
        plt.scatter([], [], c=CHANNEL_COLORS[ch], s=60, label=CHANNEL_LABELS[ch])
        for ch in CHANNEL_COLORS
    ]
    # Product legend (shape)
    product_handles = [
        plt.scatter([], [], c='gray', marker=PRODUCT_MARKERS[prod], s=60, label=prod.replace('_', ' '))
        for prod in PRODUCT_MARKERS
    ]
    legend1 = ax.legend(handles=channel_handles, title='Channel', loc='upper left')
    ax.add_artist(legend1)
    ax.legend(handles=product_handles, title='Product', loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'EDA_bind_vs_claims.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved EDA_bind_vs_claims.png")


# =============================================================================
# MAIN
# =============================================================================

def run_all_analyses():
    """Run all 8 EDA analyses."""
    print("=" * 60)
    print("INSURANCE MARKETING EDA")
    print("=" * 60)
    
    leads, search_spend, social_spend = load_data()
    
    analysis_1_credit_score(leads)
    analysis_2_age_bands(leads)
    analysis_3_cross_sell(leads)
    analysis_4_geographic(leads)
    analysis_5_early_claims(leads, search_spend, social_spend)
    analysis_6_policy_profitability(leads, search_spend, social_spend)
    analysis_7_state_claims(leads)
    analysis_8_bind_vs_claims(leads)
    
    print("\n" + "=" * 60)
    print("ALL ANALYSES COMPLETE")
    print("=" * 60)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("Files:")
    for f in sorted(OUTPUT_DIR.glob('EDA_*.png')):
        print(f"  ✓ {f.name}")


if __name__ == "__main__":
    run_all_analyses()
