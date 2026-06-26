import os, sys, json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate_insurance_data as G

# --- Life repricing: make Life slightly profitable + cut catastrophic-claim variance ---
# (mean was ~break-even; the issue was variance from rare $250k claims at modest volume)
G.CONFIG['avg_claim_severity']['Life'] = {'mean': 120000, 'std': 50000}
G.CONFIG['base_annual_premium']['Life'] = {'mean': 1950, 'std': 600}
G.CONFIG['base_claim_rate']['Life'] = 0.0075   # up from 0.005 -> Life only slightly profitable

SEED = 123
np.random.seed(SEED)
import random
random.seed(SEED)

START = datetime.strptime(G.CONFIG['start_date'], '%Y-%m-%d')
END = datetime.strptime(G.CONFIG['end_date'], '%Y-%m-%d')

CHANNELS = {
    'paid_search': dict(prefix='SEARCH', target_leads=8.2, target_cpl=45.0,
                        sat_ratio=0.75, beta=1.4, regime_lo=0.55, regime_hi=1.55),
    'paid_social': dict(prefix='SOCIAL', target_leads=6.4, target_cpl=28.0,
                        sat_ratio=0.45, beta=1.3, regime_lo=0.55, regime_hi=1.55),
    'email':       dict(prefix='EMAIL', target_leads=5.5, target_cpl=8.0,
                        sat_ratio=0.30, beta=1.3, regime_lo=0.50, regime_hi=1.60),
}
LEAD_NOISE_SIGMA = 0.06

def hill(spend, Lmax, S, beta):
    spend = np.asarray(spend, dtype=float)
    return Lmax * (spend ** beta) / (S ** beta + spend ** beta)

def solve_curve(p):
    op_spend = p['target_leads'] * p['target_cpl']
    S = op_spend / p['sat_ratio']
    rb = p['sat_ratio'] ** p['beta']
    Lmax = p['target_leads'] * (1 + rb) / rb
    return Lmax, S, op_spend

def build_spend_series(p, op_spend, rng):
    days = (END - START).days + 1
    dates = [START + timedelta(days=i) for i in range(days)]
    month_keys = sorted({(d.year, d.month) for d in dates})
    regime = {mk: op_spend * rng.uniform(p['regime_lo'], p['regime_hi']) for mk in month_keys}
    n_weeks = days // 7 + 1
    exp = np.ones(n_weeks)
    for w in range(n_weeks):
        u = rng.random()
        if u < 0.10:
            exp[w] = rng.uniform(3.0, 10.0)
        elif u < 0.17:
            exp[w] = rng.uniform(0.2, 0.45)
    rows = []
    for i, d in enumerate(dates):
        base = regime[(d.year, d.month)]
        weekend = 0.7 if d.weekday() >= 5 else 1.0
        noise = float(np.exp(rng.normal(0, 0.18)))
        spend = max(5.0, base * exp[i // 7] * weekend * noise)
        rows.append((d, spend))
    return rows

def gen_channel(channel, p, rng):
    Lmax, S, op_spend = solve_curve(p)
    spend_series = build_spend_series(p, op_spend, rng)
    spend_records, leads_list, state_records = [], [], []
    lead_counter = 1
    states = list(G.CONFIG['states'].keys())
    state_p = np.array(list(G.CONFIG['states'].values())); state_p = state_p / state_p.sum()
    for d, spend in spend_series:
        season = float(np.mean([G.SEASONALITY[pr][d.month - 1] for pr in G.CONFIG['products']]))
        lead_mu = hill(spend, Lmax, S, p['beta']) * season * float(np.exp(rng.normal(0, LEAD_NOISE_SIGMA)))
        n_leads = int(rng.poisson(max(lead_mu, 0)))
        spend_records.append({'date': d, 'spend': round(spend, 2),
            'impressions': int(spend * rng.uniform(80, 150)),
            'clicks': int(n_leads * rng.uniform(8, 15)), 'leads': n_leads,
            'cpl': round(spend / n_leads, 2) if n_leads > 0 else 0.0})
        for _ in range(n_leads):
            lead_id = f"{p['prefix']}_{lead_counter:06d}"; lead_counter += 1
            num_products = int(rng.choice([1, 2, 3], p=[0.65, 0.25, 0.10]))
            products = rng.choice(G.CONFIG['products'], size=num_products, replace=False).tolist()
            leads_list.append({'lead_id': lead_id, 'lead_date': d, 'channel': channel,
                'products': products, 'first_name': G.fake.first_name(), 'last_name': G.fake.last_name()})
            state_records.append({'lead_id': lead_id, 'date': d, 'state': str(rng.choice(states, p=state_p))})
    return pd.DataFrame(spend_records), leads_list, pd.DataFrame(state_records), (Lmax, S, op_spend)

def main(output_dir):
    rng = np.random.default_rng(SEED)
    os.makedirs(output_dir, exist_ok=True)
    all_leads, curve_info = [], {}
    for channel, p in CHANNELS.items():
        sdf, leads_list, state_df, curve = gen_channel(channel, p, rng)
        all_leads.extend(leads_list); curve_info[channel] = curve
        short = 'search' if channel == 'paid_search' else ('social' if channel == 'paid_social' else 'email')
        sdf.to_csv(os.path.join(output_dir, f'{short}_daily_spend.csv'), index=False)
        if channel != 'email':
            state_df.to_csv(os.path.join(output_dir, f'{short}_leads.csv'), index=False)
        print(f"{channel:12s} leads={len(leads_list):6d}  Lmax={curve[0]:6.1f} S=${curve[1]:7.0f} op=${curve[2]:6.0f}  daily mean=${sdf['spend'].mean():6.0f} min=${sdf['spend'].min():5.0f} max=${sdf['spend'].max():6.0f}")
    print("\nRunning funnel/underwriting (reused from v1)...")
    leads_detail = gen_lead_detail(all_leads)
    leads_detail.to_csv(os.path.join(output_dir, 'leads.csv'), index=False)
    sold = leads_detail[leads_detail['sold_date'].notna()]
    print(f"leads.csv rows={len(leads_detail):,}  sold={len(sold):,}")
    with open(os.path.join(output_dir, '_curve_info.json'), 'w') as f:
        json.dump({k: {'Lmax': v[0], 'S_half': v[1], 'op_spend': v[2]} for k, v in curve_info.items()}, f, indent=2)
    return curve_info

# ============================================================================
# CROSS-SELL DYNAMICS (v2 addition)
#   - Conversion lift on the 2nd+ product once a customer has bought one
#   - Retention/LTV uplift for customers who actually bundle (>=2 policies)
# ============================================================================
CROSS_SELL_CONV_BOOST = 1.5     # 2nd+ product converts more easily for an existing customer
BUNDLE_RETENTION_MULT = 1.9     # bundled customers churn less -> ~2x tenure (~90% vs ~80% retention)
BUNDLE_LTV_MULT = 1.9           # and are worth ~2x

def funnel_v2(lead_date, channel, product, demographics, conv_boost=1.0):
    stages = ['qualified', 'quote', 'binder', 'sold']
    keys = ['lead_to_qualified', 'qualified_to_quote', 'quote_to_binder', 'binder_to_sold']
    res = {k: None for k in ['qualified_date', 'quote_date', 'binder_date', 'sold_date']}
    res['final_status'] = 'lead'; res['ltv'] = None
    for f in ['annual_premium', 'expected_tenure_years', 'total_premium', 'has_claim',
              'has_early_claim', 'claim_count', 'total_claim_amount', 'expected_value', 'loss_ratio']:
        res[f] = None
    cur = lead_date
    for stage, key in zip(stages, keys):
        prob = min(0.95, G.calculate_conversion_probability(key, channel, product, demographics) * conv_boost)
        if np.random.random() < prob:
            d = {'qualified': (0, 3), 'quote': (1, 7), 'binder': (1, 14), 'sold': (1, 10)}[stage]
            cur = cur + timedelta(days=int(np.random.randint(*d)))
            res[f'{stage}_date'] = cur; res['final_status'] = stage
            if stage == 'sold':
                res['ltv'] = G.calculate_ltv(product, demographics, channel)
                res.update(G.simulate_policy_economics(product, demographics, channel, cur))
        else:
            break
    return res

def gen_lead_detail(all_leads):
    states = list(G.CONFIG['states'].keys())
    sp = np.array(list(G.CONFIG['states'].values())); sp = sp / sp.sum()
    rows = []
    for lead in all_leads:
        demo = G.generate_demographic_profile(lead['channel'], lead['products'][0])
        state = str(np.random.choice(states, p=sp))
        converted_any = False
        prs = []
        for product in lead['products']:
            boost = CROSS_SELL_CONV_BOOST if converted_any else 1.0
            fr = funnel_v2(lead['lead_date'], lead['channel'], product, demo, boost)
            if fr['sold_date'] is not None:
                converted_any = True
            prs.append((product, fr))
        bundled = sum(1 for _, fr in prs if fr['sold_date'] is not None) >= 2
        for product, fr in prs:
            if bundled and fr['sold_date'] is not None:
                r = BUNDLE_RETENTION_MULT
                fr['expected_tenure_years'] = round(fr['expected_tenure_years'] * r, 2)
                fr['total_claim_amount'] = round((fr['total_claim_amount'] or 0) * r, 2)
                fr['total_premium'] = round((fr['annual_premium'] or 0) * fr['expected_tenure_years'], 2)
                fr['expected_value'] = round(fr['total_premium'] - fr['total_claim_amount'], 2)
                fr['loss_ratio'] = round(fr['total_claim_amount'] / fr['total_premium'], 4) if fr['total_premium'] > 0 else 0
                fr['ltv'] = round((fr['ltv'] or 0) * BUNDLE_LTV_MULT, 2)
            rows.append({'lead_id': lead['lead_id'], 'product': product, 'channel': lead['channel'],
                'state': state, 'lead_date': lead['lead_date'],
                'first_name': lead.get('first_name'), 'last_name': lead.get('last_name'),
                'age': demo['age'], 'income_bracket': demo['income_bracket'], 'credit_score': demo['credit_score'],
                'qualified_date': fr['qualified_date'], 'quote_date': fr['quote_date'],
                'binder_date': fr['binder_date'], 'sold_date': fr['sold_date'],
                'final_status': fr['final_status'], 'ltv': fr['ltv'],
                'annual_premium': fr['annual_premium'], 'expected_tenure_years': fr['expected_tenure_years'],
                'total_premium': fr['total_premium'], 'has_claim': fr['has_claim'],
                'has_early_claim': fr['has_early_claim'], 'claim_count': fr['claim_count'],
                'total_claim_amount': fr['total_claim_amount'], 'expected_value': fr['expected_value'],
                'loss_ratio': fr['loss_ratio']})
    return pd.DataFrame(rows)
