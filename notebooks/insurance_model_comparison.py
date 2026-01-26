"""
Marketing Mix Model - Multi-Model Comparison
=============================================
Compares three approaches to estimating response curves:
1. Hill saturation function (current approach)
2. Log response function 
3. Bayesian Hill function with informative priors

This allows us to see how sensitive conclusions are to model choice.
"""

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit, minimize
import matplotlib.pyplot as plt
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

# Directory where this Python file lives
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / 'insure_co_data'
OUTPUT_DIR = BASE_DIR / 'model_comparison_outputs'
EMAIL_CPL = 8.0

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# RESPONSE CURVE FUNCTIONS
# =============================================================================

def hill_function(spend, K, S, beta):
    """Hill saturation function."""
    return K * (spend ** beta) / (S ** beta + spend ** beta)

def hill_derivative(spend, K, S, beta):
    """Derivative of Hill function (marginal ROI)."""
    numerator = K * beta * (spend ** (beta - 1)) * (S ** beta)
    denominator = (S ** beta + spend ** beta) ** 2
    return numerator / denominator

def log_function(spend, a, b):
    """Log response function: a * log(1 + spend/b)"""
    return a * np.log(1 + spend / b)

def log_derivative(spend, a, b):
    """Derivative of log function (marginal ROI)."""
    return a / (b + spend)

# =============================================================================
# DATA LOADING
# =============================================================================

def load_and_prepare_data(data_dir):
    """Load and prepare weekly aggregated data."""
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    
    leads = pd.read_csv(data_dir / 'leads.csv')
    search_spend = pd.read_csv(data_dir / 'search_daily_spend.csv')
    social_spend = pd.read_csv(data_dir / 'social_daily_spend.csv')
    
    # Convert dates
    leads['lead_date'] = pd.to_datetime(leads['lead_date'])
    leads['sold_date'] = pd.to_datetime(leads['sold_date'])
    search_spend['date'] = pd.to_datetime(search_spend['date'])
    social_spend['date'] = pd.to_datetime(social_spend['date'])
    
    # Calculate profit
    leads['policy_profit'] = leads['total_premium'] - leads['total_claim_amount']
    
    # Add week identifier
    leads['week'] = leads['lead_date'].dt.to_period('W').dt.start_time
    search_spend['week'] = search_spend['date'].dt.to_period('W').dt.start_time
    social_spend['week'] = social_spend['date'].dt.to_period('W').dt.start_time
    
    # Aggregate spend by week
    search_weekly = search_spend.groupby('week')['spend'].sum().reset_index()
    search_weekly.columns = ['week', 'search_spend']
    
    social_weekly = social_spend.groupby('week')['spend'].sum().reset_index()
    social_weekly.columns = ['week', 'social_spend']
    
    # Aggregate profit by channel and week
    sold_policies = leads[leads['sold_date'].notna()].copy()
    
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
    weekly_data = weekly_data.merge(profit_by_channel, on='week', how='outer')
    weekly_data = weekly_data.fillna(0)
    weekly_data = weekly_data.sort_values('week').reset_index(drop=True)
    
    # Ensure profit columns exist
    for ch in ['search', 'social', 'email']:
        col = f'{ch}_profit' if ch != 'search' else 'paid_search_profit'
        if col not in weekly_data.columns:
            weekly_data[col] = 0
    
    # Rename for consistency
    if 'paid_search_profit' in weekly_data.columns:
        weekly_data['search_profit'] = weekly_data['paid_search_profit']
    if 'paid_social_profit' in weekly_data.columns:
        weekly_data['social_profit'] = weekly_data['paid_social_profit']
    
    print(f"Prepared {len(weekly_data)} weeks of data")
    
    return weekly_data


# =============================================================================
# MODEL 1: HILL FUNCTION (Frequentist)
# =============================================================================

def fit_hill_model(weekly_data, channel):
    """Fit Hill saturation curve using least squares."""
    spend = weekly_data[f'{channel}_spend'].values
    profit = weekly_data[f'{channel}_profit'].values
    
    # Filter to positive values
    mask = (spend > 0) & (profit > 0)
    spend_fit = spend[mask]
    profit_fit = profit[mask]
    
    if len(spend_fit) < 10:
        return None
    
    # Initial guesses
    K_init = profit_fit.max() * 2
    S_init = np.median(spend_fit)
    beta_init = 0.8
    
    try:
        popt, pcov = curve_fit(
            hill_function,
            spend_fit,
            profit_fit,
            p0=[K_init, S_init, beta_init],
            bounds=([0, 0, 0.1], [np.inf, np.inf, 3.0]),
            maxfev=10000
        )
        
        # R-squared
        pred = hill_function(spend_fit, *popt)
        ss_res = np.sum((profit_fit - pred) ** 2)
        ss_tot = np.sum((profit_fit - profit_fit.mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        # Standard errors from covariance matrix
        se = np.sqrt(np.diag(pcov))
        
        return {
            'model': 'hill',
            'params': {'K': popt[0], 'S': popt[1], 'beta': popt[2]},
            'se': {'K': se[0], 'S': se[1], 'beta': se[2]},
            'r2': r2,
            'n_obs': len(spend_fit)
        }
    except Exception as e:
        print(f"  Hill fitting failed for {channel}: {e}")
        return None


# =============================================================================
# MODEL 2: LOG FUNCTION (Frequentist)
# =============================================================================

def fit_log_model(weekly_data, channel):
    """Fit log response curve using least squares."""
    spend = weekly_data[f'{channel}_spend'].values
    profit = weekly_data[f'{channel}_profit'].values
    
    mask = (spend > 0) & (profit > 0)
    spend_fit = spend[mask]
    profit_fit = profit[mask]
    
    if len(spend_fit) < 10:
        return None
    
    # Initial guesses
    a_init = profit_fit.max()
    b_init = np.median(spend_fit)
    
    try:
        popt, pcov = curve_fit(
            log_function,
            spend_fit,
            profit_fit,
            p0=[a_init, b_init],
            bounds=([0, 1], [np.inf, np.inf]),
            maxfev=10000
        )
        
        pred = log_function(spend_fit, *popt)
        ss_res = np.sum((profit_fit - pred) ** 2)
        ss_tot = np.sum((profit_fit - profit_fit.mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        se = np.sqrt(np.diag(pcov))
        
        return {
            'model': 'log',
            'params': {'a': popt[0], 'b': popt[1]},
            'se': {'a': se[0], 'b': se[1]},
            'r2': r2,
            'n_obs': len(spend_fit)
        }
    except Exception as e:
        print(f"  Log fitting failed for {channel}: {e}")
        return None


# =============================================================================
# MODEL 3: BAYESIAN HILL FUNCTION
# =============================================================================

def fit_bayesian_hill_model(weekly_data, channel):
    """
    Fit Bayesian Hill model using PyMC.
    Falls back to simpler approach if PyMC not available.
    """
    spend = weekly_data[f'{channel}_spend'].values
    profit = weekly_data[f'{channel}_profit'].values
    
    mask = (spend > 0) & (profit > 0)
    spend_fit = spend[mask]
    profit_fit = profit[mask]
    
    if len(spend_fit) < 10:
        return None
    
    try:
        import pymc as pm
        import arviz as az
        
        # Normalize data for better sampling
        spend_mean = spend_fit.mean()
        spend_std = spend_fit.std()
        profit_mean = profit_fit.mean()
        profit_std = profit_fit.std()
        
        spend_norm = (spend_fit - spend_mean) / spend_std
        profit_norm = (profit_fit - profit_mean) / profit_std
        
        # Use raw positive values for Hill (avoid issues with normalization)
        with pm.Model() as model:
            # Priors - informative based on data scale
            # K: max response, expect 1-5x current max profit
            K = pm.LogNormal('K', mu=np.log(profit_fit.max() * 2), sigma=0.5)
            
            # S: half-saturation, expect around current median spend
            S = pm.LogNormal('S', mu=np.log(np.median(spend_fit)), sigma=0.7)
            
            # beta: shape parameter, expect diminishing returns (0.5-2)
            beta = pm.TruncatedNormal('beta', mu=1.0, sigma=0.5, lower=0.3, upper=3.0)
            
            # Noise
            sigma = pm.HalfNormal('sigma', sigma=profit_fit.std())
            
            # Expected value
            mu = K * (spend_fit ** beta) / (S ** beta + spend_fit ** beta)
            
            # Likelihood
            likelihood = pm.Normal('y', mu=mu, sigma=sigma, observed=profit_fit)
            
            # Sample
            trace = pm.sample(1000, tune=1000, cores=1, random_seed=42, 
                            progressbar=False, return_inferencedata=True)
        
        # Extract posterior summaries
        summary = az.summary(trace, var_names=['K', 'S', 'beta'])
        
        K_mean = summary.loc['K', 'mean']
        S_mean = summary.loc['S', 'mean']
        beta_mean = summary.loc['beta', 'mean']
        
        K_sd = summary.loc['K', 'sd']
        S_sd = summary.loc['S', 'sd']
        beta_sd = summary.loc['beta', 'sd']
        
        # Credible intervals
        K_ci = (summary.loc['K', 'hdi_3%'], summary.loc['K', 'hdi_97%'])
        S_ci = (summary.loc['S', 'hdi_3%'], summary.loc['S', 'hdi_97%'])
        beta_ci = (summary.loc['beta', 'hdi_3%'], summary.loc['beta', 'hdi_97%'])
        
        # Calculate R-squared equivalent (using posterior mean)
        pred = hill_function(spend_fit, K_mean, S_mean, beta_mean)
        ss_res = np.sum((profit_fit - pred) ** 2)
        ss_tot = np.sum((profit_fit - profit_fit.mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        return {
            'model': 'bayesian_hill',
            'params': {'K': K_mean, 'S': S_mean, 'beta': beta_mean},
            'se': {'K': K_sd, 'S': S_sd, 'beta': beta_sd},
            'ci_94': {'K': K_ci, 'S': S_ci, 'beta': beta_ci},
            'r2': r2,
            'n_obs': len(spend_fit),
            'trace': trace
        }
        
    except ImportError:
        print(f"  PyMC not available, using bootstrap approximation for {channel}")
        return fit_bootstrap_hill_model(weekly_data, channel)
    except Exception as e:
        print(f"  Bayesian fitting failed for {channel}: {e}")
        return fit_bootstrap_hill_model(weekly_data, channel)


def fit_bootstrap_hill_model(weekly_data, channel, n_bootstrap=500):
    """
    Bootstrap approximation to Bayesian uncertainty.
    Used as fallback when PyMC is not available.
    """
    spend = weekly_data[f'{channel}_spend'].values
    profit = weekly_data[f'{channel}_profit'].values
    
    mask = (spend > 0) & (profit > 0)
    spend_fit = spend[mask]
    profit_fit = profit[mask]
    
    if len(spend_fit) < 10:
        return None
    
    bootstrap_params = {'K': [], 'S': [], 'beta': []}
    
    for i in range(n_bootstrap):
        # Resample with replacement
        idx = np.random.choice(len(spend_fit), size=len(spend_fit), replace=True)
        spend_boot = spend_fit[idx]
        profit_boot = profit_fit[idx]
        
        try:
            popt, _ = curve_fit(
                hill_function,
                spend_boot,
                profit_boot,
                p0=[profit_boot.max() * 2, np.median(spend_boot), 0.8],
                bounds=([0, 0, 0.1], [np.inf, np.inf, 3.0]),
                maxfev=5000
            )
            bootstrap_params['K'].append(popt[0])
            bootstrap_params['S'].append(popt[1])
            bootstrap_params['beta'].append(popt[2])
        except:
            continue
    
    if len(bootstrap_params['K']) < 100:
        print(f"  Bootstrap failed for {channel}: insufficient successful fits")
        return None
    
    # Calculate summaries
    params = {k: np.mean(v) for k, v in bootstrap_params.items()}
    se = {k: np.std(v) for k, v in bootstrap_params.items()}
    ci = {k: (np.percentile(v, 3), np.percentile(v, 97)) for k, v in bootstrap_params.items()}
    
    # R-squared
    pred = hill_function(spend_fit, params['K'], params['S'], params['beta'])
    ss_res = np.sum((profit_fit - pred) ** 2)
    ss_tot = np.sum((profit_fit - profit_fit.mean()) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    return {
        'model': 'bootstrap_hill',
        'params': params,
        'se': se,
        'ci_94': ci,
        'r2': r2,
        'n_obs': len(spend_fit)
    }


# =============================================================================
# MODEL COMPARISON AND ANALYSIS
# =============================================================================

def fit_all_models(weekly_data):
    """Fit all three models to each channel."""
    channels = ['search', 'social', 'email']
    results = {}
    
    print("\n" + "=" * 70)
    print("FITTING MODELS")
    print("=" * 70)
    
    for channel in channels:
        print(f"\n{channel.upper()}")
        print("-" * 40)
        
        results[channel] = {}
        
        # Model 1: Hill
        print("  Fitting Hill model...")
        results[channel]['hill'] = fit_hill_model(weekly_data, channel)
        if results[channel]['hill']:
            r2 = results[channel]['hill']['r2']
            print(f"    R-squared: {r2:.3f}")
        
        # Model 2: Log
        print("  Fitting Log model...")
        results[channel]['log'] = fit_log_model(weekly_data, channel)
        if results[channel]['log']:
            r2 = results[channel]['log']['r2']
            print(f"    R-squared: {r2:.3f}")
        
        # Model 3: Bayesian Hill
        print("  Fitting Bayesian Hill model...")
        results[channel]['bayesian'] = fit_bayesian_hill_model(weekly_data, channel)
        if results[channel]['bayesian']:
            r2 = results[channel]['bayesian']['r2']
            print(f"    R-squared: {r2:.3f}")
    
    return results


def calculate_marginal_roi(model_result, spend_level):
    """Calculate marginal ROI at a given spend level."""
    if model_result is None:
        return None
    
    model_type = model_result['model']
    params = model_result['params']
    
    if model_type in ['hill', 'bayesian_hill', 'bootstrap_hill']:
        return hill_derivative(spend_level, params['K'], params['S'], params['beta'])
    elif model_type == 'log':
        return log_derivative(spend_level, params['a'], params['b'])
    return None


def calculate_predicted_profit(model_result, spend_level):
    """Calculate predicted profit at a given spend level."""
    if model_result is None:
        return None
    
    model_type = model_result['model']
    params = model_result['params']
    
    if model_type in ['hill', 'bayesian_hill', 'bootstrap_hill']:
        return hill_function(spend_level, params['K'], params['S'], params['beta'])
    elif model_type == 'log':
        return log_function(spend_level, params['a'], params['b'])
    return None


def find_optimal_allocation(model_results, model_name, total_budget):
    """Find optimal budget allocation for a specific model."""
    channels = list(model_results.keys())
    
    def negative_profit(allocation):
        total = 0
        for i, ch in enumerate(channels):
            model = model_results[ch].get(model_name)
            if model:
                total += calculate_predicted_profit(model, allocation[i])
        return -total
    
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - total_budget}
    bounds = [(10, total_budget) for _ in channels]  # Min $10 per channel
    x0 = [total_budget / len(channels)] * len(channels)
    
    result = minimize(negative_profit, x0, method='SLSQP', 
                     bounds=bounds, constraints=constraints)
    
    return {ch: result.x[i] for i, ch in enumerate(channels)}


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_model_comparison(weekly_data, model_results, output_dir):
    """Create comparison plots for all models."""
    channels = ['search', 'social', 'email']
    colors = {'search': '#2E86AB', 'social': '#A23B72', 'email': '#F18F01'}
    model_styles = {'hill': '-', 'log': '--', 'bayesian': ':'}
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Response curves by channel
    for i, channel in enumerate(channels):
        ax = axes[0, i]
        
        spend = weekly_data[f'{channel}_spend'].values
        profit = weekly_data[f'{channel}_profit'].values
        
        # Scatter plot
        mask = (spend > 0) & (profit > 0)
        ax.scatter(spend[mask], profit[mask], alpha=0.4, s=20, color=colors[channel])
        
        # Fitted curves
        spend_range = np.linspace(1, spend.max() * 1.3, 200)
        
        for model_name, style in model_styles.items():
            model = model_results[channel].get(model_name)
            if model:
                pred = [calculate_predicted_profit(model, s) for s in spend_range]
                r2 = model['r2']
                label = f"{model_name} (R2={r2:.2f})"
                ax.plot(spend_range, pred, style, linewidth=2, label=label)
        
        ax.set_xlabel('Weekly Spend ($)')
        ax.set_ylabel('Weekly Profit ($)')
        ax.set_title(f'{channel.title()} Response Curves')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, None)
        ax.set_ylim(0, None)
    
    # Row 2: Marginal ROI comparison
    for i, channel in enumerate(channels):
        ax = axes[1, i]
        
        spend = weekly_data[f'{channel}_spend'].values
        current_spend = spend[spend > 0].mean()
        spend_range = np.linspace(current_spend * 0.2, current_spend * 3, 200)
        
        for model_name, style in model_styles.items():
            model = model_results[channel].get(model_name)
            if model:
                mroi = [calculate_marginal_roi(model, s) for s in spend_range]
                ax.plot(spend_range, mroi, style, linewidth=2, label=model_name)
        
        ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Break-even')
        ax.axvline(x=current_spend, color='gray', linestyle=':', alpha=0.5, label='Current spend')
        ax.set_xlabel('Weekly Spend ($)')
        ax.set_ylabel('Marginal ROI')
        ax.set_title(f'{channel.title()} Marginal ROI')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved: {output_dir}/model_comparison.png")


def plot_uncertainty(weekly_data, model_results, output_dir):
    """Plot uncertainty bands for Bayesian model."""
    channels = ['search', 'social', 'email']
    colors = {'search': '#2E86AB', 'social': '#A23B72', 'email': '#F18F01'}
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, channel in enumerate(channels):
        ax = axes[i]
        
        spend = weekly_data[f'{channel}_spend'].values
        profit = weekly_data[f'{channel}_profit'].values
        
        mask = (spend > 0) & (profit > 0)
        ax.scatter(spend[mask], profit[mask], alpha=0.4, s=20, color=colors[channel])
        
        spend_range = np.linspace(1, spend.max() * 1.3, 200)
        
        bayes_model = model_results[channel].get('bayesian')
        if bayes_model and 'ci_94' in bayes_model:
            params = bayes_model['params']
            ci = bayes_model['ci_94']
            
            # Point estimate
            pred_mean = hill_function(spend_range, params['K'], params['S'], params['beta'])
            ax.plot(spend_range, pred_mean, '-', color=colors[channel], linewidth=2, label='Posterior mean')
            
            # Uncertainty band using CI on parameters
            # Sample from approximate posterior
            n_samples = 200
            pred_samples = np.zeros((n_samples, len(spend_range)))
            
            for j in range(n_samples):
                K_samp = np.random.normal(params['K'], bayes_model['se']['K'])
                S_samp = np.random.normal(params['S'], bayes_model['se']['S'])
                beta_samp = np.random.normal(params['beta'], bayes_model['se']['beta'])
                
                # Clip to valid ranges
                K_samp = max(K_samp, 1000)
                S_samp = max(S_samp, 10)
                beta_samp = np.clip(beta_samp, 0.3, 3.0)
                
                pred_samples[j] = hill_function(spend_range, K_samp, S_samp, beta_samp)
            
            pred_lower = np.percentile(pred_samples, 3, axis=0)
            pred_upper = np.percentile(pred_samples, 97, axis=0)
            
            ax.fill_between(spend_range, pred_lower, pred_upper, 
                          color=colors[channel], alpha=0.2, label='94% CI')
        
        ax.set_xlabel('Weekly Spend ($)')
        ax.set_ylabel('Weekly Profit ($)')
        ax.set_title(f'{channel.title()} - Bayesian Uncertainty')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, None)
        ax.set_ylim(0, None)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'bayesian_uncertainty.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_dir}/bayesian_uncertainty.png")


# =============================================================================
# REPORTING
# =============================================================================

def generate_comparison_report(weekly_data, model_results, output_dir):
    """Generate comprehensive comparison report."""
    channels = ['search', 'social', 'email']
    model_names = ['hill', 'log', 'bayesian']
    
    report = []
    report.append("=" * 70)
    report.append("MARKETING MIX MODEL - MULTI-MODEL COMPARISON")
    report.append("=" * 70)
    
    # Section 1: Model Fit Summary
    report.append("\n\n1. MODEL FIT COMPARISON (R-squared)")
    report.append("-" * 50)
    report.append(f"{'Channel':<12} {'Hill':>12} {'Log':>12} {'Bayesian':>12}")
    report.append("-" * 50)
    
    for channel in channels:
        row = f"{channel:<12}"
        for model_name in model_names:
            model = model_results[channel].get(model_name)
            if model:
                row += f" {model['r2']:>11.3f}"
            else:
                row += f" {'N/A':>11}"
        report.append(row)
    
    # Section 2: Parameter Estimates
    report.append("\n\n2. PARAMETER ESTIMATES")
    report.append("-" * 50)
    
    for channel in channels:
        report.append(f"\n{channel.upper()}")
        
        for model_name in model_names:
            model = model_results[channel].get(model_name)
            if not model:
                continue
            
            report.append(f"\n  {model_name.title()} Model:")
            
            if model['model'] in ['hill', 'bayesian_hill', 'bootstrap_hill']:
                K, S, beta = model['params']['K'], model['params']['S'], model['params']['beta']
                K_se, S_se, beta_se = model['se']['K'], model['se']['S'], model['se']['beta']
                report.append(f"    K (saturation):     {K:>12,.0f} (+/- {K_se:,.0f})")
                report.append(f"    S (half-sat):       {S:>12,.0f} (+/- {S_se:,.0f})")
                report.append(f"    beta (shape):       {beta:>12.3f} (+/- {beta_se:.3f})")
                
                if 'ci_94' in model:
                    ci = model['ci_94']
                    report.append(f"    K 94% CI:           [{ci['K'][0]:,.0f}, {ci['K'][1]:,.0f}]")
                    report.append(f"    S 94% CI:           [{ci['S'][0]:,.0f}, {ci['S'][1]:,.0f}]")
                    report.append(f"    beta 94% CI:        [{ci['beta'][0]:.2f}, {ci['beta'][1]:.2f}]")
            
            elif model['model'] == 'log':
                a, b = model['params']['a'], model['params']['b']
                a_se, b_se = model['se']['a'], model['se']['b']
                report.append(f"    a (scale):          {a:>12,.0f} (+/- {a_se:,.0f})")
                report.append(f"    b (shape):          {b:>12,.0f} (+/- {b_se:,.0f})")
    
    # Section 3: Current Performance
    report.append("\n\n3. CURRENT PERFORMANCE")
    report.append("-" * 50)
    
    total_spend = 0
    total_profit = 0
    
    for channel in channels:
        spend = weekly_data[f'{channel}_spend'].sum()
        profit = weekly_data[f'{channel}_profit'].sum()
        roi = profit / spend if spend > 0 else 0
        total_spend += spend
        total_profit += profit
        report.append(f"{channel:<12} Spend: ${spend:>12,.0f}  Profit: ${profit:>12,.0f}  ROI: {roi:>6.1f}x")
    
    report.append("-" * 50)
    total_roi = total_profit / total_spend
    report.append(f"{'TOTAL':<12} Spend: ${total_spend:>12,.0f}  Profit: ${total_profit:>12,.0f}  ROI: {total_roi:>6.1f}x")
    
    # Section 4: Marginal ROI at Current Spend
    report.append("\n\n4. MARGINAL ROI AT CURRENT SPEND LEVELS")
    report.append("-" * 50)
    report.append(f"{'Channel':<12} {'Avg Spend':>12} {'Hill':>12} {'Log':>12} {'Bayesian':>12}")
    report.append("-" * 50)
    
    for channel in channels:
        spend = weekly_data[f'{channel}_spend']
        current = spend[spend > 0].mean()
        
        row = f"{channel:<12} ${current:>10,.0f}"
        
        for model_name in model_names:
            model = model_results[channel].get(model_name)
            if model:
                mroi = calculate_marginal_roi(model, current)
                row += f" {mroi:>11.1f}x"
            else:
                row += f" {'N/A':>11}"
        report.append(row)
    
    # Section 5: Optimal Allocation by Model
    report.append("\n\n5. OPTIMAL BUDGET ALLOCATION BY MODEL")
    report.append("-" * 50)
    
    total_weekly_budget = sum(
        weekly_data[f'{ch}_spend'].mean() for ch in channels
    )
    
    report.append(f"Total weekly budget: ${total_weekly_budget:,.0f}")
    report.append("")
    report.append(f"{'Channel':<12} {'Current':>12} {'Hill':>12} {'Log':>12} {'Bayesian':>12}")
    report.append("-" * 50)
    
    # Calculate current allocation
    current_alloc = {ch: weekly_data[f'{ch}_spend'].mean() for ch in channels}
    
    # Calculate optimal for each model
    optimal = {}
    for model_name in model_names:
        try:
            optimal[model_name] = find_optimal_allocation(
                model_results, model_name, total_weekly_budget
            )
        except:
            optimal[model_name] = None
    
    for channel in channels:
        row = f"{channel:<12} ${current_alloc[channel]:>10,.0f}"
        for model_name in model_names:
            if optimal[model_name]:
                row += f" ${optimal[model_name][channel]:>10,.0f}"
            else:
                row += f" {'N/A':>11}"
        report.append(row)
    
    # Section 6: Key Insights
    report.append("\n\n6. KEY INSIGHTS")
    report.append("-" * 50)
    report.append("""
The three models often give DIFFERENT optimal allocations because:

1. HILL assumes a hard saturation ceiling (K parameter)
   - Can conclude a channel is "saturated" even with limited data
   - May recommend cutting spend on high-ROI channels if it thinks they're saturated

2. LOG assumes perpetual diminishing returns with no ceiling
   - Never fully saturates
   - More conservative about cutting any channel

3. BAYESIAN HILL uses the same form as Hill but:
   - Uncertainty is explicit (wide CI = "we don't know")
   - Priors prevent extreme conclusions from sparse data
   - Compare CI widths across channels to see where data is informative

RECOMMENDATION: When models disagree significantly, trust the data limitations
over any single model's point estimate. The Bayesian uncertainty bands show
where you're extrapolating vs. interpolating.
""")
    
    report.append("=" * 70)
    
    report_text = "\n".join(report)
    
    with open(output_dir / 'comparison_report.txt', 'w') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\nSaved: {output_dir}/comparison_report.txt")
    
    return report_text


# =============================================================================
# MAIN
# =============================================================================

def main():
    # Load data
    weekly_data = load_and_prepare_data(DATA_DIR)
    
    # Fit all models
    model_results = fit_all_models(weekly_data)
    
    # Generate visualizations
    plot_model_comparison(weekly_data, model_results, OUTPUT_DIR)
    plot_uncertainty(weekly_data, model_results, OUTPUT_DIR)
    
    # Generate report
    generate_comparison_report(weekly_data, model_results, OUTPUT_DIR)
    
    # Save model results
    results_export = {}
    for channel in model_results:
        results_export[channel] = {}
        for model_name, model in model_results[channel].items():
            if model:
                export = {
                    'model': model['model'],
                    'params': {k: float(v) for k, v in model['params'].items()},
                    'r2': float(model['r2'])
                }
                if 'se' in model:
                    export['se'] = {k: float(v) for k, v in model['se'].items()}
                if 'ci_94' in model:
                    export['ci_94'] = {k: [float(v[0]), float(v[1])] for k, v in model['ci_94'].items()}
                results_export[channel][model_name] = export
    
    with open(OUTPUT_DIR / 'all_model_results.json', 'w') as f:
        json.dump(results_export, f, indent=2)
    
    print(f"\nSaved: {OUTPUT_DIR}/all_model_results.json")
    
    return weekly_data, model_results


if __name__ == "__main__":
    weekly_data, model_results = main()
