import numpy as np
import pandas as pd
import random
from scipy.stats import beta, norm, uniform, lognorm, poisson, gamma

np.random.seed(42)
random.seed(42)

n_samples = 10000

def generate_sme_data(n):
    data = {}
    segments = {
        'Micro': {'prob': 0.45, 'mu': 17.3, 'sigma': 0.6, 'min': 50e6, 'max': 300e6},
        'Small': {'prob': 0.35, 'mu': 18.2, 'sigma': 0.55, 'min': 300e6, 'max': 1.5e9},
        'Medium': {'prob': 0.15, 'mu': 19.2, 'sigma': 0.5, 'min': 1.5e9, 'max': 5e9},
        'Large': {'prob': 0.05, 'mu': 20.1, 'sigma': 0.45, 'min': 5e9, 'max': 20e9}
    }

    segment_choices = np.random.choice(list(segments.keys()), size=n, p=[s['prob'] for s in segments.values()])
    data['segment'] = segment_choices
    data['revenue_30'] = np.zeros(n)

    for seg, params in segments.items():
        mask = segment_choices == seg
        raw = lognorm.rvs(s=params['sigma'], scale=np.exp(params['mu']), size=mask.sum())
        data['revenue_30'][mask] = np.clip(raw, params['min'], params['max'])

    data['revenue_90'] = data['revenue_30'] * uniform.rvs(0.8, 1.5, size=n)
    data['revenue_growth_rate_90'] = (data['revenue_90'] - data['revenue_30']) / np.maximum(data['revenue_30'], 1)
    data['revenue_365'] = data['revenue_90'] * uniform.rvs(3, 5, size=n)

    data['num_orders_90'] = np.maximum(1, (poisson.rvs(100, size=n) + gamma.rvs(2, scale=50, size=n)).astype(int))
    data['avg_order_value_90'] = data['revenue_90'] / data['num_orders_90']

    data['cash_collected_90'] = data['revenue_90'] * beta.rvs(2, 1, size=n)
    data['revenue_to_cash_ratio_90'] = data['cash_collected_90'] / np.maximum(data['revenue_90'], 1)

    data['current_ratio'] = beta.rvs(1.5, 2, size=n) * 2
    data['quick_ratio'] = data['current_ratio'] * uniform.rvs(0.8, 1, size=n)
    data['debt_to_equity'] = beta.rvs(2, 1, size=n) * 3
    data['dscr'] = beta.rvs(1, 2, size=n) * 2
    data['liquidity_buffer_months'] = norm.rvs(loc=2, scale=1, size=n)

    data['on_time_payment_rate_12m'] = beta.rvs(2, 3, size=n)
    data['avg_payment_delay_days_12m'] = norm.rvs(loc=10, scale=5, size=n)
    data['refund_rate_90'] = beta.rvs(1, 5, size=n) * 0.1
    data['chargeback_rate_90'] = beta.rvs(1, 10, size=n) * 0.05
    data['partial_payment_frequency_90'] = np.random.poisson(5, size=n)

    data['fulfillment_time_median_30'] = norm.rvs(loc=7, scale=3, size=n)
    data['fulfillment_on_time_rate_30'] = beta.rvs(3, 2, size=n)
    data['order_defect_rate_30'] = beta.rvs(1, 5, size=n) * 0.05
    data['inventory_stockout_days_90'] = np.random.poisson(15, size=n)
    data['fulfillment_cost_per_order_90'] = norm.rvs(loc=5e6, scale=2e6, size=n)

    data['repeat_purchase_rate_90'] = beta.rvs(2, 3, size=n)
    data['avg_customer_lifetime_value_est'] = norm.rvs(loc=100e6, scale=50e6, size=n)
    data['customer_churn_rate_90'] = beta.rvs(2, 2, size=n)
    data['avg_order_frequency_per_customer_90'] = norm.rvs(loc=1.5, scale=0.5, size=n)
    data['avg_rating_score_90'] = beta.rvs(4, 1, size=n) * 5

    data['suspicious_account_flag'] = np.random.choice([0, 1], size=n, p=[0.95, 0.05])
    data['fraud_score_model_30'] = beta.rvs(1, 2, size=n)
    data['sanctions_or_politically_exposed_flag'] = np.random.choice([0, 1], size=n, p=[0.98, 0.02])
    data['aml_alert_count_12m'] = np.random.poisson(0.5, size=n)

    data['business_age_months'] = norm.rvs(loc=24, scale=12, size=n)
    data['legal_entity_type'] = np.random.choice(['sole_prop', 'llc', 'joint-stock'], size=n, p=[0.7, 0.2, 0.1])
    data['registered_office_changes_12m'] = np.random.poisson(0.2, size=n)
    data['tax_filing_up_to_date_flag'] = np.random.choice([0, 1], size=n, p=[0.1, 0.9])
    data['industry_code'] = np.random.choice(['retail', 'e-commerce', 'other'], size=n, p=[0.4, 0.5, 0.1])

    data['owner_tenure_years'] = norm.rvs(loc=5, scale=3, size=n)
    data['management_turnover_12m'] = np.random.poisson(1, size=n)
    data['owner_prior_business_failures_flag'] = np.random.choice([0, 1], size=n, p=[0.85, 0.15])
    data['related_party_transactions_flag'] = np.random.choice([0, 1], size=n, p=[0.9, 0.1])

    revenue_scaled = data['revenue_30'] / 1e9
    data['cac_90'] = lognorm.rvs(s=0.8, scale=20e6 * (1 + revenue_scaled), size=n)
    data['channel_diversification_index'] = beta.rvs(2, 2, size=n)
    data['paid_vs_organic_share_90'] = beta.rvs(1, 2, size=n)
    data['marketing_return_on_ad_spend_90'] = lognorm.rvs(s=0.5, scale=2 * (1 + revenue_scaled), size=n)

    data['supplier_concentration_top3_pct'] = beta.rvs(2, 1, size=n) * 100
    data['avg_supplier_lead_time_days_90'] = norm.rvs(loc=10, scale=5, size=n)
    data['supplier_default_flag_12m'] = np.random.choice([0, 1], size=n, p=[0.9, 0.1])
    data['procurement_payment_terms_median_days'] = norm.rvs(loc=30, scale=10, size=n)

    data['top3_skus_rev_share_90'] = beta.rvs(2, 1, size=n) * 100
    data['sku_return_rate_top3_90'] = beta.rvs(1, 5, size=n) * 0.1
    data['sku_margin_median'] = beta.rvs(3, 2, size=n) * 100
    data['sku_seasonality_strength_top3'] = beta.rvs(2, 2, size=n)

    data['industry_growth_rate_qoq'] = norm.rvs(loc=5, scale=3, size=n)
    data['local_consumer_sentiment_index'] = norm.rvs(loc=50, scale=10, size=n)
    data['exchange_rate_exposure_flag'] = np.random.choice([0, 1], size=n, p=[0.8, 0.2])
    data['regional_unemployment_rate'] = norm.rvs(loc=4, scale=1, size=n)

    data['revenue_volatility_90'] = beta.rvs(1, 2, size=n)
    data['revenue_trend_slope_90'] = norm.rvs(loc=0, scale=0.1, size=n)
    data['num_negative_cash_months_12m'] = np.random.poisson(2, size=n)
    data['feature_drift_flag_recent'] = np.random.choice([0, 1], size=n, p=[0.85, 0.15])

    data['liquidity_buffer_score'] = (data['liquidity_buffer_months'] / 6) * 100
    data['payment_reliability_score'] = (data['on_time_payment_rate_12m'] * 100) - (data['avg_payment_delay_days_12m'] / 10)
    data['business_stability_index'] = 100 - (data['revenue_volatility_90'] * 50 + data['num_negative_cash_months_12m'] * 10 + data['management_turnover_12m'] * 5)
    data['supply_chain_risk_index'] = 100 - (data['supplier_concentration_top3_pct'] / 2 + data['avg_supplier_lead_time_days_90'] / 2)

    return pd.DataFrame(data)

df = generate_sme_data(n_samples)
print(df.head())
print(df.shape)
df.to_csv("sme_data.csv", index=False)
