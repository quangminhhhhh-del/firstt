import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np

# 1. LOAD MODEL & FEATURES
logit_model = joblib.load("logit_model.pkl")
rf_model = joblib.load("rf_model.pkl")
feature_cols = joblib.load("feature_columns.pkl")

# 2. NHẬP DỮ LIỆU DOANH NGHIỆP MỚI
new_sme = {
    "segment": "small",
    "revenue_30": 120000,
    "revenue_90": 350000,
    "revenue_growth_rate_90": 0.12,
    "revenue_365": 1400000,
    "num_orders_90": 480,
    "avg_order_value_90": 730,
    "cash_collected_90": 300000,
    "revenue_to_cash_ratio_90": 1.15,
    "current_ratio": 1.6,
    "quick_ratio": 1.2,
    "debt_to_equity": 1.1,
    "dscr": 1.4,
    "liquidity_buffer_months": 2.5,
    "on_time_payment_rate_12m": 0.94,
    "avg_payment_delay_days_12m": 3,
    "refund_rate_90": 0.03,
    "chargeback_rate_90": 0.01,
    "partial_payment_frequency_90": 0.02,
    "fulfillment_time_median_30": 2,
    "fulfillment_on_time_rate_30": 0.96,
    "order_defect_rate_30": 0.01,
    "inventory_stockout_days_90": 4,
    "fulfillment_cost_per_order_90": 18,
    "repeat_purchase_rate_90": 0.42,
    "avg_customer_lifetime_value_est": 1800,
    "customer_churn_rate_90": 0.08,
    "avg_order_frequency_per_customer_90": 2.4,
    "avg_rating_score_90": 4.6,
    "suspicious_account_flag": 0,
    "fraud_score_model_30": 18,
    "sanctions_or_politically_exposed_flag": 0,
    "aml_alert_count_12m": 0,
    "business_age_months": 48,
    "legal_entity_type": "LLC",
    "registered_office_changes_12m": 0,
    "tax_filing_up_to_date_flag": 1,
    "industry_code": "E_COMMERCE",
    "owner_tenure_years": 6,
    "management_turnover_12m": 0,
    "owner_prior_business_failures_flag": 0,
    "related_party_transactions_flag": 0,
    "cac_90": 120,
    "channel_diversification_index": 0.55,
    "paid_vs_organic_share_90": 0.4,
    "marketing_return_on_ad_spend_90": 3.2,
    "supplier_concentration_top3_pct": 0.45,
    "avg_supplier_lead_time_days_90": 7,
    "supplier_default_flag_12m": 0,
    "procurement_payment_terms_median_days": 30,
    "top3_skus_rev_share_90": 0.6,
    "sku_return_rate_top3_90": 0.04,
    "sku_margin_median": 0.28,
    "sku_seasonality_strength_top3": 0.35,
    "industry_growth_rate_qoq": 0.03,
    "local_consumer_sentiment_index": 0.62,
    "exchange_rate_exposure_flag": 0,
    "regional_unemployment_rate": 0.04,
    "revenue_volatility_90": 0.18,
    "revenue_trend_slope_90": 0.06,
    "num_negative_cash_months_12m": 1,
    "feature_drift_flag_recent": 0,
    "liquidity_buffer_score": 72,
    "payment_reliability_score": 85,
    "business_stability_index": 78,
    "supply_chain_risk_index": 30
}

df_new = pd.DataFrame([new_sme])

# Đảm bảo đúng thứ tự cột
df_new = df_new.reindex(columns=feature_cols)

# 3. PREDICT
pd_rf = rf_model.predict_proba(df_new)[0, 1]
pd_logit = logit_model.predict_proba(df_new)[0, 1]

credit_score = (1 - pd_rf) * 100

# 4. MAP RISK TIER
def map_risk_tier(score):
    if score >= 90:
        return "A+"
    elif score >= 80:
        return "A"
    elif score >= 70:
        return "B+"
    elif score >= 60:
        return "B"
    elif score >= 50:
        return "C+"
    elif score >= 40:
        return "C"
    elif score >= 30:
        return "D+"
    elif score >= 20:
        return "D"
    else:
        return "E"

# 5. OUTPUT
print("\n=== CREDIT SCORING RESULT ===")
print("PD (Random Forest):", round(pd_rf, 4))
print("PD (Logistic):", round(pd_logit, 4))
print("Credit Score:", round(credit_score, 1))
print("Risk Tier:", map_risk_tier(credit_score))

# 6. SHAP EXPLANATION (FIXED FOR PIPELINE)

try:
    print("\n=== SHAP ANALYSIS ===")
    
    # Lấy preprocessor và model từ pipeline
    preprocessor = rf_model.named_steps['preprocess']
    rf_classifier = rf_model.named_steps['model']
    
    # Transform dữ liệu qua preprocessor
    X_transformed = preprocessor.transform(df_new)
    
    # Lấy tên feature sau khi transform
    # Numeric features giữ nguyên tên
    num_features = preprocessor.named_transformers_['num'].get_feature_names_out()
    
    # Categorical features sau one-hot encoding
    cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out()
    
    # Kết hợp tên features
    transformed_feature_names = list(num_features) + list(cat_features)
    
    # Tạo DataFrame với features đã transform
    X_transformed_df = pd.DataFrame(
        X_transformed, 
        columns=transformed_feature_names
    )
    
    print(f"Original features: {len(feature_cols)}")
    print(f"Transformed features: {len(transformed_feature_names)}")
    
    # Tạo SHAP explainer với model đã được train
    explainer = shap.TreeExplainer(rf_classifier)
    
    # Tính SHAP values
    shap_values = explainer.shap_values(X_transformed_df)
    
    # Debug: kiểm tra shape
    print(f"SHAP values type: {type(shap_values)}")
    if isinstance(shap_values, list):
        print(f"SHAP values list length: {len(shap_values)}")
        for i, sv in enumerate(shap_values):
            print(f"  Class {i} shape: {sv.shape}")
        shap_values_class1 = shap_values[1]  # Class 1 (default)
        expected_value = explainer.expected_value[1]
    else:
        print(f"SHAP values shape: {shap_values.shape}")
        # Nếu shape là (n_samples, n_features, n_classes), lấy class 1
        if len(shap_values.shape) == 3:
            shap_values_class1 = shap_values[:, :, 1]
        elif len(shap_values.shape) == 2 and shap_values.shape[1] == 2:
            # Nếu shape là (n_features, n_classes), transpose và lấy class 1
            shap_values_class1 = shap_values[:, 1].reshape(1, -1)
        else:
            shap_values_class1 = shap_values
        expected_value = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
    
    print(f"Final SHAP values shape for plotting: {shap_values_class1.shape}")
    
    # Tạo waterfall plot
    shap.plots.waterfall(
        shap.Explanation(
            values=shap_values_class1[0],
            base_values=expected_value,
            data=X_transformed_df.iloc[0].values,
            feature_names=transformed_feature_names
        ),
        max_display=15,
        show=False
    )
    
    plt.tight_layout()
    plt.savefig("shap_explain_new_sme.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ SHAP waterfall plot saved: shap_explain_new_sme.png")
    
    # Tạo thêm summary plot (top 20 features)
    shap.summary_plot(
        shap_values_class1,
        X_transformed_df,
        max_display=20,
        show=False
    )
    
    plt.tight_layout()
    plt.savefig("shap_summary_new_sme.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ SHAP summary plot saved: shap_summary_new_sme.png")
    
    # In ra top 10 features quan trọng nhất
    print("\n=== TOP 10 FEATURES IMPACT ===")
    feature_importance = pd.DataFrame({
        'feature': transformed_feature_names,
        'shap_value': shap_values_class1[0]
    })
    feature_importance['abs_shap'] = feature_importance['shap_value'].abs()
    feature_importance = feature_importance.sort_values('abs_shap', ascending=False)
    
    for idx, row in feature_importance.head(10).iterrows():
        impact = "increases" if row['shap_value'] > 0 else "decreases"
        print(f"{row['feature']:40s}: {row['shap_value']:+.4f} ({impact} default risk)")
    
except Exception as e:
    print(f"\n Error in SHAP analysis: {str(e)}")
    import traceback
    traceback.print_exc()