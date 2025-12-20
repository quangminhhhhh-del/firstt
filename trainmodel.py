# AI CREDIT SCORING PIPELINE - PRODUCTION READY
# SME TMĐT – Logistic + Random Forest
# Optimized: no LabelEncoder ordinal bias, robust preprocessing, deploy feature alignment, RF probability calibration

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

# NEW: robust missing handling + probability calibration
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV

# =========================
# 0. CONFIG (NEW)
# =========================
RANDOM_STATE = 42
TEST_SIZE = 0.25

# NEW: Align training features with what your WEB/APP can actually collect.
# This is the #1 fix for "web input looks good but model scores E" due to missing engineered features.
USE_DEPLOY_FEATURE_SET = True

# IMPORTANT: These are the fields you used in app.py / index.html (core + added business variables).
# The script will automatically take the intersection with your CSV columns.
DEPLOY_FEATURES = [
    # Core web inputs
    "monthly_revenue",
    "order_volume_30d",
    "refund_rate",
    "cashflow_volatility",
    "platform_rating",
    "years_in_business",
    # Added business / credit variables (if they exist in your dataset)
    "net_profit_margin",
    "debt_to_revenue",
    "monthly_debt_payment_ratio",
    "late_payment_rate",
    "cash_buffer_months",
    "customer_concentration",
    "inventory_turnover",
    "tax_compliance",
    "business_sector",
]

def make_ohe():
    """
    NEW: Backward compatible OneHotEncoder.
    - sklearn >= 1.2 uses sparse_output
    - older sklearn uses sparse
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

print("\nNOTE:")
print(" - This pipeline uses OneHotEncoder for categorical features => NO ordinal bias like LabelEncoder.")
print(" - If you ever map A=1, B=2 for a categorical feature, the model may incorrectly assume B > A.")
print(" - OneHot avoids that problem.\n")

# 1. LOAD DATA
print("=" * 60)
print("LOADING DATA...")
print("=" * 60)
df = pd.read_csv("sme_synthetic.csv")
print(f"Data shape: {df.shape}")
print(f"Columns: {len(df.columns)}")

# 2. CREATE LABEL (PROXY DEFAULT)
# Định nghĩa các features sẽ dùng để tạo label
label_features = [
    "payment_reliability_score",
    "liquidity_buffer_score",
    "fraud_score_model_30"
]

df["high_risk"] = (
    (df["payment_reliability_score"] < 40) |
    (df["liquidity_buffer_score"] < 40) |
    (df["fraud_score_model_30"] > 70)
).astype(int)

print(f"\nLabel distribution:")
print(df["high_risk"].value_counts())
print(f"High risk rate: {df['high_risk'].mean():.2%}")

y = df["high_risk"]

# Bao gồm cả các derived scores và features có tương quan cao
features_to_drop = [
    "high_risk",
    # Features đã dùng để tạo label
    "payment_reliability_score",
    "liquidity_buffer_score",
    "fraud_score_model_30",
    # Các composite scores khác (có thể derived từ label features)
    "business_stability_index",
    "supply_chain_risk_index"
]

X_full = df.drop(columns=features_to_drop, errors="ignore")

print(f"\nFeatures dropped to prevent leakage: {len([f for f in features_to_drop if f in df.columns])}")
print(f"Features for training (full set): {X_full.shape[1]}")

# NEW: Optionally restrict training features to deploy/web feature set
if USE_DEPLOY_FEATURE_SET:
    available_deploy = [c for c in DEPLOY_FEATURES if c in X_full.columns]
    missing_deploy = [c for c in DEPLOY_FEATURES if c not in X_full.columns]

    print("\n" + "=" * 60)
    print("DEPLOY FEATURE ALIGNMENT (NEW)")
    print("=" * 60)
    print(f"USE_DEPLOY_FEATURE_SET = {USE_DEPLOY_FEATURE_SET}")
    print(f"Deploy features requested: {len(DEPLOY_FEATURES)}")
    print(f"Deploy features found in CSV: {len(available_deploy)}")
    if missing_deploy:
        print(f"Missing in CSV (will be ignored): {missing_deploy}")

    # If too few deploy features exist, fallback to full set to avoid training failure
    if len(available_deploy) >= 6:
        X = X_full[available_deploy].copy()
        print(f"✓ Using DEPLOY feature set for training: {X.shape[1]} features")
    else:
        X = X_full.copy()
        print(f"⚠ Too few deploy features found -> fallback to FULL feature set: {X.shape[1]} features")
else:
    X = X_full.copy()

print(f"\nFinal training feature count: {X.shape[1]}")
print(f"Example feature columns (first 20): {X.columns.tolist()[:20]}")

# 3. TRAIN / TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"\nTrain size: {len(X_train)} (High risk: {y_train.mean():.2%})")
print(f"Test size: {len(X_test)} (High risk: {y_test.mean():.2%})")

# 4. PREPROCESSING
# NEW: also include "category" dtype in cat_cols
num_cols = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

print(f"\nNumerical features: {len(num_cols)}")
print(f"Categorical features: {len(cat_cols)}")

# NEW: robust transformers with imputers
num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", make_ohe())
])

transformers = []
if len(num_cols) > 0:
    transformers.append(("num", num_transformer, num_cols))
if len(cat_cols) > 0:
    transformers.append(("cat", cat_transformer, cat_cols))

preprocess = ColumnTransformer(
    transformers=transformers,
    remainder="drop"
)

# 5. LOGISTIC REGRESSION (BASELINE)
print("\n" + "=" * 60)
print("TRAINING LOGISTIC REGRESSION...")
print("=" * 60)

logit_pipe = Pipeline([
    ("preprocess", preprocess),
    ("model", LogisticRegression(
        max_iter=1000,
        C=0.1,  # Thêm regularization
        class_weight="balanced",
        random_state=RANDOM_STATE,
        solver='liblinear'
    ))
])

logit_pipe.fit(X_train, y_train)

# Stratified K-Fold Cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_scores_logit = cross_val_score(
    logit_pipe, X_train, y_train,
    cv=skf, scoring='roc_auc', n_jobs=-1
)

y_pred_logit = logit_pipe.predict(X_test)
y_prob_logit = logit_pipe.predict_proba(X_test)[:, 1]

print("\n=== LOGISTIC REGRESSION RESULTS ===")
print(f"Cross-validation ROC-AUC: {cv_scores_logit.mean():.4f} (+/- {cv_scores_logit.std():.4f})")
print(f"Test ROC-AUC: {roc_auc_score(y_test, y_prob_logit):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_logit))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_logit))

# 6. RANDOM FOREST (MAIN MODEL) - với regularization mạnh hơn
print("\n" + "=" * 60)
print("TRAINING RANDOM FOREST...")
print("=" * 60)

rf_pipe = Pipeline([
    ("preprocess", preprocess),
    ("model", RandomForestClassifier(
        n_estimators=200,  # Giảm từ 300
        max_depth=8,       # Giảm từ 10
        min_samples_split=100,  # Tăng từ 2 (default)
        min_samples_leaf=50,
        max_features='sqrt',  # Thêm feature sampling
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    ))
])

# CV before final fit (more efficient, avoids refit waste)
cv_scores_rf = cross_val_score(
    rf_pipe, X_train, y_train,
    cv=skf, scoring='roc_auc', n_jobs=-1
)

# NEW: RF probability calibration (business-critical)
# Split train into fit vs calibration to avoid overconfident/extreme PDs (common issue in credit scoring).
X_train_fit, X_cal, y_train_fit, y_cal = train_test_split(
    X_train, y_train,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y_train
)

rf_pipe.fit(X_train_fit, y_train_fit)

# Calibrate predicted probabilities (sigmoid = Platt scaling)
rf_calibrated = CalibratedClassifierCV(rf_pipe, cv="prefit", method="sigmoid")
rf_calibrated.fit(X_cal, y_cal)

# Raw RF (for reference)
y_prob_rf_raw = rf_pipe.predict_proba(X_test)[:, 1]
y_pred_rf_raw = (y_prob_rf_raw >= 0.5).astype(int)

# Calibrated RF (use this for credit scoring)
y_prob_rf = rf_calibrated.predict_proba(X_test)[:, 1]
y_pred_rf = (y_prob_rf >= 0.5).astype(int)

print("\n=== RANDOM FOREST RESULTS ===")
print(f"Cross-validation ROC-AUC (raw RF): {cv_scores_rf.mean():.4f} (+/- {cv_scores_rf.std():.4f})")
print(f"Test ROC-AUC (raw RF): {roc_auc_score(y_test, y_prob_rf_raw):.4f}")
print(f"Test ROC-AUC (CALIBRATED RF): {roc_auc_score(y_test, y_prob_rf):.4f}")
print(f"CV Std Dev: {cv_scores_rf.std():.4f} (lower is better)")

# Check for overfitting (raw model check)
train_score = roc_auc_score(y_train_fit, rf_pipe.predict_proba(X_train_fit)[:, 1])
test_score_raw = roc_auc_score(y_test, y_prob_rf_raw)
print(f"\nTrain ROC-AUC (raw): {train_score:.4f}")
print(f"Test ROC-AUC (raw): {test_score_raw:.4f}")
print(f"Train-Test Gap: {train_score - test_score_raw:.4f} (should be < 0.05)")

if train_score - test_score_raw > 0.05:
    print("⚠ WARNING: Model may be overfitting!")
else:
    print("✓ Model generalization looks good")

print("\nClassification Report (CALIBRATED RF):")
print(classification_report(y_test, y_pred_rf))
print("\nConfusion Matrix (CALIBRATED RF):")
print(confusion_matrix(y_test, y_pred_rf))

# 7. APPLY MODEL TO FULL DATA
print("\n" + "=" * 60)
print("APPLYING MODELS TO FULL DATASET...")
print("=" * 60)

# Logistic PD
df["pd_logit"] = logit_pipe.predict_proba(X)[:, 1]

# RF PD: save both raw and calibrated
df["pd_rf_raw"] = rf_pipe.predict_proba(X)[:, 1]
df["pd_rf"] = rf_calibrated.predict_proba(X)[:, 1]  # <- calibrated PD used for scoring

# Credit score from CALIBRATED RF (0-100 scale)
df["credit_score_ai"] = (1 - df["pd_rf"]) * 100

# 8. MAP CREDIT SCORE TO TIER
def map_risk_tier(score):
    if score >= 90:
        return "A+ (Prime Ultra-Low Risk)"
    elif score >= 80:
        return "A (Prime Low Risk)"
    elif score >= 70:
        return "B+ (Near-Prime Strong)"
    elif score >= 60:
        return "B (Near-Prime Medium)"
    elif score >= 50:
        return "C+ (Subprime Mild)"
    elif score >= 40:
        return "C (Subprime High)"
    elif score >= 30:
        return "D+ (Severe Risk)"
    elif score >= 20:
        return "D (Very Severe Risk)"
    else:
        return "E (Default/Fraud Risk)"

df["risk_tier_ai"] = df["credit_score_ai"].apply(map_risk_tier)

# 9. SAVE OUTPUT
print("\n" + "=" * 60)
print("SAVING OUTPUTS...")
print("=" * 60)

output_cols = [
    "credit_score_ai",
    "risk_tier_ai",
    "pd_rf",        # calibrated
    "pd_rf_raw",    # raw reference
    "pd_logit",
    "high_risk"
]

# Save to Excel
try:
    df.to_excel("sme_ai_credit_scoring.xlsx", index=False)
    print("✓ Saved: sme_ai_credit_scoring.xlsx")
except ImportError:
    print("⚠ openpyxl not installed. Saving to CSV instead...")
    df.to_csv("sme_ai_credit_scoring.csv", index=False)
    print("✓ Saved: sme_ai_credit_scoring.csv")

print(f"\nFinal shape: {df.shape}")
print(f"\nSample predictions:")
print(df[output_cols].head(10))

# Distribution of scores
print(f"\nCredit Score Distribution:")
print(df["credit_score_ai"].describe())
print(f"\nRisk Tier Distribution:")
print(df["risk_tier_ai"].value_counts().sort_index())

# 10. SAVE MODELS
import joblib

joblib.dump(logit_pipe, "logit_model.pkl")

# NEW:
# Save calibrated RF as the main production model to get more stable PDs.
joblib.dump(rf_calibrated, "rf_model.pkl")

# Optional: keep raw pipeline for debugging / SHAP parity
joblib.dump(rf_pipe, "rf_model_raw.pkl")

joblib.dump(X.columns.tolist(), "feature_columns.pkl")

print("\n✓ Models saved:")
print("  - logit_model.pkl")
print("  - rf_model.pkl (CALIBRATED)")
print("  - rf_model_raw.pkl (RAW, optional)")
print("  - feature_columns.pkl")

# 11. FEATURE IMPORTANCE
print("\n" + "=" * 60)
print("FEATURE IMPORTANCE ANALYSIS...")
print("=" * 60)

try:
    # Get feature importance from Random Forest (raw model inside pipeline)
    rf_model = rf_pipe.named_steps['model']

    # Get feature names after preprocessing
    preprocessor = rf_pipe.named_steps['preprocess']

    # Get feature names from transformer
    feature_names = []
    for name, trans, cols in preprocessor.transformers_:
        if name == 'num':
            feature_names.extend(cols)
        elif name == 'cat':
            # trans is a Pipeline -> get the ohe step
            ohe = trans.named_steps.get("ohe")
            if hasattr(ohe, 'get_feature_names_out'):
                feature_names.extend(ohe.get_feature_names_out(cols))

    # Create importance dataframe
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nTOP 20 MOST IMPORTANT FEATURES:")
    print(importance_df.head(20).to_string(index=False))

    # Check concentration
    top5_sum = importance_df.head(5)['importance'].sum()
    print(f"\nTop 5 features importance: {top5_sum:.2%}")
    if top5_sum > 0.80:
        print("⚠ WARNING: High feature concentration - may indicate data leakage or overfitting")

    # Save to CSV
    importance_df.to_csv("feature_importance.csv", index=False)
    print("\n✓ Feature importance saved: feature_importance.csv")

except Exception as e:
    print(f"⚠ Feature importance extraction failed: {e}")

# 12. SHAP EXPLAINABILITY (Fixed version)
print("\n" + "=" * 60)
print("GENERATING SHAP EXPLANATIONS...")
print("=" * 60)

try:
    import shap
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt

    # Transform data first
    X_train_sample = X_train.sample(min(500, len(X_train)), random_state=RANDOM_STATE)
    X_train_sample_transformed = rf_pipe.named_steps['preprocess'].transform(X_train_sample)

    # Use TreeExplainer directly on the model (not pipeline)
    rf_model = rf_pipe.named_steps['model']
    explainer = shap.TreeExplainer(rf_model)

    # Calculate SHAP values on transformed data
    shap_values = explainer.shap_values(X_train_sample_transformed)

    # For binary classification, take values for positive class
    if isinstance(shap_values, list):
        shap_values_pos = shap_values[1]
    else:
        shap_values_pos = shap_values

    # Get feature names
    preprocessor = rf_pipe.named_steps['preprocess']
    feature_names = []
    for name, trans, cols in preprocessor.transformers_:
        if name == 'num':
            feature_names.extend(cols)
        elif name == 'cat':
            ohe = trans.named_steps.get("ohe")
            if hasattr(ohe, 'get_feature_names_out'):
                feature_names.extend(ohe.get_feature_names_out(cols))

    # Create summary plot with better visualization
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values_pos,
        X_train_sample_transformed,
        feature_names=feature_names,
        show=False,
        max_display=20,
        plot_size=(12, 8)
    )

    plt.tight_layout()
    plt.savefig("shap_summary.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Create bar plot for feature importance
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values_pos,
        X_train_sample_transformed,
        feature_names=feature_names,
        plot_type="bar",
        show=False,
        max_display=20
    )
    plt.tight_layout()
    plt.savefig("shap_bar_plot.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("✓ SHAP bar plot saved: shap_bar_plot.png")
    print("✓ SHAP summary saved: shap_summary.png")

except ImportError:
    print("⚠ SHAP not installed. Install with: pip3 install shap")
except Exception as e:
    print(f"⚠ SHAP generation failed: {str(e)[:200]}")

# 13. LIME EXPLAINABILITY (Fixed version)
print("\n" + "=" * 60)
print("GENERATING LIME EXPLANATION...")
print("=" * 60)

try:
    from lime.lime_tabular import LimeTabularExplainer

    # Use only numerical features for LIME to avoid string operations
    X_train_numeric = X_train[num_cols].copy() if len(num_cols) else pd.DataFrame()
    X_test_numeric = X_test[num_cols].copy() if len(num_cols) else pd.DataFrame()

    # Fill NaNs for lime
    if len(num_cols):
        X_train_numeric = X_train_numeric.fillna(X_train_numeric.median(numeric_only=True))
        X_test_numeric = X_test_numeric.fillna(X_train_numeric.median(numeric_only=True))

    # Create a simplified pipeline for LIME
    def rf_predict_proba_lime(x):
        x_df = pd.DataFrame(x, columns=num_cols)

        # Add categorical columns with default values
        for col in cat_cols:
            # mode() may fail if empty; guard
            try:
                x_df[col] = X_train[col].mode(dropna=True)[0]
            except Exception:
                x_df[col] = "UNKNOWN"

        # Reorder columns
        x_df = x_df[X.columns]

        # IMPORTANT: use calibrated model for probabilities (business)
        return rf_calibrated.predict_proba(x_df)

    lime_explainer = LimeTabularExplainer(
        training_data=X_train_numeric.values if len(num_cols) else np.zeros((1, 1)),
        feature_names=num_cols if len(num_cols) else ["dummy"],
        class_names=["Low Risk", "High Risk"],
        mode="classification"
    )

    # Explain first high-risk case in test set
    high_risk_idx = np.where(y_test == 1)[0]
    if len(high_risk_idx) > 0 and len(num_cols) > 0:
        idx = high_risk_idx[0]

        lime_exp = lime_explainer.explain_instance(
            X_test_numeric.values[idx],
            rf_predict_proba_lime,
            num_features=min(10, len(num_cols))
        )

        lime_exp.save_to_file("lime_explanation.html")
        print(f"✓ LIME explanation saved: lime_explanation.html")
        print(f"  Explained instance {idx}: Actual={y_test.iloc[idx]}, Predicted Prob={y_prob_rf[idx]:.3f}")
    else:
        print("⚠ No high-risk cases in test set for LIME or no numeric features available")

except ImportError:
    print("⚠ LIME not installed. Install with: pip3 install lime")
except Exception as e:
    print(f"⚠ LIME generation failed: {str(e)[:200]}")

print("\n" + "=" * 60)
print("PIPELINE COMPLETED!")
print("=" * 60)

# Final model quality check
print("\n=== MODEL QUALITY ASSESSMENT ===")
print(f"Logistic Regression:")
print(f"  - CV ROC-AUC: {cv_scores_logit.mean():.4f} ± {cv_scores_logit.std():.4f}")
print(f"  - Test ROC-AUC: {roc_auc_score(y_test, y_prob_logit):.4f}")

print(f"\nRandom Forest:")
print(f"  - CV ROC-AUC (raw): {cv_scores_rf.mean():.4f} ± {cv_scores_rf.std():.4f}")
print(f"  - Test ROC-AUC (raw): {roc_auc_score(y_test, y_prob_rf_raw):.4f}")
print(f"  - Test ROC-AUC (CALIBRATED): {roc_auc_score(y_test, y_prob_rf):.4f}")
print(f"  - Train-Test Gap (raw): {train_score - test_score_raw:.4f}")

if cv_scores_rf.mean() > 0.98 and (train_score - test_score_raw) < 0.05:
    print("\n⚠ WARNING: Model performance is suspiciously high.")
    print("  This may indicate remaining data leakage in the synthetic data.")
    print("  Review feature correlations with label before production use.")
elif (train_score - test_score_raw) > 0.10:
    print("\n⚠ WARNING: Significant overfitting detected.")
    print("  Consider further hyperparameter tuning or feature selection.")
else:
    print("\n✓ Model quality looks acceptable for production.")

print("\nGenerated files:")
print("  1. sme_ai_credit_scoring.xlsx (or .csv)")
print("  2. logit_model.pkl")
print("  3. rf_model.pkl (CALIBRATED)")
print("  4. rf_model_raw.pkl (RAW, optional)")
print("  5. feature_columns.pkl")
print("  6. feature_importance.csv")
print("  7. shap_summary.png - Detailed SHAP plot")
print("  8. shap_bar_plot.png - Feature importance bar chart")
print("  9. lime_explanation.html (if successful)")
# MODEL INTEGRITY HASH 
import hashlib

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()

model_files = [
    "rf_model.pkl",
    "logit_model.pkl",
    "feature_columns.pkl"
]

hashes = {}
for f in model_files:
    hashes[f] = sha256_file(f)
    print(f"[HASH] {f}: {hashes[f]}")

# Save hashes to file for deployment verification
with open("model_hashes.txt", "w") as out:
    for k, v in hashes.items():
        out.write(f"{k}:{v}\n")

print("✓ Model hashes saved to model_hashes.txt")