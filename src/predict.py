"""
predict.py - CLI tool with Phishing Explanations (TreeSHAP).
"""
import sys
import os
import joblib
import pandas as pd
import numpy as np
from scipy.sparse import hstack

# Add src to path
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
from features import extract_features

# Feature-to-Reason Map
REASON_MAP = {
    "char_substitution_risk": "Lookalike domain detected (typosquatting)",
    "digit_density_in_hostname": "Unusually high number of digits in domain",
    "domain_entropy": "Domain name appears randomly generated",
    "num_suspicious_words": "Contains scam-related keywords (login, verify, account)",
    "brand_in_subdomain": "Impersonation risk (reputable brand in subdomain)",
    "url_length": "URL is suspiciously long",
    "path_depth": "Excessive nesting in URL path",
    "is_suspicious_tld": "Uses a domain extension frequently linked to phishing (.xyz, .top, etc.)",
    "is_professional_tld": "Professional/Tech domain extension (.io, .app, .dev)",
    "num_special_chars": "Excessive special characters in URL",
    "has_ip": "Hostname is a raw IP address (highly suspicious)"
}

def get_explanation(model, X_final, feature_names):
    """Extract top positive contributions for the specific prediction."""
    try:
        # pred_contrib=True can return a sparse matrix if X_final is sparse
        all_contribs = model.booster_.predict(X_final, pred_contrib=True)
        if hasattr(all_contribs, "toarray"):
            contribs = all_contribs.toarray()[0]
        else:
            contribs = all_contribs[0]
            
        contrib_values = contribs[:-1]
        
        # Get indices of top 3 positive contributors (driving PHISH prob UP)
        top_idx = np.argsort(contrib_values)[-3:][::-1]
        
        reasons = []
        for idx in top_idx:
            val = contrib_values[idx]
            if val > 0.05: # Only show significant drivers
                feat_name = feature_names[idx]
                reason = REASON_MAP.get(feat_name)
                if not reason:
                    # If it's a TF-IDF character pattern
                    if feat_name in feature_names[:len(feature_names)-len(REASON_MAP)]:
                        reason = f"Suspicious character pattern: '{feat_name}'"
                    else:
                        reason = f"Anatomy anomaly: {feat_name}"
                reasons.append(reason)
        return list(set(reasons)) # Deduplicate
    except Exception as e:
        return [f"Detailed analysis failed: {str(e)}"]

def main():
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py <url>")
        return

    url = sys.argv[1]
    
    if not os.path.exists("phishing_pipeline.joblib"):
        print("Error: phishing_pipeline.joblib not found. Run src/save_pipeline.py first.")
        return

    print(f"Analyzing: {url}")
    
    # Normalize input
    if not url.startswith(('http://', 'https://')):
        orig_url = url
        url = 'http://' + url.strip()
    else:
        orig_url = url

    # Load pipeline
    pipeline = joblib.load("phishing_pipeline.joblib")
    vec = pipeline["vectorizer"]
    scaler = pipeline["scaler"]
    model = pipeline["model"]
    lex_cols = pipeline.get("lex_cols", [])
    
    # 1. Extract lexical features
    lex_feat = extract_features(url)
    
    # Ensure DataFrame columns match exactly the scaler's fit state
    lex_df = pd.DataFrame([lex_feat])
    if lex_cols:
        # Sort/Reorder to match fit-time column order
        lex_df = lex_df[lex_cols]
    
    # 2. Transform
    X_tfidf = vec.transform([url])
    X_lex = scaler.transform(lex_df)
    
    # 3. Combine
    X_final = hstack([X_tfidf, X_lex]).tocsr()
    
    # 4. Predict
    prob_phish = model.predict_proba(X_final)[0, 1]
    
    # 5. Safety Override & Reason Generation
    final_pred = "PHISHING" if prob_phish >= 0.5 else "BENIGN"
    override_reason = None
    reasons = []

    # Map all feature names for SHAP
    feature_names = list(vec.get_feature_names_out()) + lex_cols

    if lex_feat["is_reputable"]:
        final_pred = "BENIGN"
        prob_phish = min(prob_phish, 0.49)
        override_reason = "Reputation Whitelist (High Trust Domain)"
    else:
        # Only extract reasons for phishing (or borderline)
        if prob_phish > 0.3:
            reasons = get_explanation(model, X_final, feature_names)

    # Secondary Override for Typosquats
    if lex_feat["char_substitution_risk"] > 0 and not lex_feat["is_reputable"]:
        if prob_phish < 0.5:
            final_pred = "PHISHING (TYPOSQUAT)"
            prob_phish = max(prob_phish, 0.75)
            override_reason = "Typosquatting Pattern Detected (e.g. 1 for l, 0 for o)"
            reasons.append(REASON_MAP["char_substitution_risk"])

    print("\n--- Model Prediction ---")
    print(f"Result: {final_pred}")
    print(f"Phishing Probability: {prob_phish:.2%}")
    if override_reason:
        print(f"Confidence Note: {override_reason}")
    
    if final_pred != "BENIGN" and reasons:
        print("\n--- Why is this phishing? ---")
        for i, r in enumerate(reasons[:3]):
            print(f"  {i+1}. {r}")

    print("\n--- Key Statistics ---")
    for k in ["url_length", "hostname_length", "domain_entropy", "char_substitution_risk", "num_suspicious_words"]:
        if k in lex_feat:
            print(f"  {k:<22}: {lex_feat[k]}")

if __name__ == "__main__":
    main()
