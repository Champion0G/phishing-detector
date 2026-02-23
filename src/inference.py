"""
src/inference.py - Production inference wrapper for the Phishing Detector.
"""
import os
import sys
import joblib
import pandas as pd
import numpy as np
from scipy.sparse import hstack

# Add root to path for local imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.features import extract_features

# Feature-to-Reason Map
REASON_MAP = {
    "char_substitution_risk": "Lookalike domain detected (typosquatting)",
    "digit_density_in_hostname": "Unusually high number of digits in domain",
    "domain_entropy": "Domain name appears randomly generated",
    "num_suspicious_words": "Contains scam-related keywords",
    "brand_in_subdomain": "Impersonation risk (brand in subdomain)",
    "url_length": "URL is suspiciously long",
    "path_depth": "Excessive nesting in URL path",
    "is_suspicious_tld": "Uses a domain extension linked to phishing",
    "is_professional_tld": "Professional/Tech domain extension",
    "has_ip": "Hostname is a raw IP address"
}

# Load the model once (Singleton-ish pattern for API)
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "phishing_v14.joblib")
if os.path.exists(MODEL_PATH):
    PIPELINE = joblib.load(MODEL_PATH)
else:
    PIPELINE = None

def get_explanation(model, X_final, feature_names):
    """Extract top positive contributions for the specific prediction."""
    try:
        all_contribs = model.booster_.predict(X_final, pred_contrib=True)
        if hasattr(all_contribs, "toarray"):
            contribs = all_contribs.toarray()[0]
        else:
            contribs = all_contribs[0]
            
        contrib_values = contribs[:-1]
        top_idx = np.argsort(contrib_values)[-3:][::-1]
        
        reasons = []
        for idx in top_idx:
            val = contrib_values[idx]
            if val > 0.05:
                feat_name = feature_names[idx]
                reason = REASON_MAP.get(feat_name)
                if not reason:
                    if feat_name in feature_names[:len(feature_names)-len(lex_cols)]:
                        reason = f"Suspicious character pattern: '{feat_name}'"
                    else:
                        reason = f"Anatomy anomaly: {feat_name}"
                reasons.append(reason)
        return list(set(reasons))
    except:
        return []

def predict_url(url: str):
    if not PIPELINE:
        return {"error": "Model not found. Build the model first."}

    # Normalize
    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url.strip()

    vec = PIPELINE["vectorizer"]
    scaler = PIPELINE["scaler"]
    model = PIPELINE["model"]
    lex_cols = PIPELINE.get("lex_cols", [])

    # 1. Extract
    lex_feat = extract_features(url)
    lex_df = pd.DataFrame([lex_feat])[lex_cols]
    
    # 2. Transform
    X_tfidf = vec.transform([url])
    X_lex = scaler.transform(lex_df)
    X_final = hstack([X_tfidf, X_lex]).tocsr()
    
    # 3. Predict
    prob_phish = float(model.predict_proba(X_final)[0, 1])
    
    # 4. Overrides & Reasons
    final_pred = "PHISHING" if prob_phish >= 0.5 else "BENIGN"
    override_reason = None
    reasons = []

    feature_names = list(vec.get_feature_names_out()) + lex_cols

    if lex_feat["is_reputable"]:
        final_pred = "BENIGN"
        prob_phish = min(prob_phish, 0.49)
        override_reason = "Reputation Whitelist"
    else:
        if prob_phish > 0.3:
            reasons = get_explanation(model, X_final, feature_names)

    if lex_feat["char_substitution_risk"] > 0 and not lex_feat["is_reputable"]:
        if prob_phish < 0.5:
            final_pred = "PHISHING (TYPOSQUAT)"
            prob_phish = max(prob_phish, 0.75)
            override_reason = "Typosquatting Pattern"
            reasons.append(REASON_MAP["char_substitution_risk"])

    return {
        "url": url,
        "result": final_pred,
        "probability": round(prob_phish * 100, 2),
        "reasons": reasons,
        "note": override_reason,
        "stats": {
            "length": lex_feat["url_length"],
            "entropy": round(lex_feat["domain_entropy"], 2),
            "is_tech_tld": bool(lex_feat["is_professional_tld"])
        }
    }
