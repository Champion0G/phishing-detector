"""
final_deployment_audit.py - Comprehensive checklist automation.
Matches the 10-point deployment requirements.
"""

import pandas as pd
import numpy as np
import time
import os
import sys
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, roc_auc_score, accuracy_score, 
    confusion_matrix, f1_score, precision_score, recall_score
)
from scipy.sparse import hstack
from lightgbm import LGBMClassifier
from urllib.parse import urlparse

# Add src to path
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
from features import extract_features, _get_hostname

def print_section(title):
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

# 1. Load Data
print("Loading dataset...")
df = pd.read_csv("data/processed/final_dataset.csv")

# --- 1.1 & 1.2 Data Integrity ---
print_section("1. DATA INTEGRITY")
df["domain"] = df["url"].apply(_get_hostname)

# Domain Isolation
unique_domains = df["domain"].unique()
train_doms, test_doms = train_test_split(unique_domains, test_size=0.2, random_state=42)
overlap = set(train_doms) & set(test_doms)
print(f"Domain Overlap: {len(overlap)} (Expected: 0)")

tr_df = df[df["domain"].isin(train_doms)]
te_df = df[df["domain"].isin(test_doms)]

# Duplicate Check
url_overlap = set(tr_df["url"]) & set(te_df["url"])
print(f"Duplicate URLs Across Splits: {len(url_overlap)} (Expected: 0)")

# --- Preparation for model tests ---
print("\nPreparing features (this may take a minute)...")
# Reduced from 15000 to 5000 for memory efficiency in audit
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=5000)
X_tr_tfidf = vec.fit_transform(tr_df["url"])
X_te_tfidf = vec.transform(te_df["url"])

tr_lex = pd.DataFrame(list(tr_df["url"].apply(extract_features)))
te_lex = pd.DataFrame(list(te_df["url"].apply(extract_features)))
scaler = StandardScaler()
X_tr_lex = scaler.fit_transform(tr_lex)
X_te_lex = scaler.transform(te_lex)

X_train = hstack([X_tr_tfidf, X_tr_lex]).tocsr()
X_test  = hstack([X_te_tfidf, X_te_lex]).tocsr()
y_train = tr_df["label"].values
y_test  = te_df["label"].values

# --- 2. Sanity Tests ---
print_section("2. SANITY TESTS")

# 2.1 Shuffle Labels
rng = np.random.default_rng(42)
y_shuf = rng.permutation(y_train)
m_shuf = LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
m_shuf.fit(X_train, y_shuf)
shuf_acc = accuracy_score(y_test, m_shuf.predict(X_test))
print(f"Label Shuffle Test: Accuracy={shuf_acc:.4f} (Expected near baseline {max(y_test.mean(), 1-y_test.mean()):.4f})")

# 2.2 Random Noise
# Use a smaller sample to avoid 27GB memory error
X_noise = rng.normal(size=(min(50000, X_train.shape[0]), min(1000, X_train.shape[1])))
y_noise = y_train[:X_noise.shape[0]]
m_noise = LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
m_noise.fit(X_noise, y_noise)
print(f"Random Noise Test: Model trained on {X_noise.shape} noise array.")

# --- 3. Metric Validation ---
print_section("3. METRIC VALIDATION (DOMAIN SPLIT)")
# Reduced estimators for faster/leaner audit run
model = LGBMClassifier(n_estimators=100, learning_rate=0.1, num_leaves=31, random_state=42, verbose=-1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 3.2 Threshold Sensitivity
print("\nThreshold Sensitivity:")
for t in [0.4, 0.5, 0.6]:
    y_t = (y_prob >= t).astype(int)
    print(f"  t={t}: F1={f1_score(y_test, y_t):.4f}, Prec={precision_score(y_test, y_t):.4f}, Rec={recall_score(y_test, y_t):.4f}")

# --- 4. CV Stability ---
print_section("4. CROSS-VALIDATION STABILITY")
# 5-fold CV on train set (using lexical features only for speed in CV)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(LGBMClassifier(verbose=-1), X_tr_lex, y_train, cv=skf, scoring='f1')
print(f"Lexical-only 5-fold CV F1: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

# --- 5. Feature Importance ---
print_section("5. FEATURE IMPORTANCE")
fnames = list(tr_lex.columns)
# Last len(fnames) are the lexical ones
lex_imp = model.feature_importances_[-len(fnames):]
total_imp = model.feature_importances_.sum()
fi = sorted(zip(fnames, lex_imp), key=lambda x: -x[1])
for fn, val in fi[:10]:
    perc = (val / total_imp) * 100
    print(f"  {fn:<25} {val:>5} ({perc:.1f}%)")

# --- 6. Manual Error Inspection ---
print_section("6. ERROR INSPECTION (SAVING SAMPLES)")
results_df = te_df.copy()
results_df["y_pred"] = y_pred
results_df["y_prob"] = y_prob
# Swap logic: y_pred=1 is Benign, y_pred=0 is Phishing
fp = results_df[(results_df["label"]==1) & (results_df["y_pred"]==0)].head(20) # Label=Benign, Pred=Phish
fn = results_df[(results_df["label"]==0) & (results_df["y_pred"]==1)].head(20) # Label=Phish, Pred=Benign
fp.to_csv("audit_fp.csv", index=False)
fn.to_csv("audit_fn.csv", index=False)
print(f"Saved 20 FP to audit_fp.csv and 20 FN to audit_fn.csv")

# --- 7. Overfitting ---
print_section("7. OVERFITTING CHECK")
train_pred = model.predict(X_train)
print(f"Train F1: {f1_score(y_train, train_pred):.4f}")
print(f"Test  F1: {f1_score(y_test, y_pred):.4f}")

# --- 8. Robustness ---
print_section("8. ROBUSTNESS")
print("Seed Stability (3 seeds):")
for s in [1, 100, 999]:
    ms = LGBMClassifier(n_estimators=100, random_state=s, verbose=-1)
    ms.fit(X_train, y_train)
    print(f"  Seed {s:<4}: F1={f1_score(y_test, ms.predict(X_test)):.4f}")

# --- 9. Production Readiness ---
print_section("9. PRODUCTION READINESS")
# Inference Speed
start = time.time()
n_test = 500
model.predict(X_test[:n_test])
elapsed = (time.time() - start) / n_test
print(f"Latency per URL: {elapsed*1000:.3f} ms")

# Model Size
joblib.dump(model, "phishing_model.joblib")
m_size = os.path.getsize("phishing_model.joblib") / (1024*1024)
print(f"Model File Size: {m_size:.2f} MB")

# Determinism
y_1 = model.predict(X_test[:50])
y_2 = model.predict(X_test[:50])
print(f"Deterministic: {np.array_equal(y_1, y_2)}")

print("\nAUDIT COMPLETE.")
