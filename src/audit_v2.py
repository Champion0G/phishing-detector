"""
audit_v2.py — Clean, robust audit of the phishing detector.
Fixes hostname extraction bug and reruns all checks.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from scipy.sparse import hstack
from lightgbm import LGBMClassifier
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
from features import extract_features, _get_hostname

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv("data/processed/final_dataset.csv")

# Use robust extraction
df["domain"] = df["url"].apply(_get_hostname)

print(f"Total rows: {len(df):,}")
print(f"Unique domains: {df['domain'].nunique():,}")
print(f"Empty domains: {(df['domain'] == '').sum():,}")

# ── Domain Split ───────────────────────────────────────────────────────────────
# We use ALL domains now
unique_domains = df["domain"].unique()
train_doms, test_doms = train_test_split(unique_domains, test_size=0.2, random_state=42)
train_dom_set = set(train_doms)
test_dom_set  = set(test_doms)

tr_df = df[df["domain"].isin(train_dom_set)]
te_df = df[df["domain"].isin(test_dom_set)]

print(f"\n=== DOMAIN SPLIT SANITY ===")
print(f"Train URLs: {len(tr_df):,}")
print(f"Test  URLs: {len(te_df):,}")
overlap = set(train_doms) & set(test_doms)
print(f"Overlap: {len(overlap)}")

# ── Feature Building ───────────────────────────────────────────────────────────
print("\nFitting TF-IDF and extracting lexical features...")
vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=15000)
X_train_tfidf = vectorizer.fit_transform(tr_df["url"])
X_test_tfidf  = vectorizer.transform(te_df["url"])

tr_lex = pd.DataFrame(list(tr_df["url"].apply(extract_features)))
te_lex = pd.DataFrame(list(te_df["url"].apply(extract_features)))

scaler = StandardScaler()
X_train_lex = scaler.fit_transform(tr_lex)
X_test_lex  = scaler.transform(te_lex)

X_train = hstack([X_train_tfidf, X_train_lex])
X_test  = hstack([X_test_tfidf, X_test_lex])
y_train = tr_df["label"].values
y_test  = te_df["label"].values

# ── Real Model ─────────────────────────────────────────────────────────────────
print("\n=== REAL MODEL (ENHANCED FEATURES) ===")
model = LGBMClassifier(n_estimators=300, learning_rate=0.1, num_leaves=31, random_state=42, verbose=-1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))
print("ROC-AUC:", round(roc_auc_score(y_test, y_prob), 6))

# ── Shuffle Sanity Test ────────────────────────────────────────────────────────
print("\n=== SHUFFLE LABEL SANITY TEST ===")
rng = np.random.default_rng(99)
y_shuffled = rng.permutation(y_train)

model_shuf = LGBMClassifier(n_estimators=100, learning_rate=0.1, num_leaves=31, random_state=42, verbose=-1)
model_shuf.fit(X_train, y_shuffled)
y_shuf_pred = model_shuf.predict(X_test)
shuf_acc = accuracy_score(y_test, y_shuf_pred)
print(f"Shuffled-label accuracy on real test set: {shuf_acc:.4f}")
base_acc = max(y_test.mean(), 1 - y_test.mean())
print(f"Baseline (majority class) accuracy: {base_acc:.4f}")

if shuf_acc < base_acc + 0.05:
    print("PASS -- Model predicts near baseline with shuffled labels.")
else:
    print("FAIL -- Model still predicts well with shuffled labels. LEAKAGE!")

# ── Feature Importance ─────────────────────────────────────────────────────────
print("\n=== TOP LEXICAL FEATURES ===")
importances = model.feature_importances_
lex_names = list(tr_lex.columns)
lex_imp = importances[-len(lex_names):]
fi = sorted(zip(lex_names, lex_imp), key=lambda x: -x[1])
for name, val in fi[:10]:
    print(f"  {name:<30} {val}")
