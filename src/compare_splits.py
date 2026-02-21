"""
Option 1 — compare_splits.py
Runs the same LightGBM model under two evaluation strategies:
  • Random split  (optimistic / train-test overlap in domain space)
  • Domain split  (realistic / unseen domains at test time)
Prints a side-by-side summary table.
"""

import pandas as pd
import numpy as np
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from scipy.sparse import hstack
from lightgbm import LGBMClassifier
from features import extract_features

# ── Shared helpers ─────────────────────────────────────────────────────────────

def build_features(train_urls, test_urls):
    """Return combined (TF-IDF + lexical) matrices for train and test."""
    # TF-IDF on character n-grams
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=15000)
    X_tr_tfidf = vec.fit_transform(train_urls)
    X_te_tfidf = vec.transform(test_urls)

    # Lexical features
    tr_lex = pd.DataFrame(list(train_urls.apply(extract_features)))
    te_lex = pd.DataFrame(list(test_urls.apply(extract_features)))

    scaler = StandardScaler()
    X_tr_lex = scaler.fit_transform(tr_lex)
    X_te_lex = scaler.transform(te_lex)

    return hstack([X_tr_tfidf, X_tr_lex]), hstack([X_te_tfidf, X_te_lex])


def train_and_eval(X_train, X_test, y_train, y_test):
    model = LGBMClassifier(n_estimators=300, learning_rate=0.1,
                           num_leaves=31, random_state=42, verbose=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, y_pred, output_dict=True)
    return {
        "accuracy":  report["accuracy"],
        "precision": report["weighted avg"]["precision"],
        "recall":    report["weighted avg"]["recall"],
        "f1":        report["weighted avg"]["f1-score"],
        "roc_auc":   roc_auc_score(y_test, y_prob),
    }


# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv("data/processed/final_dataset.csv")

# ── Random split ───────────────────────────────────────────────────────────────
print("\n[1/2] Random Split (optimistic) ...")
X_tr, X_te, y_tr, y_te = train_test_split(
    df["url"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)
X_train_r, X_test_r = build_features(X_tr, X_te)
results_random = train_and_eval(X_train_r, X_test_r, y_tr, y_te)

# ── Domain split ───────────────────────────────────────────────────────────────
print("[2/2] Domain Split (realistic) ...")
df["domain"] = df["url"].apply(lambda u: urlparse(u).hostname)
unique_domains = df["domain"].dropna().unique()
train_doms, test_doms = train_test_split(unique_domains, test_size=0.2, random_state=42)
tr_df = df[df["domain"].isin(train_doms)]
te_df = df[df["domain"].isin(test_doms)]
X_train_d, X_test_d = build_features(tr_df["url"], te_df["url"])
results_domain = train_and_eval(X_train_d, X_test_d, tr_df["label"], te_df["label"])

# ── Summary table ──────────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("  OPTION 1 — Random Split vs Domain Split (LightGBM Baseline)")
print("=" * 62)
header = f"{'Metric':<18} {'Random Split':>16} {'Domain Split':>16}  {'Delta':>8}"
print(header)
print("-" * 62)
metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
for m in metrics:
    rv = results_random[m]
    dv = results_domain[m]
    delta = dv - rv
    sign  = "+" if delta >= 0 else ""
    print(f"  {m:<16} {rv:>16.4f} {dv:>16.4f}  {sign}{delta:.4f}")
print("=" * 62)
print("\n📌 Interpretation:")
print("  Random split = same domain appears in train & test → optimistic")
print("  Domain split = model sees only NEW domains at test time → realistic")
print("  A large drop in domain-split scores means the model memorised")
print("  URL patterns specific to known domains (overfitting to domain identity).\n")
