"""
Option 2 — train_improved_features_domain_split.py
LightGBM trained on enhanced lexical features — DOMAIN SPLIT evaluation.
"""

import pandas as pd
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from scipy.sparse import hstack
from lightgbm import LGBMClassifier
from features import extract_features

print("Loading dataset...")
df = pd.read_csv("data/processed/final_dataset.csv")
df["domain"] = df["url"].apply(lambda u: urlparse(u).hostname)

# Domain-level split
unique_domains = df["domain"].dropna().unique()
train_doms, test_doms = train_test_split(unique_domains, test_size=0.2, random_state=42)
tr_df = df[df["domain"].isin(train_doms)]
te_df = df[df["domain"].isin(test_doms)]

print(f"Train size: {len(tr_df)}  |  Test size: {len(te_df)}")

print("Extracting enhanced lexical features...")
tr_lex = pd.DataFrame(list(tr_df["url"].apply(extract_features)))
te_lex = pd.DataFrame(list(te_df["url"].apply(extract_features)))

print("Fitting TF-IDF (char 3-5 grams)...")
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=15000)
X_tr_tfidf = vec.fit_transform(tr_df["url"])
X_te_tfidf = vec.transform(te_df["url"])

scaler = StandardScaler()
X_tr_lex_s = scaler.fit_transform(tr_lex)
X_te_lex_s  = scaler.transform(te_lex)

X_train = hstack([X_tr_tfidf, X_tr_lex_s])
X_test  = hstack([X_te_tfidf, X_te_lex_s])

print("Training LightGBM (enhanced features, domain split)...")
model = LGBMClassifier(n_estimators=300, learning_rate=0.1,
                       num_leaves=31, random_state=42, verbose=-1)
model.fit(X_train, tr_df["label"])

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n=== Option 2 — Enhanced Features | Domain Split ===")
print(classification_report(te_df["label"], y_pred))
print("ROC-AUC:", round(roc_auc_score(te_df["label"], y_prob), 4))
