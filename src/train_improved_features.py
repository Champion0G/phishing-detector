"""
Option 2 — train_improved_features.py
LightGBM trained on enhanced lexical features — RANDOM SPLIT evaluation.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from scipy.sparse import hstack
from lightgbm import LGBMClassifier
from features import extract_features

print("Loading dataset...")
df = pd.read_csv("data/processed/final_dataset.csv")

print("Extracting enhanced lexical features...")
lexical_df = pd.DataFrame(list(df["url"].apply(extract_features)))

X_tr_text, X_te_text, y_tr, y_te, X_tr_lex, X_te_lex = train_test_split(
    df["url"], df["label"], lexical_df,
    test_size=0.2, random_state=42, stratify=df["label"]
)

print("Fitting TF-IDF (char 3-5 grams)...")
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=15000)
X_tr_tfidf = vec.fit_transform(X_tr_text)
X_te_tfidf = vec.transform(X_te_text)

scaler = StandardScaler()
X_tr_lex_s = scaler.fit_transform(X_tr_lex)
X_te_lex_s  = scaler.transform(X_te_lex)

X_train = hstack([X_tr_tfidf, X_tr_lex_s])
X_test  = hstack([X_te_tfidf, X_te_lex_s])

print("Training LightGBM (enhanced features, random split)...")
model = LGBMClassifier(n_estimators=300, learning_rate=0.1,
                       num_leaves=31, random_state=42, verbose=-1)
model.fit(X_train, y_tr)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n=== Option 2 — Enhanced Features | Random Split ===")
print(classification_report(y_te, y_pred))
print("ROC-AUC:", round(roc_auc_score(y_te, y_prob), 4))
