"""
save_pipeline.py - Exports the full model pipeline for inference.
"""
import pandas as pd
import joblib
import os
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from scipy.sparse import hstack

# Add src to path
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
from features import extract_features, _get_hostname

print("Loading TYPOSQUAT-ROBUST dataset (V12) for final pipeline training...")
df = pd.read_csv("data/processed/final_dataset_v12.csv")
df["domain"] = df["url"].apply(_get_hostname)

# Use full dataset for production model
X_text = df["url"]
y = df["label"] # 1=Phish, 0=Benign

print("Fitting TF-IDF...")
# Increased to 15000 for better typosquatting detection
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=15000)
X_tfidf = vec.fit_transform(X_text)

print("Extracting lexical features...")
lex = pd.DataFrame(list(X_text.apply(extract_features)))
scaler = StandardScaler()
X_lex = scaler.fit_transform(lex)

X_final = hstack([X_tfidf, X_lex]).tocsr()

print("Training production model...")
# Increased num_leaves to 63 for better interaction learning (typosquatting)
model = LGBMClassifier(n_estimators=100, learning_rate=0.1, num_leaves=63, random_state=42, verbose=-1)
model.fit(X_final, y)

print("Saving pipeline components...")
pipeline = {
    "vectorizer": vec,
    "scaler": scaler,
    "model": model,
    "lex_cols": list(lex.columns)
}
joblib.dump(pipeline, "phishing_pipeline.joblib")
print("Successfully saved to phishing_pipeline.joblib")
