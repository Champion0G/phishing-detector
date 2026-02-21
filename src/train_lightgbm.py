import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from scipy.sparse import hstack
from lightgbm import LGBMClassifier
from features import extract_features

print("Loading dataset...")
df = pd.read_csv("data/processed/final_dataset.csv")

X_text = df["url"]
y = df["label"]

print("Extracting lexical features...")
lexical = df["url"].apply(lambda x: extract_features(x))
lexical_df = pd.DataFrame(list(lexical))

X_train_text, X_test_text, y_train, y_test, X_train_lex, X_test_lex = train_test_split(
    X_text,
    y,
    lexical_df,
    test_size=0.2,
    random_state=42,
    stratify=y
)

vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3,5),
    max_features=15000
)

print("Fitting TF-IDF...")
X_train_tfidf = vectorizer.fit_transform(X_train_text)
X_test_tfidf = vectorizer.transform(X_test_text)

scaler = StandardScaler()
X_train_lex = scaler.fit_transform(X_train_lex)
X_test_lex = scaler.transform(X_test_lex)

X_train_combined = hstack([X_train_tfidf, X_train_lex])
X_test_combined = hstack([X_test_tfidf, X_test_lex])

print("Training LightGBM...")
model = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.1,
    num_leaves=31,
    random_state=42
)

model.fit(X_train_combined, y_train)

y_pred = model.predict(X_test_combined)
y_prob = model.predict_proba(X_test_combined)[:,1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))