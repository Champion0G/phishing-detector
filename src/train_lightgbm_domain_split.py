import pandas as pd
import numpy as np
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

# Extract hostname
def get_domain(url):
    parsed = urlparse(url)
    return parsed.hostname

df["domain"] = df["url"].apply(get_domain)

# Get unique domains
unique_domains = df["domain"].unique()

# Split domains
train_domains, test_domains = train_test_split(
    unique_domains,
    test_size=0.2,
    random_state=42
)

train_df = df[df["domain"].isin(train_domains)]
test_df = df[df["domain"].isin(test_domains)]

print("Train size:", len(train_df))
print("Test size:", len(test_df))

X_train_text = train_df["url"]
y_train = train_df["label"]

X_test_text = test_df["url"]
y_test = test_df["label"]

# Lexical features
train_lex = train_df["url"].apply(lambda x: extract_features(x))
test_lex = test_df["url"].apply(lambda x: extract_features(x))

train_lex_df = pd.DataFrame(list(train_lex))
test_lex_df = pd.DataFrame(list(test_lex))

# TF-IDF
vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3,5),
    max_features=15000
)

print("Fitting TF-IDF...")
X_train_tfidf = vectorizer.fit_transform(X_train_text)
X_test_tfidf = vectorizer.transform(X_test_text)

scaler = StandardScaler()
X_train_lex_scaled = scaler.fit_transform(train_lex_df)
X_test_lex_scaled = scaler.transform(test_lex_df)

X_train_combined = hstack([X_train_tfidf, X_train_lex_scaled])
X_test_combined = hstack([X_test_tfidf, X_test_lex_scaled])

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