import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import joblib

print("Loading dataset...")
df = pd.read_csv("data/processed/final_dataset.csv")

# -----------------------------
# STEP 1: Take 50k subset
# -----------------------------
# df = df.sample(n=50000, random_state=42)

print("Subset size:", len(df))

# -----------------------------
# STEP 2: Split data
# -----------------------------
X = df["url"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train size:", len(X_train))
print("Test size:", len(X_test))

# -----------------------------
# STEP 3: TF-IDF (Character level)
# -----------------------------
vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3,5),
    max_features=15000
)

print("Fitting TF-IDF...")
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("TF-IDF shape:", X_train_tfidf.shape)

# -----------------------------
# STEP 4: Train Logistic Regression
# -----------------------------
model = LogisticRegression(max_iter=1000)

print("Training model...")
model.fit(X_train_tfidf, y_train)

# -----------------------------
# STEP 5: Evaluate
# -----------------------------
y_pred = model.predict(X_test_tfidf)
y_prob = model.predict_proba(X_test_tfidf)[:,1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("ROC-AUC:", roc_auc_score(y_test, y_prob))

# -----------------------------
# STEP 6: Save Model
# -----------------------------
joblib.dump(model, "model_baseline.joblib")
joblib.dump(vectorizer, "tfidf_baseline.joblib")

print("\nBaseline model saved.")
