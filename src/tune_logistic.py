import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

print("Loading dataset...")
df = pd.read_csv("data/processed/final_dataset.csv")

X = df["url"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3,5),
    max_features=15000
)

print("Fitting TF-IDF...")
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)

param_grid = {
    "C": [0.1, 0.5, 1, 2, 5]
}

print("Running GridSearch...")
grid = GridSearchCV(
    model,
    param_grid,
    cv=3,
    scoring="f1",
    verbose=2,
    n_jobs=-1
)

grid.fit(X_train_tfidf, y_train)

print("Best C:", grid.best_params_)

best_model = grid.best_estimator_

y_pred = best_model.predict(X_test_tfidf)
y_prob = best_model.predict_proba(X_test_tfidf)[:,1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("ROC-AUC:", roc_auc_score(y_test, y_prob))