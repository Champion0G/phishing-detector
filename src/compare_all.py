"""
compare_all.py — Master comparison script
Runs all 6 model variants and prints a final comparison table.

  Model                     | Split  | Acc  | Prec | Recall | F1   | ROC-AUC
  --------------------------|--------|------|------|--------|------|--------
  LightGBM (baseline)       | Random | ...  | ...  | ...    | ...  | ...
  LightGBM (baseline)       | Domain | ...  | ...  | ...    | ...  | ...
  LightGBM (enhanced feats) | Random | ...  | ...  | ...    | ...  | ...
  LightGBM (enhanced feats) | Domain | ...  | ...  | ...    | ...  | ...
  CNN+LSTM                  | Random | ...  | ...  | ...    | ...  | ...
  CNN+LSTM                  | Domain | ...  | ...  | ...    | ...  | ...
"""

import sys
import numpy as np
import pandas as pd
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, roc_auc_score, accuracy_score, f1_score
)
from scipy.sparse import hstack
from lightgbm import LGBMClassifier
from features import extract_features   # enhanced version

RESULTS = []


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def lgbm_features(tr_urls, te_urls):
    """Build TF-IDF + enhanced lexical feature matrix."""
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=15000)
    Xtr_t = vec.fit_transform(tr_urls)
    Xte_t = vec.transform(te_urls)

    tr_lex = pd.DataFrame(list(tr_urls.apply(extract_features)))
    te_lex = pd.DataFrame(list(te_urls.apply(extract_features)))

    sc = StandardScaler()
    Xtr_l = sc.fit_transform(tr_lex)
    Xte_l = sc.transform(te_lex)

    return hstack([Xtr_t, Xtr_l]), hstack([Xte_t, Xte_l])


def lgbm_baseline_features(tr_urls, te_urls):
    """Original 8-feature set for baseline comparison."""
    from features import (
        SUSPICIOUS_WORDS, _shannon_entropy, _max_consecutive_digits
    )
    import re
    from urllib.parse import urlparse as _up

    def _orig(url):
        p = _up(url)
        h = p.hostname or ""
        return {
            "url_length": len(url),
            "hostname_length": len(h),
            "num_digits": sum(c.isdigit() for c in url),
            "num_dots": url.count("."),
            "num_hyphens": url.count("-"),
            "num_special_chars": len(re.findall(r'[!@#$%^&*(),?":{}|<>]', url)),
            "has_ip": int(bool(re.match(r"\d+\.\d+\.\d+\.\d+", h))),
            "num_suspicious_words": sum(w in url.lower() for w in SUSPICIOUS_WORDS),
        }

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=15000)
    Xtr_t = vec.fit_transform(tr_urls)
    Xte_t = vec.transform(te_urls)

    tr_lex = pd.DataFrame(list(tr_urls.apply(_orig)))
    te_lex = pd.DataFrame(list(te_urls.apply(_orig)))

    sc = StandardScaler()
    Xtr_l = sc.fit_transform(tr_lex)
    Xte_l = sc.transform(te_lex)

    return hstack([Xtr_t, Xtr_l]), hstack([Xte_t, Xte_l])


def run_lgbm(X_train, X_test, y_train, y_test, label):
    print(f"  → Training: {label}")
    clf = LGBMClassifier(n_estimators=300, learning_rate=0.1,
                         num_leaves=31, random_state=42, verbose=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, y_pred, output_dict=True)
    RESULTS.append({
        "Model": label.split("|")[0].strip(),
        "Split": label.split("|")[1].strip(),
        "Accuracy":  round(report["accuracy"], 4),
        "Precision": round(report["weighted avg"]["precision"], 4),
        "Recall":    round(report["weighted avg"]["recall"], 4),
        "F1":        round(report["weighted avg"]["f1-score"], 4),
        "ROC-AUC":   round(roc_auc_score(y_test, y_prob), 4),
    })


# ═══════════════════════════════════════════════════════════════════════════════
#  Load data & build splits
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("  COMPARE ALL — Phishing Detector Model Comparison")
print("=" * 60)
print("\nLoading dataset...")
df = pd.read_csv("data/processed/final_dataset.csv")
df["domain"] = df["url"].apply(lambda u: urlparse(u).hostname)

# Random split indices
Xtr_r, Xte_r, ytr_r, yte_r = train_test_split(
    df["url"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

# Domain split
unique_domains = df["domain"].dropna().unique()
tr_doms, te_doms = train_test_split(unique_domains, test_size=0.2, random_state=42)
tr_df = df[df["domain"].isin(tr_doms)]
te_df = df[df["domain"].isin(te_doms)]


# ═══════════════════════════════════════════════════════════════════════════════
#  Option 1 — Baseline LightGBM (original 8 features)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[Option 1] Baseline LightGBM ...")
Xtr_b_r, Xte_b_r = lgbm_baseline_features(Xtr_r, Xte_r)
run_lgbm(Xtr_b_r, Xte_b_r, ytr_r, yte_r, "LightGBM Baseline | Random")

Xtr_b_d, Xte_b_d = lgbm_baseline_features(tr_df["url"], te_df["url"])
run_lgbm(Xtr_b_d, Xte_b_d, tr_df["label"], te_df["label"], "LightGBM Baseline | Domain")


# ═══════════════════════════════════════════════════════════════════════════════
#  Option 2 — Enhanced Features LightGBM (28 features)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[Option 2] Enhanced Features LightGBM ...")
Xtr_e_r, Xte_e_r = lgbm_features(Xtr_r, Xte_r)
run_lgbm(Xtr_e_r, Xte_e_r, ytr_r, yte_r, "LightGBM Enhanced | Random")

Xtr_e_d, Xte_e_d = lgbm_features(tr_df["url"], te_df["url"])
run_lgbm(Xtr_e_d, Xte_e_d, tr_df["label"], te_df["label"], "LightGBM Enhanced | Domain")


# ═══════════════════════════════════════════════════════════════════════════════
#  Option 3 — CNN + LSTM (character level)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[Option 3] CNN+LSTM (character-level) ...")
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        Embedding, Conv1D, GlobalMaxPooling1D, LSTM,
        Dense, Dropout, Bidirectional
    )
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.callbacks import EarlyStopping
    tf.get_logger().setLevel("ERROR")

    MAX_LEN    = 200
    VOCAB_SIZE = 128
    EMBED_DIM  = 32
    BATCH_SIZE = 512
    EPOCHS     = 10

    def encode(urls):
        seqs = [[min(ord(c), VOCAB_SIZE - 1) for c in u[:MAX_LEN]] for u in urls]
        return pad_sequences(seqs, maxlen=MAX_LEN, padding="post", truncating="post")

    def build_dl_model():
        m = Sequential([
            Embedding(VOCAB_SIZE, EMBED_DIM, input_length=MAX_LEN),
            Conv1D(128, 3, activation="relu", padding="same"),
            Bidirectional(LSTM(64)),
            Dropout(0.3),
            Dense(64, activation="relu"),
            Dropout(0.2),
            Dense(1, activation="sigmoid"),
        ])
        m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        return m

    def run_cnn(tr_urls, te_urls, y_tr, y_te, label):
        print(f"  → Training: {label}")
        es = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
        m = build_dl_model()
        m.fit(encode(tr_urls), y_tr,
              validation_split=0.1, epochs=EPOCHS,
              batch_size=BATCH_SIZE, callbacks=[es], verbose=0)
        y_prob = m.predict(encode(te_urls), verbose=0).flatten()
        y_pred = (y_prob >= 0.5).astype(int)
        report = classification_report(y_te, y_pred, output_dict=True)
        RESULTS.append({
            "Model":     label.split("|")[0].strip(),
            "Split":     label.split("|")[1].strip(),
            "Accuracy":  round(report["accuracy"], 4),
            "Precision": round(report["weighted avg"]["precision"], 4),
            "Recall":    round(report["weighted avg"]["recall"], 4),
            "F1":        round(report["weighted avg"]["f1-score"], 4),
            "ROC-AUC":   round(roc_auc_score(y_te, y_prob), 4),
        })

    run_cnn(Xtr_r.values, Xte_r.values, ytr_r.values, yte_r.values, "CNN+LSTM | Random")
    run_cnn(tr_df["url"].values, te_df["url"].values,
            tr_df["label"].values, te_df["label"].values, "CNN+LSTM | Domain")

except ImportError:
    print("  ⚠  TensorFlow not installed — skipping CNN+LSTM.")
    print("     Install with:  venv\\Scripts\\python.exe -m pip install tensorflow\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  Final comparison table
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  FINAL COMPARISON TABLE")
print("=" * 80)
results_df = pd.DataFrame(RESULTS)
print(results_df.to_string(index=False))
print("=" * 80)

# Highlight best F1 per split type
print("\n📊 Best F1 per split:")
for split in ["Random", "Domain"]:
    sub = results_df[results_df["Split"] == split]
    if not sub.empty:
        best = sub.loc[sub["F1"].idxmax()]
        print(f"  {split:8s}  →  {best['Model']:25s}  F1={best['F1']:.4f}  ROC-AUC={best['ROC-AUC']:.4f}")

print("\n📌 Key takeaway:")
print("  Random split == optimistic (domain overlap between train/test)")
print("  Domain split == realistic  (model sees entirely new domains)\n")
