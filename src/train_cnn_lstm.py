"""
Option 3 — train_cnn_lstm.py
Character-level CNN + LSTM deep learning model for phishing URL detection.
Evaluates under both random split and domain split.

Requirements:
    pip install tensorflow   (if not already installed)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from features import url_to_sequence, _get_hostname

# ── Try importing TensorFlow ───────────────────────────────────────────────────
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
except ImportError:
    raise ImportError(
        "TensorFlow is not installed. Run:\n"
        "  venv\\Scripts\\python.exe -m pip install tensorflow\n"
    )

# ── Config ─────────────────────────────────────────────────────────────────────
MAX_LEN   = 200       # max URL character length
VOCAB_SIZE = 128      # printable ASCII range
EMBED_DIM  = 32
BATCH_SIZE = 512
EPOCHS     = 10

def encode_urls(urls):
    seqs = [url_to_sequence(u) for u in urls]
    return pad_sequences(seqs, maxlen=MAX_LEN, padding="post", truncating="post")


# ── Model factory ──────────────────────────────────────────────────────────────
def build_model():
    model = Sequential([
        Embedding(input_dim=VOCAB_SIZE, output_dim=EMBED_DIM, input_length=MAX_LEN),
        Conv1D(128, kernel_size=3, activation="relu", padding="same"),
        Bidirectional(LSTM(64, return_sequences=False)),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def evaluate(model, X_test, y_test, label=""):
    y_prob = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)
    print(f"\n=== Option 3 — CNN+LSTM | {label} ===")
    print(classification_report(y_test, y_pred))
    auc = roc_auc_score(y_test, y_prob)
    print("ROC-AUC:", round(auc, 4))
    return y_pred, y_prob


# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv("data/processed/final_dataset.csv")

es = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)

# ══════════════════════════
#  RANDOM SPLIT
# ══════════════════════════
print("\n[1/2] Training CNN+LSTM — Random Split ...")
X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(
    df["url"].values, df["label"].values,
    test_size=0.2, random_state=42, stratify=df["label"].values
)
X_tr_enc_r = encode_urls(X_tr_r)
X_te_enc_r = encode_urls(X_te_r)

model_r = build_model()
model_r.fit(
    X_tr_enc_r, y_tr_r,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[es],
    verbose=1,
)
evaluate(model_r, X_te_enc_r, y_te_r, label="Random Split")

# ══════════════════════════
#  DOMAIN SPLIT
# ══════════════════════════
print("\n[2/2] Training CNN+LSTM — Domain Split ...")
df["domain"] = df["url"].apply(_get_hostname)
unique_domains = df["domain"].unique()
tr_doms, te_doms = train_test_split(unique_domains, test_size=0.2, random_state=42)
tr_df = df[df["domain"].isin(tr_doms)]
te_df = df[df["domain"].isin(te_doms)]

print(f"  Train: {len(tr_df):,}  |  Test: {len(te_df):,}")
X_tr_enc_d = encode_urls(tr_df["url"].values)
X_te_enc_d = encode_urls(te_df["url"].values)

model_d = build_model()
model_d.fit(
    X_tr_enc_d, tr_df["label"].values,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[es],
    verbose=1,
)
evaluate(model_d, X_te_enc_d, te_df["label"].values, label="Domain Split")
