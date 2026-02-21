import pandas as pd

print("Loading PhishTank...")
phish1 = pd.read_csv("data/raw/phishtank.csv")
phish1 = phish1[['url']]
phish1['label'] = 1

print("PhishTank:", len(phish1))

print("\nLoading Kaggle...")
kaggle = pd.read_csv("data/raw/kaggle_mixed.csv")

# Keep only phishing & benign
kaggle = kaggle[kaggle['type'].isin(['phishing', 'benign'])]

# Convert type to label
kaggle['label'] = kaggle['type'].apply(lambda x: 1 if x == 'phishing' else 0)

kaggle = kaggle[['url', 'label']]

print("Kaggle usable rows:", len(kaggle))

# Separate phishing and benign
phish2 = kaggle[kaggle['label'] == 1]
benign1 = kaggle[kaggle['label'] == 0]

print("Kaggle phishing:", len(phish2))
print("Kaggle benign:", len(benign1))

print("\nLoading Tranco...")
tranco = pd.read_csv("data/raw/tranco.csv", header=None)
tranco = tranco[[1]]
tranco.columns = ['url']
tranco['url'] = "https://" + tranco['url']
tranco['label'] = 0

print("Tranco total:", len(tranco))

# --- COMBINE PHISHING ---
all_phishing = pd.concat([phish1, phish2], ignore_index=True)
all_phishing.drop_duplicates(subset='url', inplace=True)

print("\nTotal unique phishing:", len(all_phishing))

# --- COMBINE BENIGN ---
all_benign = pd.concat([benign1, tranco], ignore_index=True)
all_benign.drop_duplicates(subset='url', inplace=True)

print("Total unique benign:", len(all_benign))

# --- BALANCE ---
phishing_count = len(all_phishing)

balanced_benign = all_benign.sample(n=phishing_count, random_state=42)

# Final dataset
final_df = pd.concat([all_phishing, balanced_benign], ignore_index=True)
final_df = final_df.sample(frac=1, random_state=42)

print("\nFinal dataset size:", len(final_df))
print(final_df['label'].value_counts())

# Save
final_df.to_csv("data/processed/final_dataset.csv", index=False)

print("\nSaved balanced dataset.")