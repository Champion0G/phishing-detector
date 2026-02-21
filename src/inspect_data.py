import pandas as pd

print("----- PHISHTANK -----")
phish = pd.read_csv("data/raw/phishtank.csv")
print(phish.head())
print("Columns:", phish.columns)
print("Rows:", len(phish))

print("\n----- TRANGO -----")
benign = pd.read_csv("data/raw/tranco.csv", header=None)
print(benign.head())
print("Rows:", len(benign))

print("\n----- KAGGLE MIXED -----")
kaggle = pd.read_csv("data/raw/kaggle_mixed.csv")
print(kaggle.head())
print("Columns:", kaggle.columns)
print("Rows:", len(kaggle))