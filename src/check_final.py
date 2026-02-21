import pandas as pd

df = pd.read_csv("data/processed/final_dataset.csv")

print(df.head())
print("\nShape:", df.shape)
print("\nLabel distribution:")
print(df["label"].value_counts())

print("\nAny missing URLs?")
print(df["url"].isna().sum())