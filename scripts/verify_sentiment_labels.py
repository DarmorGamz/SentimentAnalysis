import pandas as pd
labels_df = pd.read_csv("data/processed/sentiment_labels/sentiment_labels.csv")
print(labels_df.columns)
print(labels_df.head())