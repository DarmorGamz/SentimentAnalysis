import pandas as pd
news_df = pd.read_csv("data/processed/news_data/processed_news.csv")
print(news_df[news_df["cleaned_text"].isna()])