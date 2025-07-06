import pandas as pd
news_df = pd.read_csv("data/processed/news_data/processed_news.csv")
print(news_df.columns)
print(news_df.head())