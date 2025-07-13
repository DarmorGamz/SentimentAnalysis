# filename: preprocess.py
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import os
import logging

nltk.download("punkt", quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download("stopwords", quiet=True)

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def clean_text(text):
    """
    Clean text data by removing special characters, lowercasing, and removing stopwords.
    """
    if not isinstance(text, str) or pd.isna(text):
        return ""
    text = re.sub(r"[^\w\s]", "", text.lower())
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

def preprocess_news(news_df, output_dir="data/processed/news_data"):
    """
    Preprocess news data, handling NaN values, filtering empty text, and save per ticker.
    """
    # Replace NaN in title and description with empty strings
    news_df["title"] = news_df["title"].fillna("")
    news_df["description"] = news_df["description"].fillna("")
    
    # Combine title and description, then clean
    news_df["cleaned_text"] = (news_df["title"] + " " + news_df["description"]).apply(clean_text)
    
    # Filter out rows with empty cleaned_text
    initial_len = len(news_df)
    news_df = news_df[news_df["cleaned_text"].str.strip() != ""]
    logger.info(f"Filtered out {initial_len - len(news_df)} rows with empty cleaned_text")
    
    # Select relevant columns
    processed_news_df = news_df[["date", "ticker", "cleaned_text"]]
    
    # Verify ticker column
    if "ticker" not in processed_news_df.columns:
        logger.error("Ticker column missing in processed news DataFrame")
        raise KeyError("Ticker column missing in processed news DataFrame")
    
    # Save per ticker
    for ticker in processed_news_df["ticker"].unique():
        df_t = processed_news_df[processed_news_df["ticker"] == ticker]
        if df_t.empty:
            continue
        ticker_dir = os.path.join(output_dir, ticker)
        os.makedirs(ticker_dir, exist_ok=True)
        output_path = os.path.join(ticker_dir, "processed_news.csv")
        df_t.to_csv(output_path, index=False)
        logger.info(f"Saved processed news for {ticker} to {output_path}")
    
    return processed_news_df

def generate_sentiment_labels(stock_df, sp500_df, output_dir="data/processed/sentiment_labels"):
    """
    Generate sentiment labels for multiple tickers based on stock performance vs. S&P 500 and save per ticker.
    """
    all_labels = []
    
    for ticker in stock_df["ticker"].unique():
        try:
            logger.info(f"Generating sentiment labels for {ticker}")
            ticker_df = stock_df[stock_df["ticker"] == ticker].copy()
            if ticker_df.empty:
                logger.warning(f"No stock data for {ticker}")
                continue
            
            ticker_df["pct_change"] = ticker_df["Close"].pct_change() * 100
            sp500_df["pct_change"] = sp500_df["Close"].pct_change() * 100
            
            merged_df = pd.merge(ticker_df[["Date", "pct_change"]], sp500_df[["Date", "pct_change"]],
                                 on="Date", suffixes=("_stock", "_sp500"))
            
            if merged_df.empty:
                logger.warning(f"No matching dates for {ticker} and S&P 500")
                continue
            
            def label_sentiment(row):
                stock_change = row["pct_change_stock"]
                sp500_change = row["pct_change_sp500"]
                if stock_change > sp500_change + 0.5:
                    return "bullish"
                elif stock_change < sp500_change - 0.5:
                    return "bearish"
                else:
                    return "neutral"
            
            merged_df["sentiment"] = merged_df.apply(label_sentiment, axis=1)
            merged_df["ticker"] = ticker
            merged_df = merged_df[["Date", "ticker", "sentiment"]]
            
            # Save per ticker
            ticker_dir = os.path.join(output_dir, ticker)
            os.makedirs(ticker_dir, exist_ok=True)
            output_path = os.path.join(ticker_dir, "sentiment_labels.csv")
            merged_df.to_csv(output_path, index=False)
            logger.info(f"Saved sentiment labels for {ticker} to {output_path}")
            
            all_labels.append(merged_df)
        
        except Exception as e:
            logger.error(f"Failed to generate labels for {ticker}: {str(e)}")
            continue
    
    if not all_labels:
        logger.error("No sentiment labels generated for any ticker.")
        return None
    
    labels_df = pd.concat(all_labels, ignore_index=True)
    
    # Verify ticker column
    if "ticker" not in labels_df.columns:
        logger.error("Ticker column missing in sentiment labels DataFrame")
        raise KeyError("Ticker column missing in sentiment labels DataFrame")
    
    return labels_df

if __name__ == "__main__":
    news_df = pd.read_csv("data/raw/news_data/news_2025-06-06_2025-07-06.csv")
    stock_df = pd.read_csv("data/raw/stock_prices/stock_prices_2025-06-06_2025-07-06.csv")
    sp500_df = pd.read_csv("data/raw/sp500_data/sp500_2025-06-06_2025-07-06.csv")
    processed_news = preprocess_news(news_df)
    labels = generate_sentiment_labels(stock_df, sp500_df)