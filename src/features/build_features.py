# filename: build_features.py
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Financial sentiment lexicons
POSITIVE_WORDS = {'growth', 'profit', 'surge', 'rally', 'buy', 'strong', 'beat', 'rise', 'gain', 'upbeat', 'success', 'boom'}
NEGATIVE_WORDS = {'decline', 'loss', 'plunge', 'sell', 'weak', 'miss', 'downturn', 'drop', 'fall', 'crash', 'slump', 'warning'}

def extract_text_features(text_series: pd.Series) -> pd.DataFrame:
    """
    Extract features from text series.
    
    Returns:
        DataFrame with features: text_length, num_words, pos_count, neg_count
    """
    features = pd.DataFrame(index=text_series.index)
    features['text_length'] = text_series.str.len()
    features['num_words'] = text_series.apply(lambda x: len(word_tokenize(x)))
    
    def count_pos_words(text):
        tokens = set(word_tokenize(text.lower()))
        return len(tokens.intersection(POSITIVE_WORDS))
    
    def count_neg_words(text):
        tokens = set(word_tokenize(text.lower()))
        return len(tokens.intersection(NEGATIVE_WORDS))
    
    features['pos_count'] = text_series.apply(count_pos_words)
    features['neg_count'] = text_series.apply(count_neg_words)
    
    return features

def extract_stock_features(stock_df: pd.DataFrame, dates: pd.Series, tickers: pd.Series) -> pd.DataFrame:
    """
    Extract stock-related features for given dates and tickers.
    
    Returns:
        DataFrame with features: prev_pct_change, volume_change
    """
    features = pd.DataFrame(index=dates.index)
    features['prev_pct_change'] = np.nan
    features['volume_change'] = np.nan
    
    for ticker in stock_df['ticker'].unique():
        ticker_df = stock_df[stock_df['ticker'] == ticker].set_index('Date')
        ticker_df['pct_change'] = ticker_df['Close'].pct_change() * 100
        ticker_df['volume_change'] = ticker_df['Volume'].pct_change() * 100
        
        mask = (tickers == ticker)
        for idx in dates[mask].index:
            date = dates[idx]
            if date in ticker_df.index:
                prev_date_idx = ticker_df.index.get_loc(date) - 1
                if prev_date_idx >= 0:
                    features.at[idx, 'prev_pct_change'] = ticker_df.iloc[prev_date_idx]['pct_change']
                    features.at[idx, 'volume_change'] = ticker_df.iloc[prev_date_idx]['volume_change']
    
    return features

def build_features(data: pd.DataFrame, stock_df: pd.DataFrame, output_dir: str = "data/processed/features") -> pd.DataFrame:
    """
    Build and add features to the data DataFrame, save per ticker.
    
    Parameters:
    - data: Merged DataFrame with 'cleaned_text', 'date', 'ticker'
    - stock_df: Stock data DataFrame
    
    Returns:
        DataFrame with added features.
    """
    logger.info("Building features...")
    
    text_features = extract_text_features(data['cleaned_text'])
    stock_features = extract_stock_features(stock_df, data['date'], data['ticker'])
    
    features_df = pd.concat([text_features, stock_features], axis=1)
    enhanced_data = pd.concat([data, features_df], axis=1)
    
    # Fill NaN with 0 for stock features
    enhanced_data['prev_pct_change'] = enhanced_data['prev_pct_change'].fillna(0)
    enhanced_data['volume_change'] = enhanced_data['volume_change'].fillna(0)
    
    # Save per ticker
    for ticker in enhanced_data["ticker"].unique():
        df_t = enhanced_data[enhanced_data["ticker"] == ticker]
        if df_t.empty:
            continue
        ticker_dir = os.path.join(output_dir, ticker)
        os.makedirs(ticker_dir, exist_ok=True)
        output_path = os.path.join(ticker_dir, "features.csv")
        df_t.to_csv(output_path, index=False)
        logger.info(f"Saved features for {ticker} to {output_path}")
    
    return enhanced_data

if __name__ == "__main__":
    # Example usage
    data = pd.read_csv("path/to/merged_data.csv")  # Assume merged data from preprocess
    stock_df = pd.read_csv("data/raw/stock_prices/stock_prices.csv")
    enhanced_data = build_features(data, stock_df)