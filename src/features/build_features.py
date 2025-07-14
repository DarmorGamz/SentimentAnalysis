# filename: src/features/build_features.py
import pandas as pd

def build_features(data: pd.DataFrame, stock_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build additional features for the dataset.

    Parameters:
    - data: Preprocessed DataFrame with news and labels
    - stock_df: Stock price DataFrame

    Returns:
    - DataFrame with added features
    """
    # Add text length feature
    data['text_length'] = data['cleaned_text'].str.len()

    # Add previous close price
    stock_df['previous_close'] = stock_df.groupby('ticker')['Close'].shift(1)
    data = pd.merge(data, stock_df[['Date', 'ticker', 'previous_close']], left_on=['date', 'ticker'], right_on=['Date', 'ticker'], how='left')

    return data