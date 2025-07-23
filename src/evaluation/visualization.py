# filename: src/evaluation/visualization.py
from math import log
import os
from venv import logger
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_combined_charts(stock_df: pd.DataFrame, sp500_df: pd.DataFrame, labels_df: pd.DataFrame, ticker: str, output_dir: str) -> None:
    """
    Plot combined chart of stock price, S&P500, and sentiment labels for a ticker.
    """
    stock_t = stock_df[stock_df['ticker'] == ticker].set_index('Date')
    sp500_df = sp500_df.set_index('Date')
    labels_t = labels_df[labels_df['ticker'] == ticker].set_index('date')
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(stock_t['Close'], label=f'{ticker} Close', color='blue')
    ax.plot(sp500_df['Close'], label='S&P500 Close', color='orange')

    colors = {'positive': 'green', 'neutral': 'yellow', 'negative': 'red'}
    for sent, color in colors.items():
        df_sent = labels_t[labels_t['sentiment'] == sent]
        if not df_sent.empty:
            intersection = df_sent.index.intersection(stock_t.index)
            ax.scatter(intersection, stock_t.loc[intersection, 'Close'], color=color, label=sent, s=50)

    stock_t.index = pd.to_datetime(stock_t.index)
    df_sent.index = pd.to_datetime(df_sent.index)
    sp500_df.index = pd.to_datetime(sp500_df.index)

    ax.legend()
    plt.title(f'{ticker} Price, S&P500, and Sentiment Labels')
    plt.xlabel('Date')
    plt.ylabel('Price')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'{ticker}_combined_chart.png'))
    plt.close()

# def plot_combined_charts(stock_df: pd.DataFrame, sp500_df: pd.DataFrame, labels_df: pd.DataFrame, ticker: str, output_dir: str) -> None:
#     """
#     Plot normalized stock price, normalized S&P500, and sentiment labels for a ticker.
#     """
#     stock_t = stock_df[stock_df['ticker'] == ticker].set_index('Date')
#     sp500_df = sp500_df.set_index('Date')
#     labels_t = labels_df[labels_df['ticker'] == ticker].set_index('date')

#     # Ensure datetime indices
#     stock_t.index = pd.to_datetime(stock_t.index)
#     sp500_df.index = pd.to_datetime(sp500_df.index)
#     labels_t.index = pd.to_datetime(labels_t.index)

#     # Align on common date range
#     common_dates = stock_t.index.intersection(sp500_df.index)
#     stock_t = stock_t.loc[common_dates]
#     sp500_t = sp500_df.loc[common_dates]

#     # Normalize by first value
#     stock_norm = stock_t['Close'] / stock_t['Close'].iloc[0]
#     sp500_norm = sp500_t['Close'] / sp500_t['Close'].iloc[0]

#     fig, ax = plt.subplots(figsize=(12, 6))
#     ax.plot(stock_norm, label=f'{ticker} (normalized)', color='blue')
#     ax.plot(sp500_norm, label='S&P500 (normalized)', color='orange')

#     # Sentiment scatter points (on normalized stock line)
#     colors = {'positive': 'green', 'neutral': 'yellow', 'negative': 'red'}
#     for sent, color in colors.items():
#         df_sent = labels_t[labels_t['sentiment'] == sent]
#         if not df_sent.empty:
#             intersection = df_sent.index.intersection(stock_t.index)
#             ax.scatter(intersection, stock_norm.loc[intersection], color=color, label=sent, s=50)

#     ax.legend()
#     plt.title(f'{ticker} (Normalized) vs S&P500 and Sentiment Labels')
#     plt.xlabel('Date')
#     plt.ylabel('Normalized Price')
#     os.makedirs(output_dir, exist_ok=True)
#     plt.savefig(os.path.join(output_dir, f'{ticker}_combined_chart_normalized.png'))
#     plt.close()

def plot_sentiment_distribution(labels_df: pd.DataFrame, output_dir: str) -> None:
    """
    Plot distribution of sentiment labels.
    """
    plt.figure(figsize=(8, 6))
    sns.countplot(x='sentiment', data=labels_df)
    plt.title('Sentiment Distribution')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'sentiment_distribution.png'))
    plt.close()