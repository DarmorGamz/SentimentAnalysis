import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def plot_combined_charts(stock_df, sp500_df, labels_df, ticker, output_dir="models/naive_bayes"):
    """
    Plot stock price, S&P 500 price, and sentiment scores for a specific ticker in a single vertically stacked figure.

    Parameters:
    - stock_df: DataFrame with stock prices (columns: Date, ticker, Close, etc.)
    - sp500_df: DataFrame with S&P 500 prices (columns: Date, Close, etc.)
    - labels_df: DataFrame with sentiment labels (columns: Date, ticker, sentiment)
    - ticker: Stock ticker to plot (e.g., 'AAPL')
    - output_dir: Directory to save the plot
    """
    # Validate input DataFrames
    required_stock_cols = ["Date", "ticker", "Close"]
    required_sp500_cols = ["Date", "Close"]
    required_labels_cols = ["Date", "ticker", "sentiment"]
    
    if not all(col in stock_df.columns for col in required_stock_cols):
        logger.error(f"Missing required columns in stock_df: {stock_df.columns}")
        raise KeyError(f"Missing required columns in stock_df: {stock_df.columns}")
    if not all(col in sp500_df.columns for col in required_sp500_cols):
        logger.error(f"Missing required columns in sp500_df: {sp500_df.columns}")
        raise KeyError(f"Missing required columns in sp500_df: {sp500_df.columns}")
    if not all(col in labels_df.columns for col in required_labels_cols):
        logger.error(f"Missing required columns in labels_df: {labels_df.columns}")
        raise KeyError(f"Missing required columns in labels_df: {labels_df.columns}")
    
    # Filter stock and labels for the specific ticker
    ticker_stock_df = stock_df[stock_df["ticker"] == ticker].copy()
    ticker_labels_df = labels_df[labels_df["ticker"] == ticker].copy()
    
    if ticker_stock_df.empty:
        logger.warning(f"No stock data for {ticker}")
        return
    if ticker_labels_df.empty:
        logger.warning(f"No sentiment labels for {ticker}")
        return
    
    # Map sentiment labels to numerical scores
    sentiment_map = {"bullish": 1, "neutral": 0, "bearish": -1}
    ticker_labels_df["sentiment_score"] = ticker_labels_df["sentiment"].map(sentiment_map)
    
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # Plot 1: Stock Price
    ax1.plot(ticker_stock_df["Date"], ticker_stock_df["Close"], color="blue", label=f"{ticker} Closing Price")
    ax1.set_title(f"{ticker} Stock Price Over Time")
    ax1.set_ylabel("Closing Price (USD)")
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: S&P 500 Price
    ax2.plot(sp500_df["Date"], sp500_df["Close"], color="green", label="S&P 500 Closing Price")
    ax2.set_title("S&P 500 Price Over Time")
    ax2.set_ylabel("Closing Price (USD)")
    ax2.legend()
    ax2.grid(True)
    
    # Plot 3: Sentiment Scores
    ax3.plot(ticker_labels_df["Date"], ticker_labels_df["sentiment_score"], color="purple", marker="o", label="Sentiment Score")
    ax3.set_title("Sentiment Scores Over Time")
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Sentiment Score")
    ax3.set_yticks([-1, 0, 1])
    ax3.set_yticklabels(["Bearish", "Neutral", "Bullish"])
    ax3.legend()
    ax3.grid(True)
    
    # Rotate x-axis labels for readability
    plt.xticks(rotation=45)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save the combined plot
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{ticker}_combined_plots.png")
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Saved combined plots for {ticker} to {output_path}")

def plot_sentiment_distribution(labels_df, output_dir="models/naive_bayes"):
    """
    Plot a bar chart of sentiment label distribution across all tickers.

    Parameters:
    - labels_df: DataFrame with sentiment labels (columns: Date, ticker, sentiment)
    - output_dir: Directory to save the plot
    """
    if "sentiment" not in labels_df.columns:
        logger.error(f"Missing 'sentiment' column in labels_df: {labels_df.columns}")
        raise KeyError(f"Missing 'sentiment' column in labels_df: {labels_df.columns}")
    
    sentiment_counts = labels_df["sentiment"].value_counts()
    
    plt.figure(figsize=(8, 6))
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette=["#4CAF50", "#FFC107", "#F44336"])
    plt.title("Sentiment Label Distribution (All Tickers)")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "sentiment_distribution.png")
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Saved sentiment distribution plot to {output_path}")

if __name__ == "__main__":
    # Load data
    stock_df = pd.read_csv("data/raw/stock_prices/stock_prices_2025-06-06_2025-07-06.csv")
    sp500_df = pd.read_csv("data/raw/sp500_data/sp500_2025-06-06_2025-07-06.csv")
    labels_df = pd.read_csv("data/processed/sentiment_labels/sentiment_labels.csv")
    
    # Generate combined plots for each ticker
    tickers = stock_df["ticker"].unique()
    for ticker in tickers:
        plot_combined_charts(stock_df, sp500_df, labels_df, ticker)
    plot_sentiment_distribution(labels_df)