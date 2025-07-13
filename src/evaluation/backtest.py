import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
import shutil
from collections import Counter
from src.modeling.models import ModelType, SentimentModel
from src.modeling.naive_bayes import NaiveBayesModel
from src.modeling.bert import BertModel

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def get_model(model_type: ModelType) -> SentimentModel:
    """
    Instantiate the appropriate model based on ModelType.
    """
    if model_type == ModelType.NAIVE_BAYES:
        return NaiveBayesModel()
    elif model_type == ModelType.BERT:
        return BertModel()
    elif model_type == ModelType.VADER:
        raise NotImplementedError("VADER model is not yet implemented.")
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def get_signal(preds):
    if not preds:
        return 'neutral'
    counter = Counter(preds)
    return counter.most_common(1)[0][0]

def backtest_strategy(ticker: str, model_type: ModelType, data: pd.DataFrame, stock_df: pd.DataFrame, initial_cash: float = 10000.0, output_dir: str = 'backtest_results') -> dict:
    """
    Perform backtesting of the sentiment model for a single ticker without forward-looking bias.

    Parameters:
    - ticker: Stock ticker symbol.
    - model_type: Type of model to use (ModelType).
    - data: pd.DataFrame with 'date', 'ticker', 'cleaned_text', 'sentiment' (for training).
    - stock_df: pd.DataFrame with 'Date', 'ticker', 'Close' prices.
    - initial_cash: Starting cash amount.
    - output_dir: Directory to save results.

    Returns:
    - metrics: Dict with backtest metrics (final_value, total_return).
    """
    # Filter data for ticker
    ticker_data = data[data['ticker'] == ticker].copy()
    ticker_stock = stock_df[stock_df['ticker'] == ticker].copy()

    if ticker_data.empty or ticker_stock.empty:
        logger.warning(f"No data for {ticker}")
        return {'final_value': initial_cash, 'total_return': 0.0}

    # Ensure dates are datetime
    ticker_data['date'] = pd.to_datetime(ticker_data['date'])
    ticker_stock['Date'] = pd.to_datetime(ticker_stock['Date'])

    # Sort by date
    ticker_data = ticker_data.sort_values('date')
    ticker_stock = ticker_stock.sort_values('Date')

    # Unique dates
    unique_dates = sorted(ticker_data['date'].unique())

    if len(unique_dates) < 2:
        logger.warning(f"Insufficient dates for {ticker}")
        return {'final_value': initial_cash, 'total_return': 0.0}

    # Initialize portfolio
    cash = initial_cash
    shares = 0
    portfolio_values = []
    actions = []

    # Start from the date where we can train
    for i, date in enumerate(unique_dates[1:], start=1):
        # Past data for training
        past_dates = unique_dates[:i]
        past_data = ticker_data[ticker_data['date'].isin(past_dates)]

        if len(past_data) < 10:
            # Skip if too little data for training
            continue

        # Create and train model on past data
        model = get_model(model_type)
        X_past = past_data['cleaned_text']
        y_past = past_data['sentiment']
        temp_dir = os.path.join(output_dir, 'temp_models')
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)
        model.train(X_past, y_past, temp_dir)

        # Current day data
        current_data = ticker_data[ticker_data['date'] == date]
        if current_data.empty:
            continue

        # Predict sentiments for current day articles
        preds = model.predict(current_data['cleaned_text'])
        signal = get_signal(list(preds))

        # Get current price
        price_row = ticker_stock[ticker_stock['Date'] == date]
        if price_row.empty:
            continue
        price = price_row['Close'].iloc[0]

        # Execute trade
        action = 'hold'
        if signal == 'bullish' and shares == 0 and cash >= price:
            shares = 1
            cash -= price
            action = 'buy'
        elif signal == 'bearish' and shares > 0:
            shares = 0
            cash += price
            action = 'sell'

        # Record
        value = cash + shares * price
        portfolio_values.append((date, value))
        actions.append((date, action, signal, price))

    # Final value
    if portfolio_values:
        final_value = portfolio_values[-1][1]
    else:
        final_value = initial_cash
    total_return = ((final_value - initial_cash) / initial_cash) * 100 if initial_cash > 0 else 0.0

    # Metrics
    metrics = {
        'final_value': final_value,
        'total_return': total_return
    }

    # Save portfolio value plot
    if portfolio_values:
        os.makedirs(output_dir, exist_ok=True)
        df_port = pd.DataFrame(portfolio_values, columns=['Date', 'Value'])
        plt.figure(figsize=(10, 6))
        plt.plot(df_port['Date'], df_port['Value'], label='Portfolio Value')
        plt.title(f'Backtest Portfolio Value for {ticker} ({model_type.value})')
        plt.xlabel('Date')
        plt.ylabel('Value (USD)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f'{ticker}_backtest_portfolio.png')
        plt.savefig(plot_path)
        plt.close()
        logger.info(f'Saved backtest plot to {plot_path}')

    # Save actions to csv
    df_actions = pd.DataFrame(actions, columns=['Date', 'Action', 'Signal', 'Price'])
    actions_path = os.path.join(output_dir, f'{ticker}_backtest_actions.csv')
    df_actions.to_csv(actions_path, index=False)
    logger.info(f'Saved backtest actions to {actions_path}')

    # Save metrics
    metrics_path = os.path.join(output_dir, f'{ticker}_backtest_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write(f"Backtest Metrics for {ticker} ({model_type.value}):\n")
        f.write(f"Final Value: {final_value:.2f}\n")
        f.write(f"Total Return: {total_return:.2f}%\n")
    logger.info(f'Saved backtest metrics to {metrics_path}')

    return metrics

if __name__ == "__main__":
    # Example usage
    news_df = pd.read_csv("data/processed/news_data/processed_news.csv")
    labels_df = pd.read_csv("data/processed/sentiment_labels/sentiment_labels.csv")
    
    data = pd.merge(news_df, labels_df, left_on=["date", "ticker"], right_on=["Date", "ticker"])
    data = data.dropna(subset=["cleaned_text"])
    
    stock_df = pd.read_csv("data/raw/stock_prices/stock_prices_2025-06-06_2025-07-06.csv")  # Adjust path as needed
    
    tickers = data['ticker'].unique()
    model_type = ModelType.NAIVE_BAYES  # Use NAIVE_BAYES for speed; BERT would be slower
    
    for ticker in tickers:
        backtest_strategy(ticker, model_type, data, stock_df, output_dir="backtest_results/" + model_type.value.lower())