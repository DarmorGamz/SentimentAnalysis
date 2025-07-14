# filename: src/evaluation/backtest.py
import os
import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
from src.modeling.models import ModelType, ModelFactory

def backtest_strategy(ticker: str, model_type: ModelType, data: pd.DataFrame, stock_df: pd.DataFrame, initial_cash: float = 10000.0, output_dir: str = None) -> dict:
    """
    Backtest trading strategy based on predicted sentiments for a ticker.

    Parameters:
    - ticker: Stock ticker
    - model_type: Type of model
    - data: DataFrame with news and sentiments
    - stock_df: Stock price DataFrame
    - initial_cash: Initial investment amount
    - output_dir: Directory to save results

    Returns:
    - Dictionary of backtest metrics
    """
    data_t = data[data['ticker'] == ticker]
    stock_t = stock_df[stock_df['ticker'] == ticker].set_index('Date')

    # Load trained model
    model = ModelFactory.get_model(model_type)
    model_dir = os.path.join('models', model_type.value.lower())
    model.load(model_dir)

    # Predict sentiments
    y_pred = model.predict(data_t['cleaned_text'])

    # Simulate trading
    positions = 0
    cash = initial_cash
    portfolio = []
    dates = sorted(set(data_t['date']) & set(stock_t.index))

    for date in dates:
        sent = y_pred[data_t['date'] == date].iloc[0]
        price = stock_t.loc[date, 'Close']
        if sent == 'bullish' and positions == 0:
            shares = cash // price
            positions += shares
            cash -= shares * price
        elif sent == 'bearish' and positions > 0:
            cash += positions * price
            positions = 0
        port_value = cash + positions * price
        portfolio.append({'date': date, 'portfolio': port_value})

    df_port = pd.DataFrame(portfolio)

    # Compute metrics
    if len(df_port) < 2:
        return {'total_return': 0, 'sharpe_ratio': 0}

    returns = df_port['portfolio'].pct_change().dropna()
    total_return = (df_port['portfolio'].iloc[-1] / initial_cash) - 1
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else 0

    metrics = {'total_return': total_return, 'sharpe_ratio': sharpe}

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f'{ticker}_backtest_metrics.json'), 'w') as f:
            json.dump(metrics, f)
        df_port.plot(x='date', y='portfolio', figsize=(10, 6))
        plt.title(f'Backtest Portfolio Value for {ticker}')
        plt.savefig(os.path.join(output_dir, f'{ticker}_portfolio.png'))
        plt.close()

    return metrics