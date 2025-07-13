# filename: main.py
import os
import argparse
from datetime import datetime, timedelta
from typing import List
from enum import Enum
import pandas as pd
from sklearn.model_selection import train_test_split
import logging
import warnings

from src.data_collection.news_api import fetch_news
from src.data_collection.yfinance_data import fetch_stock_data, fetch_sp500_data
from src.data_collection.preprocess import preprocess_news, generate_sentiment_labels
from src.features.build_features import build_features
from src.evaluation.visualization import plot_combined_charts, plot_sentiment_distribution
from src.modeling.models import ModelType, SentimentModel
from src.modeling.naive_bayes import NaiveBayesModel
from src.modeling.bert import BertModel
from src.modeling.vader import VaderModel
from src.modeling.ensemble import EnsembleModel
from src.evaluation.backtest import backtest_strategy  # New import for backtest

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class ModelFactory:
    """
    Factory class for creating sentiment models based on ModelType.
    """
    @staticmethod
    def get_model(model_type: ModelType) -> SentimentModel:
        """
        Instantiate the appropriate model based on ModelType.
        """
        if model_type == ModelType.NAIVE_BAYES:
            return NaiveBayesModel()
        elif model_type == ModelType.BERT:
            return BertModel()
        elif model_type == ModelType.VADER:
            return VaderModel()
        elif model_type == ModelType.ENSEMBLE:
            return EnsembleModel()
        else:
            raise ValueError(f"Unknown model type: {model_type}")

def fetch_data(tickers: List[str], start_date: str, end_date: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Fetch news, stock, and S&P 500 data.
    
    Returns:
        Tuple of (news_df, stock_df, sp500_df)
    """
    logger.info("Fetching news data...")
    news_df = fetch_news(tickers, start_date, end_date)
    if news_df is None or news_df.empty:
        raise ValueError("Failed to fetch news data or no articles found.")
    
    logger.info("Fetching stock and S&P 500 data...")
    stock_df = fetch_stock_data(tickers, start_date, end_date)
    sp500_df = fetch_sp500_data(start_date, end_date)
    if stock_df is None or stock_df.empty or sp500_df is None or sp500_df.empty:
        raise ValueError("Failed to fetch stock or S&P 500 data.")
    
    return news_df, stock_df, sp500_df

def preprocess_and_label(news_df: pd.DataFrame, stock_df: pd.DataFrame, sp500_df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess news and generate sentiment labels, then merge them.
    
    Returns:
        Merged DataFrame ready for training.
    """
    logger.info("Preprocessing data...")
    processed_news_df = preprocess_news(news_df)
    labels_df = generate_sentiment_labels(stock_df, sp500_df)
    if processed_news_df.empty or labels_df.empty:
        raise ValueError("Preprocessing failed: Empty news or labels DataFrame.")
    
    logger.info("Merging data for training...")
    data = pd.merge(processed_news_df, labels_df, left_on=["date", "ticker"], right_on=["Date", "ticker"])
    data = data.dropna(subset=["cleaned_text"])
    if data.empty:
        raise ValueError("No valid data after preprocessing and merging.")
    
    nan_count = data["cleaned_text"].isna().sum()
    if nan_count > 0:
        logger.warning(f"Removed {nan_count} rows with NaN in cleaned_text")
    
    if len(data) < 10:
        raise ValueError(f"Dataset too small for training: {len(data)} samples")
    
    return data, labels_df

def train_and_evaluate(model_type: ModelType, data: pd.DataFrame, output_dir: str) -> dict:
    """
    Split data, train the model, and evaluate it.
    
    Returns:
        Dictionary of evaluation metrics.
    """
    X = data["cleaned_text"]
    y = data["sentiment"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    logger.info(f"Training {model_type.value} model...")
    model = ModelFactory.get_model(model_type)
    model.train(X_train, y_train, output_dir)
    
    logger.info(f"Evaluating {model_type.value} model...")
    metrics = model.evaluate(X_test, y_test, output_dir)
    
    return metrics

def generate_visualizations(tickers: List[str], stock_df: pd.DataFrame, sp500_df: pd.DataFrame, labels_df: pd.DataFrame, output_dir: str) -> None:
    """
    Generate and save visualizations.
    """
    logger.info("Generating visualizations...")
    for ticker in tickers:
        plot_combined_charts(stock_df, sp500_df, labels_df, ticker, output_dir=output_dir)
    plot_sentiment_distribution(labels_df, output_dir=output_dir)

def perform_backtest(tickers: List[str], model_type: ModelType, data: pd.DataFrame, stock_df: pd.DataFrame, output_dir: str) -> dict:
    """
    Perform backtesting for each ticker and collect metrics.
    
    Returns:
        Dictionary of backtest metrics per ticker.
    """
    logger.info("Performing backtesting...")
    backtest_results = {}
    for ticker in tickers:
        bt_metrics = backtest_strategy(ticker, model_type, data, stock_df, initial_cash=10000.0, output_dir=os.path.join(output_dir, "backtest"))
        backtest_results[ticker] = bt_metrics
    return backtest_results

def run_pipeline(tickers: List[str], start_date: str, end_date: str, model_type: ModelType, output_base_dir: str = "models") -> dict:
    """
    Run the full stock sentiment analysis pipeline for multiple tickers.

    Parameters:
    - tickers: List of stock tickers (e.g., ["AAPL", "MSFT", "GOOGL"])
    - start_date: Start date for data collection (YYYY-MM-DD)
    - end_date: End date for data collection (YYYY-MM-DD)
    - model_type: ModelType enum value (NAIVE_BAYES, BERT, VADER)
    - output_base_dir: Base directory to save model and visualization outputs
    
    Returns:
        Dictionary with evaluation metrics and backtest results.
    """
    try:
        # Create model-specific output directory
        output_dir = os.path.join(output_base_dir, model_type.value.lower())
        os.makedirs(output_dir, exist_ok=True)
        
        news_df, stock_df, sp500_df = fetch_data(tickers, start_date, end_date)
        data, labels_df = preprocess_and_label(news_df, stock_df, sp500_df)
        data = build_features(data, stock_df)
        metrics = train_and_evaluate(model_type, data, output_dir)
        generate_visualizations(tickers, stock_df, sp500_df, labels_df, output_dir)
        backtest_results = perform_backtest(tickers, model_type, data, stock_df, output_dir)
        
        logger.info("Pipeline completed successfully!")
        return {"metrics": metrics, "backtest_results": backtest_results}
    
    except ValueError as ve:
        logger.error(f"Value error in pipeline: {str(ve)}")
        raise
    except NotImplementedError as nie:
        logger.error(f"Implementation error: {str(nie)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run stock sentiment analysis pipeline.")
    parser.add_argument("--tickers", type=str, default="AAPL,MSFT,GOOGL", help="Comma-separated list of stock tickers")
    parser.add_argument("--start_date", type=str, default=(datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d"), help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, default=datetime.now().strftime("%Y-%m-%d"), help="End date (YYYY-MM-DD)")
    parser.add_argument("--model_type", type=str, default="ENSEMBLE", choices=[m.value for m in ModelType], help="Model type")
    parser.add_argument("--output_base_dir", type=str, default="models", help="Base output directory")
    
    args = parser.parse_args()
    
    tickers = args.tickers.split(",")
    model_type = ModelType[args.model_type.upper()]
    
    # Run pipeline
    run_pipeline(tickers, args.start_date, args.end_date, model_type, args.output_base_dir)