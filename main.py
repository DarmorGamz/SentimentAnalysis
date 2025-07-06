import os
from datetime import datetime, timedelta
from src.data_collection.news_api import fetch_news
from src.data_collection.yfinance_data import fetch_stock_data, fetch_sp500_data
from src.data_collection.preprocess import preprocess_news, generate_sentiment_labels
from src.evaluation.visualization import plot_combined_charts, plot_sentiment_distribution
from src.modeling.models import ModelType, SentimentModel
from src.modeling.naive_bayes import NaiveBayesModel
import pandas as pd
from sklearn.model_selection import train_test_split
import logging
import warnings

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
        pass
    elif model_type == ModelType.VADER:
        pass
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def run_pipeline(tickers, start_date, end_date, model_type: ModelType, output_base_dir="models"):
    """
    Run the full stock sentiment analysis pipeline for multiple tickers.

    Parameters:
    - tickers: List of stock tickers (e.g., ["AAPL", "MSFT", "GOOGL"])
    - start_date: Start date for data collection (YYYY-MM-DD)
    - end_date: End date for data collection (YYYY-MM-DD)
    - model_type: ModelType enum value (NAIVE_BAYES, BERT, VADER)
    - output_base_dir: Base directory to save model and visualization outputs
    """
    try:
        # Create model-specific output directory
        output_dir = os.path.join(output_base_dir, model_type.value)
        
        # Step 1: Fetch news data
        logger.info("Fetching news data...")
        news_df = fetch_news(tickers, start_date, end_date)
        if news_df is None or news_df.empty:
            raise ValueError("Failed to fetch news data or no articles found.")
        
        # Step 2: Fetch stock and S&P 500 data
        logger.info("Fetching stock and S&P 500 data...")
        stock_df = fetch_stock_data(tickers, start_date, end_date)
        sp500_df = fetch_sp500_data(start_date, end_date)
        if stock_df is None or stock_df.empty or sp500_df is None or sp500_df.empty:
            raise ValueError("Failed to fetch stock or S&P 500 data.")
        
        # Step 3: Preprocess news and generate sentiment labels
        logger.info("Preprocessing data...")
        processed_news_df = preprocess_news(news_df)
        labels_df = generate_sentiment_labels(stock_df, sp500_df)
        if processed_news_df.empty or labels_df.empty:
            raise ValueError("Preprocessing failed: Empty news or labels DataFrame.")
        
        # Step 4: Merge data for training
        logger.info("Merging data for training...")
        data = pd.merge(processed_news_df, labels_df, left_on=["date", "ticker"], right_on=["Date", "ticker"])
        if data["cleaned_text"].isna().any():
            data = data.dropna(subset=["cleaned_text"])
            logger.warning(f"Removed {data['cleaned_text'].isna().sum()} rows with NaN in cleaned_text")
        
        if len(data) < 10:
            raise ValueError(f"Dataset too small for training: {len(data)} samples")
        
        X = data["cleaned_text"]
        y = data["sentiment"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Step 5: Train and evaluate model
        logger.info(f"Training {model_type.value} model...")
        model = get_model(model_type)
        model.train(X_train, y_train, output_dir)
        
        logger.info(f"Evaluating {model_type.value} model...")
        metrics = model.evaluate(X_test, y_test, output_dir)
        
        # Step 6: Generate visualizations
        logger.info("Generating visualizations...")
        for ticker in tickers:
            plot_combined_charts(stock_df, sp500_df, labels_df, ticker, output_dir=output_dir)
        plot_sentiment_distribution(labels_df, output_dir=output_dir)
        
        logger.info("Pipeline completed successfully!")
        return metrics
    
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    # Configuration
    TICKERS = ["AAPL", "MSFT", "GOOGL"]
    END_DATE = datetime.now().strftime("%Y-%m-%d")
    START_DATE = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    MODEL_TYPE = ModelType.NAIVE_BAYES  # Change to BERT or VADER as needed
    
    # Run pipeline
    run_pipeline(TICKERS, START_DATE, END_DATE, MODEL_TYPE)