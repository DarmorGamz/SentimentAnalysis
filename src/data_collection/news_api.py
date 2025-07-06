import requests
import pandas as pd
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def fetch_news(tickers, from_date, to_date, output_dir="data/raw/news_data"):
    """
    Fetch news articles from NewsAPI for multiple tickers.

    Parameters:
    - tickers: List of stock tickers (e.g., ["AAPL", "MSFT", "GOOGL"])
    - from_date: Start date for news (YYYY-MM-DD)
    - to_date: End date for news (YYYY-MM-DD)
    - output_dir: Directory to save the output CSV file

    Returns:
    - DataFrame with columns: date, title, description, content, ticker
    """
    api_key = os.getenv("NEWSAPI_APIKEY")
    if not api_key:
        logger.error("NEWSAPI_APIKEY not found in .env file")
        raise ValueError("NEWSAPI_APIKEY not found in .env file")
    
    url = "https://newsapi.org/v2/everything"
    all_data = []
    
    for ticker in tickers:
        # Use ticker and company name for broader coverage
        query = f"{ticker} OR {ticker_to_company.get(ticker, ticker)}"
        params = {
            "q": query,
            "from": from_date,
            "to": to_date,
            "apiKey": api_key,
            "language": "en",
            "sortBy": "relevancy",
            "pageSize": 100  # Max per request in free tier
        }
        
        try:
            logger.info(f"Fetching news for {ticker} from {from_date} to {to_date}")
            response = requests.get(url, params=params)
            response.raise_for_status()  # Raise exception for bad status codes
            
            articles = response.json().get("articles", [])
            if not articles:
                logger.warning(f"No articles found for {ticker}.")
                continue
            
            # Create list of dictionaries with ticker column
            data = []
            for article in articles:
                article_data = {
                    "date": article.get("publishedAt", "").split("T")[0] if article.get("publishedAt") else "",
                    "title": article.get("title", ""),
                    "description": article.get("description", ""),
                    "content": article.get("content", ""),
                    "ticker": ticker
                }
                data.append(article_data)
            
            all_data.extend(data)
            logger.info(f"Successfully fetched {len(data)} articles for {ticker}")
        
        except requests.RequestException as e:
            logger.error(f"API request failed for {ticker}: {str(e)}")
            continue
        except Exception as e:
            logger.error(f"Unexpected error while fetching news for {ticker}: {str(e)}")
            continue
    
    if not all_data:
        logger.error("No articles found for any ticker.")
        return None
    
    # Create DataFrame and validate columns
    df = pd.DataFrame(all_data)
    required_columns = ["date", "title", "description", "content", "ticker"]
    if not all(col in df.columns for col in required_columns):
        missing_cols = [col for col in required_columns if col not in df.columns]
        logger.error(f"Missing columns in news DataFrame: {missing_cols}")
        raise KeyError(f"Missing columns in news DataFrame: {missing_cols}")
    
    # Ensure date is not empty
    df = df[df["date"].str.strip() != ""]
    
    # Save to CSV
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"news_{from_date}_{to_date}.csv")
    df.to_csv(output_path, index=False)
    logger.info(f"Saved news data to {output_path}")
    
    return df

# Map tickers to company names for better news coverage
ticker_to_company = {
    "AAPL": "Apple Inc",
    "MSFT": "Microsoft Corporation",
    "GOOGL": "Alphabet Inc"
}

if __name__ == "__main__":
    TICKERS = ["AAPL", "MSFT", "GOOGL"]
    TO_DATE = datetime.now().strftime("%Y-%m-%d")
    FROM_DATE = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    fetch_news(TICKERS, FROM_DATE, TO_DATE)