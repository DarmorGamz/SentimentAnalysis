# filename: news_api.py
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
import os
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def fetch_news_from_newsapi(tickers, from_date, to_date):
    """
    Fetch news articles from NewsAPI for multiple tickers.

    Returns:
        List of DataFrames, one per ticker.
    """
    api_key = os.getenv("NEWSAPI_APIKEY")
    if not api_key:
        logger.error("NEWSAPI_APIKEY not found in .env file")
        raise ValueError("NEWSAPI_APIKEY not found in .env file")
    
    url = "https://newsapi.org/v2/everything"
    all_dfs = []
    
    for ticker in tickers:
        query = f"{ticker} OR {ticker_to_company.get(ticker, ticker)}"
        params = {
            "q": query,
            "from": from_date,
            "to": to_date,
            "apiKey": api_key,
            "language": "en",
            "sortBy": "relevancy",
            "pageSize": 100
        }
        
        try:
            logger.info(f"Fetching NewsAPI for {ticker} from {from_date} to {to_date}")
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            articles = response.json().get("articles", [])
            if not articles:
                logger.warning(f"No articles found for {ticker} in NewsAPI.")
                continue
            
            data = []
            for article in articles:
                article_data = {
                    "date": article.get("publishedAt", "").split("T")[0] if article.get("publishedAt") else "",
                    "title": article.get("title", ""),
                    "description": article.get("description", ""),
                    "content": article.get("content", ""),
                    "ticker": ticker,
                    "source": "newsapi"
                }
                data.append(article_data)
            
            df_ticker = pd.DataFrame(data)
            all_dfs.append(df_ticker)
        
        except Exception as e:
            logger.error(f"Error fetching NewsAPI for {ticker}: {str(e)}")
            continue
    
    return all_dfs

def fetch_news_from_tickertick(tickers, from_date, to_date):
    """
    Fetch news articles from TickerTick for multiple tickers using direct API calls.

    Returns:
        List of DataFrames, one per ticker.
    """
    all_dfs = []
    from_dt = datetime.strptime(from_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    to_dt = datetime.strptime(to_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'application/json',
        'Referer': 'https://www.tickertick.com/'
    }
    base_url = "https://api.tickertick.com/feed"
    
    for ticker in tickers:
        stories = []
        last_id = None
        try:
            logger.info(f"Fetching TickerTick for {ticker} from {from_date} to {to_date}")
            while True:
                params = {
                    'q': f'(and tt:{ticker.lower()} s:sec)',
                    'n': 999
                }
                if last_id:
                    params['last'] = last_id
                
                response = requests.get(base_url, params=params, headers=headers)
                response.raise_for_status()
                data = response.json()
                
                feed = data.get('stories', [])
                if not feed:
                    break
                
                for story in feed:
                    story_time = datetime.fromtimestamp(story['time'] / 1000, tz=timezone.utc)
                    if story_time < from_dt:
                        break  # Stop if older than from_date
                    if from_dt <= story_time <= to_dt:
                        article_data = {
                            "date": story_time.strftime("%Y-%m-%d"),
                            "title": story.get('title', ''),
                            "description": story.get('description', ''),
                            "content": '',  # No content field, use description if needed
                            "ticker": ticker,
                            "source": "tickertick"
                        }
                        stories.append(article_data)
                else:
                    last_id = data.get('last_id')
                    continue
                break  # Break outer loop if stopped due to date
            
            if not stories:
                logger.warning(f"No articles found for {ticker} in TickerTick.")
                continue
            
            df_ticker = pd.DataFrame(stories)
            all_dfs.append(df_ticker)
        
        except Exception as e:
            logger.error(f"Error fetching TickerTick for {ticker}: {str(e)}")
            continue
    
    return all_dfs

def fetch_news(tickers, from_date, to_date, sources=['newsapi', 'tickertick'], output_dir="data/raw/news_data"):
    """
    Fetch news articles from specified sources for multiple tickers and save per ticker.

    Parameters:
    - tickers: List of stock tickers (e.g., ["AAPL", "MSFT", "GOOGL"])
    - from_date: Start date for news (YYYY-MM-DD)
    - to_date: End date for news (YYYY-MM-DD)
    - sources: List of sources to fetch from ('newsapi', 'tickertick')
    - output_dir: Base directory to save the output CSV files per ticker

    Returns:
    - Combined DataFrame with columns: date, title, description, content, ticker, source
    """
    all_dfs = []
    
    if 'newsapi' in sources:
        all_dfs.extend(fetch_news_from_newsapi(tickers, from_date, to_date))
    
    if 'tickertick' in sources:
        all_dfs.extend(fetch_news_from_tickertick(tickers, from_date, to_date))
    
    if not all_dfs:
        logger.error("No articles found for any ticker from any source.")
        return None
    
    # Combine all
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Ensure date is not empty
    combined_df = combined_df[combined_df["date"].str.strip() != ""]
    
    # Validate columns
    required_columns = ["date", "title", "description", "content", "ticker", "source"]
    missing_cols = [col for col in required_columns if col not in combined_df.columns]
    if missing_cols:
        logger.error(f"Missing columns in news DataFrame: {missing_cols}")
        raise KeyError(f"Missing columns in news DataFrame: {missing_cols}")
    
    # Save per ticker
    for ticker in tickers:
        df_ticker = combined_df[combined_df['ticker'] == ticker]
        if df_ticker.empty:
            continue
        ticker_dir = os.path.join(output_dir, ticker)
        os.makedirs(ticker_dir, exist_ok=True)
        output_path = os.path.join(ticker_dir, f"news_{from_date}_{to_date}.csv")
        df_ticker.to_csv(output_path, index=False)
        logger.info(f"Saved news data for {ticker} to {output_path}")
    
    return combined_df

# Map tickers to company names for better news coverage (for NewsAPI)
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