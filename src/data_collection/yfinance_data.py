import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def fetch_stock_data(tickers, start_date, end_date, output_dir="data/raw/stock_prices"):
    """
    Fetch stock price data for multiple tickers using yfinance.

    Parameters:
    - tickers: List of stock tickers (e.g., ["AAPL", "MSFT", "GOOGL"])
    - start_date: Start date for data collection (YYYY-MM-DD)
    - end_date: End date for data collection (YYYY-MM-DD)
    - output_dir: Directory to save the output CSV file

    Returns:
    - DataFrame with columns: Date, ticker, Open, High, Low, Close, Volume
    """
    all_data = []
    
    for ticker in tickers:
        try:
            logger.info(f"Fetching data for {ticker} from {start_date} to {end_date}")
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)
            
            if df.empty:
                logger.warning(f"No data found for {ticker}")
                continue
            
            df.reset_index(inplace=True)
            df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
            df["ticker"] = ticker
            df = df[["Date", "ticker", "Open", "High", "Low", "Close", "Volume"]]
            all_data.append(df)
            logger.info(f"Successfully fetched data for {ticker}")
        
        except Exception as e:
            logger.error(f"Failed to fetch data for {ticker}: {str(e)}")
            continue
    
    if not all_data:
        logger.error("No stock data found for any ticker.")
        return None
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Save to CSV
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"stock_prices_{start_date}_{end_date}.csv")
    combined_df.to_csv(output_path, index=False)
    logger.info(f"Saved stock data to {output_path}")
    
    return combined_df

def fetch_sp500_data(start_date, end_date, output_dir="data/raw/sp500_data"):
    """
    Fetch S&P 500 data as benchmark.

    Parameters:
    - start_date: Start date for data collection (YYYY-MM-DD)
    - end_date: End date for data collection (YYYY-MM-DD)
    - output_dir: Directory to save the output CSV file

    Returns:
    - DataFrame with columns: Date, Open, High, Low, Close, Volume
    """
    try:
        logger.info(f"Fetching S&P 500 data from {start_date} to {end_date}")
        sp500 = yf.Ticker("^GSPC")
        df = sp500.history(start=start_date, end=end_date)
        
        if df.empty:
            logger.error("No data found for S&P 500")
            return None
        
        df.reset_index(inplace=True)
        df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
        df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
        
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"sp500_{start_date}_{end_date}.csv")
        df.to_csv(output_path, index=False)
        logger.info(f"Saved S&P 500 data to {output_path}")
        return df
    
    except Exception as e:
        logger.error(f"Failed to fetch S&P 500 data: {str(e)}")
        return None

if __name__ == "__main__":
    TICKERS = ["AAPL", "MSFT", "GOOGL"]
    END_DATE = datetime.now().strftime("%Y-%m-%d")
    START_DATE = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    fetch_stock_data(TICKERS, START_DATE, END_DATE)
    fetch_sp500_data(START_DATE, END_DATE)