from math import log
import os
import argparse
from datetime import datetime, timedelta
import pandas as pd
from sklearn.model_selection import train_test_split
import logging
import warnings
import torch
import torch.nn as nn
import numpy as np
from transformers import BertTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from typing import List

from src.data_collection.news_api import fetch_news
from src.data_collection.yfinance_data import fetch_stock_data, fetch_sp500_data
from src.data_collection.preprocess import preprocess_news
from src.features.build_features import build_features
from src.evaluation.visualization import plot_combined_charts, plot_sentiment_distribution
from src.evaluation.backtest import backtest_strategy

# Suppress warnings
warnings.filterwarnings("ignore")

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class FinBERTLSTM(nn.Module):
    def __init__(self, embedding_dim, price_features, hidden_size, num_layers, output_size=1):
        super().__init__()
        input_size = embedding_dim + price_features
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class LSTMEnsemble(nn.Module):
    def __init__(self, embedding_dim, price_features, hidden_sizes, num_layers):
        super().__init__()
        self.models = nn.ModuleList([FinBERTLSTM(embedding_dim, price_features, hs, num_layers) for hs in hidden_sizes])

    def forward(self, x):
        outputs = [m(x) for m in self.models]
        return torch.mean(torch.stack(outputs), dim=0)

class SequenceDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

def generate_finbert_sentiments_and_embeddings(data, model, tokenizer, device):
    sentiments = []
    embeddings = []
    model.eval()
    id2label = model.config.id2label
    with torch.no_grad():
        for text in data['cleaned_text']:
            if pd.isna(text) or not text:
                sentiments.append('neutral')
                embeddings.append(np.zeros(768))
                continue
            inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512).to(device)
            outputs = model(**inputs)
            logits = outputs.logits
            sentiment_idx = torch.argmax(logits, dim=1).item()
            sentiment = id2label[sentiment_idx]
            emb = model.bert(**inputs).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            sentiments.append(sentiment)
            embeddings.append(emb)
    data['sentiment'] = sentiments
    data['embeddings'] = embeddings
    return data

def aggregate_daily(data):
    def mean_embedding(embs):
        return np.mean(np.stack(embs), axis=0) if len(embs) > 0 else np.zeros(768)
    
    agg_df = data.groupby(['date', 'ticker']).agg({
        'embeddings': mean_embedding,
        'sentiment': lambda x: x.mode()[0] if not x.empty else 'neutral',
        'cleaned_text': ' '.join
    }).reset_index()
    return agg_df

def prepare_sequences(data, stock_df, seq_length=10):
    sequences = []
    targets = []
    tickers = data['ticker'].unique()
    for ticker in tickers:
        ticker_data = data[data['ticker'] == ticker].sort_values('date')
        ticker_stock = stock_df[stock_df['ticker'] == ticker].sort_values('Date')
        merged = pd.merge(ticker_data, ticker_stock, left_on='date', right_on='Date')
        for i in range(len(merged) - seq_length):
            seq_emb = np.stack(merged['embeddings'].iloc[i:i+seq_length])
            seq_prices = merged[['Open', 'High', 'Low', 'Close', 'Volume']].iloc[i:i+seq_length].values
            seq = np.concatenate([seq_emb, seq_prices], axis=1)
            target = merged['Close'].iloc[i+seq_length]
            sequences.append(seq)
            targets.append(target)
    return np.array(sequences), np.array(targets)

def train_lstm_ensemble(model, train_loader, optimizer, criterion, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        for seq, target in train_loader:
            seq, target = seq.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(seq.float())
            loss = criterion(output.squeeze(), target.float())
            loss.backward()
            optimizer.step()

def evaluate_lstm(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for seq, target in test_loader:
            seq, target = seq.to(device), target.to(device)
            output = model(seq.float())
            loss = criterion(output.squeeze(), target.float())
            total_loss += loss.item()
    return total_loss / len(test_loader)

def generate_visualizations(tickers: List[str], stock_df: pd.DataFrame, sp500_df: pd.DataFrame, labels_df: pd.DataFrame, output_dir: str) -> None:
    """
    Generate and save visualizations.
    """
    logger.info("Generating visualizations...")
    for ticker in tickers:
        plot_combined_charts(stock_df, sp500_df, labels_df, ticker, output_dir=output_dir)
    plot_sentiment_distribution(labels_df, output_dir=output_dir)

def run_hybrid_pipeline(tickers, start_date, end_date, output_base_dir="models"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    
    # Fetch data
    logger.info("Fetching data...")
    news_df = fetch_news(tickers, start_date, end_date, sources=["tickertick"])
    stock_df = fetch_stock_data(tickers, start_date, end_date)
    sp500_df = fetch_sp500_data(start_date, end_date)
    
    # Preprocess
    logger.info("Preprocessing data...")
    data = preprocess_news(news_df)
    data = data.dropna(subset=["cleaned_text"])
    
    # Load FinBERT
    logger.info("Loading FinBERT model...")
    tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')
    finbert_model = AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert').to(device)
    
    # Generate sentiments and embeddings
    logger.info("Generating sentiments and embeddings...")
    data = generate_finbert_sentiments_and_embeddings(data, finbert_model, tokenizer, device)
    
    # Aggregate daily
    logger.info("Aggregating daily data...")
    data = aggregate_daily(data)
    
    # Build features
    logger.info("Building features...")
    data = build_features(data, stock_df)
    
    # Prepare sequences
    logger.info("Preparing sequences...")
    sequences, targets = prepare_sequences(data, stock_df)
    
    # Split
    logger.info("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(sequences, targets, test_size=0.2)
    
    # Create datasets and dataloaders
    logger.info("Creating datasets and dataloaders...")
    train_dataset = SequenceDataset(torch.tensor(X_train), torch.tensor(y_train))
    test_dataset = SequenceDataset(torch.tensor(X_test), torch.tensor(y_test))
    
    logger.info(f"Train dataset size: {len(train_dataset)}, Test dataset size: {len(test_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Model
    logger.info("Initializing LSTM ensemble model...")
    embedding_dim = 768
    price_features = 5  # Open, High, Low, Close, Volume
    hidden_sizes = [128, 256, 512]
    num_layers = 2
    ensemble = LSTMEnsemble(embedding_dim, price_features, hidden_sizes, num_layers).to(device)
    
    optimizer = torch.optim.Adam(ensemble.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Train
    logger.info("Training LSTM ensemble model...")
    train_lstm_ensemble(ensemble, train_loader, optimizer, criterion, device)
    
    # Evaluate
    logger.info("Evaluating LSTM ensemble model...")
    metrics = {'mse': evaluate_lstm(ensemble, test_loader, criterion, device)}
    
    # Save
    logger.info("Saving model...")
    output_dir = os.path.join(output_base_dir, "finbert_lstm")
    os.makedirs(output_dir, exist_ok=True)
    torch.save(ensemble.state_dict(), os.path.join(output_dir, "model.pth"))
    
    # Visualizations and backtest
    logger.info("Generating visualizations and backtest results...")
    generate_visualizations(tickers, stock_df, sp500_df, data, output_dir)
    backtest_results = {}  # Implement adapted backtest if needed
    
    return {"metrics": metrics, "backtest_results": backtest_results}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", type=str, default="AAPL,MSFT,GOOGL")
    parser.add_argument("--start_date", type=str, default=(datetime.now() - timedelta(days=360)).strftime("%Y-%m-%d"))
    parser.add_argument("--end_date", type=str, default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--output_base_dir", type=str, default="models")
    
    args = parser.parse_args()
    tickers = args.tickers.split(",")
    run_hybrid_pipeline(tickers, args.start_date, args.end_date, args.output_base_dir)