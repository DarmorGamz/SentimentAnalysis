# filename: models.py
from enum import Enum
from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional

class ModelType(Enum):
    NAIVE_BAYES = "naive_bayes"
    BERT = "bert"
    VADER = "vader"
    ENSEMBLE = "ensemble"
    FINANCIALBERT = "financial_bert"

class SentimentModel(ABC):
    """
    Abstract base class for sentiment analysis models.
    """
    @abstractmethod
    def train(self, X: pd.Series, y: pd.Series, output_dir: str) -> None:
        """
        Train the model on the given data.

        Parameters:
        - X: Series of cleaned text data
        - y: Series of sentiment labels (bullish, neutral, bearish)
        - output_dir: Directory to save the trained model
        """
        pass

    @abstractmethod
    def predict(self, X: pd.Series) -> pd.Series:
        """
        Predict sentiment labels for the given text data.

        Parameters:
        - X: Series of cleaned text data

        Returns:
        - Series of predicted sentiment labels
        """
        pass

    @abstractmethod
    def evaluate(self, X: pd.Series, y: pd.Series, output_dir: str) -> dict:
        """
        Evaluate the model and save metrics.

        Parameters:
        - X: Series of cleaned text data
        - y: Series of true sentiment labels
        - output_dir: Directory to save metrics and plots

        Returns:
        - Dictionary of evaluation metrics (e.g., accuracy, precision, recall, f1)
        """
        pass