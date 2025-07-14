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
    def predict_proba(self, X: pd.Series) -> pd.DataFrame:
        """
        Predict sentiment probabilities for the given text data.

        Parameters:
        - X: Series of cleaned text data

        Returns:
        - DataFrame of predicted sentiment probabilities with columns ['bullish', 'neutral', 'bearish']
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

    @abstractmethod
    def load(self, output_dir: str) -> None:
        """
        Load the trained model from the given directory.

        Parameters:
        - output_dir: Directory to load the trained model from
        """
        pass

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
            from src.modeling.naive_bayes import NaiveBayesModel
            return NaiveBayesModel()
        elif model_type == ModelType.BERT:
            from src.modeling.bert import BertModel
            return BertModel()
        elif model_type == ModelType.VADER:
            from src.modeling.vader import VaderModel
            return VaderModel()
        elif model_type == ModelType.ENSEMBLE:
            from src.modeling.ensemble import EnsembleModel
            return EnsembleModel()
        elif model_type == ModelType.FINANCIALBERT:
            from src.modeling.financial_bert import FinancialBertModel
            return FinancialBertModel()
        else:
            raise ValueError(f"Unknown model type: {model_type}")